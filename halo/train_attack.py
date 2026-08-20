from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .attacks.eval import (
    dump_badnets_triggered_jsonl,
    dump_triggered_padding_jsonl,
    evaluate_badnets_attack,
    evaluate_padding_attack,
)
from .attacks.prototypes import build_layerwise_prototypes, online_update_layer_protos_and_radii
from .attacks.triggers import RNNPadGenerator, UniversalPadTrigger, apply_badnets_insert_batch, attack_mask, parse_insert_lens
from .checkpoints import model_kwargs_from_args, save_classifier, save_joint, state_dict_from_ckpt
from .config import FeatureConfig
from .data import (
    FlowBatch,
    FlowDataset,
    build_label_map,
    build_poison_source_ids,
    collate_flows,
    dump_train_subset_jsonl,
)
from .features import apply_dir_sign, classifier_inputs, compute_state_ids
from .models import FSNetClassifier
from .utils import evaluate_clean, get_device, set_seed


def _load_or_build_model(args, config, num_classes: int, device: torch.device):
    if not args.clean_model_path:
        model = FSNetClassifier(
            config.num_states,
            num_classes,
            d_model=args.d_model,
            backbone=args.backbone,
            hidden_dim=args.rnn_hidden,
            num_layers=args.rnn_layers,
            dropout=args.dropout,
            max_pkt_len=config.max_pkt_len,
            transformer_heads=args.transformer_heads,
            cnn_kernel_size=args.cnn_kernel_size,
        ).to(device)
        return model, config

    if not os.path.isfile(args.clean_model_path):
        raise FileNotFoundError(args.clean_model_path)
    ckpt = torch.load(args.clean_model_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    clean_config = FeatureConfig(
        use_dir=not bool(ckpt_args.get("no_dir", getattr(args, "no_dir", False))),
        use_dt_bucket=not bool(ckpt_args.get("no_dt_bucket", getattr(args, "no_dt_bucket", False))),
        max_pkt_len=float(ckpt_args.get("max_pkt_len", config.max_pkt_len)),
    )
    kwargs = model_kwargs_from_args(ckpt_args)
    kwargs["max_pkt_len"] = clean_config.max_pkt_len
    model = FSNetClassifier(clean_config.num_states, num_classes, **kwargs).to(device)
    model.load_state_dict(state_dict_from_ckpt(ckpt))
    print(f"[Model] initialized from clean checkpoint: {args.clean_model_path}")
    return model, clean_config


def _build_generator(args, config):
    if args.attack == "badnets":
        return None
    if args.attack in {"trojanflow", "halo"}:
        return RNNPadGenerator(
            input_dim=4,
            hidden_dim=args.gen_hidden,
            num_layers=args.gen_layers,
            max_pkt_len=config.max_pkt_len,
            only_server=args.only_server,
            rnn_type=args.gen_rnn_type,
        )
    if args.attack == "uap":
        return UniversalPadTrigger(
            max_len=args.max_len,
            max_pkt_len=config.max_pkt_len,
            only_server=args.only_server,
            uap_max_bytes=args.uap_max_bytes,
            uap_init_bytes=args.uap_init_bytes,
            uap_integer=args.uap_integer,
        )
    raise ValueError(f"Unknown attack: {args.attack}")


def _uap_delta_summary(trigger: UniversalPadTrigger) -> str:
    with torch.no_grad():
        delta = trigger.delta_bytes()
        kth = max(1, int(0.95 * delta.numel()))
        return f"delta_mean={delta.mean().item():.4f} delta_p95={delta.kthvalue(kth).values.item():.4f}"


def _preoptimize_uap(args, model, trigger, train_loader, config, device, target_class, poison_source_ids) -> None:
    if args.attack != "uap" or args.uap_opt_epochs <= 0:
        return
    print(f"[UAP] pre-optimizing trigger for {args.uap_opt_epochs} epochs on frozen classifier")
    for p in model.parameters():
        p.requires_grad_(False)
    model.train()
    trigger.train()
    opt = torch.optim.Adam(trigger.parameters(), lr=args.uap_lr if args.uap_lr > 0 else args.lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.uap_opt_epochs + 1):
        total = 0
        loss_sum = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            poison = attack_mask(batch.labels, target_class, poison_source_ids, poison_rate=1.0)
            if not poison.any():
                continue
            pad = trigger(batch.lengths, batch.times, batch.dirs, batch.can_modify, batch.mask)
            pad = pad * poison.unsqueeze(1).float()
            lengths_trig = torch.clamp(batch.lengths + pad, max=config.max_pkt_len)
            state_ids = compute_state_ids(batch.times, batch.dirs, batch.mask, config, device)
            lengths_signed = apply_dir_sign(lengths_trig, batch.dirs, config)
            logits = model(state_ids, lengths_signed, batch.mask)
            target_labels = torch.full_like(batch.labels, int(target_class))
            attack_loss = criterion(logits[poison], target_labels[poison])
            denom = (batch.mask.float() * poison.unsqueeze(1).float()).sum() + 1e-8
            pad_loss = args.lambda_pad * (pad.sum() / denom / config.max_pkt_len)
            loss = attack_loss + pad_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trigger.parameters(), args.grad_clip)
            opt.step()
            total += batch.labels.size(0)
            loss_sum += loss.item() * batch.labels.size(0)
        print(f"[UAP] preopt_epoch={epoch} loss={loss_sum / max(1,total):.4f} {_uap_delta_summary(trigger)}")
    for p in model.parameters():
        p.requires_grad_(True)
    if args.freeze_uap_after_opt:
        for p in trigger.parameters():
            p.requires_grad_(False)
        print("[UAP] trigger frozen after pre-optimization")


def train_attack(args) -> None:
    set_seed(args.seed)
    label2id, id2label, label_key = build_label_map(args.train_jsonl, args.task, args.app_min, args.app_max)
    poison_source_ids = build_poison_source_ids(args.poison_source_labels, label2id, label_key)
    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = FlowDataset(args.train_jsonl, label_key, label2id, args.max_len)
    valid_ds = FlowDataset(args.valid_jsonl, label_key, label2id, args.max_len)
    dump_train_subset_jsonl(
        train_ds,
        id2label,
        os.path.join(args.save_dir, "train_subset_per_class.jsonl"),
        args.subset_per_class,
        args.seed,
    )
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_flows)
    valid_loader = DataLoader(valid_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_flows)
    proto_loader = DataLoader(train_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_flows)

    device = get_device(args.cpu)
    config = FeatureConfig.from_args(args)
    model, config = _load_or_build_model(args, config, len(label2id), device)
    generator = _build_generator(args, config)
    if generator is not None:
        generator = generator.to(device)
    print(f"[Info] device={device} attack={args.attack} feature_config={config}")

    insert_lens = parse_insert_lens(args.bd_insert_lens)
    layer_protos = layer_radii = None
    if args.attack == "halo":
        layer_protos, layer_radii = build_layerwise_prototypes(
            model,
            proto_loader,
            config,
            device,
            target_class=args.target_class,
            num_prototypes=args.num_target_prototypes,
            max_samples=args.max_proto_samples,
            radius_percentile=args.proto_radius_percentile,
            core_fraction=args.proto_core_fraction,
            density_fraction=args.proto_density_fraction,
            micro_cluster_factor=args.proto_micro_cluster_factor,
            seed=args.seed,
        )

    _preoptimize_uap(args, model, generator, train_loader, config, device, args.target_class, poison_source_ids)

    params = list(model.parameters())
    if generator is not None:
        params += [p for p in generator.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_score = float("inf") if args.best_metric == "align" else -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        if args.attack == "halo" and args.proto_refresh_interval > 0 and epoch % args.proto_refresh_interval == 0:
            layer_protos, layer_radii = build_layerwise_prototypes(
                model,
                proto_loader,
                config,
                device,
                target_class=args.target_class,
                num_prototypes=args.num_target_prototypes,
                max_samples=args.max_proto_samples,
                radius_percentile=args.proto_radius_percentile,
                core_fraction=args.proto_core_fraction,
                density_fraction=args.proto_density_fraction,
                micro_cluster_factor=args.proto_micro_cluster_factor,
                seed=args.seed + epoch,
            )

        model.train()
        if generator is not None:
            generator.train()
        sums = dict(total=0, loss=0.0, clean=0.0, poison=0.0, pad=0.0, align=0.0, repulse=0.0, poison_examples=0)

        for step, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)
            state_ids, lengths_signed, mask = classifier_inputs(batch, config, device)

            if args.attack == "halo":
                logits_clean, feats_clean = model.forward_with_features(state_ids, lengths_signed, mask)
            else:
                logits_clean = model(state_ids, lengths_signed, mask)
                feats_clean = None
            clean_loss = criterion(logits_clean, batch.labels)
            loss = clean_loss
            poison_loss = torch.tensor(0.0, device=device)
            pad_loss = torch.tensor(0.0, device=device)
            align_loss = torch.tensor(0.0, device=device)
            repulse_loss = torch.tensor(0.0, device=device)

            poison = attack_mask(batch.labels, args.target_class, poison_source_ids, args.poison_rate)

            if args.attack == "halo" and args.lambda_repulse > 0:
                preds_clean = logits_clean.argmax(dim=1)
                non_target = (batch.labels != args.target_class) & (preds_clean == batch.labels)
                if non_target.any():
                    terms = []
                    for layer_id, feat in enumerate(feats_clean):
                        h = feat[non_target]
                        protos = layer_protos[layer_id].to(device)
                        dist = torch.sqrt(((h.unsqueeze(1) - protos.unsqueeze(0)) ** 2).sum(dim=-1) + 1e-8)
                        nearest_dist, nearest_proto = dist.min(dim=1)
                        radii = layer_radii[layer_id].to(device)
                        margin = radii[nearest_proto] + args.repulse_delta
                        terms.append(torch.relu(margin - nearest_dist).mean())
                    if terms:
                        repulse_loss = args.lambda_repulse * torch.stack(terms).mean()
                        loss = loss + repulse_loss

                if args.proto_online_momentum > 0:
                    target_clean = batch.labels == args.target_class
                    if target_clean.any():
                        layer_protos, layer_radii = online_update_layer_protos_and_radii(
                            layer_protos,
                            layer_radii,
                            [feat[target_clean].detach() for feat in feats_clean],
                            momentum=args.proto_online_momentum,
                            radius_percentile=args.proto_radius_percentile,
                        )

            if args.poison_rate > 0 and poison.any():
                target_labels = torch.full_like(batch.labels, int(args.target_class))
                if args.attack == "badnets":
                    res = apply_badnets_insert_batch(
                        batch.lengths,
                        batch.times,
                        batch.dirs,
                        batch.can_modify,
                        batch.mask,
                        poison,
                        insert_lens=insert_lens,
                        after_kth_server=args.bd_after_kth_server,
                        dt_mode=args.bd_dt_mode,
                        dt_const=args.bd_dt_const,
                        max_pkt_len=config.max_pkt_len,
                    )
                    effective = poison & res.applied
                    if effective.any():
                        trig_batch = FlowBatch(res.lengths, res.times, res.dirs, res.can_modify, res.mask, batch.labels)
                        state_trig, len_trig_signed, mask_trig = classifier_inputs(trig_batch, config, device)
                        logits_trig = model(state_trig, len_trig_signed, mask_trig)
                        poison_loss = args.lambda_poison * criterion(logits_trig[effective], target_labels[effective])
                        loss = loss + poison_loss
                else:
                    pad = torch.clamp(generator(batch.lengths, batch.times, batch.dirs, batch.can_modify, batch.mask), min=0.0)
                    pad = pad * poison.unsqueeze(1).float()
                    lengths_trig = torch.clamp(batch.lengths + pad, max=config.max_pkt_len)
                    lengths_trig_signed = apply_dir_sign(lengths_trig, batch.dirs, config)
                    if args.attack == "halo":
                        logits_trig, feats_trig = model.forward_with_features(state_ids, lengths_trig_signed, mask)
                    else:
                        logits_trig = model(state_ids, lengths_trig_signed, mask)
                        feats_trig = None
                    poison_loss = args.lambda_poison * criterion(logits_trig[poison], target_labels[poison])
                    denom = (batch.mask.float() * poison.unsqueeze(1).float()).sum() + 1e-8
                    pad_loss = args.lambda_pad * (pad.sum() / denom / config.max_pkt_len)
                    loss = loss + poison_loss + pad_loss

                    if args.attack == "halo" and args.lambda_align > 0:
                        terms = []
                        for layer_id, feat in enumerate(feats_trig):
                            h = feat[poison]
                            protos = layer_protos[layer_id].to(device)
                            if protos.size(0) == 1:
                                center = protos[0].unsqueeze(0).expand_as(h)
                            else:
                                dist2 = ((h.unsqueeze(1) - protos.unsqueeze(0)) ** 2).sum(dim=-1)
                                weights = torch.softmax(-dist2 / max(args.proto_soft_tau, 1e-6), dim=1)
                                center = (weights.unsqueeze(-1) * protos.unsqueeze(0)).sum(dim=1)
                            terms.append(nn.functional.mse_loss(h, center))
                        if terms:
                            align_loss = args.lambda_align * torch.stack(terms).mean()
                            loss = loss + align_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            bsz = batch.labels.size(0)
            sums["total"] += bsz
            sums["loss"] += loss.item() * bsz
            sums["clean"] += clean_loss.item() * bsz
            sums["poison"] += poison_loss.item() * bsz
            sums["pad"] += pad_loss.item() * bsz
            sums["align"] += align_loss.item() * bsz
            sums["repulse"] += repulse_loss.item() * bsz
            sums["poison_examples"] += int(poison.sum().item())

            if args.log_interval > 0 and step % args.log_interval == 0:
                msg = (
                    f"[Train] epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={sums['loss']/max(1,sums['total']):.4f} "
                    f"clean={sums['clean']/max(1,sums['total']):.4f} "
                    f"poison={sums['poison']/max(1,sums['total']):.4f} "
                    f"align={sums['align']/max(1,sums['total']):.4f} "
                    f"repulse={sums['repulse']/max(1,sums['total']):.4f} "
                    f"pad={sums['pad']/max(1,sums['total']):.6f} "
                    f"poison_frac={sums['poison_examples']/max(1,sums['total']):.3f}"
                )
                if args.attack == "uap":
                    msg += " " + _uap_delta_summary(generator)
                print(msg)

        val_loss, val_acc = evaluate_clean(model, valid_loader, config, device)
        avg_align = sums["align"] / max(1, sums["total"])
        print(
            f"[Eval] epoch={epoch} train_loss={sums['loss']/max(1,sums['total']):.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} avg_align={avg_align:.4f}"
        )

        if args.save_every_epoch:
            save_joint(
                os.path.join(args.save_dir, f"joint_epoch_{epoch:03d}.pt"),
                model,
                generator,
                label2id,
                id2label,
                args,
                epoch=epoch,
                val_acc=val_acc,
                align_loss=avg_align,
            )

        current_score = avg_align if args.best_metric == "align" else val_acc
        improved = current_score < best_score if args.best_metric == "align" else current_score > best_score
        if improved:
            best_score = current_score
            best_epoch = epoch
            save_joint(
                os.path.join(args.save_dir, "best_joint_model.pt"),
                model,
                generator,
                label2id,
                id2label,
                args,
                epoch=epoch,
                val_acc=val_acc,
                align_loss=avg_align,
            )
            save_classifier(
                os.path.join(args.save_dir, "best_classifier_model.pt"),
                model,
                label2id,
                id2label,
                args,
                epoch=epoch,
                val_acc=val_acc,
                align_loss=avg_align,
            )
            if generator is not None:
                torch.save(
                    {"state_dict": generator.state_dict(), "args": vars(args), "epoch": epoch, "val_acc": val_acc},
                    os.path.join(args.save_dir, "best_generator_model.pt"),
                )
            print(f"[Save] best updated by {args.best_metric}: epoch={epoch} score={current_score:.4f}")

    print(f"[Result] best_epoch={best_epoch} best_{args.best_metric}={best_score:.4f}")
    _evaluate_and_dump(args, model, generator, config, label2id, id2label, label_key, poison_source_ids, insert_lens, device)


def _evaluate_and_dump(args, model, generator, config, label2id, id2label, label_key, poison_source_ids, insert_lens, device):
    if not args.test_jsonl:
        return
    test_ds = FlowDataset(args.test_jsonl, label_key, label2id, args.max_len)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_flows)
    clean_loss, clean_acc = evaluate_clean(model, test_loader, config, device)
    print(f"[Test-Clean] clean_loss={clean_loss:.4f} clean_acc={clean_acc:.4f}")
    if args.attack == "badnets":
        metrics = evaluate_badnets_attack(
            model,
            test_loader,
            config,
            device,
            target_class=args.target_class,
            poison_source_ids=poison_source_ids,
            insert_lens=insert_lens,
            after_kth_server=args.bd_after_kth_server,
            dt_mode=args.bd_dt_mode,
            dt_const=args.bd_dt_const,
        )
    else:
        metrics = evaluate_padding_attack(model, generator, test_loader, config, device, args.target_class, poison_source_ids)
    atk_loss, asr, _triggered_samples = metrics
    print(f"[Test-Attack] attack_loss={atk_loss:.4f} asr={asr:.4f}")
    if args.attack == "badnets":
        dump_badnets_triggered_jsonl(
            test_loader,
            config,
            device,
            id2label,
            os.path.join(args.save_dir, "test_triggered_poisoned.jsonl"),
            args.target_class,
            poison_source_ids,
            insert_lens=insert_lens,
            after_kth_server=args.bd_after_kth_server,
            dt_mode=args.bd_dt_mode,
            dt_const=args.bd_dt_const,
        )
    else:
        dump_triggered_padding_jsonl(
            generator,
            test_loader,
            config,
            device,
            id2label,
            os.path.join(args.save_dir, "test_triggered_poisoned.jsonl"),
            args.target_class,
            poison_source_ids,
        )


def build_parser(default_attack: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train FS-Net backdoor attacks.")
    if default_attack in {None, "halo"}:
        default_lambda_pad = 0e-8
    elif default_attack == "trojanflow":
        default_lambda_pad = 3e1
    elif default_attack == "uap":
        default_lambda_pad = 0.0
    else:
        default_lambda_pad = 1e-4
    p.add_argument("--attack", choices=["badnets", "trojanflow", "uap", "halo"], default=default_attack or "halo")
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--valid_jsonl", required=True)
    p.add_argument("--test_jsonl", default=None)
    p.add_argument("--task", choices=["vpn", "service", "app"], required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--clean_model_path", default=None)
    p.add_argument("--app_min", type=int, default=None)
    p.add_argument("--app_max", type=int, default=None)
    p.add_argument("--max_len", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--rnn_hidden", type=int, default=128)
    p.add_argument("--rnn_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_pkt_len", type=float, default=1500.0)
    p.add_argument("--backbone", choices=["gru", "lstm", "cnn", "transformer"], default="gru")
    p.add_argument("--transformer_heads", type=int, default=4)
    p.add_argument("--cnn_kernel_size", type=int, default=3)
    p.add_argument("--gen_rnn_type", choices=["gru", "lstm"], default="lstm")
    p.add_argument("--gen_hidden", type=int, default=64)
    p.add_argument("--gen_layers", type=int, default=1)
    p.add_argument("--only_server", action="store_true")
    p.add_argument("--no_dir", action="store_true")
    p.add_argument("--no_dt_bucket", action="store_true")
    p.add_argument("--target_class", type=int, default=0)
    p.add_argument("--poison_rate", type=float, default=0.1)
    p.add_argument("--poison_source_labels", default="")
    p.add_argument("--lambda_poison", type=float, default=1.0)
    if default_attack == "halo":
        p.add_argument("--lambda_pad", type=float, default=default_lambda_pad)
    else:
        p.set_defaults(lambda_pad=default_lambda_pad)
    p.add_argument("--lambda_align", type=float, default=1.0)
    p.add_argument("--lambda_repulse", type=float, default=0.0)
    p.add_argument("--repulse_delta", type=float, default=0.2)
    p.add_argument("--num_target_prototypes", type=int, default=4)
    p.add_argument("--max_proto_samples", type=int, default=5000)
    p.add_argument("--proto_soft_tau", type=float, default=0.5)
    p.add_argument("--proto_refresh_interval", type=int, default=0)
    p.add_argument("--proto_online_momentum", type=float, default=0.0)
    p.add_argument("--proto_radius_percentile", type=float, default=80.0)
    p.add_argument("--proto_core_fraction", type=float, default=0.7)
    p.add_argument("--proto_density_fraction", type=float, default=0.3)
    p.add_argument("--proto_micro_cluster_factor", type=int, default=4)
    p.add_argument("--bd_insert_lens", default="400,500,600")
    p.add_argument("--bd_after_kth_server", type=int, default=2)
    p.add_argument("--bd_dt_mode", choices=["split", "const"], default="split")
    p.add_argument("--bd_dt_const", type=float, default=1e-3)
    p.add_argument("--uap_max_bytes", type=float, default=256.0)
    p.add_argument("--uap_init_bytes", type=float, default=0.0)
    p.add_argument("--uap_integer", action="store_true")
    p.add_argument("--uap_opt_epochs", type=int, default=0)
    p.add_argument("--uap_lr", type=float, default=0.0)
    p.add_argument("--freeze_uap_after_opt", action="store_true")
    p.add_argument("--best_metric", choices=["val_acc", "align"], default=None)
    p.add_argument("--save_every_epoch", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--subset_per_class", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--log_interval", type=int, default=50)
    return p


def main(default_attack: str | None = None) -> None:
    args = build_parser(default_attack=default_attack).parse_args()
    if default_attack is not None:
        args.attack = default_attack
    elif args.attack == "halo":
        args.lambda_pad = 0e-8
    elif args.attack == "trojanflow":
        args.lambda_pad = 3e1
    elif args.attack == "uap":
        args.lambda_pad = 0.0
    else:
        args.lambda_pad = 1e-4
    if args.best_metric is None:
        args.best_metric = "align" if args.attack == "halo" else "val_acc"
    train_attack(args)


if __name__ == "__main__":
    main()
