from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from .checkpoints import save_classifier
from .config import FeatureConfig
from .data import FlowDataset, build_label_map, collate_flows, dump_train_subset_jsonl
from .features import classifier_inputs
from .models import FSNetClassifier
from .utils import evaluate_clean, get_device, set_seed


def train_clean(args) -> None:
    set_seed(args.seed)
    label2id, id2label, label_key = build_label_map(args.train_jsonl, args.task, args.app_min, args.app_max)
    os.makedirs(args.save_dir, exist_ok=True)
    train_ds = FlowDataset(args.train_jsonl, label_key, label2id, args.max_len)
    valid_ds = FlowDataset(args.valid_jsonl, label_key, label2id, args.max_len)
    dump_train_subset_jsonl(
        train_ds,
        id2label,
        os.path.join(args.save_dir, "train_subset_per_class.jsonl"),
        samples_per_class=args.subset_per_class,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_flows
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_flows
    )
    device = get_device(args.cpu)
    config = FeatureConfig.from_args(args)
    print(f"[Info] device={device} feature_config={config}")
    model = FSNetClassifier(
        config.num_states,
        len(label2id),
        d_model=args.d_model,
        backbone=args.backbone,
        hidden_dim=args.rnn_hidden,
        num_layers=args.rnn_layers,
        dropout=args.dropout,
        max_pkt_len=config.max_pkt_len,
        transformer_heads=args.transformer_heads,
        cnn_kernel_size=args.cnn_kernel_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = -1.0
    best_epoch = -1
    checkpoint_name = (
        "final_classifier_model.pt" if args.checkpoint_policy == "last" else "best_classifier_model.pt"
    )
    checkpoint_path = os.path.join(args.save_dir, checkpoint_name)
    final_val_acc = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        loss_sum = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)
            state_ids, lengths_signed, mask = classifier_inputs(batch, config, device)
            logits = model(state_ids, lengths_signed, mask)
            loss = criterion(logits, batch.labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            total += batch.labels.size(0)
            loss_sum += loss.item() * batch.labels.size(0)
            if args.log_interval > 0 and step % args.log_interval == 0:
                print(f"[Train] epoch={epoch} step={step}/{len(train_loader)} loss={loss_sum / max(1,total):.4f}")

        val_loss, val_acc = evaluate_clean(model, valid_loader, config, device)
        final_val_acc = val_acc
        print(
            f"[Eval] epoch={epoch} train_loss={loss_sum / max(1,total):.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if args.checkpoint_policy == "best_val" and val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            save_classifier(
                checkpoint_path,
                model,
                label2id,
                id2label,
                args,
                epoch=epoch,
                val_acc=val_acc,
                num_classes=len(label2id),
            )
            print(f"[Save] best classifier updated at epoch={epoch}")
        elif args.checkpoint_policy == "last" and epoch == args.epochs:
            save_classifier(
                checkpoint_path,
                model,
                label2id,
                id2label,
                args,
                epoch=epoch,
                val_acc=val_acc,
                num_classes=len(label2id),
            )
            print(f"[Save] final classifier saved at fixed epoch={epoch}")

    if args.checkpoint_policy == "best_val":
        print(f"[Result] best_val_acc={best_acc:.4f} epoch={best_epoch}")
    else:
        print(f"[Result] checkpoint_policy=last epoch={args.epochs} val_acc={final_val_acc:.4f}")
    if args.test_jsonl:
        test_ds = FlowDataset(args.test_jsonl, label_key, label2id, args.max_len)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_flows
        )
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        test_loss, test_acc = evaluate_clean(model, test_loader, config, device)
        print(f"[Test] loss={test_loss:.4f} acc={test_acc:.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a clean FS-Net-style traffic classifier.")
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--valid_jsonl", required=True)
    p.add_argument("--test_jsonl", default=None)
    p.add_argument("--task", choices=["vpn", "service", "app"], required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--app_min", type=int, default=None)
    p.add_argument("--app_max", type=int, default=None)
    p.add_argument("--max_len", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--checkpoint_policy", choices=["best_val", "last"], default="best_val")
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
    p.add_argument("--no_dir", action="store_true")
    p.add_argument("--no_dt_bucket", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--subset_per_class", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--log_interval", type=int, default=50)
    return p


def main() -> None:
    train_clean(build_parser().parse_args())


if __name__ == "__main__":
    main()
