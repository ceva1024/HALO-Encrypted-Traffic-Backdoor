from __future__ import annotations

import json
import os
from typing import Dict

import torch
import torch.nn as nn

from ..data import FlowBatch
from ..features import apply_dir_sign, classifier_inputs, compute_state_ids
from .triggers import apply_badnets_insert_batch


def _source_mask(labels: torch.Tensor, target_class: int, poison_source_ids=None) -> torch.Tensor:
    if poison_source_ids is None:
        src = torch.ones_like(labels, dtype=torch.bool)
    else:
        src = torch.zeros_like(labels, dtype=torch.bool)
        for cid in poison_source_ids:
            src |= labels == int(cid)
    return src & (labels != int(target_class))


def evaluate_padding_attack(model, generator, data_loader, config, device, target_class: int, poison_source_ids=None):
    model.eval()
    generator.eval()
    criterion = nn.CrossEntropyLoss()
    totals = dict(all=0, src=0, success=0)
    loss_sum = 0.0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            attack = _source_mask(batch.labels, target_class, poison_source_ids)
            pad = torch.clamp(generator(batch.lengths, batch.times, batch.dirs, batch.can_modify, batch.mask), min=0.0)
            pad = pad * attack.unsqueeze(1).float()
            lengths_trig = torch.clamp(batch.lengths + pad, max=config.max_pkt_len)

            state_ids = compute_state_ids(batch.times, batch.dirs, batch.mask, config, device=device)
            lengths_signed = apply_dir_sign(lengths_trig, batch.dirs, config)
            logits = model(state_ids, lengths_signed, batch.mask)
            loss = criterion(logits, batch.labels)
            preds = logits.argmax(dim=1)

            bsz = batch.labels.size(0)
            totals["all"] += bsz
            loss_sum += loss.item() * bsz

            if attack.any():
                preds_src = preds[attack]
                totals["src"] += preds_src.size(0)
                totals["success"] += (preds_src == int(target_class)).sum().item()

    return (
        loss_sum / max(1, totals["all"]),
        totals["success"] / max(1, totals["src"]),
        totals["src"],
    )


def evaluate_badnets_attack(
    model,
    data_loader,
    config,
    device,
    target_class: int,
    poison_source_ids=None,
    insert_lens=(400.0, 500.0, 600.0),
    after_kth_server: int = 2,
    dt_mode: str = "split",
    dt_const: float = 1e-3,
):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    totals = dict(all=0, src=0, success=0)
    loss_sum = 0.0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            attack = _source_mask(batch.labels, target_class, poison_source_ids)
            res = apply_badnets_insert_batch(
                batch.lengths,
                batch.times,
                batch.dirs,
                batch.can_modify,
                batch.mask,
                attack,
                insert_lens=insert_lens,
                after_kth_server=after_kth_server,
                dt_mode=dt_mode,
                dt_const=dt_const,
                max_pkt_len=config.max_pkt_len,
            )
            effective = attack & res.applied
            trig_batch = FlowBatch(res.lengths, res.times, res.dirs, res.can_modify, res.mask, batch.labels)
            state_ids, lengths_signed, mask = classifier_inputs(trig_batch, config, device)
            logits = model(state_ids, lengths_signed, mask)
            loss = criterion(logits, batch.labels)
            preds = logits.argmax(dim=1)
            bsz = batch.labels.size(0)
            totals["all"] += bsz
            loss_sum += loss.item() * bsz

            if effective.any():
                preds_src = preds[effective]
                totals["src"] += preds_src.size(0)
                totals["success"] += (preds_src == int(target_class)).sum().item()

    return (
        loss_sum / max(1, totals["all"]),
        totals["success"] / max(1, totals["src"]),
        totals["src"],
    )


def dump_triggered_padding_jsonl(
    generator,
    data_loader,
    config,
    device,
    id2label: Dict[int, object],
    out_path: str,
    target_class: int,
    poison_source_ids=None,
) -> None:
    generator.eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f, torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            attack = _source_mask(batch.labels, target_class, poison_source_ids)
            if not attack.any():
                continue
            pad = torch.clamp(generator(batch.lengths, batch.times, batch.dirs, batch.can_modify, batch.mask), min=0.0)
            pad = pad * attack.unsqueeze(1).float()
            lengths_trig = torch.clamp(batch.lengths + pad, max=config.max_pkt_len)
            for i in range(batch.labels.size(0)):
                if not bool(attack[i]):
                    continue
                valid = batch.mask[i].bool()
                label_id = int(batch.labels[i].item())
                obj = {
                    "lengths_clean": batch.lengths[i][valid].detach().cpu().tolist(),
                    "lengths_triggered": lengths_trig[i][valid].detach().cpu().tolist(),
                    "times": batch.times[i][valid].detach().cpu().tolist(),
                    "dirs": batch.dirs[i][valid].detach().cpu().tolist(),
                    "can_modify": batch.can_modify[i][valid].detach().cpu().tolist(),
                    "orig_label_id": label_id,
                    "orig_label_raw": id2label.get(label_id, label_id),
                    "target_label_id": int(target_class),
                    "target_label_raw": id2label.get(int(target_class), int(target_class)),
                }
                f.write(json.dumps(obj) + "\n")
                written += 1
    print(f"[Dump] wrote {written} triggered samples to {out_path}")


def dump_badnets_triggered_jsonl(
    data_loader,
    config,
    device,
    id2label: Dict[int, object],
    out_path: str,
    target_class: int,
    poison_source_ids=None,
    insert_lens=(400.0, 500.0, 600.0),
    after_kth_server: int = 2,
    dt_mode: str = "split",
    dt_const: float = 1e-3,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f, torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            attack = _source_mask(batch.labels, target_class, poison_source_ids)
            if not attack.any():
                continue
            res = apply_badnets_insert_batch(
                batch.lengths,
                batch.times,
                batch.dirs,
                batch.can_modify,
                batch.mask,
                attack,
                insert_lens=insert_lens,
                after_kth_server=after_kth_server,
                dt_mode=dt_mode,
                dt_const=dt_const,
                max_pkt_len=config.max_pkt_len,
            )
            effective = attack & res.applied
            for i in range(batch.labels.size(0)):
                if not bool(effective[i]):
                    continue
                clean_valid = batch.mask[i].bool()
                trig_valid = res.mask[i].bool()
                label_id = int(batch.labels[i].item())
                obj = {
                    "lengths_clean": batch.lengths[i][clean_valid].detach().cpu().tolist(),
                    "lengths_triggered": res.lengths[i][trig_valid].detach().cpu().tolist(),
                    "times": res.times[i][trig_valid].detach().cpu().tolist(),
                    "dirs": res.dirs[i][trig_valid].detach().cpu().tolist(),
                    "can_modify": res.can_modify[i][trig_valid].detach().cpu().tolist(),
                    "orig_label_id": label_id,
                    "orig_label_raw": id2label.get(label_id, label_id),
                    "target_label_id": int(target_class),
                    "target_label_raw": id2label.get(int(target_class), int(target_class)),
                }
                f.write(json.dumps(obj) + "\n")
                written += 1
    print(f"[Dump] wrote {written} BadNets triggered samples to {out_path}")
