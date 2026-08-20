from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from halo.attacks.eval import dump_triggered_padding_jsonl, evaluate_padding_attack
from halo.checkpoints import build_model_from_checkpoint
from halo.data import (
    CleanSubsetDataset,
    FlowDataset,
    TriggeredDataset,
    build_poison_source_ids,
    collate_flows,
)
from halo.defenses.beatrix import build_stats as beatrix_build_stats
from halo.defenses.beatrix import score_loader as beatrix_score_loader
from halo.defenses.common import auc_score
from halo.defenses.ted import collect_layer_features, mahalanobis_scores, rank_sequences
from halo.train_attack import _build_generator
from halo.utils import evaluate_clean, get_device


TASK_TO_LABEL_KEY = {"app": "app", "vpn": "vpn_flag", "service": "service"}


def _safe_float(value):
    if value is None:
        return None
    return float(value)


def _joint_paths(trial_dir: str, limit: int | None = None):
    paths = sorted(glob.glob(os.path.join(trial_dir, "joint_epoch_*.pt")))
    if not paths and os.path.isfile(os.path.join(trial_dir, "best_joint_model.pt")):
        paths = [os.path.join(trial_dir, "best_joint_model.pt")]
    if limit is not None:
        paths = paths[: int(limit)]
    return paths


def _load_model_and_generator(joint_path: str, device):
    model, info = build_model_from_checkpoint(joint_path, device)
    ckpt = info["ckpt"]
    args = dict(info["args"])
    ns = Namespace(**args)
    generator = _build_generator(ns, info["config"]).to(device)
    state = ckpt.get("generator_state_dict")
    if state is None:
        raise KeyError(f"{joint_path} has no generator_state_dict")
    generator.load_state_dict(state)
    generator.eval()
    return model, generator, info


def _evaluate_detectors(model, config, clean_subset_jsonl, triggered_jsonl, batch_size, max_len, device):
    clean_loader = DataLoader(
        CleanSubsetDataset(clean_subset_jsonl, max_len),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_flows,
    )
    trig_loader = DataLoader(
        TriggeredDataset(triggered_jsonl, max_len),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_flows,
    )

    b_stats = beatrix_build_stats(model, clean_loader, config, device, max_order=3)
    b_clean, _ = beatrix_score_loader(model, clean_loader, b_stats, config, device, max_order=3, use_pred_label=True)
    b_trig, _ = beatrix_score_loader(model, trig_loader, b_stats, config, device, max_order=3, use_pred_label=True)
    b_auc = auc_score(
        np.concatenate([np.zeros_like(b_clean), np.ones_like(b_trig)]),
        np.concatenate([b_clean, b_trig]),
    )
    b_fpr5_thr = float(np.percentile(b_clean, 95.0)) if b_clean.size else float("nan")
    b_tpr5 = float((b_trig > b_fpr5_thr).mean()) if b_trig.size else 0.0

    clean_feats, clean_labels, _, _ = collect_layer_features(model, clean_loader, config, device)
    trig_feats, _, _, trig_targets = collect_layer_features(model, trig_loader, config, device)
    ranks_clean = rank_sequences(clean_feats, clean_labels, clean_feats, clean_labels, exclude_self=True)
    ranks_trig = rank_sequences(trig_feats, trig_targets, clean_feats, clean_labels)
    ted_clean, mu, _, inv = mahalanobis_scores(ranks_clean)
    centered = ranks_trig - mu.reshape(1, -1)
    ted_trig = np.einsum("ij,jk,ik->i", centered, inv, centered)
    ted_auc = auc_score(
        np.concatenate([np.zeros_like(ted_clean), np.ones_like(ted_trig)]),
        np.concatenate([ted_clean, ted_trig]),
    )
    ted_thr = float(np.percentile(ted_clean, 95.0)) if ted_clean.size else float("nan")
    ted_tpr5 = float((ted_trig > ted_thr).mean()) if ted_trig.size else 0.0

    return {
        "beatrix_auc": _safe_float(b_auc),
        "beatrix_tpr_fpr5": b_tpr5,
        "ted_auc": _safe_float(ted_auc),
        "ted_tpr_fpr5": ted_tpr5,
    }


def evaluate_joint(
    joint_path: str,
    trial_dir: str,
    out_dir: str,
    device,
    eval_jsonl: str | None = None,
    eval_name: str = "test",
):
    model, generator, info = _load_model_and_generator(joint_path, device)
    args = dict(info["args"])
    config = info["config"]
    label2id = info["label2id"]
    id2label = info["id2label"]
    label_key = TASK_TO_LABEL_KEY[args["task"]]
    poison_source_ids = build_poison_source_ids(args.get("poison_source_labels", ""), label2id, label_key)
    batch_size = int(args.get("batch_size", 64))
    max_len = int(args.get("max_len", 128))

    eval_path = os.path.abspath(eval_jsonl or args["test_jsonl"])
    eval_ds = FlowDataset(eval_path, label_key, label2id, max_len)
    eval_loader = DataLoader(eval_ds, batch_size, shuffle=False, num_workers=0, collate_fn=collate_flows)
    clean_loss, clean_acc = evaluate_clean(model, eval_loader, config, device)
    atk_loss, asr, triggered_samples = evaluate_padding_attack(
        model, generator, eval_loader, config, device, args["target_class"], poison_source_ids
    )

    stem = os.path.splitext(os.path.basename(joint_path))[0]
    epoch_dir = os.path.join(out_dir, stem)
    os.makedirs(epoch_dir, exist_ok=True)
    triggered_jsonl = os.path.join(epoch_dir, f"{eval_name}_triggered_poisoned.jsonl")
    dump_triggered_padding_jsonl(
        generator,
        eval_loader,
        config,
        device,
        id2label,
        triggered_jsonl,
        args["target_class"],
        poison_source_ids,
    )
    detectors = _evaluate_detectors(
        model,
        config,
        os.path.join(trial_dir, "train_subset_per_class.jsonl"),
        triggered_jsonl,
        batch_size,
        max_len,
        device,
    )
    ckpt = info["ckpt"]
    row = {
        "joint_path": joint_path,
        "epoch": int(ckpt.get("epoch", -1)),
        "val_acc": _safe_float(ckpt.get("val_acc")),
        "align_loss": _safe_float(ckpt.get("align_loss")),
        "clean_loss": _safe_float(clean_loss),
        "clean_acc": _safe_float(clean_acc),
        "attack_loss": _safe_float(atk_loss),
        "asr": _safe_float(asr),
        "triggered_samples": int(triggered_samples),
    }
    row.update(detectors)
    with open(os.path.join(epoch_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, sort_keys=True)
    return row


def main():
    parser = argparse.ArgumentParser(description="Evaluate HALO joint checkpoints with ASR, Beatrix, and TED metrics.")
    parser.add_argument("--trial_dir", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Evaluate only this checkpoint instead of discovering every joint_epoch checkpoint.",
    )
    parser.add_argument(
        "--eval_jsonl",
        default=None,
        help="Optional evaluation split. Defaults to the test_jsonl stored in each checkpoint.",
    )
    parser.add_argument(
        "--eval_name",
        default="test",
        help="Filename prefix for the triggered dump, e.g. 'valid' or 'test'.",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    trial_dir = os.path.abspath(args.trial_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(trial_dir, "tuning_eval"))
    os.makedirs(out_dir, exist_ok=True)
    device = get_device(args.cpu)
    rows = []
    joint_paths = [os.path.abspath(args.checkpoint)] if args.checkpoint else _joint_paths(trial_dir, args.limit)
    for joint_path in joint_paths:
        print(f"[EvalJoint] {joint_path}")
        rows.append(
            evaluate_joint(
                joint_path,
                trial_dir,
                out_dir,
                device,
                eval_jsonl=args.eval_jsonl,
                eval_name=args.eval_name,
            )
        )

    csv_path = os.path.join(out_dir, "summary.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[EvalJoint] wrote {csv_path}")
        ranked = sorted(
            rows,
            key=lambda r: (
                not (r["asr"] >= 0.95 and r["clean_acc"] >= 0.93),
                r["beatrix_auc"] + r["ted_auc"] + r["beatrix_tpr_fpr5"] + r["ted_tpr_fpr5"],
                -r["asr"],
                -r["clean_acc"],
            ),
        )
        print("[EvalJoint] top candidates:")
        for row in ranked[:8]:
            print(
                "  epoch={epoch} clean={clean_acc:.4f} asr={asr:.4f} "
                "b_auc={beatrix_auc:.4f} b_tpr5={beatrix_tpr_fpr5:.4f} "
                "ted_auc={ted_auc:.4f} ted_tpr5={ted_tpr_fpr5:.4f}".format(**row)
            )


if __name__ == "__main__":
    main()
