from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..checkpoints import build_model_from_checkpoint
from ..data import CleanSubsetDataset, TriggeredDataset, collate_flows
from ..features import apply_dir_sign, compute_state_ids
from ..utils import get_device
from .common import auc_score, plot_score_hist_and_roc


def packet_embeddings(model, state_ids, lengths_signed):
    return model.packet_embeddings(state_ids, lengths_signed)


def gram_vector(seq_feat: torch.Tensor, mask: torch.Tensor, max_order: int = 3) -> np.ndarray:
    valid = mask.bool()
    x = seq_feat[valid]
    if x.size(0) == 0:
        raise RuntimeError("empty sequence")
    x = x - x.mean(dim=0, keepdim=True)
    f = x.transpose(0, 1)
    d = f.size(0)
    gram = (f @ f.transpose(0, 1)) / (x.size(0) * d)
    idx = torch.triu_indices(d, d, device=gram.device)
    base = gram[idx[0], idx[1]]
    parts = [base]
    for power in range(2, max_order + 1):
        parts.append(base.pow(power))
    return torch.cat(parts).detach().cpu().numpy()


def build_stats(model, loader, config, device, max_order: int = 3):
    cls2vecs = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            state_ids = compute_state_ids(batch.times, batch.dirs, batch.mask, config, device)
            lengths_signed = apply_dir_sign(batch.lengths, batch.dirs, config)
            seq_feats = packet_embeddings(model, state_ids, lengths_signed)
            for i in range(batch.labels.size(0)):
                cls2vecs[int(batch.labels[i].item())].append(gram_vector(seq_feats[i], batch.mask[i], max_order))

    centers, dist_med, dist_mad = {}, {}, {}
    for cls, vecs in cls2vecs.items():
        x = np.stack(vecs, axis=0)
        center = np.median(x, axis=0)
        dists = np.abs(x - center).mean(axis=1)
        centers[cls] = center
        dist_med[cls] = float(np.median(dists))
        dist_mad[cls] = float(np.median(np.abs(dists - dist_med[cls])) + 1e-12)
    return {"centers": centers, "dist_med": dist_med, "dist_mad": dist_mad}


def score_loader(model, loader, stats, config, device, max_order: int = 3, use_pred_label: bool = True):
    scores = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            state_ids = compute_state_ids(batch.times, batch.dirs, batch.mask, config, device)
            lengths_signed = apply_dir_sign(batch.lengths, batch.dirs, config)
            logits = model(state_ids, lengths_signed, batch.mask)
            preds = logits.argmax(dim=1)
            seq_feats = packet_embeddings(model, state_ids, lengths_signed)
            for i in range(batch.labels.size(0)):
                cls = int(preds[i].item()) if use_pred_label else int(batch.labels[i].item())
                if cls not in stats["centers"]:
                    continue
                vec = gram_vector(seq_feats[i], batch.mask[i], max_order)
                d = float(np.abs(vec - stats["centers"][cls]).mean())
                z = (d - stats["dist_med"][cls]) / (stats["dist_mad"][cls] + 1e-12)
                scores.append(z)
                labels.append(cls)
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def build_parser():
    p = argparse.ArgumentParser(description="Beatrix-style Gram-matrix detector for FS-Net traffic models.")
    p.add_argument("--cls_ckpt", "--ckpt", dest="ckpt", required=True)
    p.add_argument("--clean_subset_jsonl", "--clean_jsonl", dest="clean_jsonl", required=True)
    p.add_argument("--triggered_jsonl", "--inspect_jsonl", dest="triggered_jsonl", required=True)
    p.add_argument("--output_dir", "--out_dir", "--save_dir", dest="out_dir", required=True)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_order", type=int, default=3)
    p.add_argument("--mad_k", type=float, default=3.0)
    p.add_argument("--cpu", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device(args.cpu)
    model, info = build_model_from_checkpoint(args.ckpt, device)
    config = info["config"]
    clean_loader = DataLoader(
        CleanSubsetDataset(args.clean_jsonl, args.max_len), args.batch_size, shuffle=False, collate_fn=collate_flows
    )
    trig_loader = DataLoader(
        TriggeredDataset(args.triggered_jsonl, args.max_len, prefer_lengths_key=True),
        args.batch_size,
        shuffle=False,
        collate_fn=collate_flows,
    )
    stats = build_stats(model, clean_loader, config, device, args.max_order)
    clean_scores, _ = score_loader(model, clean_loader, stats, config, device, args.max_order, use_pred_label=True)
    trig_scores, _ = score_loader(model, trig_loader, stats, config, device, args.max_order, use_pred_label=True)
    y_true = np.concatenate([np.zeros_like(clean_scores), np.ones_like(trig_scores)])
    y_score = np.concatenate([clean_scores, trig_scores])
    auc = auc_score(y_true, y_score)
    print(f"[Beatrix] clean_n={clean_scores.size} trig_n={trig_scores.size} auc={auc:.4f}")
    np.savez_compressed(
        os.path.join(args.out_dir, "beatrix_scores.npz"),
        clean=clean_scores,
        triggered=trig_scores,
    )
    plot_score_hist_and_roc(
        clean_scores,
        trig_scores,
        args.out_dir,
        prefix="beatrix",
        score_label="Anomaly Score ($z$)",
        reference_threshold=args.mad_k,
        target_fpr=0.05,
        paper_style=True,
        histogram_stem="beatrix_",
    )


if __name__ == "__main__":
    main()
