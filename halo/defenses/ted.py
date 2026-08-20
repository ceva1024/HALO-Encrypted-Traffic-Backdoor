from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..checkpoints import build_model_from_checkpoint
from ..data import CleanSubsetDataset, TriggeredDataset, collate_flows
from ..features import classifier_inputs
from ..utils import get_device
from .common import auc_score, get_pyplot, plot_score_hist_and_roc


def collect_layer_features(model, loader, config, device):
    per_layer = None
    labels = []
    pred_labels = []
    target_labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            state_ids, lengths_signed, mask = classifier_inputs(batch, config, device)
            logits, feats = model.forward_with_features(state_ids, lengths_signed, mask)
            if per_layer is None:
                per_layer = [[] for _ in feats]
            for i, feat in enumerate(feats):
                per_layer[i].append(feat.detach().cpu().numpy())
            labels.append(batch.labels.cpu().numpy())
            pred_labels.append(logits.argmax(dim=1).cpu().numpy())
            if batch.target_labels is not None:
                target_labels.append(batch.target_labels.cpu().numpy())
    feats_np = [np.concatenate(chunks, axis=0) for chunks in per_layer]
    labels_np = np.concatenate(labels)
    preds_np = np.concatenate(pred_labels)
    targets_np = np.concatenate(target_labels) if target_labels else labels_np
    return feats_np, labels_np, preds_np, targets_np


def rank_sequences(sample_feats, query_labels, ref_feats, ref_labels, exclude_self: bool = False):
    ranks = []
    for layer_id, q in enumerate(sample_feats):
        ref = ref_feats[layer_id]
        layer_ranks = np.zeros(q.shape[0], dtype=np.float64)
        for i in range(q.shape[0]):
            dist = ((ref - q[i]) ** 2).sum(axis=1)
            if exclude_self and q.shape[0] == ref.shape[0]:
                dist[i] = np.inf
            order = np.argsort(dist)
            hit = np.where(ref_labels[order] == query_labels[i])[0]
            layer_ranks[i] = float(hit[0] + 1 if hit.size else len(order) + 1)
        ranks.append(layer_ranks)
    return np.stack(ranks, axis=1)


def mahalanobis_scores(x: np.ndarray, eps: float = 1e-6):
    mu = x.mean(axis=0)
    cov = np.cov(x.T)
    cov = np.atleast_2d(cov) + eps * np.eye(x.shape[1])
    inv = np.linalg.pinv(cov)
    centered = x - mu.reshape(1, -1)
    scores = np.einsum("ij,jk,ik->i", centered, inv, centered)
    return scores, mu, cov, inv


def _normalize_ranks_for_plot(ranks: np.ndarray, db_size: int) -> np.ndarray:
    if ranks.size == 0:
        return ranks.astype(np.float64)
    denom = max(1.0, float(db_size) - 1.0)
    norm = (ranks.astype(np.float64) - 1.0) / denom
    return np.clip(norm, 0.0, 1.0)


def _cluster_trajectories(y_fill: np.ndarray, n_clusters: int, seed: int):
    if y_fill.shape[0] == 0:
        return None
    actual_k = min(int(n_clusters), y_fill.shape[0])
    if actual_k <= 1:
        return np.zeros(y_fill.shape[0], dtype=np.int64), actual_k
    try:
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=actual_k, random_state=seed, n_init=10).fit_predict(y_fill)
        return labels, actual_k
    except Exception as exc:
        print(f"[Plot] trajectory clustering unavailable; plotting one bundle: {exc}")
        return np.zeros(y_fill.shape[0], dtype=np.int64), 1


def _trajectory_colors(color_theme: str, n_colors: int):
    if color_theme == "Reds":
        palette = [
            "#8b0000",
            "#0072b2",
            "#009e73",
            "#d55e00",
            "#cc79a7",
            "#e69f00",
            "#56b4e9",
            "#000000",
        ]
    elif color_theme == "Blues":
        palette = [
            "#004488",
            "#d55e00",
            "#009e73",
            "#cc79a7",
            "#e69f00",
            "#56b4e9",
            "#882255",
            "#000000",
        ]
    else:
        palette = [
            "#0072b2",
            "#d55e00",
            "#009e73",
            "#cc79a7",
            "#e69f00",
            "#56b4e9",
            "#882255",
            "#000000",
        ]
    return [palette[i % len(palette)] for i in range(max(1, int(n_colors)))]


def _trajectory_bundle_data(
    ranks: np.ndarray,
    db_size: int,
    n_clusters: int,
    color_theme: str,
    seed: int = 42,
    omit_cluster_orders=None,
    merge_eps: float = 0.01,
    downsample_smallest_visible_cluster: float | None = None,
):
    if ranks.size == 0:
        return None
    norm = _normalize_ranks_for_plot(ranks, db_size)
    valid = ~np.all(np.isnan(norm), axis=1)
    if not valid.any():
        return None
    med = np.nanmedian(norm[valid], axis=0)
    y_fill = np.where(np.isnan(norm), med.reshape(1, -1), norm)[valid]
    labels, actual_k = _cluster_trajectories(y_fill, n_clusters, seed)

    xs = np.arange(1, ranks.shape[1] + 1)
    groups = [[k] for k in range(actual_k) if (labels == k).any()]
    merge_eps = float(merge_eps)
    if merge_eps > 0.0:
        while len(groups) > 1:
            medians = [
                np.median(y_fill[np.isin(labels, group)], axis=0)
                for group in groups
            ]
            best = None
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    dist = float(np.linalg.norm(medians[i] - medians[j]))
                    if best is None or dist < best[0]:
                        best = (dist, i, j)
            if best is None or best[0] > merge_eps:
                break
            _, i, j = best
            groups[i] = groups[i] + groups[j]
            del groups[j]

    group_rows = [y_fill[np.isin(labels, group)] for group in groups]
    order = sorted(
        range(len(group_rows)),
        key=lambda i: (-group_rows[i].shape[0], tuple(np.median(group_rows[i], axis=0))),
    )
    colors = _trajectory_colors(color_theme, len(order))

    bundles = []
    omit_cluster_orders = {int(i) for i in (omit_cluster_orders or [])}
    visible_order = [
        (order_idx + 1, group_idx)
        for order_idx, group_idx in enumerate(order)
        if (order_idx + 1) not in omit_cluster_orders
    ]
    smallest_visible_display_idx = visible_order[-1][0] if visible_order else None
    downsample_keep = None
    if downsample_smallest_visible_cluster is not None:
        downsample_keep = float(downsample_smallest_visible_cluster)
        if not (0.0 < downsample_keep <= 1.0):
            downsample_keep = None

    prepared_bundles = []
    for order_idx, group_idx in enumerate(order):
        display_idx = order_idx + 1
        if display_idx in omit_cluster_orders:
            continue
        cluster_rows = group_rows[group_idx]
        if cluster_rows.size == 0:
            continue
        plot_rows = cluster_rows
        if downsample_keep is not None and display_idx == smallest_visible_display_idx:
            keep_n = max(1, int(np.ceil(cluster_rows.shape[0] * downsample_keep)))
            if keep_n < cluster_rows.shape[0]:
                rng = np.random.default_rng(seed + display_idx)
                keep_idx = np.sort(rng.choice(cluster_rows.shape[0], size=keep_n, replace=False))
                plot_rows = cluster_rows[keep_idx]
        prepared_bundles.append((order_idx, display_idx, plot_rows))

    total_plot_rows = sum(plot_rows.shape[0] for _, _, plot_rows in prepared_bundles)
    for order_idx, display_idx, plot_rows in prepared_bundles:
        frac = float(plot_rows.shape[0] / max(1, total_plot_rows))
        bundles.append(
            {
                "q25": np.percentile(plot_rows, 25, axis=0),
                "q50": np.percentile(plot_rows, 50, axis=0),
                "q75": np.percentile(plot_rows, 75, axis=0),
                "frac": frac,
                "color": colors[order_idx % len(colors)],
                "label": f"C{display_idx} ({frac * 100:.1f}%)",
            }
        )
    return xs, bundles


def _draw_trajectory_bundles(ax, xs, bundles):
    for item in bundles:
        ax.fill_between(xs, item["q25"], item["q75"], color=item["color"], alpha=0.10, linewidth=0)
        ax.plot(
            xs,
            item["q50"],
            color=item["color"],
            linewidth=1.6 + 3.5 * item["frac"],
            label=item["label"],
        )


def plot_trajectory_bundle(
    ranks: np.ndarray,
    db_size: int,
    out_prefix: str,
    n_clusters: int,
    color_theme: str,
    seed: int = 42,
    merge_eps: float = 0.01,
    omit_cluster_orders=None,
) -> None:
    plt = get_pyplot()
    if plt is None:
        return
    if ranks.size == 0:
        print(f"[Plot] skip {out_prefix}: empty ranks")
        return

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    bundle_data = _trajectory_bundle_data(
        ranks,
        db_size,
        n_clusters,
        color_theme,
        seed,
        omit_cluster_orders=omit_cluster_orders,
        merge_eps=merge_eps,
    )
    if bundle_data is None:
        print(f"[Plot] skip {out_prefix}: no valid ranks")
        return
    xs, bundles = bundle_data

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    _draw_trajectory_bundles(ax, xs, bundles)

    ax.set_xlabel("Layer Index ($l$)")
    ax.set_ylabel(r"Normalized Rank ($\tilde{r}^{(l)}$)")
    ax.set_xticks(xs)
    ax.set_ylim(-0.01, 0.40)
    ax.set_yticks(np.linspace(0.0, 0.4, 5))
    ncol = 3 if len(bundles) > 4 else 2
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=True, framealpha=0.92, ncol=ncol)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_prefix}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] saved trajectory bundle to {out_prefix}.pdf")


def plot_trajectory_panel(panels, out_prefix: str, y_max: float = 0.4, seed: int = 42) -> None:
    plt = get_pyplot()
    if plt is None:
        return
    if not panels:
        print(f"[Plot] skip {out_prefix}: no panels")
        return

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    with plt.rc_context(
        {
            "font.size": 17,
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
        }
    ):
        fig = plt.figure(figsize=(12.8, 7.4))
        gs = fig.add_gridspec(2, 6, hspace=1.08, wspace=0.48)
        slots = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6], gs[1, 1:3], gs[1, 3:5]]

        for idx, panel in enumerate(panels[: len(slots)]):
            ax = fig.add_subplot(slots[idx])
            bundle_data = _trajectory_bundle_data(
                panel["ranks"],
                int(panel["db_size"]),
                int(panel.get("n_clusters", 4)),
                panel.get("color_theme", "Reds"),
                omit_cluster_orders=panel.get("omit_cluster_orders"),
                merge_eps=float(panel.get("merge_eps", 0.01)),
                downsample_smallest_visible_cluster=panel.get("downsample_smallest_visible_cluster"),
                seed=seed,
            )
            if bundle_data is None:
                ax.text(0.5, 0.5, "No valid ranks", ha="center", va="center", transform=ax.transAxes)
                xs = np.arange(1, 4)
                bundles = []
            else:
                xs, bundles = bundle_data
                _draw_trajectory_bundles(ax, xs, bundles)

            ax.set_xlim(float(xs[0]) - 0.1, float(xs[-1]) + 0.1)
            ax.set_ylim(-0.01, float(y_max))
            ax.set_xticks(xs)
            ax.set_yticks(np.linspace(0.0, float(y_max), 5))
            ax.set_xlabel("Layer Index ($l$)")
            if idx not in {0, 3}:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("Normalized Rank", fontsize=20)
            ax.text(
                0.0,
                1.035,
                panel["title"],
                ha="left",
                va="bottom",
                transform=ax.transAxes,
                fontsize=17,
                fontweight="bold",
            )
            if bundles:
                ncol = 3 if len(bundles) > 4 else 2
                fontsize = 14.5 if len(bundles) > 4 else 16
                ax.legend(
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.20),
                    ncol=ncol,
                    frameon=True,
                    framealpha=0.92,
                    fontsize=fontsize,
                    handlelength=2.1,
                    columnspacing=0.9,
                    borderpad=0.25,
                )

        fig.subplots_adjust(left=0.105, right=0.995, top=0.87, bottom=0.11)
        for ext in ("pdf", "png"):
            fig.savefig(f"{out_prefix}.{ext}", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    print(f"[Plot] saved trajectory panel to {out_prefix}.pdf")


def build_parser():
    p = argparse.ArgumentParser(description="TED-style layer trajectory detector for FS-Net traffic models.")
    p.add_argument("--cls_ckpt", "--ckpt", dest="ckpt", required=True)
    p.add_argument("--clean_subset_jsonl", "--clean_jsonl", dest="clean_jsonl", required=True)
    p.add_argument("--triggered_jsonl", "--inspect_jsonl", dest="triggered_jsonl", default=None)
    p.add_argument("--output_dir", "--out_dir", dest="out_dir", required=True)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--threshold_percentile", type=float, default=95.0)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--clean_only", action="store_true", help="Only generate clean TED rank trajectories for a clean model.")
    p.add_argument("--cpu", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device(args.cpu)
    model, info = build_model_from_checkpoint(args.ckpt, device)
    config = info["config"]
    clean_loader = DataLoader(CleanSubsetDataset(args.clean_jsonl, args.max_len), args.batch_size, False, collate_fn=collate_flows)

    clean_feats, clean_labels, clean_preds, _ = collect_layer_features(model, clean_loader, config, device)
    ranks_clean = rank_sequences(clean_feats, clean_labels, clean_feats, clean_labels, exclude_self=True)
    clean_scores, mu, cov, inv = mahalanobis_scores(ranks_clean)
    threshold_percentile = float(args.threshold_percentile)
    if args.alpha is not None:
        threshold_percentile = 100.0 * (1.0 - float(args.alpha))
    threshold = np.percentile(clean_scores, threshold_percentile)

    if args.clean_only:
        print(f"[TED] clean_only clean_n={clean_scores.size} threshold={threshold:.4f}")
        np.savez_compressed(
            os.path.join(args.out_dir, "ted_scores.npz"),
            ranks_clean=ranks_clean,
            clean_scores=clean_scores,
            threshold=np.asarray(threshold, dtype=np.float64),
            db_size=np.asarray(clean_labels.size, dtype=np.int64),
        )
        plot_trajectory_bundle(
            ranks_clean,
            db_size=clean_labels.size,
            out_prefix=os.path.join(args.out_dir, "traj_clean"),
            n_clusters=4,
            color_theme="Blues",
            omit_cluster_orders=[3, 4],
        )
        return

    if not args.triggered_jsonl:
        raise SystemExit("--triggered_jsonl is required unless --clean_only is set.")

    trig_loader = DataLoader(TriggeredDataset(args.triggered_jsonl, args.max_len), args.batch_size, False, collate_fn=collate_flows)
    trig_feats, _, _, trig_targets = collect_layer_features(model, trig_loader, config, device)
    ranks_trig = rank_sequences(trig_feats, trig_targets, clean_feats, clean_labels)
    centered = ranks_trig - mu.reshape(1, -1)
    trig_scores = np.einsum("ij,jk,ik->i", centered, inv, centered)
    auc = auc_score(np.concatenate([np.zeros_like(clean_scores), np.ones_like(trig_scores)]), np.concatenate([clean_scores, trig_scores]))
    target_fpr = 1.0 - threshold_percentile / 100.0
    tpr_at_target_fpr = float((trig_scores > threshold).mean()) if trig_scores.size else 0.0
    print(
        f"[TED] clean_n={clean_scores.size} trig_n={trig_scores.size} "
        f"auc={auc:.4f} target_fpr={target_fpr:.4f} "
        f"tpr_at_target_fpr={tpr_at_target_fpr:.4f} threshold={threshold:.4f}"
    )
    np.savez_compressed(
        os.path.join(args.out_dir, "ted_scores.npz"),
        ranks_clean=ranks_clean,
        ranks_triggered=ranks_trig,
        clean_scores=clean_scores,
        triggered_scores=trig_scores,
        threshold=np.asarray(threshold, dtype=np.float64),
        db_size=np.asarray(clean_labels.size, dtype=np.int64),
    )
    plot_score_hist_and_roc(
        clean_scores,
        trig_scores,
        args.out_dir,
        prefix="ted",
        score_label="Mahalanobis Score",
        reference_threshold=float(threshold),
        target_fpr=target_fpr,
    )
    plot_trajectory_bundle(
        ranks_clean,
        db_size=clean_labels.size,
        out_prefix=os.path.join(args.out_dir, "traj_clean"),
        n_clusters=4,
        color_theme="Blues",
        omit_cluster_orders=[3, 4],
    )
    plot_trajectory_bundle(
        ranks_trig,
        db_size=clean_labels.size,
        out_prefix=os.path.join(args.out_dir, "traj_triggered"),
        n_clusters=4,
        color_theme="Reds",
    )


if __name__ == "__main__":
    main()
