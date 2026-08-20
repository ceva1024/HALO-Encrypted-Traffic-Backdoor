from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ..features import classifier_inputs


@dataclass(frozen=True)
class ScoreDistributionPanel:
    title: str
    clean_scores: np.ndarray
    trig_scores: np.ndarray
    reference_threshold: float | None = None


def robust_z(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return (values - med) / (1.4826 * mad + eps)


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    pos_ranks = ranks[: pos.size]
    return float((pos_ranks.sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = max(1, int((y_true == 1).sum()))
    neg = max(1, int((y_true == 0).sum()))
    thresholds = np.r_[np.inf, np.unique(y_score)[::-1], -np.inf]
    fpr = []
    tpr = []
    for thr in thresholds:
        pred = y_score >= thr
        fp = int(((pred) & (y_true == 0)).sum())
        tp = int(((pred) & (y_true == 1)).sum())
        fpr.append(fp / neg)
        tpr.append(tp / pos)
    return np.asarray(fpr), np.asarray(tpr), thresholds


def get_pyplot():
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = os.path.join("/tmp", f"matplotlib-{os.getuid()}")
        os.makedirs(mpl_config_dir, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = mpl_config_dir
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Plot] matplotlib unavailable; skipping plots: {exc}")
        return None

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _score_plot_rc(paper_style: bool = False) -> dict[str, float | int | str]:
    if not paper_style:
        return {}
    return {
        "font.size": 14,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 1.8,
    }


def _score_hist_bins(clean_scores: np.ndarray, trig_scores: np.ndarray, bins_count: int = 60) -> np.ndarray:
    combined = np.concatenate([clean_scores, trig_scores])
    lo = float(np.percentile(combined, 0.5))
    hi = float(np.percentile(combined, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.min(combined))
        hi = float(np.max(combined))
    pad = max(1e-6, 0.05 * (hi - lo))
    return np.linspace(lo - pad, hi + pad, bins_count)


def _draw_score_hist_panel(
    ax,
    clean_scores: np.ndarray,
    trig_scores: np.ndarray,
    *,
    bins_count: int,
    target_fpr: float,
    reference_threshold: float | None = None,
    include_reference_threshold: bool = True,
):
    fpr_threshold = float(np.percentile(clean_scores, (1.0 - target_fpr) * 100.0))
    bins = _score_hist_bins(clean_scores, trig_scores, bins_count=bins_count)

    clean_hist = ax.hist(clean_scores, bins=bins, color="#377eb8", alpha=0.70, density=True, label="Clean")
    trig_hist = ax.hist(trig_scores, bins=bins, color="#e6550d", alpha=0.62, density=True, label="Triggered")
    threshold_line = ax.axvline(
        fpr_threshold,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label=f"{int(target_fpr * 100)}% FPR threshold",
    )
    handles = [clean_hist[2][0], trig_hist[2][0], threshold_line]
    labels = ["Clean", "Triggered", f"{int(target_fpr * 100)}% FPR threshold"]

    if include_reference_threshold and reference_threshold is not None and np.isfinite(reference_threshold):
        reference_line = ax.axvline(
            float(reference_threshold),
            color="#4d4d4d",
            linestyle=":",
            linewidth=1.3,
            label="Reference threshold",
        )
        handles.append(reference_line)
        labels.append("Reference threshold")

    return handles, labels


def plot_score_hist_and_roc(
    clean_scores: np.ndarray,
    trig_scores: np.ndarray,
    out_dir: str,
    prefix: str,
    score_label: str = "Anomaly Score",
    reference_threshold: float | None = None,
    target_fpr: float = 0.05,
    paper_style: bool = False,
    include_histogram: bool = True,
    histogram_stem: str | None = None,
) -> None:
    plt = get_pyplot()
    if plt is None:
        return
    clean_scores = np.asarray(clean_scores, dtype=np.float64)
    trig_scores = np.asarray(trig_scores, dtype=np.float64)
    if clean_scores.size == 0 or trig_scores.size == 0:
        print(f"[Plot] skip {prefix} plots: empty score arrays")
        return

    os.makedirs(out_dir, exist_ok=True)
    fpr_threshold = float(np.percentile(clean_scores, (1.0 - target_fpr) * 100.0))

    if include_histogram:
        with plt.rc_context(_score_plot_rc(paper_style)):
            hist_figsize = (6.4, 4.6) if paper_style else (6.0, 4.2)
            fig, ax = plt.subplots(figsize=hist_figsize)
            _draw_score_hist_panel(
                ax,
                clean_scores,
                trig_scores,
                bins_count=60,
                target_fpr=target_fpr,
                reference_threshold=reference_threshold,
                include_reference_threshold=True,
            )
            ax.set_xlabel(score_label)
            ax.set_ylabel("Density")
            ax.legend(frameon=False)
            fig.tight_layout()
            output_stem = histogram_stem or f"{prefix}_zscore_hist"
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(out_dir, f"{output_stem}.{ext}"), bbox_inches="tight")
            plt.close(fig)

    y_true = np.concatenate([np.zeros_like(clean_scores), np.ones_like(trig_scores)])
    y_score = np.concatenate([clean_scores, trig_scores])
    fpr, tpr, _ = roc_curve_points(y_true, y_score)
    auc = auc_score(y_true, y_score)
    with plt.rc_context(_score_plot_rc(paper_style)):
        roc_figsize = (5.2, 4.8) if paper_style else (4.8, 4.4)
        fig, ax = plt.subplots(figsize=roc_figsize)
        ax.plot(fpr, tpr, color="#1b4f72", linewidth=1.8, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.0)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"{prefix}_roc.{ext}"), bbox_inches="tight")
        plt.close(fig)

    tpr_at_fpr = float((trig_scores > fpr_threshold).mean())
    plot_names = "hist/ROC" if include_histogram else "ROC"
    print(
        f"[Plot] saved {prefix} {plot_names} to {out_dir}; "
        f"threshold@FPR={target_fpr:.0%}={fpr_threshold:.6f} TPR={tpr_at_fpr:.4f}"
    )


def plot_score_hist_grid(
    panels: Sequence[ScoreDistributionPanel],
    out_prefix: str,
    *,
    score_label: str = "Anomaly Score ($z$)",
    target_fpr: float = 0.05,
    ncols: int = 2,
    bins_count: int = 60,
    include_reference_threshold: bool = False,
    figsize: tuple[float, float] = (7.2, 4.2),
) -> None:
    plt = get_pyplot()
    if plt is None:
        return
    if not panels:
        raise ValueError("plot_score_hist_grid requires at least one panel")
    if ncols <= 0:
        raise ValueError("ncols must be positive")

    clean_panels: list[ScoreDistributionPanel] = []
    for panel in panels:
        clean_scores = np.asarray(panel.clean_scores, dtype=np.float64)
        trig_scores = np.asarray(panel.trig_scores, dtype=np.float64)
        if clean_scores.size == 0 or trig_scores.size == 0:
            raise ValueError(f"panel {panel.title!r} has empty score arrays")
        clean_panels.append(
            ScoreDistributionPanel(
                title=panel.title,
                clean_scores=clean_scores,
                trig_scores=trig_scores,
                reference_threshold=panel.reference_threshold,
            )
        )

    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    nrows = int(np.ceil(len(clean_panels) / ncols))
    with plt.rc_context(
        {
            **_score_plot_rc(True),
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10.5,
        }
    ):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        legend_handles = None
        legend_labels = None
        flat_axes = axes.ravel()

        for idx, ax in enumerate(flat_axes):
            if idx >= len(clean_panels):
                ax.axis("off")
                continue

            panel = clean_panels[idx]
            handles, labels = _draw_score_hist_panel(
                ax,
                panel.clean_scores,
                panel.trig_scores,
                bins_count=bins_count,
                target_fpr=target_fpr,
                reference_threshold=panel.reference_threshold,
                include_reference_threshold=include_reference_threshold,
            )
            if legend_handles is None:
                legend_handles = handles
                legend_labels = labels
            ax.set_title(panel.title, pad=3, fontweight="bold")
            ax.set_xlabel(score_label)
            if idx % ncols == 0:
                ax.set_ylabel("Density")
            ax.tick_params(axis="both", which="major", pad=2)

        if legend_handles is not None and legend_labels is not None:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                ncol=len(legend_labels),
                frameon=False,
                bbox_to_anchor=(0.5, 0.985),
                columnspacing=1.15,
                handlelength=1.8,
            )
        fig.subplots_adjust(left=0.095, right=0.99, top=0.855, bottom=0.12, wspace=0.26, hspace=0.72)

        for ext in ("pdf", "png"):
            fig.savefig(f"{out_prefix}.{ext}", bbox_inches="tight")
        plt.close(fig)

    print(f"[Plot] saved score distribution grid to {out_prefix}.pdf and {out_prefix}.png")


def collect_penultimate(model, loader, config, device):
    feats = []
    labels = []
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            state_ids, lengths_signed, mask = classifier_inputs(batch, config, device)
            logits, layer_feats = model.forward_with_features(state_ids, lengths_signed, mask)
            feats.append(layer_feats[-1].detach().cpu())
            labels.append(batch.labels.detach().cpu())
            preds.append(logits.argmax(dim=1).detach().cpu())
    return torch.cat(feats).numpy(), torch.cat(labels).numpy(), torch.cat(preds).numpy()
