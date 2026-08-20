import os
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import math

from ..config import DT_THRESHOLDS_DEFAULT, FeatureConfig
from ..features import apply_dir_sign as _apply_dir_sign
from ..features import compute_state_ids as _compute_state_ids
from ..models import FSNetClassifier


_FEATURE_CONFIG = FeatureConfig()


def configure_feature_flags(use_dir: bool = True, use_dt_bucket: bool = True):
    global _FEATURE_CONFIG
    _FEATURE_CONFIG = FeatureConfig(use_dir=use_dir, use_dt_bucket=use_dt_bucket)


def get_num_states(dt_thresholds=DT_THRESHOLDS_DEFAULT, use_dt_bucket: bool = True, use_dir: bool = True):
    config = FeatureConfig(use_dir=use_dir, use_dt_bucket=use_dt_bucket, dt_thresholds=dt_thresholds)
    return config.num_dt_buckets, config.num_dirs, config.num_states


def compute_state_ids(
    times: torch.Tensor,
    dirs: torch.Tensor,
    mask: torch.Tensor,
    dt_thresholds,
    num_dt_buckets: int,
    state_pad_id: int,
    device: torch.device,
) -> torch.Tensor:
    use_dt_bucket = int(num_dt_buckets) > 1
    use_dir = int(state_pad_id) > int(num_dt_buckets)
    config = FeatureConfig(use_dir=use_dir, use_dt_bucket=use_dt_bucket, dt_thresholds=dt_thresholds)
    return _compute_state_ids(times, dirs, mask, config, device=device)


def apply_dir_sign(lengths: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    return _apply_dir_sign(lengths, dirs, _FEATURE_CONFIG)


@dataclass
class Batch:
    lengths: torch.Tensor
    times: torch.Tensor
    dirs: torch.Tensor
    can_modify: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor
    sources: List[str]


class JsonlFlowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        max_len: int,
        label2id: Optional[Dict[Any, int]] = None,
        label_key: Optional[str] = None,
        use_triggered_lengths: bool = False,
        force_label_id: Optional[int] = None,
        encoding: str = "utf-8",
    ):
        super().__init__()
        self.path = path
        self.max_len = int(max_len)
        self.label2id = label2id or {}
        self.label_key = label_key
        self.use_triggered_lengths = bool(use_triggered_lengths)
        self.force_label_id = force_label_id
        self.encoding = encoding

        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)

        self.offsets: List[int] = []
        with open(self.path, "rb") as f:
            off = 0
            for line in f:
                self.offsets.append(off)
                off += len(line)

        if len(self.offsets) == 0:
            raise RuntimeError(f"Empty jsonl: {self.path}")

        first = self._read_json_at(0)
        self.has_triggered = ("lengths_triggered" in first)
        self.has_standard = ("lengths" in first) or ("lengths_clean" in first)

        if self.use_triggered_lengths and not self.has_triggered:
            raise ValueError(
                "--inspect_use_triggered_lengths was set, but jsonl does not contain 'lengths_triggered'."
            )

        if self.label_key is None and self.has_standard:
            for k in ["vpn_flag", "service", "app", "label", "label_id"]:
                if k in first:
                    self.label_key = k
                    break

        if self.has_standard and self.label_key is None and self.force_label_id is None:
            raise ValueError(
                "Could not infer label_key from jsonl. Please pass --label_key explicitly."
            )

    def __len__(self):
        return len(self.offsets)

    def _read_json_at(self, idx: int) -> Dict[str, Any]:
        with open(self.path, "rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
        line = line.decode(self.encoding, errors="ignore").strip()
        return json.loads(line)

    def _pad_1d(self, arr: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        x = np.zeros((self.max_len,), dtype=np.float32)
        m = np.zeros((self.max_len,), dtype=np.bool_)
        L = min(len(arr), self.max_len)
        if L > 0:
            x[:L] = np.asarray(arr[:L], dtype=np.float32)
            m[:L] = True
        return x, m

    def __getitem__(self, idx: int):
        obj = self._read_json_at(idx)

        if self.use_triggered_lengths:
            lengths = obj["lengths_triggered"]
        else:
            lengths = obj["lengths"] if "lengths" in obj else obj.get("lengths_clean", None)
            if lengths is None:
                raise KeyError("Cannot find 'lengths' (or 'lengths_clean') in json record.")

        times = obj.get("times", obj.get("times_triggered"))
        dirs = obj.get("dirs", obj.get("dirs_triggered"))
        can_modify = obj.get("can_modify", obj.get("can_modify_triggered"))
        if times is None or dirs is None or can_modify is None:
            raise KeyError("Record must contain keys: times, dirs, can_modify.")

        lengths_pad, mask = self._pad_1d(lengths)
        times_pad, _ = self._pad_1d(times)
        dirs_pad, _ = self._pad_1d(dirs)
        can_pad, _ = self._pad_1d(can_modify)

        if self.force_label_id is not None:
            label_id = int(self.force_label_id)
        elif "target_label_id" in obj and self.use_triggered_lengths:
            label_id = int(obj["target_label_id"])
        else:
            raw = obj[self.label_key]
            if raw in self.label2id:
                label_id = int(self.label2id[raw])
            else:
                if isinstance(raw, (int, np.integer)):
                    label_id = int(raw)
                else:
                    raise KeyError(f"Label '{raw}' not found in label2id and not an id.")

        return {
            "lengths": lengths_pad,
            "times": times_pad,
            "dirs": dirs_pad,
            "can_modify": can_pad,
            "mask": mask,
            "label": label_id,
        }


class OffsetDataset(torch.utils.data.Dataset):
    def __init__(self, ds: torch.utils.data.Dataset, source: str = "dataset"):
        super().__init__()
        self.ds = ds
        self.source = str(source)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i: int):
        obj = dict(self.ds[i])
        obj["source"] = self.source
        return obj


def collate_batch(samples: List[Dict[str, Any]]) -> Batch:
    def stack_float(key):
        return torch.from_numpy(np.stack([s[key] for s in samples], axis=0)).float()

    def stack_bool(key):
        return torch.from_numpy(np.stack([s[key] for s in samples], axis=0)).bool()

    lengths = stack_float("lengths")
    times = stack_float("times")
    dirs = stack_float("dirs")
    can_mod = stack_float("can_modify")
    mask = stack_bool("mask")
    labels = torch.tensor([int(s["label"]) for s in samples], dtype=torch.long)
    sources = [str(s.get("source", "unknown")) for s in samples]

    return Batch(lengths, times, dirs, can_mod, mask, labels, sources)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    s = (x * m).sum(dim=1)
    c = m.sum(dim=1).clamp(min=1e-8)
    return s / c


@torch.no_grad()
def extract_penultimate(
    model: FSNetClassifier,
    state_ids: torch.Tensor,
    lengths_signed: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    s_emb = model.state_emb(state_ids)
    norm_len = torch.clamp(lengths_signed / model.max_pkt_len, min=-1.0, max=1.0)
    len_feat = model.len_mlp(norm_len.unsqueeze(-1))
    x = s_emb + len_feat

    mode = model.backbone.mode

    if mode in ["gru", "lstm"]:
        x = model.backbone.dropout(x)
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        if mode == "gru":
            _, h_n = model.backbone.rnn(packed)
        else:
            _, (h_n, _) = model.backbone.rnn(packed)

        num_layers = model.backbone.num_layers
        hidden_dim = model.backbone.hidden_dim
        h_n = h_n.view(num_layers, 2, -1, hidden_dim)
        h_fwd = h_n[-1, 0]
        h_bwd = h_n[-1, 1]
        h = torch.cat([h_fwd, h_bwd], dim=1)
        return h

    elif mode == "cnn":
        x = model.backbone.dropout(x)
        x = x * mask.unsqueeze(-1).float()
        y = x.transpose(1, 2)
        for layer in model.backbone.cnn:
            y = layer(y)
        y_t = y.transpose(1, 2)
        h = masked_mean(y_t, mask)
        return h

    else:
        x = model.backbone.dropout(x)
        key_padding_mask = ~mask
        y = x
        for layer in model.backbone.encoder.layers:
            y = layer(y, src_key_padding_mask=key_padding_mask)
        h = masked_mean(y, mask)
        return h


def _set_dropout_train_only(model: nn.Module, enabled: bool):
    model.eval()
    if enabled:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()


@torch.no_grad()
def extract_features_batch(
    model: FSNetClassifier,
    batch: Batch,
    device: torch.device,
    dt_thresholds,
    num_dt_buckets: int,
    state_pad_id: int,
    mc_dropout: bool = False,
) -> torch.Tensor:
    lengths = batch.lengths.to(device)
    times = batch.times.to(device)
    dirs = batch.dirs.to(device)
    mask = batch.mask.to(device)

    state_ids = compute_state_ids(
        times, dirs, mask, dt_thresholds, num_dt_buckets, state_pad_id, device
    )
    lengths_signed = apply_dir_sign(lengths, dirs)

    if mc_dropout:
        _set_dropout_train_only(model, enabled=True)
    else:
        model.eval()

    h = extract_penultimate(model, state_ids, lengths_signed, mask)
    return h


def compute_class_means(
    model: FSNetClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    dt_thresholds,
    num_dt_buckets: int,
    state_pad_id: int,
    num_classes: int,
    mc_samples: int = 1,
    seed: int = 0,
) -> torch.Tensor:
    torch.manual_seed(seed)
    sums = None
    counts = torch.zeros((num_classes,), dtype=torch.long, device=device)

    for batch in loader:
        labels = batch.labels.to(device)

        if mc_samples <= 1:
            h = extract_features_batch(
                model, batch, device, dt_thresholds, num_dt_buckets, state_pad_id, mc_dropout=False
            )
        else:
            reps = []
            for _ in range(mc_samples):
                reps.append(
                    extract_features_batch(
                        model, batch, device, dt_thresholds, num_dt_buckets, state_pad_id, mc_dropout=True
                    )
                )
            h = torch.stack(reps, dim=0).mean(dim=0)

        if sums is None:
            D = h.shape[1]
            sums = torch.zeros((num_classes, D), dtype=torch.float64, device=device)

        for c in range(num_classes):
            idx = (labels == c)
            if idx.any():
                sums[c] += h[idx].double().sum(dim=0)
                counts[c] += idx.sum()

    if sums is None:
        raise RuntimeError("No samples in clean loader to compute class means.")

    mu = torch.zeros_like(sums, dtype=torch.float64)
    for c in range(num_classes):
        if counts[c].item() > 0:
            mu[c] = sums[c] / counts[c].double()
        else:
            mu[c].zero_()

    return mu.float()


def estimate_covariances(
    model: FSNetClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    dt_thresholds,
    num_dt_buckets: int,
    state_pad_id: int,
    mu_class: torch.Tensor,
    mc_samples: int = 20,
    reg_scale: float = 1e-4,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert mc_samples >= 2, "Need mc_samples>=2 to estimate within-sample covariance."
    torch.manual_seed(seed)

    D = mu_class.shape[1]
    scatter_eps = torch.zeros((D, D), dtype=torch.float64, device=device)
    sum_r = torch.zeros((D,), dtype=torch.float64, device=device)
    sum_rr = torch.zeros((D, D), dtype=torch.float64, device=device)
    N = 0

    for batch in loader:
        labels = batch.labels.to(device)
        B = labels.shape[0]

        reps = []
        for _ in range(mc_samples):
            reps.append(
                extract_features_batch(
                    model, batch, device, dt_thresholds, num_dt_buckets, state_pad_id, mc_dropout=True
                )
            )
        H = torch.stack(reps, dim=0)
        fbar = H.mean(dim=0)

        E = (H - fbar.unsqueeze(0)).reshape(-1, D).double()
        scatter_eps += E.t().mm(E)

        mu_y = mu_class[labels]
        r = (fbar - mu_y).double()
        sum_r += r.sum(dim=0)
        sum_rr += r.t().mm(r)
        N += B

    if N <= 1:
        raise RuntimeError("Not enough clean samples to estimate covariances.")

    S_eps = scatter_eps / (float(N) * float(mc_samples - 1))

    mean_r = sum_r / float(N)
    S_mu = (sum_rr - float(N) * mean_r.view(-1, 1).mm(mean_r.view(1, -1))) / float(max(N - 1, 1))

    def reg_cov(S: torch.Tensor) -> torch.Tensor:
        tr = torch.trace(S).item()
        eps = reg_scale * (tr / max(D, 1) + 1e-12)
        return S + eps * torch.eye(D, dtype=S.dtype, device=S.device)

    S_eps = reg_cov(S_eps)
    S_mu = reg_cov(S_mu)

    return S_eps.float(), S_mu.float()


def build_decomposition_matrix(S_eps: torch.Tensor, S_mu: torch.Tensor) -> torch.Tensor:
    S_sum = S_mu + S_eps
    A = torch.linalg.solve(S_sum, S_mu).transpose(0, 1)
    return A


def kmeans2_whitened(z: torch.Tensor, iters: int = 50, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    N, D = z.shape
    if N < 2:
        assign = torch.zeros((N,), dtype=torch.long, device=z.device)
        centers = z[:1].repeat(2, 1)
        return assign, centers

    idx0 = torch.randint(0, N, (1,), device=z.device).item()
    c0 = z[idx0:idx0 + 1]
    dist = ((z - c0) ** 2).sum(dim=1)
    idx1 = int(dist.argmax().item())
    c1 = z[idx1:idx1 + 1]
    centers = torch.cat([c0, c1], dim=0)

    for _ in range(iters):
        dist2_0 = ((z - centers[0]) ** 2).sum(dim=1)
        dist2_1 = ((z - centers[1]) ** 2).sum(dim=1)
        assign = (dist2_1 < dist2_0).long()

        new_centers = centers.clone()
        for k in [0, 1]:
            idx = (assign == k)
            if idx.any():
                new_centers[k] = z[idx].mean(dim=0)
            else:
                rid = torch.randint(0, N, (1,), device=z.device).item()
                new_centers[k] = z[rid]

        if torch.norm(new_centers - centers).item() < 1e-5:
            centers = new_centers
            break
        centers = new_centers

    return assign, centers


def _logsumexp2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = torch.maximum(a, b)
    return m + torch.log(torch.exp(a - m) + torch.exp(b - m) + 1e-12)


def _stable_cov_root(Sigma: torch.Tensor) -> torch.Tensor:
    Sigma = 0.5 * (Sigma + Sigma.t())
    D = Sigma.shape[0]
    eye = torch.eye(D, dtype=Sigma.dtype, device=Sigma.device)
    base = torch.trace(Sigma).abs() / max(D, 1)
    jitter = torch.clamp(base * 1e-6, min=torch.tensor(1e-8, dtype=Sigma.dtype, device=Sigma.device))

    for i in range(8):
        L, info = torch.linalg.cholesky_ex(Sigma + (jitter * (10.0 ** i)) * eye)
        if int(info.item()) == 0:
            return L

    eigvals, eigvecs = torch.linalg.eigh(Sigma)
    eigvals = eigvals.clamp(min=float(jitter.item()))
    return eigvecs.mm(torch.diag(torch.sqrt(eigvals))).mm(eigvecs.t())


def class_lrt_score(
    X: torch.Tensor,
    Sigma: torch.Tensor,
    iters: int = 100,
    seed: int = 0,
    tol: float = 1e-5,
    pi_floor: float = 1e-3,
    lrt_dim: int = 32,
    bic_lambda: float = 1.0,
    min_pi: float = 0.03,
) -> float:
    torch.manual_seed(seed)
    N = int(X.shape[0])
    if N <= 2:
        return 0.0

    L = _stable_cov_root(Sigma)
    z = torch.linalg.solve(L, X.t()).t()

    D_full = int(z.shape[1])
    if int(lrt_dim) > 0 and int(lrt_dim) < D_full:
        zc = z - z.mean(dim=0, keepdim=True)
        _, _, Vh = torch.linalg.svd(zc, full_matrices=False)
        Vk = Vh[: int(lrt_dim)].transpose(0, 1)
        z = zc.mm(Vk)

    z0 = z.mean(dim=0, keepdim=True)
    d0 = ((z - z0) ** 2).sum(dim=1)
    ll0 = (-0.5 * d0).sum()

    init_assign, centers = kmeans2_whitened(z, iters=50, seed=seed + 17)
    m0 = centers[0].detach()
    m1 = centers[1].detach()
    pi = init_assign.float().mean().clamp(pi_floor, 1.0 - pi_floor)

    ll1_old = None
    for _ in range(int(iters)):
        d_m0 = ((z - m0.view(1, -1)) ** 2).sum(dim=1)
        d_m1 = ((z - m1.view(1, -1)) ** 2).sum(dim=1)

        logp0 = torch.log(pi) - 0.5 * d_m0
        logp1 = torch.log(1.0 - pi) - 0.5 * d_m1
        log_norm = _logsumexp2(logp0, logp1)
        ll1 = log_norm.sum()

        r0 = torch.exp(logp0 - log_norm)
        r1 = 1.0 - r0

        n0 = r0.sum().clamp(min=1e-8)
        n1 = r1.sum().clamp(min=1e-8)
        pi = (n0 / float(N)).clamp(pi_floor, 1.0 - pi_floor)
        m0 = (r0.unsqueeze(1) * z).sum(dim=0) / n0
        m1 = (r1.unsqueeze(1) * z).sum(dim=0) / n1

        if ll1_old is not None and torch.abs(ll1 - ll1_old).item() < float(tol):
            break
        ll1_old = ll1

    d_eff = int(z.shape[1])
    lrt = float((2.0 * (ll1 - ll0)).item())
    penalty = 0.0
    if float(bic_lambda) > 0.0:
        penalty = float(bic_lambda) * float(d_eff + 1) * float(math.log(max(N, 2)))

    lrt_pen = lrt - penalty

    pi_small = float(min(float(pi.item()), float(1.0 - pi.item())))
    if float(min_pi) > 0.0 and pi_small < float(min_pi):
        lrt_pen = 0.0

    if lrt_pen < 0.0:
        lrt_pen = 0.0

    Jbar = float(lrt_pen / float(N))
    if not np.isfinite(Jbar):
        Jbar = 0.0

    return Jbar


def robust_outlier_zscores(
    values: List[float],
    two_sided: bool = False,
    clip_z: float = 50.0,
) -> List[float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return []

    med = float(np.median(x))
    abs_dev = np.abs(x - med)
    mad = float(np.median(abs_dev))

    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale < 1e-9:
        q1, q3 = np.percentile(x, [25.0, 75.0])
        iqr = float(q3 - q1)
        scale = 0.7413 * iqr

    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(np.std(x, ddof=1)) if x.size > 1 else 1.0

    scale = float(max(scale, 1e-12))

    if two_sided:
        z = abs_dev / scale
    else:
        z = (x - med) / scale
        z = np.maximum(z, 0.0)

    return np.minimum(z, float(clip_z)).tolist()


def load_ckpt(ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("state_dict", None)
    if sd is None:
        sd = ckpt.get("model_state_dict", None)
    if sd is None:
        raise KeyError("Checkpoint must contain 'state_dict' or 'model_state_dict'.")
    ckpt["__state_dict__"] = sd
    return ckpt


def build_model_from_ckpt(ckpt: Dict[str, Any], device: torch.device) -> Tuple[FSNetClassifier, Dict[str, Any]]:
    args = ckpt.get("args", {}) or {}
    label2id = ckpt.get("label2id", None) or {}
    num_classes = len(label2id) if len(label2id) > 0 else int(args.get("num_classes", 0))

    use_dir = not bool(args.get("no_dir", False))
    use_dt_bucket = not bool(args.get("no_dt_bucket", False))
    configure_feature_flags(use_dir=use_dir, use_dt_bucket=use_dt_bucket)

    dt_thresholds = DT_THRESHOLDS_DEFAULT
    num_dt_buckets, _, num_states = get_num_states(
        dt_thresholds=dt_thresholds,
        use_dt_bucket=use_dt_bucket,
        use_dir=use_dir,
    )
    state_pad_id = num_states

    model = FSNetClassifier(
        num_states=num_states,
        num_classes=num_classes,
        d_model=int(args.get("d_model", 64)),
        backbone=str(args.get("backbone", "gru")),
        hidden_dim=int(args.get("rnn_hidden", 128)),
        num_layers=int(args.get("rnn_layers", 2)),
        dropout=float(args.get("dropout", 0.1)),
        max_pkt_len=float(args.get("max_pkt_len", 1500.0)),
        transformer_heads=int(args.get("transformer_heads", 4)),
        cnn_kernel_size=int(args.get("cnn_kernel_size", 3)),
    ).to(device)

    model.load_state_dict(ckpt["__state_dict__"])
    model.eval()

    info = {
        "args": args,
        "label2id": label2id,
        "id2label": ckpt.get("id2label", None) or {},
        "dt_thresholds": dt_thresholds,
        "num_dt_buckets": num_dt_buckets,
        "num_states": num_states,
        "state_pad_id": state_pad_id,
    }
    return model, info


def parse_args():
    p = argparse.ArgumentParser(description="SCAn detection for FS-Net-like models (traffic sequences).")

    p.add_argument("--ckpt", "--cls_ckpt", dest="ckpt", type=str, required=True, help="Path to classifier checkpoint (.pt).")
    p.add_argument(
        "--clean_jsonl",
        "--clean_subset_jsonl",
        dest="clean_jsonl",
        type=str,
        required=True,
        help="Trusted clean subset jsonl (per-class).",
    )
    p.add_argument(
        "--inspect_jsonl",
        "--triggered_jsonl",
        dest="inspect_jsonl",
        type=str,
        required=True,
        help="Dataset jsonl to scan (labeled).",
    )
    p.add_argument("--out_dir", "--output_dir", dest="out_dir", type=str, required=True, help="Output directory.")

    p.add_argument(
        "--mix_base_jsonl",
        type=str,
        default=None,
        help=(
            "If set, we CONCATENATE this base jsonl BEFORE inspect_jsonl, and scan the mixed dataset. "
            "This is handy when inspect_jsonl contains mostly one class (e.g., triggered dumps). "
            "Note: clean_jsonl is still used as the trusted clean set for covariance estimation."
        ),
    )

    p.add_argument("--max_len", type=int, default=128, help="Max packets per flow.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--label_key", type=str, default=None, help="Label key for standard jsonl (if not inferable).")

    p.add_argument("--inspect_use_triggered_lengths", action="store_true",
                   help="Use 'lengths_triggered' from triggered-dump jsonl.")
    p.add_argument("--force_label_id", type=int, default=None,
                   help="Force all inspected samples to this label id (useful for triggered dumps).")

    p.add_argument("--mc_samples", type=int, default=20,
                   help="Number of MC-dropout variants per clean sample (>=2).")
    p.add_argument("--reg_scale", type=float, default=1e-4,
                   help="Diagonal regularization scale for covariance matrices.")
    p.add_argument("--kmeans_iters", type=int, default=50, help="Iterations for 2-means.")

    p.add_argument(
        "--lrt_dim",
        type=int,
        default=32,
        help="Run the per-class EM/LRT in a PCA-reduced whitened space of this dimension (0 = use full D).",
    )
    p.add_argument(
        "--bic_lambda",
        type=float,
        default=1.0,
        help=(
            "Apply a BIC-style penalty to the 2-subgroup model: "
            "J̄ = max(0, (2ΔlogL - bic_lambda*(d+1)logN)/N). Set 0 to disable."
        ),
    )
    p.add_argument(
        "--min_pi",
        type=float,
        default=0.03,
        help=(
            "Minimum mixture weight for the smaller subgroup. "
            "If min(pi,1-pi) < min_pi, treat the split as 'a few outliers' and set J̄=0."
        ),
    )
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--scan_classes", type=str, default="all",
                   help="Comma-separated class ids to scan, or 'all'.")

    p.add_argument(
        "--scan_min_samples",
        "--min_samples",
        dest="scan_min_samples",
        type=int,
        default=20,
        help="Minimum samples per class to compute the per-class LRT statistic (SCAn). Classes below this are skipped.",
    )

    p.add_argument("--threshold_z", type=float, default=2.0,
                   help="Robust z threshold for flagging contaminated classes.")


    p.add_argument(
        "--outlier_two_sided",
        action="store_true",
        help="If set, use two-sided robust z = |x - median|/MAD (default is one-sided, only large J̄).",
    )
    p.add_argument(
        "--filter_no_inspect",
        action="store_true",
        help="(Useful with --mix_base_jsonl) Ignore classes that contain no (or too few) samples from source='inspect'.",
    )
    p.add_argument(
        "--min_inspect_count",
        type=int,
        default=1,
        help="Minimum number of 'inspect' samples required for a class to be considered when --filter_no_inspect is set.",
    )
    p.add_argument("--cpu", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[Info] device={device}")

    ckpt = load_ckpt(args.ckpt, device=device)
    model, info = build_model_from_ckpt(ckpt, device=device)

    label2id = info["label2id"]
    id2label = info["id2label"]
    num_classes = len(label2id)
    dt_thresholds = info["dt_thresholds"]
    num_dt_buckets = info["num_dt_buckets"]
    state_pad_id = info["state_pad_id"]

    clean_ds_raw = JsonlFlowDataset(
        args.clean_jsonl, max_len=args.max_len,
        label2id=label2id, label_key=args.label_key,
        use_triggered_lengths=False, force_label_id=None,
    )
    clean_ds = OffsetDataset(clean_ds_raw, source="trusted_clean")
    clean_loader = torch.utils.data.DataLoader(
        clean_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_batch, pin_memory=(device.type == "cuda")
    )

    inspect_ds_raw = JsonlFlowDataset(
        args.inspect_jsonl, max_len=args.max_len,
        label2id=label2id, label_key=args.label_key,
        use_triggered_lengths=args.inspect_use_triggered_lengths,
        force_label_id=args.force_label_id,
    )

    if args.mix_base_jsonl is not None and str(args.mix_base_jsonl).strip() != "":
        mix_base_ds = JsonlFlowDataset(
            args.mix_base_jsonl, max_len=args.max_len,
            label2id=label2id, label_key=args.label_key,
            use_triggered_lengths=False, force_label_id=None,
        )
        base_wrapped = OffsetDataset(mix_base_ds, source="mix_base")
        insp_wrapped = OffsetDataset(inspect_ds_raw, source="inspect")
        inspect_ds = torch.utils.data.ConcatDataset([base_wrapped, insp_wrapped])
        print(
            f"[Data] Using MIXED inspected dataset: mix_base_jsonl({len(mix_base_ds)}) + "
            f"inspect_jsonl({len(inspect_ds_raw)}) = total({len(inspect_ds)})"
        )
    else:
        inspect_ds = OffsetDataset(inspect_ds_raw, source="inspect")
        print(f"[Data] Using inspected dataset only: inspect_jsonl({len(inspect_ds_raw)})")

    inspect_loader = torch.utils.data.DataLoader(
        inspect_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_batch, pin_memory=(device.type == "cuda")
    )

    print("[SCAn] Estimating class means from trusted clean subset ...")
    mu_class = compute_class_means(
        model, clean_loader, device, dt_thresholds, num_dt_buckets, state_pad_id,
        num_classes=num_classes, mc_samples=max(1, min(args.mc_samples, 5)), seed=args.seed
    )
    print("[SCAn] mu_class shape:", tuple(mu_class.shape))

    print("[SCAn] Estimating covariances (S_eps, S_mu) via MC-dropout variants ...")
    S_eps, S_mu = estimate_covariances(
        model, clean_loader, device, dt_thresholds, num_dt_buckets, state_pad_id,
        mu_class=mu_class.to(device),
        mc_samples=max(2, args.mc_samples),
        reg_scale=args.reg_scale,
        seed=args.seed,
    )
    print("[SCAn] S_eps/S_mu shape:", tuple(S_eps.shape), tuple(S_mu.shape))

    A = build_decomposition_matrix(S_eps.to(device), S_mu.to(device))
    Sigma_hat = A.mm(S_eps.to(device)).mm(A.t())
    D = Sigma_hat.shape[0]
    Sigma_hat = Sigma_hat + (args.reg_scale * torch.trace(Sigma_hat) / max(D, 1) + 1e-12) * torch.eye(D, device=device)
    Sigma_hat = Sigma_hat.float()

    print("[SCAn] Extracting decomposed features r_hat for inspected dataset ...")
    feats_by_class: List[List[np.ndarray]] = [[] for _ in range(num_classes)]
    inspect_counts = {c: 0 for c in range(num_classes)}

    for batch in inspect_loader:
        labels = batch.labels.to(device)
        h = extract_features_batch(model, batch, device, dt_thresholds, num_dt_buckets, state_pad_id, mc_dropout=False)
        mu_y = mu_class.to(device)[labels]
        r_hat = (h - mu_y).mm(A.t())

        r_hat_cpu = r_hat.detach().cpu().numpy().astype(np.float32)
        labels_cpu = labels.detach().cpu().numpy().astype(np.int32)

        for i in range(r_hat_cpu.shape[0]):
            c = int(labels_cpu[i])
            feats_by_class[c].append(r_hat_cpu[i])
            if str(batch.sources[i]) == "inspect":
                inspect_counts[c] += 1

    if args.scan_classes.strip().lower() == "all":
        scan_set = list(range(num_classes))
    else:
        scan_set = [int(x) for x in args.scan_classes.split(",") if x.strip()]

    print("[SCAn] Computing per-class statistics ...")
    jbars: List[float] = []
    for c in range(num_classes):
        if c not in scan_set or len(feats_by_class[c]) < int(args.scan_min_samples):
            jbars.append(float("nan"))
            continue

        xc = torch.from_numpy(np.stack(feats_by_class[c], axis=0)).to(device)
        jbars.append(
            class_lrt_score(
                xc,
                Sigma_hat,
                iters=args.kmeans_iters,
                seed=args.seed + 13 * c,
                lrt_dim=int(args.lrt_dim),
                bic_lambda=float(args.bic_lambda),
                min_pi=float(args.min_pi),
            )
        )

    valid_classes = [c for c in scan_set if np.isfinite(jbars[c])]
    valid_vals = [float(jbars[c]) for c in valid_classes]
    z_scores = robust_outlier_zscores(valid_vals, two_sided=args.outlier_two_sided)
    z_map = {c: float(z_scores[i]) for i, c in enumerate(valid_classes)}

    detected = [
        c
        for c in valid_classes
        if z_map[c] >= float(args.threshold_z)
        and (not args.filter_no_inspect or inspect_counts.get(c, 0) >= int(args.min_inspect_count))
    ]
    detected_sorted = sorted(detected, key=lambda c: z_map[c], reverse=True)

    top_class = max(valid_classes, key=lambda c: z_map[c]) if valid_classes else None
    target_class = int(args.force_label_id) if args.force_label_id is not None else None
    target_robust_z = z_map.get(target_class) if target_class is not None else None

    report: Dict[str, Any] = {
        "ckpt": os.path.abspath(args.ckpt),
        "clean_jsonl": os.path.abspath(args.clean_jsonl),
        "inspect_jsonl": os.path.abspath(args.inspect_jsonl),
        "mix_base_jsonl": os.path.abspath(args.mix_base_jsonl) if args.mix_base_jsonl else None,
        "inspect_use_triggered_lengths": bool(args.inspect_use_triggered_lengths),
        "force_label_id": args.force_label_id,
        "max_len": int(args.max_len),
        "mc_samples": int(args.mc_samples),
        "reg_scale": float(args.reg_scale),
        "threshold_z": float(args.threshold_z),
        "target_class_id": target_class,
        "target_detected": bool(target_class in detected_sorted) if target_class is not None else None,
        "target_robust_z": target_robust_z,
        "top_suspect_class_id": top_class,
        "top_robust_z": z_map.get(top_class) if top_class is not None else None,
        "detected_classes": [],
        "class_stats": {},
    }

    for c in scan_set:
        report["class_stats"][str(c)] = {
            "class_id": c,
            "class_name": id2label.get(c, c),
            "n_samples": len(feats_by_class[c]),
            "n_inspect": int(inspect_counts.get(c, 0)),
            "robust_z": z_map.get(c),
        }

    for c in detected_sorted:
        report["detected_classes"].append(
            {
                "class_id": c,
                "class_name": id2label.get(c, c),
                "n_samples": len(feats_by_class[c]),
                "n_inspect": int(inspect_counts.get(c, 0)),
                "robust_z": z_map[c],
            }
        )

    out_path = os.path.join(args.out_dir, "scan_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[Done] Report saved to: {out_path}")
    if target_class is not None:
        print(
            f"[SCAn] target_class={target_class} target_detected={report['target_detected']} "
            f"target_robust_z={target_robust_z} threshold_z={args.threshold_z}"
        )
    if top_class is not None:
        print(f"[SCAn] top_suspect_class={top_class} top_robust_z={z_map[top_class]:.4f}")

    if not detected_sorted:
        print("[Result] No contaminated class detected (robust_z < threshold).")
    else:
        print("[Result] Detected contaminated classes (sorted):")
        for item in report["detected_classes"]:
            print("  -", item)


if __name__ == "__main__":
    main()
