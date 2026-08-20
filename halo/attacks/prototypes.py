from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from ..features import classifier_inputs


def kmeans_np(x: np.ndarray, k: int, max_iters: int = 50, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    if k <= 0:
        raise ValueError("k must be positive")
    if n <= k:
        labels = np.arange(n)
        return x.copy(), labels
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iters):
        dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = dist2.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k):
            members = x[new_labels == j]
            if members.size:
                new_centers[j] = members.mean(axis=0)
            else:
                new_centers[j] = x[rng.integers(0, n)]
        shift = float(np.linalg.norm(new_centers - centers))
        centers = new_centers
        labels = new_labels
        if shift < 1e-4:
            break
    return centers, labels


def _knn_mean_distance(x: np.ndarray, ref: np.ndarray, k: int = 5, exclude_self: bool = False) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros(n, dtype=np.float32)
    chunk = 512
    kk = min(k + (1 if exclude_self else 0), ref.shape[0])
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d2 = ((x[start:stop, None, :] - ref[None, :, :]) ** 2).sum(axis=-1)
        vals = np.partition(d2, kth=min(kk - 1, d2.shape[1] - 1), axis=1)[:, :kk]
        if exclude_self and x is ref and vals.shape[1] > 1:
            vals = vals[:, 1:]
        out[start:stop] = np.sqrt(vals + 1e-8).mean(axis=1)
    return out


def _nearest_distance(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros(n, dtype=np.float32)
    chunk = 512
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d2 = ((x[start:stop, None, :] - ref[None, :, :]) ** 2).sum(axis=-1)
        out[start:stop] = np.sqrt(d2.min(axis=1) + 1e-8)
    return out


def _density_peak_prototypes(
    target_x: np.ndarray,
    other_x: np.ndarray | None,
    num_prototypes: int,
    seed: int,
    core_fraction: float = 0.7,
    density_fraction: float = 0.3,
    micro_cluster_factor: int = 4,
) -> np.ndarray:
    if target_x.shape[0] <= num_prototypes:
        return target_x.copy()

    x = target_x
    if other_x is not None and other_x.shape[0] > 0 and x.shape[0] > 8:
        intra = _knn_mean_distance(x, x, k=min(5, x.shape[0] - 1), exclude_self=True)
        inter = _nearest_distance(x, other_x)
        margin = inter - intra
        keep = max(num_prototypes, int(np.ceil(core_fraction * x.shape[0])))
        idx = np.argsort(margin)[-keep:]
        x_core = x[idx]
    else:
        x_core = x

    density = 1.0 / (_knn_mean_distance(x_core, x_core, k=min(5, max(1, x_core.shape[0] - 1)), exclude_self=True) + 1e-8)
    k_big = min(max(num_prototypes * max(1, int(micro_cluster_factor)), num_prototypes), x_core.shape[0])
    centers, labels = kmeans_np(x_core, k_big, seed=seed)
    sizes = np.array([(labels == j).sum() for j in range(k_big)], dtype=np.float32)

    selected = []
    first = int(np.argmax(sizes))
    selected.append(first)
    while len(selected) < min(num_prototypes, k_big):
        best_j = None
        best_score = -1.0
        for j in range(k_big):
            if j in selected or sizes[j] <= 0:
                continue
            diversity = min(np.linalg.norm(centers[j] - centers[s]) for s in selected)
            score = float(sizes[j] * diversity)
            if score > best_score:
                best_score = score
                best_j = j
        if best_j is None:
            break
        selected.append(best_j)

    protos = []
    for cluster_id in selected:
        members_idx = np.where(labels == cluster_id)[0]
        if members_idx.size == 0:
            protos.append(centers[cluster_id])
            continue
        member_density = density[members_idx]
        top_n = max(1, int(np.ceil(density_fraction * members_idx.size)))
        top_idx = members_idx[np.argsort(member_density)[-top_n:]]
        protos.append(x_core[top_idx].mean(axis=0))
    return np.stack(protos, axis=0)


def _prototype_radii(
    target_x: np.ndarray,
    protos: np.ndarray,
    radius_percentile: float,
) -> np.ndarray:
    dist = np.sqrt(((target_x[:, None, :] - protos[None, :, :]) ** 2).sum(axis=-1) + 1e-8)
    assign = dist.argmin(axis=1)
    radii = []
    for proto_id in range(protos.shape[0]):
        local = dist[assign == proto_id, proto_id]
        if local.size == 0:
            local = dist[:, proto_id]
        radii.append(float(np.percentile(local, radius_percentile)))
    return np.asarray(radii, dtype=np.float32)


def build_layerwise_prototypes(
    model,
    data_loader,
    config,
    device: torch.device,
    target_class: int,
    num_prototypes: int = 4,
    max_samples: int = 5000,
    radius_percentile: float = 80.0,
    core_fraction: float = 0.7,
    density_fraction: float = 0.3,
    micro_cluster_factor: int = 4,
    seed: int = 0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    model.eval()
    target_chunks = None
    other_chunks = None
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            state_ids, lengths_signed, mask = classifier_inputs(batch, config, device)
            feats = model.extract_layer_features(state_ids, lengths_signed, mask)
            if target_chunks is None:
                target_chunks = [[] for _ in feats]
                other_chunks = [[] for _ in feats]
            tgt_mask = batch.labels == int(target_class)
            other_mask = ~tgt_mask
            for layer_id, feat in enumerate(feats):
                if tgt_mask.any():
                    target_chunks[layer_id].append(feat[tgt_mask].detach().cpu().numpy())
                if other_mask.any():
                    other_chunks[layer_id].append(feat[other_mask].detach().cpu().numpy())

    if target_chunks is None:
        raise RuntimeError(f"No data was available while building prototypes for target_class={target_class}")

    layer_protos: List[torch.Tensor] = []
    layer_radii: List[torch.Tensor] = []
    for layer_id, chunks in enumerate(target_chunks):
        if not chunks:
            raise RuntimeError(f"No target-class samples for layer {layer_id}")
        target_x = np.concatenate(chunks, axis=0)
        other_x = np.concatenate(other_chunks[layer_id], axis=0) if other_chunks[layer_id] else None

        if target_x.shape[0] > max_samples:
            target_x = target_x[rng.choice(target_x.shape[0], size=max_samples, replace=False)]
        if other_x is not None and other_x.shape[0] > max_samples:
            other_x = other_x[rng.choice(other_x.shape[0], size=max_samples, replace=False)]

        protos = _density_peak_prototypes(
            target_x,
            other_x,
            num_prototypes,
            seed=seed + layer_id,
            core_fraction=core_fraction,
            density_fraction=density_fraction,
            micro_cluster_factor=micro_cluster_factor,
        )
        radii = _prototype_radii(target_x, protos, radius_percentile)
        layer_protos.append(torch.tensor(protos.astype(np.float32), device=device))
        layer_radii.append(torch.tensor(radii, dtype=torch.float32, device=device))
        print(
            f"[Prototypes] layer={layer_id} target_n={target_x.shape[0]} "
            f"protos={protos.shape} radius_p{radius_percentile:g}="
            f"{float(radii.mean()):.4f}+/-{float(radii.std()):.4f}"
        )

    return layer_protos, layer_radii


def _torch_percentile(values: torch.Tensor, percentile: float) -> torch.Tensor:
    if values.numel() == 0:
        raise ValueError("values must be non-empty")
    q = min(max(float(percentile) / 100.0, 0.0), 1.0)
    kth = max(1, min(values.numel(), int(np.ceil(q * values.numel()))))
    return values.kthvalue(kth).values


def online_update_layer_protos_and_radii(
    layer_protos,
    layer_radii,
    feats_tgt_per_layer,
    momentum: float = 0.1,
    radius_percentile: float = 80.0,
):
    if momentum <= 0.0:
        return layer_protos, layer_radii
    with torch.no_grad():
        for layer_id, feats in enumerate(feats_tgt_per_layer):
            if feats is None or feats.numel() == 0:
                continue
            protos = layer_protos[layer_id].to(feats.device)
            radii = layer_radii[layer_id].to(feats.device)
            dist = torch.sqrt(((feats.unsqueeze(1) - protos.unsqueeze(0)) ** 2).sum(dim=-1) + 1e-8)
            assign = dist.argmin(dim=1)
            for k in range(protos.size(0)):
                idx = assign == k
                if idx.any():
                    protos[k].mul_(1.0 - momentum).add_(momentum * feats[idx].mean(dim=0))
            dist = torch.sqrt(((feats.unsqueeze(1) - protos.unsqueeze(0)) ** 2).sum(dim=-1) + 1e-8)
            assign = dist.argmin(dim=1)
            for k in range(protos.size(0)):
                idx = assign == k
                if idx.any():
                    radius = _torch_percentile(dist[idx, k], radius_percentile)
                    radii[k].mul_(1.0 - momentum).add_(momentum * radius)
            layer_protos[layer_id] = protos
            layer_radii[layer_id] = radii
    return layer_protos, layer_radii


def online_update_layer_protos(layer_protos, feats_tgt_per_layer, momentum: float = 0.1):
    layer_radii = [torch.zeros(proto.size(0), device=proto.device) for proto in layer_protos]
    layer_protos, _ = online_update_layer_protos_and_radii(
        layer_protos,
        layer_radii,
        feats_tgt_per_layer,
        momentum=momentum,
    )
    return layer_protos
