from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import torch

from .config import FeatureConfig
from .models import FSNetClassifier


def state_dict_from_ckpt(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state = ckpt.get("state_dict", None)
    if state is None:
        state = ckpt.get("model_state_dict", None)
    if state is None:
        raise KeyError("Checkpoint must contain 'state_dict' or 'model_state_dict'.")
    return state


def model_kwargs_from_args(args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "d_model": int(args.get("d_model", 64)),
        "backbone": str(args.get("backbone", "gru")),
        "hidden_dim": int(args.get("rnn_hidden", args.get("hidden_dim", 128))),
        "num_layers": int(args.get("rnn_layers", args.get("num_layers", 2))),
        "dropout": float(args.get("dropout", 0.1)),
        "max_pkt_len": float(args.get("max_pkt_len", 1500.0)),
        "transformer_heads": int(args.get("transformer_heads", 4)),
        "cnn_kernel_size": int(args.get("cnn_kernel_size", 3)),
    }


def build_model_from_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[FSNetClassifier, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {}) or {}
    label2id = ckpt.get("label2id", {}) or {}
    num_classes = len(label2id) if label2id else int(args.get("num_classes", 0))
    if num_classes <= 0:
        raise ValueError("Cannot infer num_classes from checkpoint.")

    config = FeatureConfig(
        use_dir=not bool(args.get("no_dir", False)),
        use_dt_bucket=not bool(args.get("no_dt_bucket", False)),
        max_pkt_len=float(args.get("max_pkt_len", 1500.0)),
    )
    kwargs = model_kwargs_from_args(args)
    kwargs["max_pkt_len"] = config.max_pkt_len
    model = FSNetClassifier(config.num_states, num_classes, **kwargs).to(device)
    model.load_state_dict(state_dict_from_ckpt(ckpt))
    model.eval()
    info = {
        "ckpt": ckpt,
        "args": args,
        "label2id": label2id,
        "id2label": ckpt.get("id2label", {}) or {},
        "config": config,
        "model_kwargs": kwargs,
    }
    return model, info


def save_classifier(path: str, model, label2id, id2label, args, **extra) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "label2id": label2id,
        "id2label": id2label,
        "args": vars(args) if not isinstance(args, dict) else args,
    }
    payload.update(extra)
    torch.save(payload, path)


def save_joint(path: str, model, generator, label2id, id2label, args, **extra) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "generator_state_dict": None if generator is None else generator.state_dict(),
        "label2id": label2id,
        "id2label": id2label,
        "args": vars(args) if not isinstance(args, dict) else args,
    }
    payload.update(extra)
    torch.save(payload, path)
