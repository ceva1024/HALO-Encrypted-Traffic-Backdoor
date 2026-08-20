from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset


TASK_TO_LABEL_KEY = {
    "vpn": "vpn_flag",
    "service": "service",
    "app": "app",
}


@dataclass
class FlowBatch:
    lengths: torch.Tensor
    times: torch.Tensor
    dirs: torch.Tensor
    can_modify: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor
    target_labels: torch.Tensor | None = None

    def to(self, device: torch.device) -> "FlowBatch":
        return FlowBatch(
            lengths=self.lengths.to(device),
            times=self.times.to(device),
            dirs=self.dirs.to(device),
            can_modify=self.can_modify.to(device),
            mask=self.mask.to(device),
            labels=self.labels.to(device),
            target_labels=None if self.target_labels is None else self.target_labels.to(device),
        )


class FlowDataset(Dataset):
    def __init__(self, jsonl_path: str, label_key: str, label2id: Dict, max_len: int = 128):
        self.jsonl_path = jsonl_path
        self.label_key = label_key
        self.label2id = label2id
        self.max_len = int(max_len)
        self.samples: List[Dict] = []
        self._load()

    def _label_from_obj(self, obj: Dict):
        if self.label_key == "vpn_flag":
            return int(obj.get("vpn_flag"))
        return obj.get(self.label_key)

    def _load(self) -> None:
        skipped = 0
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                lengths = obj.get("lengths", [])
                times = obj.get("times", [])
                dirs = obj.get("dirs", [])
                can_modify = obj.get("can_modify", [])
                if not (lengths and times and dirs and can_modify):
                    skipped += 1
                    continue
                n = min(len(lengths), len(times), len(dirs), len(can_modify), self.max_len)
                if n <= 0:
                    skipped += 1
                    continue
                raw_label = self._label_from_obj(obj)
                if raw_label not in self.label2id:
                    skipped += 1
                    continue
                self.samples.append(
                    {
                        "lengths": lengths[:n],
                        "times": times[:n],
                        "dirs": dirs[:n],
                        "can_modify": can_modify[:n],
                        "label": int(self.label2id[raw_label]),
                        "_pad_to": self.max_len,
                    }
                )
        print(f"[Dataset] {self.jsonl_path}: loaded={len(self.samples)} skipped={skipped}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class CleanSubsetDataset(Dataset):
    def __init__(self, jsonl_path: str, max_len: int = 128):
        self.samples: List[Dict] = []
        self.max_len = int(max_len)
        skipped = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                lengths = obj.get("lengths", [])
                times = obj.get("times", [])
                dirs = obj.get("dirs", [])
                can_modify = obj.get("can_modify", [1.0] * len(lengths))
                n = min(len(lengths), len(times), len(dirs), len(can_modify), self.max_len)
                if n <= 0 or "label_id" not in obj:
                    skipped += 1
                    continue
                self.samples.append(
                    {
                        "lengths": lengths[:n],
                        "times": times[:n],
                        "dirs": dirs[:n],
                        "can_modify": can_modify[:n],
                        "label": int(obj["label_id"]),
                        "_pad_to": self.max_len,
                    }
                )
        print(f"[CleanSubsetDataset] {jsonl_path}: loaded={len(self.samples)} skipped={skipped}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class TriggeredDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        max_len: int = 128,
        use_triggered_lengths: bool = True,
        prefer_lengths_key: bool = False,
    ):
        self.samples: List[Dict] = []
        self.max_len = int(max_len)
        self.use_triggered_lengths = bool(use_triggered_lengths)
        self.prefer_lengths_key = bool(prefer_lengths_key)
        skipped = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if self.prefer_lengths_key:
                    lengths = obj.get("lengths", obj.get("lengths_triggered", []))
                else:
                    len_key = "lengths_triggered" if self.use_triggered_lengths else "lengths_clean"
                    lengths = obj.get(len_key, obj.get("lengths", obj.get("lengths_triggered", [])))
                times = obj.get("times", obj.get("times_triggered", []))
                dirs = obj.get("dirs", obj.get("dirs_triggered", []))
                can_modify = obj.get("can_modify", obj.get("can_modify_triggered", [1.0] * len(lengths)))
                n = min(len(lengths), len(times), len(dirs), len(can_modify), self.max_len)
                if n <= 0:
                    skipped += 1
                    continue
                self.samples.append(
                    {
                        "lengths": lengths[:n],
                        "times": times[:n],
                        "dirs": dirs[:n],
                        "can_modify": can_modify[:n],
                        "label": int(obj.get("orig_label_id", obj.get("label_id", 0))),
                        "target_label": int(obj.get("target_label_id", obj.get("label_id", 0))),
                        "_pad_to": self.max_len,
                    }
                )
        print(f"[TriggeredDataset] {jsonl_path}: loaded={len(self.samples)} skipped={skipped}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def _tensor1d(values: Iterable, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(list(values), dtype=dtype)


def collate_flows(batch: List[Dict]) -> FlowBatch:
    seq_lens = [len(s["lengths"]) for s in batch]
    batch_size = len(batch)
    max_t = max(max(seq_lens), max(int(s.get("_pad_to", 0)) for s in batch))

    lengths = torch.zeros(batch_size, max_t, dtype=torch.float32)
    times = torch.zeros_like(lengths)
    dirs = torch.zeros_like(lengths)
    can_modify = torch.zeros_like(lengths)
    mask = torch.zeros(batch_size, max_t, dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)
    target_labels = torch.zeros(batch_size, dtype=torch.long)
    has_targets = any("target_label" in s for s in batch)

    for i, sample in enumerate(batch):
        n = seq_lens[i]
        lengths[i, :n] = _tensor1d(sample["lengths"], torch.float32)
        times[i, :n] = _tensor1d(sample["times"], torch.float32)
        dirs[i, :n] = _tensor1d(sample["dirs"], torch.float32)
        can_modify[i, :n] = _tensor1d(sample["can_modify"], torch.float32)
        mask[i, :n] = True
        labels[i] = int(sample["label"])
        target_labels[i] = int(sample.get("target_label", sample["label"]))

    return FlowBatch(lengths, times, dirs, can_modify, mask, labels, target_labels if has_targets else None)


def build_label_map(train_jsonl: str, task: str, app_min: int | None = None, app_max: int | None = None):
    if task not in TASK_TO_LABEL_KEY:
        raise ValueError(f"Unknown task '{task}', expected one of {sorted(TASK_TO_LABEL_KEY)}")
    label_key = TASK_TO_LABEL_KEY[task]

    allowed_app = None
    if task == "app" and app_min is not None and app_max is not None:
        if app_min > app_max:
            raise ValueError(f"app_min({app_min}) > app_max({app_max})")
        allowed_app = set(range(int(app_min), int(app_max) + 1))

    labels = set()
    with open(train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            raw = int(obj.get("vpn_flag")) if label_key == "vpn_flag" else obj.get(label_key)
            if allowed_app is not None and raw not in allowed_app:
                continue
            labels.add(raw)

    if label_key == "vpn_flag":
        label2id = {0: 0, 1: 1}
    else:
        label2id = {label: i for i, label in enumerate(sorted(labels))}
    id2label = {v: k for k, v in label2id.items()}
    print(f"[LabelMap] task={task} label_key={label_key} num_classes={len(label2id)}")
    print(f"[LabelMap] {label2id}")
    return label2id, id2label, label_key


def parse_label_token(token: str, label2id: Dict, label_key: str):
    if label_key == "vpn_flag":
        return int(token)
    sample_key = next(iter(label2id.keys()))
    return int(token) if isinstance(sample_key, int) else token


def build_poison_source_ids(raw_labels: str, label2id: Dict, label_key: str) -> Optional[set[int]]:
    if not raw_labels:
        print("[Backdoor] poison_source_labels unset: all non-target classes are eligible.")
        return None
    source_ids: set[int] = set()
    for token in [x.strip() for x in raw_labels.split(",") if x.strip()]:
        try:
            raw_label = parse_label_token(token, label2id, label_key)
        except ValueError:
            print(f"[Backdoor] skip unparsable source label: {token}")
            continue
        if raw_label not in label2id:
            print(f"[Backdoor] skip source label not in label map: {raw_label}")
            continue
        source_ids.add(int(label2id[raw_label]))
    print(f"[Backdoor] poison_source_ids={sorted(source_ids)}")
    return source_ids


def dump_train_subset_jsonl(
    dataset: FlowDataset,
    id2label: Dict[int, object],
    out_path: str,
    samples_per_class: int = 100,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        by_class[int(sample["label"])].append(idx)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for label_id, indices in sorted(by_class.items()):
            chosen = indices if len(indices) <= samples_per_class else rng.sample(indices, samples_per_class)
            for idx in chosen:
                sample = dataset.samples[idx]
                obj = {
                    "lengths": sample["lengths"],
                    "times": sample["times"],
                    "dirs": sample["dirs"],
                    "can_modify": sample["can_modify"],
                    "label_id": int(label_id),
                    "label_raw": id2label[label_id],
                }
                f.write(json.dumps(obj) + "\n")
                written += 1
    print(f"[Dump] wrote {written} clean subset samples to {out_path}")
