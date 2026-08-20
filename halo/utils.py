from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from .features import classifier_inputs


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cpu: bool = False) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")


def evaluate_clean(model, data_loader, config, device: torch.device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            state_ids, lengths_signed, mask = classifier_inputs(batch, config, device)
            logits = model(state_ids, lengths_signed, mask)
            loss = criterion(logits, batch.labels)
            preds = logits.argmax(dim=1)
            total += batch.labels.size(0)
            correct += (preds == batch.labels).sum().item()
            loss_sum += loss.item() * batch.labels.size(0)
    return loss_sum / max(1, total), correct / max(1, total)
