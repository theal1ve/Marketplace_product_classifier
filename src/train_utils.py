"""Общие части для скриптов обучения: датасет, токенизация, веса классов."""
import numpy as np
import torch

from src.project_utils import MAX_LENGTH, NUM_LABELS


class EncodedDataset(torch.utils.data.Dataset):
    """Токенизированные тексты + метки; паддинг делает DataCollatorWithPadding."""

    def __init__(self, enc: dict, labels: np.ndarray):
        self.ids = enc["input_ids"]
        self.mask = enc["attention_mask"]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        return {
            "input_ids": self.ids[i],
            "attention_mask": self.mask[i],
            "labels": int(self.labels[i]),
        }


def encode(tok, texts: list, chunk: int = 50_000) -> dict:
    ids, mask = [], []
    for i in range(0, len(texts), chunk):
        out = tok(texts[i:i + chunk], truncation=True, max_length=MAX_LENGTH)
        ids += out["input_ids"]
        mask += out["attention_mask"]
    return {"input_ids": ids, "attention_mask": mask}


def smooth_weights(y: np.ndarray) -> torch.Tensor:
    """Те же сглаженные веса, что в ноутбуке 02: корень, нормировка, clamp."""
    counts = np.maximum(np.bincount(y, minlength=NUM_LABELS), 1)
    raw = len(y) / (NUM_LABELS * counts)
    w = torch.sqrt(torch.tensor(raw, dtype=torch.float32))
    w = w / w.mean()
    return torch.clamp(w, min=0.5, max=5.0)