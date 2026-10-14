"""Общий инференс для скриптов: предсказания батчами с сортировкой по длине."""
import time

import numpy as np
import torch

from src.project_utils import MAX_LENGTH


@torch.no_grad()
def predict(model, tok, texts, device, batch_size=128) -> np.ndarray:
    """Предсказания батчами; тексты отсортированы по длине, чтобы меньше паддинга."""
    order = np.argsort([len(t) for t in texts])
    preds = np.empty(len(texts), dtype=np.int64)
    model.eval()
    use_amp = device.type == "cuda"
    n_batches = -(-len(texts) // batch_size)
    t_start = time.time()
    for b, i in enumerate(range(0, len(texts), batch_size)):
        if b % 10 == 0:
            print(f"    предсказание: батч {b}/{n_batches}  ({time.time() - t_start:.0f} с)", flush=True)
        idx = order[i:i + batch_size]
        enc = tok(
            [texts[j] for j in idx], truncation=True, max_length=MAX_LENGTH,
            padding=True, return_tensors="pt",
        ).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(**enc).logits
        preds[idx] = logits.argmax(dim=-1).cpu().numpy()
    return preds