
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_utils import apply_split, load_filtered, texts_and_labels
from src.project_utils import MAX_LENGTH, MODEL_NAME

_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub(" ", s.lower()).strip()


def titles(part) -> pd.Series:
    return part.select_columns(["title"]).to_pandas()["title"].fillna("")


@torch.no_grad()
def predict(model, tok, texts, device, batch_size=256) -> np.ndarray:
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


def metrics(y, p) -> dict:
    if len(y) == 0:
        return {"n": 0}
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "f1_macro": float(f1_score(y, p, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, p, average="weighted", zero_division=0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="artifacts/split_ids.parquet")
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "v2_weighted"))
    ap.add_argument("--out", default="results/leak_check.json")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="батч для предсказаний (при нехватке видеопамяти уменьши до 64)")
    args = ap.parse_args()

    t0 = time.time()

    def log(msg: str) -> None:
        print(f"|{time.time() - t0:6.0f}s| {msg}", flush=True)

    log("загрузка датасета и split")
    ds, _ = load_filtered()
    split_df = pd.read_parquet(args.split)
    train, _, test = apply_split(ds, split_df)

    log("сборка текстов")
    train_texts, y_train = texts_and_labels(train)
    test_texts, y_test = texts_and_labels(test)
    tn_train = train_texts.map(norm)
    tn_test = test_texts.map(norm)
    title_train = titles(train).map(norm)
    title_test = titles(test).map(norm)

    log("поиск дублей")
    dup_text = tn_test.isin(set(tn_train)).to_numpy()
    dup_title = title_test.isin(set(title_train)).to_numpy()

    pairs = set(zip(tn_train.tolist(), y_train.tolist()))
    dup_pairs = zip(tn_test[dup_text].tolist(), y_test[dup_text].tolist())
    agree = np.array([(t, int(y)) in pairs for t, y in dup_pairs], dtype=bool)
    label_agreement = float(agree.mean()) if len(agree) else None

    n = len(y_test)
    leak = {
        "test_size": n,
        "dup_text_count": int(dup_text.sum()),
        "dup_text_share": float(dup_text.mean()),
        "dup_title_count": int(dup_title.sum()),
        "dup_title_share": float(dup_title.mean()),
        "dup_text_label_agreement": label_agreement,
    }
    log(f"дубли по тексту: {leak['dup_text_count']} ({leak['dup_text_share']:.2%}), "
        f"по title: {leak['dup_title_count']} ({leak['dup_title_share']:.2%})")
    if label_agreement is not None:
        log(f"у дублей по тексту метка совпадает с train в {label_agreement:.2%} случаев")

    log("загрузка модели и предсказания на test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"устройство: {device}")
    try:
        tok = AutoTokenizer.from_pretrained(args.model_dir)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    preds = predict(model, tok, test_texts.tolist(), device, batch_size=args.batch_size)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "v2_test_preds.npy", preds)

    results = {
        "leak": leak,
        "v2_all": metrics(y_test, preds),
        "v2_without_text_dups": metrics(y_test[~dup_text], preds[~dup_text]),
        "v2_without_title_dups": metrics(y_test[~dup_title], preds[~dup_title]),
        "v2_only_text_dups": metrics(y_test[dup_text], preds[dup_text]),
    }

    print()
    print(f"{'подмножество':<28}{'n':>8}{'accuracy':>10}{'macro F1':>10}")
    for key in ("v2_all", "v2_without_text_dups", "v2_without_title_dups", "v2_only_text_dups"):
        m = results[key]
        if m["n"]:
            print(f"{key:<28}{m['n']:>8}{m['accuracy']:>10.4f}{m['f1_macro']:>10.4f}")
    print(f"\nДля сверки: в README test Accuracy 0.9734, Macro F1 0.9493 "
          f"(здесь fp16, допустима разница в 4-м знаке)")

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"сохранено: {args.out}")


if __name__ == "__main__":
    main()