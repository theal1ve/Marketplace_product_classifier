
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import precision_recall_fscore_support


def fast_scores(y: np.ndarray, p: np.ndarray, k: int):
    """Accuracy и Macro F1 через bincount (быстро, для bootstrap).

    Macro F1 усредняется по классам, присутствующим в y или p, как в sklearn по умолчанию."""
    tp = np.bincount(y[y == p], minlength=k)
    pc = np.bincount(p, minlength=k)
    tc = np.bincount(y, minlength=k)
    den = pc + tc
    mask = den > 0
    f1 = np.zeros(k)
    f1[mask] = 2 * tp[mask] / den[mask]
    return float((y == p).mean()), float(f1[mask].mean())


def get_v2_val_preds(args, split_df, y_split) -> np.ndarray:
    path = Path(args.v2_preds)
    if path.exists():
        print(f"V2: беру готовые предсказания {path}")
        return np.load(path)

    print("V2: предсказаний на validation нет, считаю (нужны веса и лучше GPU)")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.data_utils import apply_split, load_filtered, texts_and_labels
    from src.infer_utils import predict
    from src.project_utils import MODEL_NAME

    ds, _ = load_filtered()
    _, val, _ = apply_split(ds, split_df)
    texts, y_val = texts_and_labels(val)
    assert (y_val == y_split).all(), "метки validation не совпали со split"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"устройство: {device}")
    try:
        tok = AutoTokenizer.from_pretrained(args.model_dir)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    preds = predict(model, tok, texts.tolist(), device, batch_size=128)
    path.parent.mkdir(exist_ok=True)
    np.save(path, preds)
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="artifacts/split_ids.parquet")
    ap.add_argument("--labels", default="artifacts/label_names.json")
    ap.add_argument("--tfidf-preds", default="results/tfidf_val_preds.npy")
    ap.add_argument("--v2-preds", default="results/v2_val_preds.npy")
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "v2_weighted"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/compare_tfidf_v2_val.json")
    args = ap.parse_args()

    t0 = time.time()
    label_names = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    k = len(label_names)
    split_df = pd.read_parquet(args.split)
    y = split_df.loc[split_df["split"] == "val", "labels"].to_numpy()

    tfidf_path = Path(args.tfidf_preds)
    if not tfidf_path.exists():
        sys.exit(f"Нет {tfidf_path}. Сначала запусти scripts/02_tfidf_baseline.py "
                 f"(обновлённая версия сохраняет предсказания).")
    pa = np.load(tfidf_path)  # A = TF-IDF
    pb = get_v2_val_preds(args, split_df, y)  # B = V2
    assert len(pa) == len(pb) == len(y), f"длины не совпали: {len(pa)}, {len(pb)}, {len(y)}"

    # ---- метрики и сверка ----
    acc_a, f1_a = fast_scores(y, pa, k)
    acc_b, f1_b = fast_scores(y, pb, k)
    print(f"\n{'модель':<10}{'accuracy':>10}{'macro F1':>10}")
    print(f"{'TF-IDF':<10}{acc_a:>10.4f}{f1_a:>10.4f}   (ожидалось 0.9697 / 0.9496)")
    print(f"{'V2':<10}{acc_b:>10.4f}{f1_b:>10.4f}   (ожидалось 0.9724 / 0.9530, fp16 даёт разницу в 4-м знаке)")

    # ---- парный bootstrap ----
    rng = np.random.default_rng(777)
    n = len(y)
    d_acc, d_f1 = np.empty(args.n_boot), np.empty(args.n_boot)
    for i in range(args.n_boot):
        idx = rng.integers(0, n, n)
        a1, f1a = fast_scores(y[idx], pa[idx], k)
        b1, f1b = fast_scores(y[idx], pb[idx], k)
        d_acc[i], d_f1[i] = b1 - a1, f1b - f1a
    ci = lambda d: (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    print(f"\nПарный bootstrap ({args.n_boot} повторов), разность V2 - TF-IDF:")
    print(f"  accuracy: {acc_b - acc_a:+.4f}  95% ДИ [{ci(d_acc)[0]:+.4f}, {ci(d_acc)[1]:+.4f}]")
    print(f"  macro F1: {f1_b - f1_a:+.4f}  95% ДИ [{ci(d_f1)[0]:+.4f}, {ci(d_f1)[1]:+.4f}]")
    print("  Если интервал включает 0 - различие нельзя отличить от шума.")

    # ---- согласие и Мак-Немар ----
    ca, cb = pa == y, pb == y
    both = int((ca & cb).sum())
    only_a = int((ca & ~cb).sum())
    only_b = int((~ca & cb).sum())
    neither = int((~ca & ~cb).sum())
    disc = only_a + only_b
    p_mcnemar = float(binomtest(min(only_a, only_b), disc, 0.5).pvalue) if disc else 1.0
    print(f"\nСогласие на validation ({n} примеров):")
    print(f"  обе верны: {both}   обе ошиблись: {neither}")
    print(f"  только TF-IDF верен: {only_a}   только V2 верен: {only_b}")
    print(f"  тест Мак-Немара (точный): p = {p_mcnemar:.4g}")

    # ---- по классам ----
    labels = np.arange(k)
    _, _, f_a, sup = precision_recall_fscore_support(y, pa, labels=labels, zero_division=0)
    _, _, f_b, _ = precision_recall_fscore_support(y, pb, labels=labels, zero_division=0)
    table = pd.DataFrame({"class": label_names, "n_val": sup, "f1_tfidf": f_a,
                          "f1_v2": f_b, "diff_v2_minus_tfidf": f_b - f_a})
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 40)
    print("\nКлассы, где V2 лучше всего:")
    print(table.sort_values("diff_v2_minus_tfidf", ascending=False).head(8).round(4).to_string(index=False))
    print("\nКлассы, где TF-IDF лучше всего:")
    print(table.sort_values("diff_v2_minus_tfidf").head(8).round(4).to_string(index=False))
    print("\nВнимание: у классов с n_val < 50 F1 очень шумный: одна ошибка двигает F1 на несколько пунктов.")

    out = {
        "tfidf": {"accuracy": acc_a, "f1_macro": f1_a},
        "v2": {"accuracy": acc_b, "f1_macro": f1_b},
        "bootstrap": {"n_boot": args.n_boot,
                      "diff_accuracy": {"mean": acc_b - acc_a, "ci95": ci(d_acc)},
                      "diff_f1_macro": {"mean": f1_b - f1_a, "ci95": ci(d_f1)}},
        "agreement": {"both_correct": both, "both_wrong": neither,
                      "only_tfidf_correct": only_a, "only_v2_correct": only_b,
                      "mcnemar_p": p_mcnemar},
        "per_class": table.round(4).to_dict(orient="records"),
    }
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n|{time.time() - t0:.0f}s| сохранено: {args.out}")


if __name__ == "__main__":
    main()