
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from src.data_utils import SEED, apply_split, load_filtered, texts_and_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TF-IDF + LinearSVC baseline")
    p.add_argument("--split", default="artifacts/split_ids.parquet")
    p.add_argument("--max-train", type=int, default=0,
                   help="взять подвыборку train (0 = весь train)")
    p.add_argument("--fit-sample", type=int, default=300_000,
                   help="на скольких train-текстах строить словарь TF-IDF")
    p.add_argument("--max-features", type=int, default=300_000)
    p.add_argument("--C", type=float, default=0.5)
    p.add_argument("--n-jobs", type=int, default=4,
                   help="параллельные бинарные классификаторы (больше = быстрее и больше RAM)")
    p.add_argument("--eval-test", action="store_true",
                   help="дополнительно оценить на test (делать один раз, в конце)")
    return p.parse_args()


def evaluate(clf, X, y, label_names):
    labels = list(range(len(label_names)))
    pred = clf.predict(X)
    report = classification_report(
        y, pred, labels=labels, target_names=label_names,
        output_dict=True, zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "per_class_f1": {n: round(report[n]["f1-score"], 4) for n in label_names},
    }
    return metrics, pred


def main() -> None:
    args = parse_args()
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"|{time.time() - t0:6.0f}s| {msg}", flush=True)

    log("загрузка датасета и split")
    ds, label_names = load_filtered()
    split_df = pd.read_parquet(args.split)
    train, val, test = apply_split(ds, split_df)
    if args.max_train:
        train = train.shuffle(seed=SEED).select(range(args.max_train))

    log("сборка текстов (title + очищенный description)")
    X_train_text, y_train = texts_and_labels(train)
    X_val_text, y_val = texts_and_labels(val)
    log(f"train: {len(y_train)}, val: {len(y_val)}")

    # Словарь строим на подвыборке train: bigram-словарь на всех 1.26M текстов
    # может не поместиться в RAM. Val и test в словаре не участвуют.
    log("TF-IDF: построение словаря")
    n_fit = min(args.fit_sample, len(X_train_text))
    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_features=args.max_features,
        sublinear_tf=True, dtype=np.float32,
    )
    vec.fit(X_train_text.sample(n=n_fit, random_state=SEED))
    log(f"словарь: {len(vec.vocabulary_)} признаков")

    log("TF-IDF: преобразование train и val")
    X_train = vec.transform(X_train_text)
    X_val = vec.transform(X_val_text)

    log(f"обучение LinearSVC (C={args.C}, n_jobs={args.n_jobs})")
    clf = OneVsRestClassifier(
        LinearSVC(C=args.C, dual=False), n_jobs=args.n_jobs
    )
    clf.fit(X_train, y_train)
    fit_seconds = round(time.time() - t0)
    log("обучение завершено")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    suffix = "_smoke" if args.max_train else ""

    val_metrics, val_pred = evaluate(clf, X_val, y_val, label_names)
    np.save(out_dir / f"tfidf_val_preds{suffix}.npy", val_pred)
    results = {
        "config": {k: v for k, v in vars(args).items() if k != "split"},
        "n_train": int(len(y_train)),
        "seconds_total": fit_seconds,
        "val": val_metrics,
    }
    v = results["val"]
    log(f"VAL  accuracy={v['accuracy']:.4f}  macro_f1={v['f1_macro']:.4f}  "
        f"weighted_f1={v['f1_weighted']:.4f}")

    if args.eval_test:
        X_test_text, y_test = texts_and_labels(test)
        test_metrics, test_pred = evaluate(clf, vec.transform(X_test_text), y_test, label_names)
        np.save(out_dir / f"tfidf_test_preds{suffix}.npy", test_pred)
        results["test"] = test_metrics
        t = results["test"]
        log(f"TEST accuracy={t['accuracy']:.4f}  macro_f1={t['f1_macro']:.4f}  "
            f"weighted_f1={t['f1_weighted']:.4f}")

    name = "tfidf_baseline_smoke.json" if args.max_train else "tfidf_baseline.json"
    (out_dir / name).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"результаты сохранены: results/{name}")


if __name__ == "__main__":
    main()