
import argparse
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub(" ", s.lower()).strip()


def conflict_stats(texts_norm: pd.Series, y: np.ndarray, label_names: list, top: int = 10) -> dict:
    """Статистика конфликтующих меток у одинаковых нормализованных текстов."""
    df = pd.DataFrame({"t": texts_norm.to_numpy(), "y": y})
    uniq = df.drop_duplicates(["t", "y"])
    n_labels = uniq.groupby("t")["y"].nunique()
    conflict_idx = n_labels[n_labels > 1].index

    in_conflict = df["t"].isin(conflict_idx)
    groups = uniq[uniq["t"].isin(conflict_idx)].groupby("t")["y"].apply(lambda s: sorted(s))
    pairs = Counter()
    for labs in groups:
        for a, b in itertools.combinations(labs, 2):
            pairs[(label_names[a], label_names[b])] += 1

    return {
        "n_rows": int(len(df)),
        "n_distinct_texts": int(df["t"].nunique()),
        "n_conflicting_texts": int(len(conflict_idx)),
        "rows_in_conflicting_texts": int(in_conflict.sum()),
        "share_rows_in_conflict": float(in_conflict.mean()),
        "top_conflicting_pairs": [
            {"class_a": a, "class_b": b, "texts": int(c)} for (a, b), c in pairs.most_common(top)
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="artifacts/split_ids.parquet")
    ap.add_argument("--labels", default="artifacts/label_names.json")
    ap.add_argument("--v2-preds", default="results/v2_val_preds.npy")
    ap.add_argument("--n-pairs", type=int, default=5, help="сколько топ-путаниц разбирать с примерами")
    ap.add_argument("--n-examples", type=int, default=15)
    args = ap.parse_args()

    if not Path(args.v2_preds).exists():
        sys.exit(f"Нет {args.v2_preds}. Сначала запусти scripts/06_compare_models.py "
                 f"(он посчитает и сохранит предсказания V2 на validation).")

    from src.data_utils import apply_split, load_filtered, texts_and_labels

    label_names = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    split_df = pd.read_parquet(args.split)
    ds, _ = load_filtered()
    train, val, _ = apply_split(ds, split_df)

    # ---- 1. конфликты в train ----
    train_texts, y_train = texts_and_labels(train)
    stats = conflict_stats(train_texts.map(norm), y_train, label_names)
    print("Конфликты меток у одинаковых текстов в train:")
    print(f"  различных текстов: {stats['n_distinct_texts']}, из них с конфликтом: "
          f"{stats['n_conflicting_texts']}")
    print(f"  строк train в конфликтующих текстах: {stats['rows_in_conflicting_texts']} "
          f"({stats['share_rows_in_conflict']:.2%})")
    print("  топ конфликтующих пар классов:")
    for r in stats["top_conflicting_pairs"]:
        print(f"    {r['texts']:>6}  {r['class_a']}  <->  {r['class_b']}")

    # ---- 2. путаницы V2 на validation ----
    preds = np.load(args.v2_preds)
    y_val = split_df.loc[split_df["split"] == "val", "labels"].to_numpy()
    assert len(preds) == len(y_val), "длины предсказаний и меток validation не совпали"
    err = preds != y_val
    n_err = int(err.sum())
    conf = Counter(zip(y_val[err].tolist(), preds[err].tolist()))
    top_pairs = conf.most_common(10)
    print(f"\nОшибок V2 на validation: {n_err} из {len(y_val)}")
    print("Топ путаниц (истина -> предсказание):")
    cum = 0
    confusions = []
    for (t, p), c in top_pairs:
        cum += c
        confusions.append({"true": label_names[t], "pred": label_names[p], "errors": c})
        print(f"  {c:>5}  {label_names[t]}  ->  {label_names[p]}")
    print(f"  топ-10 путаниц покрывают {cum / n_err:.1%} всех ошибок")

    # ---- 3. примеры ----
    titles = val.select_columns(["title"]).to_pandas()["title"].fillna("").to_numpy()
    rng = np.random.default_rng(777)
    rows = []
    for (t, p), _ in top_pairs[:args.n_pairs]:
        idx = np.where((y_val == t) & (preds == p))[0]
        take = rng.choice(idx, size=min(args.n_examples, len(idx)), replace=False)
        for i in take:
            rows.append({"true": label_names[t], "pred": label_names[p], "title": titles[i]})
    ex = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    ex.to_csv("results/error_examples.csv", index=False, encoding="utf-8-sig")
    print(f"\nПримеры ошибок: results/error_examples.csv ({len(ex)} строк). "
          f"Прочитай их и реши: ошибка модели или спорная разметка?")

    out = {"train_conflicts": stats, "val_errors": n_err, "top_confusions": confusions,
           "top10_share_of_errors": cum / n_err}
    Path("results/taxonomy_noise.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("сохранено: results/taxonomy_noise.json")


if __name__ == "__main__":
    main()