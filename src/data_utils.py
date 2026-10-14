"""Данные без torch: загрузка, фильтрация, воспроизводимый split.

Порядок строк и label id совпадают с ноутбуками 01-04, поэтому split,
сохранённый scripts/01_save_split.py, один и тот же локально и на Kaggle.
"""
import re

from datasets import ClassLabel, load_dataset

SEED = 777
DATASET_NAME = "nyuuzyou/ke-products"
TARGET = "greatgrandparent_category"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_html(text):
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def load_filtered():
    """Датасет без пустого target + колонки labels и row_idx. Возвращает (ds, label_names)."""
    ds = load_dataset(DATASET_NAME)["train"]
    ds = ds.filter(lambda e: e[TARGET] is not None)
    label_names = sorted(set(ds[TARGET]))
    label2id = {name: i for i, name in enumerate(label_names)}
    ds = ds.map(lambda e: {"labels": label2id[e[TARGET]]})
    ds = ds.cast_column(
        "labels", ClassLabel(num_classes=len(label_names), names=label_names)
    )
    # row_idx - позиция строки в отфильтрованном датасете; по ней split восстанавливается
    ds = ds.add_column("row_idx", list(range(len(ds))))
    return ds, label_names


def make_splits(ds):
    """Тот же split, что в ноутбуке 01: 90/5/5, stratify по labels, seed=777."""
    first = ds.train_test_split(
        test_size=0.10, seed=SEED, stratify_by_column="labels"
    )
    second = first["test"].train_test_split(
        test_size=0.5, seed=SEED, stratify_by_column="labels"
    )
    return first["train"], second["train"], second["test"]


def apply_split(ds, split_df):
    """Применить сохранённый split (parquet с колонками split, row_idx) к ds."""
    parts = []
    for name in ("train", "val", "test"):
        idx = split_df.loc[split_df["split"] == name, "row_idx"].tolist()
        parts.append(ds.select(idx))
    return tuple(parts)


def texts_and_labels(part):
    """Текст как в build_text: title + очищенный description. Возвращает (Series, ndarray)."""
    df = part.select_columns(["title", "description", "labels"]).to_pandas()
    title = df["title"].fillna("")
    desc = df["description"].map(clean_html)
    return (title + " " + desc).str.strip(), df["labels"].to_numpy()