
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data_utils import load_filtered, make_splits

OUT = Path("artifacts")

# Размеры из README проекта
EXPECTED_SIZES = {"train": 1261675, "val": 70093, "test": 70094}

# Контрольные значения из распечаток ноутбука 01 (split, класс, число строк).
# Если все совпали - split воспроизведён точно как в исходных экспериментах.
FINGERPRINT = [
    ("val", "Автотовары", 2498),
    ("val", "Аксессуары", 9534),
    ("test", "Одежда", 7940),
    ("test", "Гигиена и подгузники", 33),
    ("test", "Компьютерная техника", 48),
]


def main() -> None:
    OUT.mkdir(exist_ok=True)

    ds, label_names = load_filtered()
    print(f"После фильтрации: {len(ds)} строк, {len(label_names)} классов")

    train, val, test = make_splits(ds)
    frames = []
    for name, part in (("train", train), ("val", val), ("test", test)):
        df = part.select_columns(["row_idx", "id", "labels"]).to_pandas()
        df.insert(0, "split", name)
        frames.append(df)
    split_df = pd.concat(frames, ignore_index=True)

    split_df.to_parquet(OUT / "split_ids.parquet", index=False)
    (OUT / "label_names.json").write_text(
        json.dumps(label_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- проверки ----
    print("\nРазмеры:")
    sizes_ok = True
    for name, expected in EXPECTED_SIZES.items():
        got = int((split_df["split"] == name).sum())
        ok = got == expected
        sizes_ok &= ok
        print(f"  {name}: {got} (ожидалось {expected}) {'OK' if ok else 'НЕ СОВПАДАЕТ'}")

    print("\nКонтрольные значения:")
    fp_ok = True
    for split, label, expected in FINGERPRINT:
        sub = split_df[split_df["split"] == split]
        got = int((sub["labels"].map(lambda i: label_names[i]) == label).sum())
        ok = got == expected
        fp_ok &= ok
        print(f"  {split} / {label}: {got} (ожидалось {expected}) {'OK' if ok else 'НЕ СОВПАДАЕТ'}")

    print(f"\nrow_idx уникальны: {split_df['row_idx'].is_unique}")
    print(f"Уникальных id: {split_df['id'].nunique()} из {len(split_df)}")

    if sizes_ok and fp_ok:
        print("\nSplit воспроизведён точно. Файлы сохранены в artifacts/")
    else:
        print(
            "\nВНИМАНИЕ: split отличается от исходного. Возможная причина - другая "
            "версия datasets. Метрики из README с этим split сравнивать нельзя."
        )


if __name__ == "__main__":
    main()