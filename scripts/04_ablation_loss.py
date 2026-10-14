
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments

from src.data_utils import SEED, apply_split, load_filtered, texts_and_labels
from src.project_utils import (
    MODEL_NAME, NUM_LABELS, WeightedTrainer,
    compute_metrics, make_model, set_seed,
)
from src.train_utils import EncodedDataset, encode, smooth_weights


def summarize(runs: dict, seeds: list) -> None:
    print("\nMacro F1 на validation по seed'ам:")
    print(f"{'seed':>6}{'CE':>10}{'weighted CE':>14}{'разница':>10}")
    diffs, ce_all, w_all = [], [], []
    for s in seeds:
        a, b = runs.get(f"ce_s{s}"), runs.get(f"weighted_ce_s{s}")
        if not a or not b:
            continue
        ce_all.append(a["f1_macro"])
        w_all.append(b["f1_macro"])
        diffs.append(b["f1_macro"] - a["f1_macro"])
        print(f"{s:>6}{a['f1_macro']:>10.4f}{b['f1_macro']:>14.4f}{diffs[-1]:>+10.4f}")
    if not diffs:
        return

    def ms(v):
        return f"{statistics.mean(v):.4f} ± {statistics.stdev(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}"

    print(f"\nCE:           {ms(ce_all)}")
    print(f"weighted CE:  {ms(w_all)}")
    print(f"разница:      {ms(diffs)}  (n={len(diffs)} seed'а)")
    print("Если |средняя разница| сопоставима со стандартным отклонением - "
          "утверждать улучшение нельзя.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="artifacts/split_ids.parquet")
    ap.add_argument("--n-train", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[777, 778, 779])
    ap.add_argument("--arms", nargs="+", default=["ce", "weighted_ce"],
                    choices=["ce", "weighted_ce"],
                    help="какие ветки обучать (для подбора lr хватает одной: ce)")
    ap.add_argument("--out", default="results/ablation_loss.json")
    args = ap.parse_args()

    t0 = time.time()

    def log(msg: str) -> None:
        print(f"|{time.time() - t0:6.0f}s| {msg}", flush=True)

    log("загрузка датасета и split")
    ds, _ = load_filtered()
    split_df = pd.read_parquet(args.split)
    train, val, _ = apply_split(ds, split_df)  # test не трогаем

    # Одна и та же подвыборка train для всех прогонов: меняется только seed обучения
    train = train.shuffle(seed=SEED).select(range(min(args.n_train, len(train))))
    X_tr, y_tr = texts_and_labels(train)
    X_val, y_val = texts_and_labels(val)
    log(f"train: {len(y_tr)}, val: {len(y_val)}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    log("токенизация")
    train_ds = EncodedDataset(encode(tok, X_tr.tolist()), y_tr)
    val_ds = EncodedDataset(encode(tok, X_val.tolist()), y_val)
    collator = DataCollatorWithPadding(tokenizer=tok)

    # warmup 10% шагов задаём числом: параметр warmup_ratio убран в новых версиях transformers
    steps_per_epoch = -(-len(train_ds) // args.batch_size)
    warmup_steps = int(0.1 * steps_per_epoch * args.epochs)
    log(f"шагов на эпоху: {steps_per_epoch}, warmup: {warmup_steps}")

    arms = {
        "ce": torch.ones(NUM_LABELS),            # веса = 1  -> обычная CE
        "weighted_ce": smooth_weights(y_tr),     # те же веса, что в V2
    }
    arms = {k: v for k, v in arms.items() if k in args.arms}

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    results = {"config": vars(args), "runs": {}}
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        pc = prev.get("config", {})
        if all(pc.get(k) == getattr(args, k) for k in ("n_train", "epochs", "lr", "batch_size")):
            results = prev
            log(f"найдены готовые прогоны: {list(results['runs'])}")

    for seed in args.seeds:
        for arm, weights in arms.items():
            key = f"{arm}_s{seed}"
            if key in results["runs"]:
                log(f"пропуск {key} (уже посчитан)")
                continue
            log(f"обучение {key}")
            set_seed(seed)
            model = make_model()
            targs = TrainingArguments(
                output_dir=f"runs/ablation/{key}",
                num_train_epochs=args.epochs,
                learning_rate=args.lr,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=128,
                weight_decay=0.01,
                lr_scheduler_type="linear",
                warmup_steps=warmup_steps,
                eval_strategy="no",
                save_strategy="no",
                fp16=torch.cuda.is_available(),
                logging_steps=200,
                seed=seed,
                data_seed=seed,
                report_to="none",
                remove_unused_columns=False,
                dataloader_num_workers=0,
            )
            trainer = WeightedTrainer(
                model=model, args=targs, train_dataset=train_ds,
                processing_class=tok, data_collator=collator,
                compute_metrics=compute_metrics, class_weights=weights,
            )
            t_run = time.time()
            trainer.train()
            m = trainer.evaluate(val_ds)
            m = {k.replace("eval_", ""): float(v) for k, v in m.items()
                 if isinstance(v, (int, float))}
            m["train_seconds"] = round(time.time() - t_run)
            results["runs"][key] = m
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            log(f"{key}: accuracy={m['accuracy']:.4f} macro_f1={m['f1_macro']:.4f} "
                f"({m['train_seconds']} с)")

            del trainer, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summarize(results["runs"], args.seeds)
    log(f"результаты: {args.out}")
    print("Примечание: eval_loss двух веток сравнивать нельзя (loss считается с разными весами) - "
          "сравнивай accuracy и F1.")


if __name__ == "__main__":
    main()