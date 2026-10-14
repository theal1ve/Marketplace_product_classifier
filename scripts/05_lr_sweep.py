
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments

from src.data_utils import SEED, apply_split, load_filtered, texts_and_labels
from src.project_utils import (
    MODEL_NAME, NUM_LABELS, WeightedTrainer,
    compute_metrics, make_model, set_seed,
)
from src.train_utils import EncodedDataset, encode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="artifacts/split_ids.parquet")
    ap.add_argument("--n-train", type=int, default=20_000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--lrs", type=float, nargs="+", default=[5e-5, 1e-4, 3e-4, 1e-3])
    ap.add_argument("--out", default="results/lr_sweep.json")
    args = ap.parse_args()

    t0 = time.time()

    def log(msg: str) -> None:
        print(f"|{time.time() - t0:6.0f}s| {msg}", flush=True)

    log("загрузка датасета и split")
    ds, _ = load_filtered()
    split_df = pd.read_parquet(args.split)
    train, val, _ = apply_split(ds, split_df)  # test не трогаем

    train = train.shuffle(seed=SEED).select(range(min(args.n_train, len(train))))
    X_tr, y_tr = texts_and_labels(train)
    X_val, y_val = texts_and_labels(val)
    log(f"train: {len(y_tr)}, val: {len(y_val)}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    log("токенизация")
    train_ds = EncodedDataset(encode(tok, X_tr.tolist()), y_tr)
    val_ds = EncodedDataset(encode(tok, X_val.tolist()), y_val)
    collator = DataCollatorWithPadding(tokenizer=tok)

    steps_per_epoch = -(-len(train_ds) // args.batch_size)
    warmup_steps = int(0.1 * steps_per_epoch * args.epochs)
    log(f"шагов на эпоху: {steps_per_epoch}, warmup: {warmup_steps}")

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    results = {"config": vars(args), "runs": {}}
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        pc = prev.get("config", {})
        if all(pc.get(k) == getattr(args, k) for k in ("n_train", "epochs", "batch_size", "seed")):
            results["runs"] = prev.get("runs", {})
            log(f"найдены готовые значения lr: {list(results['runs'])}")

    for lr in args.lrs:
        key = f"{lr:g}"
        if key in results["runs"]:
            log(f"пропуск lr={key} (уже посчитан)")
            continue
        log(f"обучение lr={key}")
        set_seed(args.seed)
        model = make_model()
        targs = TrainingArguments(
            output_dir=f"runs/lr_sweep/{key}",
            num_train_epochs=args.epochs,
            learning_rate=lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=128,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=warmup_steps,
            eval_strategy="no",
            save_strategy="no",
            fp16=torch.cuda.is_available(),
            logging_steps=200,
            seed=args.seed,
            data_seed=args.seed,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )
        trainer = WeightedTrainer(
            model=model, args=targs, train_dataset=train_ds,
            processing_class=tok, data_collator=collator,
            compute_metrics=compute_metrics,
            class_weights=torch.ones(NUM_LABELS),  # обычная CE
        )
        t_run = time.time()
        train_out = trainer.train()
        m = trainer.evaluate(val_ds)
        m = {k.replace("eval_", ""): float(v) for k, v in m.items()
             if isinstance(v, (int, float))}
        m["train_loss"] = float(train_out.training_loss)
        m["train_seconds"] = round(time.time() - t_run)
        results["runs"][key] = m
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"lr={key}: accuracy={m['accuracy']:.4f} macro_f1={m['f1_macro']:.4f} "
            f"train_loss={m['train_loss']:.3f} ({m['train_seconds']} с)")

        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- итог ----
    runs = results["runs"]
    ordered = sorted(runs.items(), key=lambda kv: kv[1]["f1_macro"], reverse=True)
    print(f"\n{'lr':>8}{'accuracy':>10}{'macro F1':>10}{'train loss':>12}")
    for key, m in ordered:
        flag = "  <- расходится?" if not math.isfinite(m["train_loss"]) else ""
        print(f"{key:>8}{m['accuracy']:>10.4f}{m['f1_macro']:>10.4f}{m['train_loss']:>12.3f}{flag}")

    if ordered:
        best = ordered[0][0]
        grid = sorted(float(k) for k in runs)
        print(f"\nЛучший lr по validation Macro F1: {best}")
        if float(best) in (grid[0], grid[-1]) and len(grid) > 1:
            print("Лучшее значение на краю сетки: расширь её в эту сторону "
                  "(например, добавь соседние значения через --lrs).")
        print("Это же значение lr используй для обеих веток в 04_ablation_loss.py (--lr).")
    log(f"результаты: {args.out}")


if __name__ == "__main__":
    main()