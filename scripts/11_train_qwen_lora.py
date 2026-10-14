
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, TrainingArguments,
)

from src.data_utils import SEED, apply_split, load_filtered, texts_and_labels
from src.project_utils import NUM_LABELS, WeightedTrainer, compute_metrics, set_seed
from src.train_utils import EncodedDataset, encode, smooth_weights

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@torch.no_grad()
def predict_logits(model, collator, ds, device, bs: int, amp: bool) -> np.ndarray:
    """Логиты по всему датасету; тексты отсортированы по длине, чтобы меньше паддинга."""
    n = len(ds)
    order = np.argsort([len(x) for x in ds.ids])
    out = np.empty((n, NUM_LABELS), dtype=np.float32)
    model.eval()
    t0 = time.time()
    n_batches = -(-n // bs)
    for b, i in enumerate(range(0, n, bs)):
        if b % 50 == 0:
            print(f"    предсказание: батч {b}/{n_batches}  ({time.time() - t0:.0f} с)", flush=True)
        idx = order[i:i + bs]
        batch = collator([{"input_ids": ds.ids[j], "attention_mask": ds.mask[j]} for j in idx])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(**batch).logits
        out[idx] = logits.float().cpu().numpy()
    return out


def evaluate_part(name, model, collator, ds, y, device, args, amp, tag) -> dict:
    t0 = time.time()
    logits = predict_logits(model, collator, ds, device, args.eval_batch, amp)
    seconds = time.time() - t0
    preds = logits.argmax(axis=-1)
    metrics = {k: float(v) for k, v in compute_metrics((logits, y)).items()}
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / f"qwen_lora_{tag}_{name}_preds.npy", preds)
    np.save(out_dir / f"qwen_lora_{tag}_{name}_logits.npy", logits.astype(np.float16))
    metrics["predict_seconds"] = round(seconds)
    metrics["predict_samples_per_s"] = round(len(y) / seconds, 1)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="artifacts/split_ids.parquet")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--val-n", type=int, default=0, help="оценить на подвыборке validation (0 = весь)")
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--fp32", action="store_true", help="без fp16 (если в fp16 loss = nan)")
    ap.add_argument("--eval-test", action="store_true", help="также оценить на test (один раз, в конце)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"n{args.n_train}"

    t0 = time.time()

    def log(msg: str) -> None:
        print(f"|{time.time() - t0:6.0f}s| {msg}", flush=True)

    assert torch.cuda.is_available(), "Нужен GPU (на Kaggle: Accelerator = GPU T4)"
    device = torch.device("cuda")
    amp = not args.fp32
    set_seed(args.seed)

    log("загрузка датасета и split")
    ds, _ = load_filtered()
    train, val, test = apply_split(ds, pd.read_parquet(args.split))
    train = train.shuffle(seed=SEED).select(range(min(args.n_train, len(train))))
    if args.val_n:
        val = val.shuffle(seed=SEED).select(range(args.val_n))
    X_tr, y_tr = texts_and_labels(train)
    X_val, y_val = texts_and_labels(val)
    log(f"train: {len(y_tr)}, val: {len(y_val)}")

    log(f"токенизатор и модель: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    train_ds = EncodedDataset(encode(tok, X_tr.tolist()), y_tr)
    val_ds = EncodedDataset(encode(tok, X_val.tolist()), y_val)
    collator = DataCollatorWithPadding(tokenizer=tok)
    lens = np.array([len(x) for x in train_ds.ids])
    log(f"токенов на карточку (Qwen): среднее {lens.mean():.0f}, доля усечённых до 256: "
        f"{(lens >= 256).mean():.1%}")

    # Веса грузим и приводим к fp32: fp16-автокаст работает с fp32-весами, адаптеры LoRA обучаются в fp32
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=NUM_LABELS)
    model = model.float()
    model.config.pad_token_id = tok.pad_token_id
    lora = LoraConfig(task_type="SEQ_CLS", r=args.r, lora_alpha=args.alpha,
                      lora_dropout=args.dropout, target_modules=LORA_TARGETS)
    model = get_peft_model(model, lora)
    n_total = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"параметров всего {n_total:,}, обучаемых {n_train_params:,} ({n_train_params / n_total:.2%})")

    eff_batch = args.batch_size * args.grad_accum
    steps_per_epoch = -(-len(train_ds) // eff_batch)
    warmup_steps = int(0.05 * steps_per_epoch * args.epochs)
    log(f"шагов на эпоху: {steps_per_epoch}, warmup: {warmup_steps}, эффективный батч: {eff_batch}")

    targs = TrainingArguments(
        output_dir=f"runs/qwen_lora_{tag}/trainer",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,
        eval_strategy="no",
        save_strategy="no",
        fp16=amp,
        logging_steps=50,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = WeightedTrainer(
        model=model, args=targs, train_dataset=train_ds, processing_class=tok,
        data_collator=collator, compute_metrics=compute_metrics,
        class_weights=smooth_weights(y_tr),
    )

    log("обучение")
    t_train = time.time()
    train_out = trainer.train()
    train_seconds = round(time.time() - t_train)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    tm = train_out.metrics
    log(f"обучение завершено: {train_seconds} с, {tm.get('train_samples_per_second', float('nan')):.1f} карточек/с, "
        f"train_loss {train_out.training_loss:.3f}, пик памяти GPU {peak_gb:.1f} ГБ")
    if not math.isfinite(train_out.training_loss):
        log("ВНИМАНИЕ: loss = nan. Перезапусти с --fp32")

    adapter_dir = Path(f"runs/qwen_lora_{tag}/adapter")
    model.save_pretrained(adapter_dir)

    log("оценка на validation")
    model.to(device)
    results = {
        "config": {k: v for k, v in vars(args).items() if k != "split"},
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_val)),
        "params_total": int(n_total),
        "params_trainable": int(n_train_params),
        "train_seconds": train_seconds,
        "train_samples_per_second": float(tm.get("train_samples_per_second", float("nan"))),
        "train_loss": float(train_out.training_loss),
        "peak_gpu_gb": round(peak_gb, 2),
        "mean_tokens": float(lens.mean()),
        "share_truncated": float((lens >= 256).mean()),
        "val": evaluate_part("val", model, collator, val_ds, y_val, device, args, amp, tag),
    }
    v = results["val"]
    log(f"VAL  accuracy={v['accuracy']:.4f}  macro_f1={v['f1_macro']:.4f}  "
        f"weighted_f1={v['f1_weighted']:.4f}  ({v['predict_samples_per_s']} карточек/с)")

    if args.eval_test:
        X_te, y_te = texts_and_labels(test)
        test_ds = EncodedDataset(encode(tok, X_te.tolist()), y_te)
        results["test"] = evaluate_part("test", model, collator, test_ds, y_te, device, args, amp, tag)
        t = results["test"]
        log(f"TEST accuracy={t['accuracy']:.4f}  macro_f1={t['f1_macro']:.4f}")

    out = Path(f"results/qwen_lora_{tag}_val.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"результаты: {out}")

    if args.n_train < 20_000:
        full = 100_000 / max(train_out.metrics.get("train_samples_per_second", 1), 1e-9)
        pred = 70_093 / max(v["predict_samples_per_s"], 1e-9)
        print(f"\nОценка полного запуска на 100k: обучение ~{full / 60:.0f} мин, "
              f"оценка на всём validation ~{pred / 60:.0f} мин (грубо, экстраполяция по короткому прогону)")


if __name__ == "__main__":
    main()
