
import argparse
import io
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_utils import SEED, apply_split, load_filtered, texts_and_labels
from src.project_utils import MAX_LENGTH, MODEL_NAME


def sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def run_batch(model, tok, batch_texts, device, amp: bool) -> np.ndarray:
    enc = tok(batch_texts, truncation=True, max_length=MAX_LENGTH,
              padding=True, return_tensors="pt").to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        logits = model(**enc).logits
    return logits.argmax(dim=-1).cpu().numpy()


def single_latency(model, tok, texts, device, amp, n) -> dict:
    for t in texts[:10]:  # прогрев
        run_batch(model, tok, [t], device, amp)
    times = []
    for t in texts[:n]:
        sync(device)
        t0 = time.perf_counter()
        run_batch(model, tok, [t], device, amp)
        sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "mean_ms": float(np.mean(times)), "n": len(times)}


def throughput(model, tok, texts, device, amp, bs=64):
    order = np.argsort([len(t) for t in texts])
    preds = np.empty(len(texts), dtype=np.int64)
    sync(device)
    t0 = time.perf_counter()
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        preds[idx] = run_batch(model, tok, [texts[j] for j in idx], device, amp)
    sync(device)
    dt = time.perf_counter() - t0
    return len(texts) / dt, preds


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="artifacts/split_ids.parquet")
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "v2_weighted"))
    ap.add_argument("--n", type=int, default=2000, help="карточек для GPU-замеров и точности")
    ap.add_argument("--n-cpu", type=int, default=500, help="карточек для CPU-замеров")
    ap.add_argument("--single", type=int, default=200, help="запросов для latency batch=1")
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args()

    ds, _ = load_filtered()
    _, val, _ = apply_split(ds, pd.read_parquet(args.split))
    val = val.shuffle(seed=SEED).select(range(args.n))
    texts_s, y = texts_and_labels(val)
    texts = texts_s.tolist()

    try:
        tok = AutoTokenizer.from_pretrained(args.model_dir)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    model_dir = Path(args.model_dir)
    n_params = None
    results = {
        "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "n_samples": args.n,
        "size_on_disk_mb": round(dir_size_mb(model_dir), 1),
        "variants": {},
    }

    # ---- CPU fp32 (эталон для сравнения предсказаний) ----
    cpu = torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(cpu).eval()
    n_params = sum(p.numel() for p in model.parameters())
    results["parameters"] = int(n_params)
    nc = min(args.n_cpu, args.n)
    print(f"CPU fp32: {nc} карточек, потоков {torch.get_num_threads()}")
    lat = single_latency(model, tok, texts, cpu, False, min(args.single, nc))
    tps, p_cpu = throughput(model, tok, texts[:nc], cpu, False)
    results["variants"]["cpu_fp32"] = {
        **lat, "throughput_per_s": tps, "accuracy": float((p_cpu == y[:nc]).mean())}

    # ---- CPU int8 (динамическое квантование Linear) ----
    try:
        qmodel = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8).eval()
        buf = io.BytesIO()
        torch.save(qmodel.state_dict(), buf)
        print("CPU int8")
        lat = single_latency(qmodel, tok, texts, cpu, False, min(args.single, nc))
        tps, p_q = throughput(qmodel, tok, texts[:nc], cpu, False)
        results["variants"]["cpu_int8"] = {
            **lat, "throughput_per_s": tps, "accuracy": float((p_q == y[:nc]).mean()),
            "agreement_with_fp32": float((p_q == p_cpu).mean()),
            "size_mb": round(buf.getbuffer().nbytes / 1e6, 1)}
    except Exception as e:  # квантование зависит от версии torch и платформы
        print(f"int8-квантование недоступно: {e}")

    # ---- GPU fp16 ----
    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        model_g = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(gpu).eval()
        print(f"GPU fp16 ({torch.cuda.get_device_name(0)}): {args.n} карточек")
        lat = single_latency(model_g, tok, texts, gpu, True, args.single)
        tps, p_g = throughput(model_g, tok, texts, gpu, True)
        results["gpu"] = torch.cuda.get_device_name(0)
        results["variants"]["gpu_fp16"] = {
            **lat, "throughput_per_s": tps, "accuracy": float((p_g == y).mean())}

    print(f"\nРазмер на диске: {results['size_on_disk_mb']} МБ, параметров: {results['parameters']:,}")
    print(f"{'вариант':<10}{'p50, мс':>9}{'p95, мс':>9}{'карт/с':>10}{'accuracy':>10}")
    for name, v in results["variants"].items():
        print(f"{name:<10}{v['p50_ms']:>9.1f}{v['p95_ms']:>9.1f}"
              f"{v['throughput_per_s']:>10.0f}{v['accuracy']:>10.4f}")
    if "cpu_int8" in results["variants"]:
        v = results["variants"]["cpu_int8"]
        print(f"int8: размер {v['size_mb']} МБ, совпадение с fp32 на {v['agreement_with_fp32']:.1%} карточек")
    print("\nНа маленькой выборке accuracy шумная: сравнивай варианты по совпадению предсказаний.")

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"сохранено: {args.out}")


if __name__ == "__main__":
    main()