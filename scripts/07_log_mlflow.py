
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlflow

EXPERIMENT = "marketplace-product-classifier"
VAL_KEYS = {"accuracy", "f1_macro", "f1_weighted", "loss"}

# V1, V2, V3 из README / ноутбука 02 (validation); исходные запуски, залогировать живьём уже нельзя
REPORTED = [
    ("v1_baseline_ce", {"model": "rubert-tiny2", "input": "title+description", "loss": "cross_entropy"},
     {"accuracy": 0.968242, "f1_macro": 0.938793, "f1_weighted": 0.968488}),
    ("v2_weighted_ce", {"model": "rubert-tiny2", "input": "title+description", "loss": "weighted_ce",
                        "lr": "2e-05", "epochs": "5", "batch_size": "16", "seed": "777"},
     {"accuracy": 0.972408, "f1_macro": 0.952999, "f1_weighted": 0.972413}),
    ("v3_weighted_ce_attributes", {"model": "rubert-tiny2", "input": "title+description+attributes",
                                   "loss": "weighted_ce", "lr": "2e-05", "epochs": "5", "batch_size": "16",
                                   "seed": "777"},
     {"accuracy": 0.972594, "f1_macro": 0.952233, "f1_weighted": 0.972604}),
]


def load_json(path: str):
    p = Path(path)
    if not p.exists():
        print(f"  пропуск: нет {path}")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def clean_params(d: dict) -> dict:
    return {k: str(v) for k, v in d.items() if not isinstance(v, (dict, list))}


def clean_metrics(m: dict, prefix_keys=VAL_KEYS, prefix="val_") -> dict:
    out = {}
    for k, v in m.items():
        if isinstance(v, (int, float)) and math.isfinite(v):
            out[f"{prefix}{k}" if k in prefix_keys else k] = float(v)
    return out


def log_run(name: str, group: str, params: dict, metrics: dict, artifact: str | None = None) -> None:
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"group": group, "logged": "retroactive"})
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if artifact and Path(artifact).exists():
            mlflow.log_artifact(artifact)
    print(f"  залогировано: {group} / {name}")


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)

    print("Исходные эксперименты V1/V2/V3 (из README):")
    for name, params, metrics in REPORTED:
        log_run(name, "main_experiments", params, clean_metrics(metrics))

    print("TF-IDF бейзлайн:")
    base = load_json("results/tfidf_baseline.json")
    if base:
        metrics = clean_metrics(base["val"])
        if "test" in base:
            metrics.update(clean_metrics(base["test"], prefix_keys=VAL_KEYS, prefix="test_"))
        params = {"model": "tfidf_linearsvc", **clean_params(base.get("config", {})),
                  "n_train": base.get("n_train")}
        metrics["seconds_total"] = float(base.get("seconds_total", 0))
        log_run("tfidf_linearsvc", "baseline", params, metrics, "results/tfidf_baseline.json")

    print("Подбор learning rate:")
    sweep = load_json("results/lr_sweep.json")
    if sweep:
        cfg = clean_params(sweep.get("config", {}))
        for lr, m in sweep["runs"].items():
            log_run(f"lr_{lr}", "lr_sweep", {**cfg, "lr": lr, "loss": "cross_entropy"}, clean_metrics(m),
                    "results/lr_sweep.json")

    print("Абляция loss (CE против weighted CE):")
    abl = load_json("results/ablation_loss.json")
    if abl:
        cfg = clean_params(abl.get("config", {}))
        for key, m in abl["runs"].items():
            arm, seed = key.rsplit("_s", 1)
            params = {**cfg, "arm": arm, "seed": seed}
            log_run(key, "ablation_loss", params, clean_metrics(m), "results/ablation_loss.json")

    print("Проверка пересечений train/test (V2):")
    leak = load_json("results/leak_check.json")
    if leak:
        metrics = {}
        for subset in ("v2_all", "v2_without_text_dups", "v2_without_title_dups", "v2_only_text_dups"):
            for k, v in leak.get(subset, {}).items():
                if k != "n" and isinstance(v, (int, float)):
                    metrics[f"test_{subset}_{k}"] = float(v)
        metrics["dup_text_share"] = leak["leak"]["dup_text_share"]
        metrics["dup_title_share"] = leak["leak"]["dup_title_share"]
        log_run("v2_leak_check", "diagnostics", {"model": "v2_weighted", "split": "test"}, metrics,
                "results/leak_check.json")

    print("Сравнение V2 и TF-IDF:")
    cmp_ = load_json("results/compare_tfidf_v2_val.json")
    if cmp_:
        b = cmp_["bootstrap"]
        metrics = {
            "diff_accuracy": b["diff_accuracy"]["mean"],
            "diff_accuracy_ci_low": b["diff_accuracy"]["ci95"][0],
            "diff_accuracy_ci_high": b["diff_accuracy"]["ci95"][1],
            "diff_f1_macro": b["diff_f1_macro"]["mean"],
            "diff_f1_macro_ci_low": b["diff_f1_macro"]["ci95"][0],
            "diff_f1_macro_ci_high": b["diff_f1_macro"]["ci95"][1],
            "mcnemar_p": cmp_["agreement"]["mcnemar_p"],
        }
        log_run("v2_vs_tfidf_val", "diagnostics", {"part": "val", "n_boot": b["n_boot"]}, metrics,
                "results/compare_tfidf_v2_val.json")

    print("\nГотово. Запусти: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("Сделай скриншот таблицы runs для README. В .gitignore добавь mlflow.db и mlruns/.")


if __name__ == "__main__":
    main()