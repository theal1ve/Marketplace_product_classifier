
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.project_utils import MAX_LENGTH, MODEL_NAME


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "v2_weighted"))
    ap.add_argument("--labels", default="artifacts/label_names.json")
    ap.add_argument("--card", default="MODEL_CARD.md")
    ap.add_argument("--out", default="hub_export")
    ap.add_argument("--push", action="store_true", help="загрузить на Hub (иначе только пробный прогон)")
    ap.add_argument("--repo-id", default=None, help="ИМЯ_ПОЛЬЗОВАТЕЛЯ/название-репозитория")
    ap.add_argument("--private", action="store_true", help="создать приватный репозиторий")
    args = ap.parse_args()

    label_names = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    assert model.config.num_labels == len(label_names), (
        f"в модели {model.config.num_labels} классов, а в label_names.json {len(label_names)}")
    model.config.id2label = {i: n for i, n in enumerate(label_names)}
    model.config.label2id = {n: i for i, n in enumerate(label_names)}
    try:
        tok = AutoTokenizer.from_pretrained(args.model_dir)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.save_pretrained(out)
    tok.save_pretrained(out)

    card = Path(args.card)
    if card.exists():
        shutil.copy(card, out / "README.md")
    else:
        print(f"ВНИМАНИЕ: нет {args.card}, репозиторий будет без model card")

    # ---- проверка: загружаем экспорт и предсказываем с названиями классов ----
    m2 = AutoModelForSequenceClassification.from_pretrained(out).eval()
    t2 = AutoTokenizer.from_pretrained(out)
    examples = [
        "Смартфон Samsung Galaxy A55 256 ГБ, 8 ГБ RAM",
        "Корм для кошек сухой, 10 кг",
        "Детская футболка хлопок для мальчика",
    ]
    print("\nПроверка экспорта:")
    with torch.no_grad():
        for text in examples:
            enc = t2(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
            pred = m2(**enc).logits.argmax(-1).item()
            print(f"  {text} -> {m2.config.id2label[pred]}")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"\nПапка {out}/ собрана ({size:.0f} МБ): {sorted(p.name for p in out.iterdir())}")

    if not args.push:
        print("Пробный прогон завершён. Для загрузки добавь --push --repo-id ИМЯ/репозиторий")
        return

    if not args.repo_id:
        sys.exit("Для --push укажи --repo-id ИМЯ_ПОЛЬЗОВАТЕЛЯ/название-репозитория")
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, exist_ok=True, private=args.private)
    api.upload_folder(folder_path=str(out), repo_id=args.repo_id,
                      commit_message="Add RuBERT-tiny2 marketplace category classifier (V2)")
    print(f"\nГотово: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()