import random
import re

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, Trainer

SEED = 777
MODEL_NAME = "cointegrated/rubert-tiny2"
DATASET_NAME = "nyuuzyou/ke-products"
NUM_LABELS = 39
MAX_LENGTH = 256

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_html(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_text(example):
    title = example["title"] or ""
    description = clean_html(example["description"] or "")
    return {"text": f"{title} {description}".strip()}

def build_text_with_attributes(example):
    attributes = example["attributes"] or []
    attributes_text = " ".join(str(x) for x in attributes if x)
    return {"text_v3": f"{example['text']} {attributes_text}".strip()}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
    }

def make_model(label_names=None, label2id=None):
    kwargs = {"num_labels": NUM_LABELS}
    if label_names is not None and label2id is not None:
        kwargs.update({"id2label": {i: x for i, x in enumerate(label_names)}, "label2id": label2id})
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, **kwargs)

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # smooth_weights задаётся в notebook перед запуском обучения.
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def predict_category(text, model, tokenizer, label_names):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    return label_names[outputs.logits.argmax(dim=-1).item()]
