import torch
from transformers import AutoTokenizer, BertForSequenceClassification
import os
from functools import lru_cache
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "bert_model")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_model():
    try:
        if os.path.exists(MODEL_DIR):
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_DIR,
                use_fast=True,
                local_files_only=True
            )
            model = BertForSequenceClassification.from_pretrained(
                MODEL_DIR,
                local_files_only=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

        model.to(DEVICE)
        model.eval()

        if DEVICE.type == "cuda":
            model.half()

        return tokenizer, model

    except Exception as e:
        raise RuntimeError(f"BERT load failed: {e}")


def _confidence(probs):
    probs = np.array(probs)
    top2 = np.sort(probs)[-2:]
    margin = top2[-1] - top2[-2] if len(top2) > 1 else top2[-1]
    return float(np.clip((top2[-1] * 0.7 + margin * 0.3) * 100, 0, 100))


def _risk(conf):
    if conf >= 85:
        return "Critical"
    elif conf >= 70:
        return "High"
    elif conf >= 50:
        return "Moderate"
    return "Low"


def _severity(label, conf):
    if label == "High Stress" and conf >= 80:
        return "Severe"
    elif conf >= 65:
        return "Elevated"
    return "Normal"


def _advice(risk):
    return {
        "Critical": "Immediate professional consultation required",
        "High": "Reduce workload and monitor closely",
        "Moderate": "Practice stress management techniques",
        "Low": "Maintain a healthy routine"
    }.get(risk, "Maintain balance")


def _explanation(label):
    return f"{label} inferred using contextual semantic understanding via transformer model"


def predict_stress(text):
    text = str(text).strip()

    if not text or len(text.split()) < 2:
        return "Invalid", 0.0, "Low", "Normal", "Provide meaningful input"

    try:
        tokenizer, model = load_model()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            if DEVICE.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]

        pred = int(np.argmax(probs))
        conf = _confidence(probs)

        label = "Low Stress" if pred == 0 else "High Stress"

        risk = _risk(conf)
        severity = _severity(label, conf)
        advice = _advice(risk)

        return label, round(conf, 2), risk, severity, advice

    except Exception as e:
        return "Error", 0.0, "Low", "Normal", str(e)