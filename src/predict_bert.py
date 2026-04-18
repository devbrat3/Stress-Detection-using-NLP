import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "bert_model")

@lru_cache(maxsize=1)
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def predict_stress(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    confidence = float(confidence.item() * 100)
    prediction = int(prediction.item())

    if prediction == 0:
        label = "Low Stress"
    else:
        label = "High Stress"

    return label, round(confidence, 2)