import joblib
import os
from functools import lru_cache
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

def _confidence(probs):
    probs = np.array(probs)
    return float(np.max(probs) * 100)

def _risk_level(conf):
    if conf < 50:
        return "Low"
    elif conf < 75:
        return "Moderate"
    else:
        return "High"

def predict_stress(text):
    text = str(text).strip()

    if not text:
        return "Invalid", 0.0, "Low"

    try:
        pred = model.predict([text])[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
            conf = _confidence(probs)
        else:
            conf = 70.0

        if pred == 0:
            label = "Low Stress"
        elif pred == 1:
            label = "High Stress"
        else:
            label = "Medium Stress"

        risk = _risk_level(conf)

        return label, round(conf, 2), risk

    except Exception:
        return "Error", 0.0, "Low"