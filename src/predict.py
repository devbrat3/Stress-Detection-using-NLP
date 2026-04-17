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

def _calibrate_confidence(probs):
    probs = np.array(probs)
    top2 = np.sort(probs)[-2:]
    margin = top2[-1] - top2[-2] if len(top2) > 1 else top2[-1]
    confidence = (top2[-1] * 0.7 + margin * 0.3) * 100
    return float(np.clip(confidence, 0, 100))

def predict_stress(text):
    text = str(text).strip()

    if not text or len(text.split()) < 2:
        return "Insufficient Input", 0.0

    try:
        prediction = model.predict([text])[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            confidence = _calibrate_confidence(probabilities)
        else:
            confidence = 70.0

        if prediction == 0:
            label = "Low Stress"
        elif prediction == 1:
            label = "High Stress"
        else:
            label = f"Unknown ({prediction})"

        if confidence < 55:
            label = "Uncertain"

        return label, round(confidence, 2)

    except Exception:
        return "Prediction Error", 0.0