import joblib
import os
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

def predict_stress(text):
    text = str(text).strip()

    if not text:
        return "Invalid Input", 0.0

    try:
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        confidence = float(max(probabilities) * 100)

        if prediction == 0:
            label = "Low Stress"
        elif prediction == 1:
            label = "High Stress"
        else:
            label = f"Unknown ({prediction})"

        return label, round(confidence, 2)

    except Exception:
        return "Prediction Error", 0.0