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

def _risk(conf):
    if conf > 80:
        return "Critical"
    elif conf > 60:
        return "High"
    elif conf > 40:
        return "Moderate"
    else:
        return "Low"

def _severity(label, conf):
    if "High" in label and conf > 75:
        return "Severe"
    elif conf > 60:
        return "Elevated"
    else:
        return "Normal"

def _recommendation(risk):
    if risk == "Critical":
        return "Immediate professional consultation recommended"
    elif risk == "High":
        return "Take rest and monitor stress closely"
    elif risk == "Moderate":
        return "Practice relaxation techniques"
    else:
        return "Maintain healthy routine"

def predict_stress(text):
    text = str(text).strip()

    if not text:
        return "Invalid", 0.0, "Low", "Normal", [], "No input"

    try:
        pred = model.predict([text])[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
            conf = float(np.max(probs) * 100)
        else:
            conf = 70.0

        if pred == 0:
            label = "Low Stress"
        elif pred == 1:
            label = "High Stress"
        else:
            label = "Medium Stress"

        risk = _risk(conf)
        severity = _severity(label, conf)
        advice = _recommendation(risk)

        # explainability
        try:
            tfidf = model.named_steps["tfidf"]
            feature_names = np.array(tfidf.get_feature_names_out())
            vec = tfidf.transform([text]).toarray()[0]

            idx = vec.argsort()[-5:][::-1]
            keywords = feature_names[idx]
        except:
            keywords = []

        return label, round(conf, 2), risk, severity, list(keywords), advice

    except:
        return "Error", 0.0, "Low", "Normal", [], "Error"