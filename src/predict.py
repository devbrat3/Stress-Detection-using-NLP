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
    if conf >= 85:
        return "Critical"
    elif conf >= 70:
        return "High"
    elif conf >= 50:
        return "Moderate"
    return "Low"


def _severity(label, conf):
    if "High" in label and conf >= 80:
        return "Severe"
    elif conf >= 65:
        return "Elevated"
    return "Normal"


def _recommendation(risk):
    return {
        "Critical": "Immediate clinical attention recommended",
        "High": "Monitor closely and reduce workload",
        "Moderate": "Apply stress management techniques",
        "Low": "Maintain healthy routine"
    }.get(risk, "Maintain balance")


def _confidence(probs):
    probs = np.array(probs)
    top2 = np.sort(probs)[-2:]
    margin = top2[-1] - top2[-2] if len(top2) > 1 else top2[-1]
    return float(np.clip((top2[-1] * 0.7 + margin * 0.3) * 100, 0, 100))


def _extract_keywords(text):
    try:
        tfidf = model.named_steps["tfidf"]
        feature_names = np.array(tfidf.get_feature_names_out())
        vec = tfidf.transform([text]).toarray()[0]
        idx = vec.argsort()[-5:][::-1]
        return feature_names[idx].tolist()
    except:
        return []


def _explanation(label, keywords):
    if not keywords:
        return "Prediction based on overall linguistic pattern"
    return f"{label} inferred due to signals like: {', '.join(keywords[:3])}"


def predict_stress(text):
    text = str(text).strip()

    if not text or len(text.split()) < 2:
        return {
            "label": "Invalid",
            "confidence": 0.0,
            "risk": "Low",
            "severity": "Normal",
            "keywords": [],
            "advice": "Provide meaningful input",
            "explanation": "Insufficient input",
            "model": "ML"
        }

    try:
        pred = model.predict([text])[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text])[0]
            conf = _confidence(probs)
        else:
            conf = 70.0

        label_map = {
            0: "Low Stress",
            1: "High Stress",
            2: "Moderate Stress"
        }

        label = label_map.get(pred, "Moderate Stress")

        risk = _risk(conf)
        severity = _severity(label, conf)
        advice = _recommendation(risk)

        keywords = _extract_keywords(text)
        explanation = _explanation(label, keywords)

        return {
            "label": label,
            "confidence": round(conf, 2),
            "risk": risk,
            "severity": severity,
            "keywords": keywords,
            "advice": advice,
            "explanation": explanation,
            "model": "ML"
        }

    except Exception as e:
        return {
            "label": "Error",
            "confidence": 0.0,
            "risk": "Low",
            "severity": "Normal",
            "keywords": [],
            "advice": "System error",
            "explanation": str(e),
            "model": "ML"
        }