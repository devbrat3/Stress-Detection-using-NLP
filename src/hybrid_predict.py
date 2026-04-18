from predict import predict_stress as ml_predict
from predict_bert import predict_stress as bert_predict


def _risk(conf):
    if conf >= 85:
        return "Critical"
    elif conf >= 70:
        return "High"
    elif conf >= 50:
        return "Moderate"
    return "Low"


def _reliability(agreement, conf):
    if agreement and conf >= 75:
        return "High"
    if not agreement and conf < 60:
        return "Low"
    return "Medium"


def _fuse_confidence(ml_conf, bert_conf, agree):
    if agree:
        return (ml_conf * 0.3 + bert_conf * 0.7)
    return max(ml_conf, bert_conf)


def _select_label(ml_out, bert_out, agree):
    if agree:
        return bert_out["label"]
    return bert_out["label"] if bert_out["confidence"] >= ml_out["confidence"] else ml_out["label"]


def _merge_keywords(ml_out, bert_out):
    k1 = ml_out.get("keywords", [])
    k2 = bert_out.get("signals", [])
    return list(set(k1 + k2))[:5]


def _explanation(label, keywords):
    if not keywords:
        return "Prediction based on contextual language understanding"
    return f"{label} inferred from signals: {', '.join(keywords[:3])}"


def hybrid_predict(text, mode="AUTO"):

    text = str(text).strip()

    if not text or len(text.split()) < 2:
        return {
            "label": "Invalid",
            "confidence": 0.0,
            "risk": "Low",
            "severity": "Normal",
            "advice": "Provide meaningful input",
            "model": "None",
            "agreement": "N/A",
            "reliability": "Low",
            "keywords": [],
            "explanation": "Insufficient input",
            "ml_output": {},
            "bert_output": {}
        }

    try:
        ml_out = ml_predict(text)
    except Exception as e:
        ml_out = {
            "label": "Error",
            "confidence": 0.0,
            "risk": "Low",
            "severity": "Normal",
            "keywords": [],
            "advice": "ML failure",
            "explanation": str(e),
            "model": "ML"
        }

    try:
        bert_label, bert_conf, bert_risk, bert_severity, bert_advice = bert_predict(text)
        bert_out = {
            "label": bert_label,
            "confidence": bert_conf,
            "risk": bert_risk,
            "severity": bert_severity,
            "advice": bert_advice,
            "signals": [],
            "model": "BERT"
        }
    except Exception as e:
        bert_out = {
            "label": "Error",
            "confidence": 0.0,
            "risk": "Low",
            "severity": "Normal",
            "advice": "BERT failure",
            "signals": [],
            "model": "BERT"
        }

    if mode == "ML":
        return {**ml_out, "model": "ML", "agreement": "N/A", "reliability": "Medium"}

    if mode == "BERT":
        return {
            **bert_out,
            "model": "BERT",
            "agreement": "N/A",
            "reliability": "High",
            "keywords": [],
            "explanation": "Transformer-based contextual prediction"
        }

    agree = ml_out["label"] == bert_out["label"]

    final_conf = _fuse_confidence(ml_out["confidence"], bert_out["confidence"], agree)
    final_label = _select_label(ml_out, bert_out, agree)

    reliability = _reliability(agree, final_conf)
    risk = _risk(final_conf)

    keywords = _merge_keywords(ml_out, bert_out)
    explanation = _explanation(final_label, keywords)

    return {
        "label": final_label,
        "confidence": round(final_conf, 2),
        "risk": risk,
        "severity": bert_out["severity"],
        "advice": bert_out["advice"],
        "model": "Hybrid",
        "agreement": "Yes" if agree else "No",
        "reliability": reliability,
        "keywords": keywords,
        "explanation": explanation,
        "ml_output": ml_out,
        "bert_output": bert_out
    }