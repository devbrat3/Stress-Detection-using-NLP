from predict import predict_stress as ml_predict
from predict_bert import predict_stress as bert_predict

def _risk_level(conf):
    if conf > 80:
        return "Critical"
    elif conf > 65:
        return "High"
    elif conf > 50:
        return "Moderate"
    return "Low"

def hybrid_predict(text, mode="AUTO"):

    text = str(text).strip()

    if not text:
        return {
            "label": "Invalid",
            "confidence": 0.0,
            "risk": "Low",
            "severity": "None",
            "advice": "No input",
            "model": "None",
            "agreement": "N/A",
            "reliability": "Low"
        }

    # -------- RUN BOTH MODELS --------
    try:
        ml_label, ml_conf, ml_risk = ml_predict(text)
    except:
        ml_label, ml_conf, ml_risk = "Error", 0, "Low"

    try:
        bert_label, bert_conf, bert_risk, bert_severity, bert_advice = bert_predict(text)
    except:
        bert_label, bert_conf, bert_risk, bert_severity, bert_advice = "Error", 0, "Low", "Normal", "Fallback"

    # -------- MODE LOGIC --------
    if mode == "ML":
        return {
            "label": ml_label,
            "confidence": ml_conf,
            "risk": ml_risk,
            "severity": "Basic",
            "advice": "General monitoring",
            "model": "ML",
            "agreement": "N/A",
            "reliability": "Medium"
        }

    if mode == "BERT":
        return {
            "label": bert_label,
            "confidence": bert_conf,
            "risk": bert_risk,
            "severity": bert_severity,
            "advice": bert_advice,
            "model": "BERT",
            "agreement": "N/A",
            "reliability": "High"
        }

    # -------- CONSENSUS SYSTEM --------
    agree = ml_label == bert_label

    if agree:
        final_label = bert_label
        final_conf = (ml_conf * 0.3 + bert_conf * 0.7)
        reliability = "High"
    else:
        # disagreement → choose higher confidence
        if bert_conf >= ml_conf:
            final_label = bert_label
            final_conf = bert_conf
        else:
            final_label = ml_label
            final_conf = ml_conf

        reliability = "Medium"

    risk = _risk_level(final_conf)

    # -------- FINAL OUTPUT --------
    return {
        "label": final_label,
        "confidence": round(final_conf, 2),
        "risk": risk,
        "severity": bert_severity,
        "advice": bert_advice,
        "model": "Hybrid",
        "agreement": "Yes" if agree else "No",
        "reliability": reliability
    }