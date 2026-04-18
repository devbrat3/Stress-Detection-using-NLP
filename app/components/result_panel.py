import streamlit as st


def _confidence_band(conf):
    if conf > 85:
        return "Very High"
    elif conf > 70:
        return "High"
    elif conf > 50:
        return "Moderate"
    return "Low"


def _trust_score(conf, reliability, agreement):
    score = conf

    if reliability == "Low":
        score -= 20
    elif reliability == "Medium":
        score -= 10

    if agreement == "No":
        score -= 15

    return round(max(score, 0), 2)


def _clinical_reasoning(r):
    label = r["label"]
    conf = r["confidence"]

    if label == "High Stress" and conf > 80:
        return "Strong linguistic stress indicators detected with high confidence"
    elif label == "High Stress":
        return "Moderate stress signals detected in text patterns"
    elif label == "Low Stress" and conf > 80:
        return "Stable emotional indicators with consistent patterns"
    else:
        return "Low intensity emotional signals detected"


def _explain_keywords(keywords):
    if not keywords:
        return "No dominant keywords detected"
    return ", ".join(keywords[:5])


def render_result(r):

    if not r:
        st.info("No result yet")
        return

    st.markdown("## 🧾 AI Clinical Result")

    label = r["label"]
    confidence = r["confidence"]
    risk = r["risk"]
    severity = r["severity"]
    reliability = r["reliability"]
    agreement = r["agreement"]
    advice = r["advice"]
    model = r["model"]

    # ---------- PRIMARY RESULT ----------
    col1, col2 = st.columns([1.2, 1])

    with col1:

        if "Low" in label:
            st.success(label)
        elif "High" in label:
            st.error(label)
        else:
            st.warning(label)

        st.progress(int(confidence))

        st.metric("Confidence", f"{confidence}%")
        st.metric("Risk Level", risk)
        st.metric("Severity", severity)

    with col2:
        st.markdown("### 🧠 Model Info")
        st.metric("Model Used", model)
        st.metric("Reliability", reliability)
        st.metric("Agreement", agreement)

    st.markdown("---")

    # ---------- TRUST SCORE ----------
    trust = _trust_score(confidence, reliability, agreement)

    st.markdown("### 🔐 Trust Score")
    st.metric("AI Trust Score", f"{trust}%")

    if trust < 50:
        st.error("Low confidence decision → review needed")
    elif trust < 70:
        st.warning("Moderate trust → interpret carefully")
    else:
        st.success("High trust prediction")

    # ---------- CLINICAL REASONING ----------
    st.markdown("### 🧠 Clinical Reasoning")

    reasoning = _clinical_reasoning(r)
    st.info(reasoning)

    # ---------- EXPLAINABILITY ----------
    st.markdown("### 🔍 Explainability")

    keywords = r.get("keywords", [])
    st.write(f"Key Indicators: {_explain_keywords(keywords)}")

    st.write(f"Confidence Band: {_confidence_band(confidence)}")

    # ---------- DECISION SUPPORT ----------
    st.markdown("### ⚕️ Clinical Decision Support")

    if risk == "Critical":
        st.error("Immediate clinical attention recommended")
    elif risk == "High":
        st.warning("Monitor closely and apply intervention strategies")
    elif risk == "Moderate":
        st.info("Recommend relaxation and monitoring")
    else:
        st.success("Stable condition")

    # ---------- RECOMMENDATION ----------
    st.markdown("### 💡 AI Recommendation")
    st.success(advice)

    # ---------- ALERT ENGINE ----------
    st.markdown("### 🚨 Alert Engine")

    if agreement == "No":
        st.warning("Model disagreement detected → second opinion advised")

    if reliability == "Low":
        st.error("Low reliability → prediction may be unstable")

    if confidence < 50:
        st.warning("Low confidence prediction")

    # ---------- SUMMARY ----------
    st.markdown("### 📄 Final Summary")

    summary = f"""
Stress Level: {label}  
Confidence: {confidence}%  
Risk: {risk}  
Severity: {severity}  
Trust Score: {trust}%  
Advice: {advice}
"""
    st.text(summary)