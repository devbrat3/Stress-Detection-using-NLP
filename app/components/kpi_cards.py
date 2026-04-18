import streamlit as st
import numpy as np

def _color(metric, value):
    if metric == "severity":
        return "#ff4b4b" if value == "Severe" else "#ffa500" if value == "Elevated" else "#00c853"
    if metric == "reliability":
        return "#00c853" if value == "High" else "#ffa500" if value == "Medium" else "#ff4b4b"
    if metric == "agreement":
        return "#00c853" if value == "Yes" else "#ff4b4b"
    return "#00c6ff"

def _trend(history):
    if len(history) < 2:
        return 0
    return history[-1][1] - history[-2][1]

def _volatility(history):
    if len(history) < 3:
        return 0
    vals = [c for _, c in history[-5:]]
    return round(np.std(vals), 2)

def _risk_score(conf, severity):
    weight = {"Severe": 1.0, "Elevated": 0.7, "Normal": 0.3}
    return round(conf * weight.get(severity, 0.5), 2)

def render_kpis(r, history):

    st.markdown("## 📊 Clinical Intelligence Panel")

    severity = r.get("severity", "Normal")
    reliability = r.get("reliability", "Medium")
    agreement = r.get("agreement", "Yes")
    confidence = r["confidence"]
    uncertainty = round(100 - confidence, 2)

    trend = _trend(history)
    volatility = _volatility(history)
    risk_score = _risk_score(confidence, severity)

    # ---------- KPI CARDS ----------
    c1, c2, c3, c4 = st.columns(4)

    def card(title, value, color, sub=""):
        return f"""
        <div style="background:#1e1e2f;padding:18px;border-radius:16px;
        box-shadow:0 8px 20px rgba(0,0,0,0.4);text-align:center">
            <div style="font-size:26px;font-weight:bold;color:{color}">
                {value}
            </div>
            <div style="color:#9aa0a6">{title}</div>
            <div style="font-size:12px;color:gray">{sub}</div>
        </div>
        """

    c1.markdown(card("Severity", severity, _color("severity", severity)), unsafe_allow_html=True)
    c2.markdown(card("Reliability", reliability, _color("reliability", reliability)), unsafe_allow_html=True)
    c3.markdown(card("Agreement", agreement, _color("agreement", agreement)), unsafe_allow_html=True)
    c4.markdown(card("Uncertainty", f"{uncertainty}%", "#00c6ff"), unsafe_allow_html=True)

    st.markdown("---")

    # ---------- ADVANCED METRICS ----------
    st.markdown("### 🧠 Advanced Intelligence Metrics")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Confidence", f"{confidence}%")
    m2.metric("Risk Score", risk_score)
    m3.metric("Trend Δ", f"{trend:+.2f}")
    m4.metric("Volatility", volatility)

    # ---------- INTERPRETATION ----------
    st.markdown("### 🔍 Clinical Interpretation")

    col1, col2, col3 = st.columns(3)

    with col1:
        if severity == "Severe":
            st.error("Critical stress condition")
        elif severity == "Elevated":
            st.warning("Elevated stress level")
        else:
            st.success("Normal condition")

    with col2:
        if reliability == "Low":
            st.error("Low reliability")
        elif reliability == "Medium":
            st.warning("Moderate reliability")
        else:
            st.success("High reliability")

    with col3:
        if agreement == "No":
            st.warning("Model disagreement")
        else:
            st.success("Model consensus")

    # ---------- BEHAVIOR ANALYSIS ----------
    st.markdown("### 📈 Behavioral Insights")

    if volatility > 15:
        st.error("Highly unstable stress pattern")
    elif volatility > 8:
        st.warning("Fluctuating condition")
    else:
        st.success("Stable behavior")

    if trend > 5:
        st.error("Rapid increase detected")
    elif trend > 0:
        st.warning("Increasing trend")
    elif trend < -5:
        st.success("Strong improvement")
    elif trend < 0:
        st.info("Improving condition")
    else:
        st.info("Stable")

    # ---------- PROGRESS ----------
    st.markdown("### 📊 Confidence Distribution")

    st.progress(int(confidence))

    # ---------- MICRO GRAPH ----------
    if len(history) > 2:
        st.markdown("### 📉 Recent Pattern")

        vals = [c for _, c in history[-10:]]
        st.line_chart(vals)