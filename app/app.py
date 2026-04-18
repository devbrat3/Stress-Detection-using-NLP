import streamlit as st
import sys
import os
import pandas as pd
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.extend([
    os.path.join(BASE_DIR, "src"),
    os.path.join(BASE_DIR, "components")
])

from hybrid_predict import hybrid_predict
from input_panel import render_input
from result_panel import render_result
from kpi_cards import render_kpis
from charts import render_charts

# 🔥 NEW IMPORT (IMPORTANT)
from intelligence_engine import (
    generate_alerts,
    generate_insights,
    generate_intelligence,
    generate_recommendation,
    generate_summary
)


# ---------- CONFIG ----------
st.set_page_config(
    page_title="AI Mental Health Platform",
    layout="wide"
)


# ---------- SESSION ----------
if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

if "latency" not in st.session_state:
    st.session_state.latency = 0


# ---------- HEADER ----------
st.markdown("""
<div style="padding:18px;border-radius:12px;
background:linear-gradient(90deg,#00c6ff,#0072ff);color:white">
<h3>🧠 AI Clinical Intelligence Platform</h3>
<p>Decision Support • Real-time Monitoring • Predictive Intelligence</p>
</div>
""", unsafe_allow_html=True)


# ---------- SIDEBAR ----------
with st.sidebar:

    st.title("🧠 System Control")

    page = st.radio(
        "Navigation",
        ["Dashboard", "Analytics", "Monitor", "Reports", "System"]
    )

    model_mode = st.selectbox("Model Mode", ["AUTO", "ML", "BERT"])

    st.markdown("---")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history, columns=["Label", "Confidence"])

        st.metric("Sessions", len(df))
        st.metric("Avg Stress", f"{round(df['Confidence'].mean(),2)}%")

    st.caption(f"Latency: {st.session_state.latency} ms")


# ================= DASHBOARD =================
if page == "Dashboard":

    st.markdown("## 🏥 Clinical Decision Dashboard")

    # ---------- INPUT + RESULT ----------
    left, right = st.columns([2, 1])

    with left:
        text, analyze = render_input()

    with right:

        if analyze and text.strip():

            start = time.time()

            result = hybrid_predict(text, model_mode)

            st.session_state.latency = int((time.time() - start) * 1000)

            st.session_state.result = result
            st.session_state.history.append(
                (result["label"], result["confidence"])
            )

        render_result(st.session_state.result)

    st.markdown("---")

    # ---------- KPI ----------
    if st.session_state.result:
        render_kpis(st.session_state.result, st.session_state.history)

    st.markdown("---")

    # ================= 🧠 INTELLIGENCE LAYER =================
    if st.session_state.result:

        result = st.session_state.result
        history = st.session_state.history

        alerts = generate_alerts(result, history)
        insights = generate_insights(result, history)
        intelligence = generate_intelligence(result, history)
        recommendation = generate_recommendation(result)
        summary = generate_summary(result, history)

        st.markdown("## 🧠 Intelligence Layer")

        col1, col2 = st.columns(2)

        # ---------- ALERTS ----------
        with col1:
            st.markdown("### 🚨 Alerts")

            if alerts:
                for a in alerts:
                    st.error(a)
            else:
                st.success("No critical alerts")

        # ---------- INSIGHTS ----------
        with col2:
            st.markdown("### 🔍 Insights")

            if insights:
                for i in insights:
                    st.info(i)
            else:
                st.info("No major insights")

        st.markdown("---")

        # ---------- INTELLIGENCE METRICS ----------
        i1, i2 = st.columns(2)

        i1.metric("Risk Score", intelligence["risk_score"])
        i2.metric("Volatility", intelligence["volatility"])

        st.markdown("---")

        # ---------- RECOMMENDATION ----------
        st.markdown("### 💡 Recommendation")
        st.success(recommendation)

        # ---------- SUMMARY ----------
        st.markdown("### 📄 Clinical Summary")
        st.code(summary)

    st.markdown("---")

    # ---------- TREND ----------
    if len(st.session_state.history) > 2:

        st.markdown("### 📈 Trend Analysis")

        df = pd.DataFrame(st.session_state.history, columns=["Label", "Confidence"])
        st.line_chart(df["Confidence"])


# ================= ANALYTICS =================
elif page == "Analytics":

    st.markdown("## 📊 Advanced Analytics")

    render_charts(st.session_state.history)


# ================= MONITOR =================
elif page == "Monitor":

    st.markdown("## 👤 Patient Monitoring")

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history, columns=["Label", "Confidence"])

        c1, c2 = st.columns([2, 1])

        with c1:
            st.dataframe(df, use_container_width=True)
            st.line_chart(df["Confidence"])

        with c2:
            avg = df["Confidence"].mean()

            st.metric("Average Stress", f"{round(avg,2)}%")

            if avg > 75:
                st.error("Chronic stress condition")
            elif avg > 50:
                st.warning("Moderate stress")
            else:
                st.success("Stable")

    else:
        st.info("No monitoring data available")


# ================= REPORTS =================
elif page == "Reports":

    st.markdown("## 📄 Clinical Reports")

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history, columns=["Label", "Confidence"])

        st.download_button("⬇ CSV", df.to_csv(index=False), "report.csv")
        st.download_button("⬇ JSON", df.to_json(), "report.json")

        st.dataframe(df)

    else:
        st.info("No data available")


# ================= SYSTEM =================
elif page == "System":

    st.markdown("## ⚙️ System Intelligence")

    st.markdown("""
### 🧠 Architecture
- Hybrid AI (ML + BERT)
- Intelligence Engine
- Decision Support Layer

### ⚡ Capabilities
- Real-time prediction
- Behavioral analysis
- Risk scoring
- Alert system

### 🚀 Deployment
- Streamlit modular system
- Scalable architecture
""")

    st.success("System Fully Operational")