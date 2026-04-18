import streamlit as st
import sys
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from hybrid_predict import hybrid_predict

st.set_page_config(page_title="Stress Detection AI", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


def gauge(value):
    fig, ax = plt.subplots()
    ax.pie(
        [value, 100 - value],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.3)
    )
    ax.text(0, 0, f"{value}%", ha="center", va="center", fontsize=18)
    return fig


def trend(history):
    if len(history) < 3:
        return "Insufficient"
    vals = [c for _, c in history[-5:]]
    return "Increasing" if vals[-1] > vals[0] else "Decreasing" if vals[-1] < vals[0] else "Stable"


def pattern(history):
    if len(history) < 5:
        return "Insufficient"
    vals = [c for _, c in history]
    if all(v > 70 for v in vals[-3:]):
        return "Chronic"
    if all(v < 40 for v in vals[-3:]):
        return "Stable"
    return "Fluctuating"


def uncertainty(conf):
    return round(100 - conf, 2)


with st.sidebar:
    st.title("🧠 Mental Health AI")

    page = st.radio("Navigation", ["Dashboard", "Analytics", "Monitor", "Reports", "System"])
    model_mode = st.selectbox("Model Mode", ["AUTO", "ML", "BERT"])

    st.markdown("---")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history, columns=["Label", "Confidence"])
        st.metric("Avg Stress", f"{round(df['Confidence'].mean(),2)}%")
        st.metric("Sessions", len(df))

    st.caption("Clinical AI v5")


# ================= DASHBOARD =================
if page == "Dashboard":

    st.markdown("## 🏥 Clinical Intelligence Dashboard")

    c1, c2 = st.columns([2.5, 1])

    with c1:
        text = st.text_area("Patient Input", height=150)
        run = st.button("Analyze", use_container_width=True)

    with c2:
        st.markdown("### Result")

        if run and text.strip():
            with st.spinner("Processing..."):
                res = hybrid_predict(text, model_mode)

            st.session_state.last_result = res
            st.session_state.history.append((res["label"], res["confidence"]))

        if st.session_state.last_result:
            r = st.session_state.last_result

            (st.success if "Low" in r["label"] else st.error if "High" in r["label"] else st.warning)(r["label"])
            st.progress(int(r["confidence"]))
            st.metric("Confidence", f"{r['confidence']}%")
            st.metric("Risk", r["risk"])
            st.caption(r["model"])

    st.markdown("---")

    if st.session_state.last_result:
        r = st.session_state.last_result

        # -------- CORE METRICS --------
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Severity", r["severity"])
        m2.metric("Reliability", r["reliability"])
        m3.metric("Agreement", r["agreement"])
        m4.metric("Uncertainty", f"{uncertainty(r['confidence'])}%")

        if r["agreement"] == "No":
            st.warning("Model disagreement")
        if r["reliability"] == "Low":
            st.error("Low reliability")

        # -------- VISUAL GRID --------
        v1, v2 = st.columns(2)

        with v1:
            st.markdown("### Stress Gauge")
            st.pyplot(gauge(r["confidence"]))

        with v2:
            fig, ax = plt.subplots()
            ax.bar(["Confidence", "Uncertainty"], [r["confidence"], uncertainty(r["confidence"])])
            st.pyplot(fig)

        # -------- TEXT ANALYSIS --------
        st.markdown("### Indicators")
        for w, f in Counter([w for w in text.lower().split() if len(w) > 3]).most_common(5):
            st.write(f"{w} ({f})")

        # -------- INTELLIGENCE --------
        t = trend(st.session_state.history)
        p = pattern(st.session_state.history)

        st.markdown("### Intelligence")
        st.write(f"Trend: {t}")
        st.write(f"Pattern: {p}")

        if len(st.session_state.history) > 1:
            prev = st.session_state.history[-2][1]
            st.metric("Δ Change", f"{r['confidence'] - prev:+.2f}%")

        # -------- EXPLAINABILITY --------
        st.markdown("### AI Explanation")
        st.info(r.get("explanation", "N/A"))

        # -------- ALERT ENGINE --------
        st.markdown("### Alert")

        if r["risk"] == "Critical":
            st.error("Immediate intervention required")
        elif r["risk"] == "High":
            st.warning("High stress detected")
        else:
            st.success("Stable")

        # -------- RECOMMENDATION --------
        st.markdown("### Recommendation")
        st.info(r["advice"])

        # -------- SUMMARY --------
        st.markdown("### Summary")
        st.text(
            f"Stress: {r['label']} | Confidence: {r['confidence']}% | Trend: {t} | Pattern: {p}"
        )


# ================= ANALYTICS =================
elif page == "Analytics":

    st.title("📊 Analytics")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history, columns=["Label", "Confidence"])

        a1, a2 = st.columns(2)
        a1.line_chart(df["Confidence"])

        fig, ax = plt.subplots()
        df["Label"].value_counts().plot(kind="bar", ax=ax)
        a2.pyplot(fig)

        st.dataframe(df.describe())

    else:
        st.info("No data")


# ================= MONITOR =================
elif page == "Monitor":

    st.title("👤 Monitoring")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history, columns=["Label", "Confidence"])

        st.dataframe(df)
        st.line_chart(df["Confidence"])

        avg = df["Confidence"].mean()
        st.metric("Average Stress", f"{round(avg,2)}%")

        (st.error if avg > 75 else st.warning if avg > 50 else st.success)(
            "Chronic" if avg > 75 else "Moderate" if avg > 50 else "Stable"
        )

    else:
        st.info("No data")


# ================= REPORTS =================
elif page == "Reports":

    st.title("📄 Reports")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history, columns=["Label", "Confidence"])
        st.download_button("Download", df.to_csv(index=False), "report.csv")
        st.dataframe(df)
    else:
        st.info("No data")


# ================= SYSTEM =================
elif page == "System":

    st.title("⚙️ System")

    st.markdown("""
- Hybrid AI Engine (ML + BERT)
- Explainable AI
- Trend + Pattern Intelligence
- Clinical Decision Support
- Real-time Monitoring
""")

    st.success("System Operational")