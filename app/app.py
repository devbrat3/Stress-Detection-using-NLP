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

# -------- SIDEBAR --------
with st.sidebar:
    st.title("🧠 Mental Health AI")

    page = st.radio(
        "Navigation",
        ["Dashboard", "Analytics", "Patient Monitor", "Reports", "System"]
    )

    model_mode = st.selectbox(
        "Model Mode",
        ["AUTO", "ML", "BERT"]
    )

    st.markdown("---")
    st.caption("Clinical AI Monitoring System v2")

# ================= DASHBOARD =================
if page == "Dashboard":

    st.title("🏥 Clinical Stress Detection Dashboard")

    left, right = st.columns([2.7, 1])

    # -------- INPUT --------
    with left:
        text = st.text_area(
            "Patient Input",
            height=180,
            placeholder="Enter patient thoughts..."
        )

        analyze = st.button("Analyze", use_container_width=True)

    # -------- RESULT PANEL --------
    with right:
        st.markdown("### 🧾 Clinical Output")

        if analyze and text.strip():

            result = hybrid_predict(text, model_mode)

            st.session_state.last_result = result
            st.session_state.history.append(
                (result["label"], result["confidence"])
            )

        if st.session_state.last_result:

            r = st.session_state.last_result

            if "Low" in r["label"]:
                st.success(r["label"])
            elif "High" in r["label"]:
                st.error(r["label"])
            else:
                st.warning(r["label"])

            st.progress(int(r["confidence"]))
            st.metric("Confidence", f"{r['confidence']}%")
            st.metric("Risk", r["risk"])

            st.caption(f"Model: {r['model']}")

    st.markdown("---")

    # -------- ADVANCED PANEL --------
    if st.session_state.last_result:

        r = st.session_state.last_result

        col1, col2, col3 = st.columns(3)

        col1.metric("Severity", r["severity"])
        col2.metric("Reliability", r["reliability"])
        col3.metric("Agreement", r["agreement"])

        # -------- ALERTS --------
        if r["agreement"] == "No":
            st.warning("⚠️ Model disagreement detected")

        if r["reliability"] == "Low":
            st.error("Low prediction reliability")

        # -------- KEYWORDS --------
        st.markdown("### 🧠 Key Indicators")

        words = [w for w in text.lower().split() if len(w) > 3]
        for w, f in Counter(words).most_common(5):
            st.write(f"🔹 {w} ({f})")

        # -------- VISUALS --------
        v1, v2 = st.columns(2)

        with v1:
            fig, ax = plt.subplots()
            ax.barh(["Stress"], [r["confidence"]])
            ax.set_xlim(0, 100)
            st.pyplot(fig)

        with v2:
            fig2, ax2 = plt.subplots()
            ax2.pie(
                [r["confidence"], 100 - r["confidence"]],
                labels=["Stress", "Neutral"],
                autopct="%1.1f%%"
            )
            st.pyplot(fig2)

        # -------- RECOMMENDATION --------
        st.markdown("### 💡 Clinical Recommendation")
        st.info(r["advice"])

        # -------- TREND --------
        if len(st.session_state.history) > 3:

            recent = [c for _, c in st.session_state.history[-5:]]

            st.markdown("### 📈 Behavioral Insight")

            if recent[-1] > recent[0]:
                st.warning("Increasing stress trend")
            else:
                st.success("Stable or improving trend")

# ================= ANALYTICS =================
elif page == "Analytics":

    st.title("📊 Clinical Analytics")

    if st.session_state.history:

        df = pd.DataFrame(
            st.session_state.history,
            columns=["Label", "Confidence"]
        )

        c1, c2 = st.columns(2)

        c1.line_chart(df["Confidence"])

        fig, ax = plt.subplots()
        df["Label"].value_counts().plot(kind="bar", ax=ax)
        c2.pyplot(fig)

        st.dataframe(df.describe())

    else:
        st.info("No data")

# ================= PATIENT MONITOR =================
elif page == "Patient Monitor":

    st.title("👤 Patient Monitoring")

    if st.session_state.history:

        df = pd.DataFrame(
            st.session_state.history,
            columns=["Label", "Confidence"]
        )

        st.dataframe(df, use_container_width=True)

        st.line_chart(df["Confidence"])

        avg = df["Confidence"].mean()

        st.metric("Average Stress", f"{round(avg,2)}%")

        if avg > 75:
            st.error("Chronic high stress detected")
        elif avg > 50:
            st.warning("Moderate stress pattern")
        else:
            st.success("Stable condition")

    else:
        st.info("No patient data")

# ================= REPORTS =================
elif page == "Reports":

    st.title("📄 Clinical Reports")

    if st.session_state.history:

        df = pd.DataFrame(
            st.session_state.history,
            columns=["Label", "Confidence"]
        )

        st.download_button(
            "Download Report",
            df.to_csv(index=False),
            "stress_report.csv"
        )

    else:
        st.info("No data")

# ================= SYSTEM =================
elif page == "System":

    st.title("⚙️ System Overview")

    st.markdown("""
    - Hybrid AI System (ML + BERT)
    - Clinical Decision Support
    - Real-time Monitoring
    - Deployed on Streamlit Cloud
    """)

    st.info("This system assists in early stress detection.")