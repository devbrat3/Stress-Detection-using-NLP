import streamlit as st
import pandas as pd
from datetime import datetime

def render_sidebar():

    if "history" not in st.session_state:
        st.session_state.history = []

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    with st.sidebar:

        # ---------- HEADER ----------
        st.markdown("## 🧠 Mental Health AI")
        st.caption("Clinical Intelligence Platform")

        st.markdown("---")

        # ---------- NAV ----------
        page = st.radio(
            "Navigation",
            ["Dashboard", "Analytics", "Monitor", "Reports", "System"],
            index=0
        )

        # ---------- MODEL CONTROL ----------
        st.markdown("### 🧪 Model Control")

        model_mode = st.selectbox(
            "Engine",
            ["AUTO", "ML", "BERT"],
            index=0
        )

        perf_mode = st.toggle("⚡ Performance Mode", value=True)
        explain_mode = st.toggle("🧠 Explainability", value=True)

        # ---------- SESSION CONTROL ----------
        st.markdown("### 🧾 Session Control")

        col1, col2, col3 = st.columns(3)

        if col1.button("Reset"):
            st.session_state.history.clear()
            st.session_state.last_result = None
            st.rerun()

        if col2.button("Demo"):
            st.session_state.history.extend([
                ("Low Stress", 22),
                ("Moderate Stress", 48),
                ("High Stress", 81)
            ])
            st.rerun()

        if col3.button("Undo"):
            if st.session_state.history:
                st.session_state.history.pop()
                st.rerun()

        st.markdown("---")

        # ---------- LIVE METRICS ----------
        st.markdown("### 📊 Live Metrics")

        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history, columns=["Label", "Confidence"])

            avg = round(df["Confidence"].mean(), 2)
            last = df["Confidence"].iloc[-1]
            max_val = df["Confidence"].max()
            min_val = df["Confidence"].min()

            m1, m2 = st.columns(2)
            m1.metric("Avg", f"{avg}%")
            m2.metric("Last", f"{last}%")

            m3, m4 = st.columns(2)
            m3.metric("Max", f"{max_val}%")
            m4.metric("Min", f"{min_val}%")

            st.metric("Sessions", len(df))

            # ---------- TREND ----------
            if len(df) > 2:
                trend = "↑ Increasing" if last > avg else "↓ Decreasing" if last < avg else "→ Stable"
                st.caption(f"Trend: {trend}")

            # ---------- MINI ANALYTICS ----------
            st.line_chart(df["Confidence"], height=100)

        else:
            st.info("No session data available")

        st.markdown("---")

        # ---------- ALERT MONITOR ----------
        st.markdown("### 🚨 Alert Monitor")

        if st.session_state.last_result:
            r = st.session_state.last_result

            if r["risk"] == "Critical":
                st.error("Critical Condition")
            elif r["risk"] == "High":
                st.warning("High Stress")
            else:
                st.success("Stable")

        else:
            st.caption("No active alerts")

        st.markdown("---")

        # ---------- SYSTEM HEALTH ----------
        st.markdown("### ⚙️ System Health")

        st.success("● System Active")
        st.caption(f"Time: {datetime.now().strftime('%H:%M:%S')}")

        if perf_mode:
            st.caption("Mode: Optimized ⚡")
        else:
            st.caption("Mode: Standard")

        # ---------- MODEL INFO ----------
        st.markdown("### 🧠 Model Info")

        if model_mode == "AUTO":
            st.write("Hybrid AI (ML + BERT)")
        elif model_mode == "ML":
            st.write("Logistic Regression")
        else:
            st.write("Transformer Model")

        # ---------- FOOTER ----------
        st.markdown("---")
        st.caption("Clinical AI v9 | Devbrat Kumar Saha")

    return page, model_mode, perf_mode, explain_mode