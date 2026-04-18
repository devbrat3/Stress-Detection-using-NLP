import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def _forecast(values):
    if len(values) < 5:
        return None
    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    future = intercept + slope * (len(values) + 1)
    return round(future, 2), slope


def _zones(val):
    if val > 75:
        return "Critical"
    elif val > 50:
        return "Moderate"
    else:
        return "Stable"


def render_charts(history):

    if not history:
        st.info("No analytics data available")
        return

    df = pd.DataFrame(history, columns=["Label", "Confidence"])
    df["Session"] = range(1, len(df) + 1)
    df["Uncertainty"] = 100 - df["Confidence"]
    df["Rolling"] = df["Confidence"].rolling(3).mean()
    df["Expanding"] = df["Confidence"].expanding().mean()
    df["Volatility"] = df["Confidence"].rolling(3).std()

    st.markdown("## 📊 Clinical Intelligence Dashboard")

    # ================= ROW 1 =================
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 📈 Stress Evolution Engine")

        fig, ax = plt.subplots()
        ax.plot(df["Session"], df["Confidence"], marker="o", linewidth=2)
        ax.plot(df["Session"], df["Rolling"], linestyle="--")
        ax.plot(df["Session"], df["Expanding"], linestyle=":")
        ax.axhline(75, linestyle="--")
        ax.axhline(50, linestyle="--")
        ax.set_title("Stress Progression with Zones")
        ax.grid(True)
        st.pyplot(fig)

    with c2:
        st.markdown("### 📊 Stress Composition")

        fig, ax = plt.subplots()
        df["Label"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

    # ================= ROW 2 =================
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("### 📉 Confidence Density")

        fig, ax = plt.subplots()
        ax.hist(df["Confidence"], bins=15)
        ax.set_title("Confidence Density Distribution")
        st.pyplot(fig)

    with c4:
        st.markdown("### ⚖️ Confidence vs Uncertainty Map")

        fig, ax = plt.subplots()
        ax.scatter(df["Confidence"], df["Uncertainty"])
        ax.set_title("Decision Boundary View")
        st.pyplot(fig)

    # ================= ROW 3 =================
    c5, c6 = st.columns(2)

    with c5:
        st.markdown("### 🔥 Heatmap Intelligence")

        heat = np.array(df["Confidence"]).reshape(-1, 1)

        fig, ax = plt.subplots()
        ax.imshow(heat, aspect="auto")
        st.pyplot(fig)

    with c6:
        st.markdown("### 📊 Volatility Engine")

        fig, ax = plt.subplots()
        ax.plot(df["Volatility"])
        ax.set_title("Stress Volatility")
        st.pyplot(fig)

    # ================= ROW 4 =================
    st.markdown("### 📊 Statistical Core")

    stats = df["Confidence"].describe()
    st.dataframe(stats)

    # ================= ROW 5 =================
    st.markdown("### 🧠 Behavioral Intelligence")

    avg = df["Confidence"].mean()
    std = df["Confidence"].std()
    peak = df["Confidence"].max()

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Average", f"{round(avg,2)}%")
    m2.metric("Peak", f"{round(peak,2)}%")
    m3.metric("Volatility", f"{round(std,2)}")
    m4.metric("Zone", _zones(avg))

    # ================= ROW 6 =================
    st.markdown("### 🔍 AI Interpretation")

    if avg > 75:
        st.error("Chronic stress condition")
    elif avg > 50:
        st.warning("Moderate stress pattern")
    else:
        st.success("Healthy condition")

    if std > 15:
        st.error("Highly unstable pattern")
    elif std > 8:
        st.warning("Fluctuating stress")
    else:
        st.success("Stable behavior")

    # ================= ROW 7 =================
    st.markdown("### 🔮 Predictive Intelligence")

    forecast = _forecast(df["Confidence"].values)

    if forecast:
        future, slope = forecast

        c1, c2 = st.columns(2)
        c1.metric("Next Session Prediction", f"{future}%")
        c2.metric("Trend Slope", f"{round(slope,2)}")

        if slope > 2:
            st.error("High future risk")
        elif slope > 0:
            st.warning("Increasing stress expected")
        else:
            st.success("Improving trend")

    # ================= ROW 8 =================
    st.markdown("### 🧠 Decision Support")

    if avg > 70 and std > 10:
        st.error("Immediate intervention recommended")
    elif avg > 50:
        st.warning("Monitor patient regularly")
    else:
        st.success("Maintain routine care")