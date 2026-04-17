import streamlit as st
import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from predict import predict_stress

st.set_page_config(page_title="Stress Detection AI", page_icon="🧠", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
<style>
.main-title {font-size:38px;font-weight:700;}
.sub {color:#94a3b8;}
.metric {font-size:20px;font-weight:600;}
.block {padding:15px;border-radius:10px;background:#0f172a;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🧠 Mental AI")
    page = st.radio("", ["Dashboard", "Analytics", "About"])

if page == "Dashboard":

    st.markdown('<div class="main-title">Stress Detection Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Real-time NLP-based stress analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2.2, 1])

    with col1:
        text = st.text_area("Input Text", height=200, placeholder="Type your thoughts...")
        btn = st.button("Analyze", use_container_width=True)

    with col2:
        st.markdown("### Result")

        if btn:
            with st.spinner("Analyzing..."):
                time.sleep(0.8)
                label, confidence = predict_stress(text)
                st.session_state.history.append((label, confidence))

                if label == "Low Stress":
                    st.success("🟢 Low Stress")
                elif label == "High Stress":
                    st.error("🔴 High Stress")
                else:
                    st.info(label)

                st.progress(int(confidence))
                st.markdown(f"**Confidence:** {confidence}%")

                if confidence > 80:
                    st.error("High Risk")
                elif confidence > 50:
                    st.warning("Moderate Risk")
                else:
                    st.success("Low Risk")

    st.markdown("---")

    if btn and text.strip():

        st.markdown("### Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Stress", label)
        c2.metric("Confidence", f"{confidence}%")
        c3.metric("Words", len(text.split()))

        st.markdown("### Text Insights")
        words = text.split()
        common = Counter(words).most_common(5)
        for w, f in common:
            st.write(f"{w} ({f})")

        st.markdown("### Confidence Distribution")
        fig1, ax1 = plt.subplots()
        ax1.pie([confidence, 100-confidence], labels=["Stress", "Neutral"], autopct="%1.1f%%")
        st.pyplot(fig1)

    st.markdown("---")

    if st.session_state.history:
        st.markdown("### Trend")

        df = pd.DataFrame(st.session_state.history, columns=["Label", "Confidence"])
        st.line_chart(df["Confidence"])

        st.markdown("### Recent Results")
        for i, (l, c) in enumerate(st.session_state.history[-5:]):
            st.write(f"{i+1}. {l} - {c}%")

elif page == "Analytics":

    st.title("Model Analytics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "Logistic Regression")
    col2.metric("Features", "TF-IDF")
    col3.metric("Status", "Active")

    st.markdown("---")
    st.write("This module processes text using NLP and predicts stress level.")

elif page == "About":

    st.title("About")

    st.write("""
    AI-based stress detection system using NLP and Machine Learning.

    - Real-time prediction
    - Interactive dashboard
    - Deployable architecture
    """)