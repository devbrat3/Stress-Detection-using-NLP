import streamlit as st
import sys
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from predict import predict_stress

st.set_page_config(page_title="Stress Detection", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🏥 Stress Detection Dashboard")

col1, col2 = st.columns([2, 1])

with col1:
    text = st.text_area("Enter text", height=200)
    btn = st.button("Analyze")

with col2:
    st.markdown("### Result")

    if btn:
        label, confidence, risk = predict_stress(text)

        st.session_state.history.append((label, confidence))

        if "Low" in label:
            st.success(label)
        elif "High" in label:
            st.error(label)
        else:
            st.warning(label)

        st.progress(int(confidence))
        st.write(f"Confidence: {confidence}%")
        st.write(f"Risk Level: {risk}")

st.markdown("---")

if btn and text.strip():
    st.markdown("### Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Stress", label)
    c2.metric("Confidence", f"{confidence}%")
    c3.metric("Risk", risk)

st.markdown("---")

if st.session_state.history:
    st.markdown("### Stress Trend")

    df = pd.DataFrame(st.session_state.history, columns=["Label", "Confidence"])
    st.line_chart(df["Confidence"])

    st.markdown("### Recent History")
    for i, (l, c) in enumerate(st.session_state.history[-5:]):
        st.write(f"{i+1}. {l} - {c}%")