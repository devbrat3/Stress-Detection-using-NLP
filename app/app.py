import streamlit as st
import sys
import os

# Fix path to import from src
sys.path.append(os.path.abspath("../src"))

from predict import predict_stress

# Page settings
st.set_page_config(page_title="Stress Detection System", layout="wide")

# Title
st.title("🧠 Stress Detection System")
st.subheader("AI-Based Mental Health Analysis")

# Input section
st.markdown("### Enter your text below:")
user_input = st.text_area("Type here...", height=150)

# Button
if st.button("Analyze Stress"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = predict_stress(user_input)

        st.markdown("### Result:")
        
        if result == "Low Stress":
            st.success(f"Stress Level: {result}")
        elif result == "High Stress":
            st.error(f"Stress Level: {result}")
        else:
            st.info(f"Stress Level: {result}")