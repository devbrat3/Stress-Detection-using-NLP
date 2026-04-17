import streamlit as st
import sys
import os

# ✅ Base path (deployment safe)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from predict import predict_stress

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Stress Detection System",
    page_icon="🧠",
    layout="wide"
)

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.title("🧠 AI Mental Health")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Go to:", ["Home", "About", "How it Works"])
    st.markdown("---")
    st.markdown("### Developer")
    st.info("Devbrat Kumar Saha")

# ------------------- HOME PAGE -------------------
if page == "Home":

    st.title("🏥 Stress Detection Dashboard")
    st.markdown("AI-powered system for detecting stress from text")

    # Layout split
    col1, col2 = st.columns([2, 1])

    # -------- INPUT SECTION --------
    with col1:
        st.markdown("### 📝 Enter Text")

        user_input = st.text_area(
            "Type your thoughts or message here:",
            height=200,
            placeholder="Example: I feel overwhelmed with my work and deadlines..."
        )

        analyze_btn = st.button("🔍 Analyze Stress", use_container_width=True)

    # -------- RESULT SECTION --------
    with col2:
        st.markdown("### 📊 Analysis Result")

        if analyze_btn:

            if user_input.strip() == "":
                st.warning("⚠️ Please enter some text")
            else:
                result = predict_stress(user_input)

                # Display styled result
                if result == "Low Stress":
                    st.success("🟢 Low Stress")
                    st.progress(25)

                elif result == "High Stress":
                    st.error("🔴 High Stress")
                    st.progress(90)

                else:
                    st.info(result)

                # Additional info
                st.markdown("### 🧠 Insights")
                st.write("AI has analyzed linguistic patterns and emotional indicators.")

# ------------------- ABOUT PAGE -------------------
elif page == "About":

    st.title("📘 About This Project")

    st.markdown("""
    This system uses **Natural Language Processing (NLP)** and **Machine Learning**
    to detect stress levels from textual input.

    ### 🔍 Features:
    - Real-time stress detection
    - AI-powered text analysis
    - Scalable architecture

    ### 🧠 Models Used:
    - Logistic Regression
    - TF-IDF Vectorization

    ### 🎯 Goal:
    To provide early detection of stress for better mental health awareness.
    """)

# ------------------- HOW IT WORKS -------------------
elif page == "How it Works":

    st.title("⚙️ How It Works")

    st.markdown("""
    ### 🧩 Pipeline:

    1. **User Input**
       - Text entered by user

    2. **Preprocessing**
       - Cleaning
       - Stopword removal
       - Lemmatization

    3. **Feature Extraction**
       - TF-IDF Vectorization

    4. **Model Prediction**
       - Logistic Regression model predicts stress level

    5. **Output**
       - Stress level displayed with visual indicators
    """)