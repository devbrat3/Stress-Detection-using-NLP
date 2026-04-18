import streamlit as st
from collections import Counter
import re
import numpy as np

SUGGESTIONS = [
    "I feel overwhelmed with work and cannot focus",
    "I am anxious and unable to sleep properly",
    "Everything feels stressful and heavy",
    "I feel calm and productive today"
]

EMOTION_MAP = {
    "stress": ["overwhelmed", "pressure", "burden"],
    "anxiety": ["anxious", "panic", "nervous"],
    "depression": ["sad", "hopeless", "low"],
    "positive": ["calm", "happy", "relaxed"]
}


def _clean(text):
    return re.sub(r"[^a-zA-Z\s]", "", text.lower())


def _keywords(text):
    words = _clean(text).split()
    words = [w for w in words if len(w) > 4]
    return Counter(words).most_common(6)


def _emotion_score(text):
    text = _clean(text)
    scores = {k: 0 for k in EMOTION_MAP}

    for category, words in EMOTION_MAP.items():
        for w in words:
            if w in text:
                scores[category] += 1

    total = sum(scores.values()) + 1
    normalized = {k: round(v / total * 100, 2) for k, v in scores.items()}

    return normalized


def _complexity(text):
    words = text.split()
    if not words:
        return 0

    avg_len = np.mean([len(w) for w in words])
    diversity = len(set(words)) / len(words)

    score = (avg_len * 5 + diversity * 50)
    return round(min(score, 100), 2)


def _readability(text):
    words = text.split()
    if len(words) < 5:
        return "Low"
    elif len(words) < 15:
        return "Medium"
    return "High"


def render_input():

    st.markdown("## 🧠 Smart Clinical Input System")

    # ---------- INPUT ----------
    text = st.text_area(
        "Patient Description",
        height=170,
        placeholder="Describe thoughts, emotions, or symptoms in detail..."
    )

    # ---------- ACTION BAR ----------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("📋 Sample"):
            text = SUGGESTIONS[0]

    with c2:
        if st.button("🧠 Stress"):
            text = SUGGESTIONS[2]

    with c3:
        if st.button("😊 Positive"):
            text = SUGGESTIONS[3]

    with c4:
        analyze = st.button("🔍 Analyze", use_container_width=True)

    # ---------- LIVE METRICS ----------
    if text:
        words = len(text.split())
        chars = len(text)
        complexity = _complexity(text)
        readability = _readability(text)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Words", words)
        m2.metric("Chars", chars)
        m3.metric("Complexity", f"{complexity}%")
        m4.metric("Clarity", readability)

        st.progress(int(complexity))

    # ---------- VALIDATION ----------
    if text:
        if len(text.split()) < 5:
            st.warning("Low detail → model confidence may drop")

        if len(text.split()) > 200:
            st.warning("Too long → noise may affect prediction")

    # ---------- NLP ANALYSIS ----------
    if text:
        st.markdown("### 🧠 Pre-AI Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Keywords**")
            for w, f in _keywords(text):
                st.write(f"{w} ({f})")

        with col2:
            st.markdown("**Emotion Distribution**")
            emotions = _emotion_score(text)

            for k, v in emotions.items():
                st.progress(int(v))
                st.caption(f"{k}: {v}%")

    # ---------- COGNITIVE SIGNALS ----------
    if text:
        st.markdown("### 🔍 Cognitive Signals")

        if "not" in text.lower():
            st.info("Negation detected → possible negative cognitive bias")

        if len(set(text.split())) < len(text.split()) * 0.5:
            st.warning("Low vocabulary diversity → repetitive thought pattern")

    # ---------- AI GUIDANCE ----------
    if text:
        st.markdown("### 💡 AI Guidance")

        if len(text.split()) < 10:
            st.info("Add more details → improves prediction accuracy")

        if "feel" not in text.lower():
            st.info("Include emotions → helps model interpret mental state")

    return text, analyze