import streamlit as st


def render_intelligence(result, history, alerts, insights, intelligence, recommendation, summary):

    st.markdown("## 🧠 Clinical Intelligence")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🚨 Alerts")
        if alerts:
            for a in alerts:
                st.warning(a)
        else:
            st.success("No critical alerts")

    with c2:
        st.markdown("### 📊 Intelligence Score")
        st.metric("Risk Score", intelligence["risk_score"])
        st.metric("Volatility", intelligence["volatility"])

    st.markdown("---")

    st.markdown("### 🧠 Insights")
    for i in insights:
        st.info(i)

    st.markdown("---")

    st.markdown("### 💡 Recommendation")
    st.success(recommendation)

    st.markdown("---")

    st.markdown("### 📄 Summary")
    st.code(summary)