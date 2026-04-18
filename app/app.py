import streamlit as st
import sys
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from hybrid_predict import hybrid_predict

st.set_page_config(page_title="AI Mental Health Platform", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
.block-container {padding-top: 1rem;}
.card {
    background-color:#1e1e2f;
    padding:16px;
    border-radius:12px;
    text-align:center;
    box-shadow:0 4px 12px rgba(0,0,0,0.3);
}
.big {font-size:22px;font-weight:600;color:#00c6ff;}
.small {color:#9aa0a6;}
</style>
""", unsafe_allow_html=True)

# ---------- STATE ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "last" not in st.session_state:
    st.session_state.last = None

# ---------- HELPERS ----------
def gauge(v):
    fig, ax = plt.subplots()
    ax.pie([v, 100-v], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.35))
    ax.text(0,0,f"{v}%", ha='center', va='center', fontsize=20)
    return fig

def trend(h):
    if len(h) < 3: return "Insufficient"
    vals = [c for _,c in h[-5:]]
    return "Increasing" if vals[-1]>vals[0] else "Decreasing" if vals[-1]<vals[0] else "Stable"

def pattern(h):
    if len(h)<5: return "Insufficient"
    vals = [c for _,c in h]
    if all(v>70 for v in vals[-3:]): return "Chronic"
    if all(v<40 for v in vals[-3:]): return "Stable"
    return "Fluctuating"

def uncertainty(c): return round(100-c,2)

# ---------- HEADER ----------
st.markdown("""
<div style="padding:18px;border-radius:12px;
background:linear-gradient(90deg,#00c6ff,#0072ff);color:white">
<h3>🧠 AI Mental Health Platform</h3>
<p>Clinical Intelligence Dashboard</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("Navigation")
    page = st.radio("", ["Dashboard","Analytics","Monitor","Reports","System"])
    mode = st.selectbox("Model Mode", ["AUTO","ML","BERT"])

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history, columns=["Label","Confidence"])
        st.metric("Avg Stress", f"{round(df['Confidence'].mean(),2)}%")
        st.metric("Sessions", len(df))

    st.caption("Clinical AI v6")

# ================= DASHBOARD =================
if page == "Dashboard":

    left, right = st.columns([2.5,1])

    with left:
        text = st.text_area("Patient Input", height=140)
        run = st.button("Analyze", use_container_width=True)

    with right:
        st.markdown("### Result")

        if run and text.strip():
            res = hybrid_predict(text, mode)
            st.session_state.last = res
            st.session_state.history.append((res["label"], res["confidence"]))

        if st.session_state.last:
            r = st.session_state.last
            (st.success if "Low" in r["label"] else st.error if "High" in r["label"] else st.warning)(r["label"])
            st.progress(int(r["confidence"]))
            st.metric("Confidence", f"{r['confidence']}%")
            st.metric("Risk", r["risk"])
            st.caption(r["model"])

    st.markdown("---")

    if st.session_state.last:
        r = st.session_state.last

        # ---------- KPI CARDS ----------
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"<div class='card'><div class='big'>{r['severity']}</div><div class='small'>Severity</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card'><div class='big'>{r['reliability']}</div><div class='small'>Reliability</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card'><div class='big'>{r['agreement']}</div><div class='small'>Agreement</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='card'><div class='big'>{uncertainty(r['confidence'])}%</div><div class='small'>Uncertainty</div></div>", unsafe_allow_html=True)

        # ---------- ALERT BAR ----------
        if r["agreement"]=="No":
            st.warning("⚠️ Model disagreement detected")
        if r["reliability"]=="Low":
            st.error("Low prediction reliability")

        st.markdown("---")

        # ---------- VISUAL GRID ----------
        v1,v2,v3 = st.columns([1,1,1])

        with v1:
            st.markdown("#### Stress Gauge")
            st.pyplot(gauge(r["confidence"]))

        with v2:
            fig, ax = plt.subplots()
            ax.bar(["Confidence","Uncertainty"],
                   [r["confidence"],uncertainty(r["confidence"])])
            st.pyplot(fig)

        with v3:
            st.markdown("#### Trend")
            if len(st.session_state.history)>2:
                df = pd.DataFrame(st.session_state.history, columns=["Label","Confidence"])
                st.line_chart(df["Confidence"])
            else:
                st.info("Not enough data")

        # ---------- INTELLIGENCE PANEL ----------
        st.markdown("### Intelligence")

        t = trend(st.session_state.history)
        p = pattern(st.session_state.history)

        i1,i2,i3 = st.columns(3)
        i1.metric("Trend", t)
        i2.metric("Pattern", p)

        if len(st.session_state.history)>1:
            prev = st.session_state.history[-2][1]
            i3.metric("Δ Change", f"{r['confidence']-prev:+.2f}%")

        # ---------- TEXT INSIGHTS ----------
        st.markdown("### NLP Insights")
        for w,f in Counter([w for w in text.lower().split() if len(w)>3]).most_common(5):
            st.write(f"{w} ({f})")

        # ---------- EXPLAINABILITY ----------
        st.markdown("### AI Explanation")
        st.info(r.get("explanation","N/A"))

        # ---------- ALERT ENGINE ----------
        st.markdown("### Clinical Alert")
        if r["risk"]=="Critical":
            st.error("Immediate intervention required")
        elif r["risk"]=="High":
            st.warning("High stress detected")
        else:
            st.success("Stable condition")

        # ---------- RECOMMENDATION ----------
        st.markdown("### Recommendation")
        st.success(r["advice"])

        # ---------- SUMMARY ----------
        st.markdown("### Summary")
        st.code(f"{r['label']} | {r['confidence']}% | Trend:{t} | Pattern:{p}")

# ================= ANALYTICS =================
elif page == "Analytics":

    st.title("Analytics")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history, columns=["Label","Confidence"])

        a1,a2 = st.columns(2)
        a1.line_chart(df["Confidence"])

        fig, ax = plt.subplots()
        df["Label"].value_counts().plot(kind="bar", ax=ax)
        a2.pyplot(fig)

        st.dataframe(df.describe())
    else:
        st.info("No data")

# ================= MONITOR =================
elif page == "Monitor":

    st.title("Patient Monitoring")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history, columns=["Label","Confidence"])
        st.dataframe(df, use_container_width=True)
        st.line_chart(df["Confidence"])

        avg = df["Confidence"].mean()
        st.metric("Average Stress", f"{round(avg,2)}%")

        (st.error if avg>75 else st.warning if avg>50 else st.success)(
            "Chronic" if avg>75 else "Moderate" if avg>50 else "Stable"
        )
    else:
        st.info("No data")

# ================= REPORTS =================
elif page == "Reports":

    st.title("Reports")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history, columns=["Label","Confidence"])
        st.download_button("Download CSV", df.to_csv(index=False), "report.csv")
        st.dataframe(df)
    else:
        st.info("No data")

# ================= SYSTEM =================
elif page == "System":

    st.title("System Overview")

    st.markdown("""
- Hybrid AI Engine (ML + BERT)  
- Explainable AI  
- Trend Intelligence  
- Clinical Decision Support  
- Real-time Monitoring  
""")

    st.success("System Operational")