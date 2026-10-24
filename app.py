
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="ThalScreen AI",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #0F172A, #0F766E);
    color: white;
}
.hero h1 {
    color: white;
    margin-bottom: 8px;
}
.hero p {
    color: #CCFBF1;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🧬 ThalScreen AI")
    st.caption("Research Demonstration Platform")
    st.divider()
    st.write("📊 Research Dashboard")
    st.write("🧠 ML Prediction — Upcoming")
    st.write("🔍 Explainable AI — Upcoming")
    st.write("🌐 Federated Learning — Upcoming")
    st.info("Research demonstration only.")

st.markdown("""
<div class="hero">
<h1>ThalScreen AI</h1>
<p>Explainable Machine Learning for
β-Thalassemia Trait Screening</p>
</div>
""", unsafe_allow_html=True)

st.write("")
st.subheader("Research Overview")

a, b, c, d = st.columns(4)

a.metric("Total Samples", "467")
b.metric("βTT Samples", "143")
c.metric("Anemia Samples", "324")
d.metric("Formula Inputs", "47")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Dataset Distribution")
    df = pd.DataFrame({
        "Class": ["β-Thalassemia Trait", "Anemia"],
        "Samples": [143, 324]
    })
    fig = px.pie(
        df,
        names="Class",
        values="Samples",
        hole=0.55,
        color_discrete_sequence=["#0D9488", "#94A3B8"]
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Primary Model Performance")
    st.metric("Nested CV Accuracy", "95.07%")
    st.metric("Nested CV Sensitivity", "94.36%")
    st.metric("Nested CV ROC-AUC", "0.9823")
    st.caption(
        "Fully nested 5-fold cross-validation "
        "results; not external clinical validation."
    )

st.divider()
st.subheader("Research Highlights")

st.markdown("""
- **47 formula-based inputs** evaluated
- **Feature subset comparisons:** Top-3, 4, 6, 10, 19
- **Random Forest** with nested cross-validation
- **SHAP** explainability analysis
- **Simulated federated learning** comparison
- **Probability calibration** analysis
""")

st.warning(
    "This platform is for research demonstration. "
    "It is not a clinical diagnostic device."
)
