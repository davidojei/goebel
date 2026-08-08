import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Goebel — Predictive Maintenance Platform", layout="wide")

API_BASE = "http://127.0.0.1:8000"

# ---- Custom styling: white background, blue accents, card-based layout ----
st.markdown("""
<style>
    .main { background-color: #FFFFFF; }
    h1, h2, h3 { color: #0F172A; font-weight: 600; }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card h4 { margin: 0 0 0.3rem 0; color: #64748B; font-size: 0.85rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.03em; }
    .metric-card p { margin: 0; font-size: 1.8rem; font-weight: 700; color: #0F172A; }
    .status-ok { color: #16A34A; font-weight: 600; }
    .status-warn { color: #DC2626; font-weight: 600; }
    .stTabs [data-baseweb="tab"] {
        background-color: #F8FAFC;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        color: #0F172A !important;
}
    .stTabs [data-baseweb="tab"] p {
        color: #0F172A !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Goebel")
st.caption("Multi-Asset Predictive Maintenance Platform")

def metric_card(label, value, col):
    col.markdown(f'<div class="metric-card"><h4>{label}</h4><p>{value}</p></div>', unsafe_allow_html=True)

def plot_feature_impacts(top_features, title):
    features = [f["feature"] for f in top_features][::-1]
    impacts = [f["impact"] for f in top_features][::-1]
    colors = ["#2563EB" if v >= 0 else "#DC2626" for v in impacts]

    fig = go.Figure(go.Bar(
        x=impacts, y=features, orientation="h",
        marker_color=colors
    ))
    fig.update_layout(
        title=title, height=280, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Impact on prediction", gridcolor="#E2E8F0"),
        yaxis=dict(title=""),
        font=dict(color="#0F172A")
    )
    return fig

def check_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

# ---- Sidebar: API status ----
with st.sidebar:
    st.subheader("System Status")
    if check_api_health():
        st.markdown('<span class="status-ok">● API Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">● API Offline</span>', unsafe_allow_html=True)
        st.caption("Run `uvicorn api.main:app --reload` in the goebel folder")

tab1, tab2, tab3, tab4 = st.tabs(["Turbine RUL", "Bearing Fault", "Hydraulic Health", "IMS Bearing RUL"])


def render_turbine_tab():
    st.subheader("Turbine Engine — Remaining Useful Life")

    samples = pd.read_csv("sample_data/turbine_samples.csv")
    feature_cols = [c for c in samples.columns if c != "true_RUL"]

    selected_idx = st.selectbox(
        "Select a sample engine reading",
        options=samples.index,
        format_func=lambda i: f"Sample {i} (true RUL: {samples.loc[i, 'true_RUL']:.0f} cycles)"
    )

    if st.button("Predict RUL", type="primary"):
        features = samples.loc[selected_idx, feature_cols].tolist()
        response = requests.post(f"{API_BASE}/predict/turbine", json={"features": features})

        if response.status_code == 200:
            result = response.json()
            col1, col2, col3 = st.columns(3)
            metric_card("Predicted RUL", f"{result['predicted_rul']:.0f} cycles", col1)
            metric_card("True RUL", f"{samples.loc[selected_idx, 'true_RUL']:.0f} cycles", col2)
            error = abs(result['predicted_rul'] - samples.loc[selected_idx, 'true_RUL'])
            metric_card("Absolute Error", f"{error:.1f} cycles", col3)

            st.plotly_chart(plot_feature_impacts(result["top_features"], "What drove this prediction"), use_container_width=True)
        else:
            st.error(f"API error: {response.text}")

with tab1:
    render_turbine_tab()