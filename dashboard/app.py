import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(page_title="Goebel — Predictive Maintenance Platform", layout="wide")

APP_DIR = Path(__file__).parent  # robust — works no matter where you launch streamlit from
SAMPLE_DIR = APP_DIR / "sample_data"
API_BASE = "http://127.0.0.1:8000"

# ---- Claude-inspired theme ----
st.markdown("""
<style>
    .stApp { background-color: #F9F8F6; }
    [data-testid="stSidebar"] { background-color: #F0EEE6; border-right: 1px solid #E3E1D9; }
    h1, h2, h3 { color: #2D2D2A; font-weight: 600; }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E3E1D9;
        border-left: 4px solid #D97757;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card h4 { margin: 0 0 0.3rem 0; color: #8A8778; font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-card p { margin: 0; font-size: 1.8rem; font-weight: 700; color: #2D2D2A; }
    .status-ok { color: #4A7C59; font-weight: 600; }
    .status-warn { color: #C15B4A; font-weight: 600; }
    .stButton button {
        background-color: #D97757; color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.5rem 1.5rem;
    }
    .stButton button:hover { background-color: #C15B4A; }
    .shap-panel {
        background-color: #FFFFFF; border: 1px solid #E3E1D9; border-radius: 10px;
        padding: 1rem 1.2rem; margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, col):
    col.markdown(f'<div class="metric-card"><h4>{label}</h4><p>{value}</p></div>', unsafe_allow_html=True)


def plot_feature_impacts(top_features, title):
    features = [f["feature"] for f in top_features][::-1]
    impacts = [f["impact"] for f in top_features][::-1]
    colors = ["#D97757" if v >= 0 else "#8A8778" for v in impacts]
    fig = go.Figure(go.Bar(x=impacts, y=features, orientation="h", marker_color=colors))
    fig.update_layout(
        title=title, height=280, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Impact on prediction", gridcolor="#E3E1D9"),
        yaxis=dict(title=""), font=dict(color="#2D2D2A")
    )
    return fig


def check_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def shap_panel(top_features, title="Why this prediction"):
    st.markdown(f'<div class="shap-panel"><h4 style="margin-top:0;">{title}</h4></div>', unsafe_allow_html=True)
    st.plotly_chart(plot_feature_impacts(top_features, ""), use_container_width=True)


# ---- Sidebar navigation ----
from streamlit_option_menu import option_menu

with st.sidebar:
    st.markdown("## Goebel")
    st.caption("Predictive Maintenance Platform")

    page = option_menu(
        menu_title=None,
        options=["Turbine RUL", "Bearing Fault", "Hydraulic Health", "IMS Bearing RUL"],
        icons=["wind", "gear", "droplet", "tools"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "#F0EEE6"},
            "icon": {"color": "#8A8778", "font-size": "16px"},
            "nav-link": {
                "font-size": "15px", "color": "#2D2D2A", "text-align": "left",
                "margin": "2px 0", "padding": "10px 12px", "border-radius": "8px",
                "--hover-color": "#E3E1D9",
            },
            "nav-link-selected": {"background-color": "#D97757", "color": "white", "font-weight": "600"},
        }
    )

    st.markdown("---")
    st.markdown("#### System Status")
    if check_api_health():
        st.markdown('<span class="status-ok">● API Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">● API Offline</span>', unsafe_allow_html=True)
        st.caption("Run `uvicorn api.main:app --reload` from the goebel folder")

# ==================== TURBINE ====================
if page == "Turbine RUL":
    st.title("Turbine Engine — Remaining Useful Life")

    samples = pd.read_csv(SAMPLE_DIR / "turbine_samples.csv")
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
            shap_panel(result["top_features"])
        else:
            st.error(f"API error: {response.text}")


# ==================== BEARING ====================
elif page == "Bearing Fault":
    st.title("Bearing — Fault Classification")

    samples = pd.read_csv(SAMPLE_DIR / "bearing_samples.csv")
    feature_cols = [c for c in samples.columns if c != "true_fault_type"]

    selected_idx = st.selectbox(
        "Select a sample bearing reading",
        options=samples.index,
        format_func=lambda i: f"Sample {i} (true fault: {samples.loc[i, 'true_fault_type']})"
    )

    if st.button("Classify Fault", type="primary"):
        features = samples.loc[selected_idx, feature_cols].tolist()
        response = requests.post(f"{API_BASE}/predict/bearing", json={"features": features})

        if response.status_code == 200:
            result = response.json()
            col1, col2, col3 = st.columns(3)
            metric_card("Predicted Fault", result["predicted_fault_type"], col1)
            metric_card("True Fault", samples.loc[selected_idx, "true_fault_type"], col2)
            metric_card("Confidence", f"{result['confidence']*100:.1f}%", col3)
            shap_panel(result["top_features"])
        else:
            st.error(f"API error: {response.text}")


# ==================== HYDRAULIC ====================
elif page == "Hydraulic Health":
    st.title("Hydraulic System — Component Health")

    with open(SAMPLE_DIR / "hydraulic_samples.json") as f:
        samples = json.load(f)

    selected_idx = st.selectbox(
        "Select a sample cycle",
        options=range(len(samples)),
        format_func=lambda i: f"Cycle {samples[i]['row_idx']}"
    )

    if st.button("Assess All Components", type="primary"):
        sample = samples[selected_idx]
        response = requests.post(f"{API_BASE}/predict/hydraulic", json={
            "cooler_features": sample["cooler_features"],
            "valve_features": sample["valve_features"],
            "pump_features": sample["pump_features"],
            "accumulator_features": sample["accumulator_features"],
        })

        if response.status_code == 200:
            result = response.json()
            targets = [
                ("Cooler", "cooler", "true_cooler"),
                ("Valve", "valve", "true_valve"),
                ("Pump", "pump", "true_pump"),
                ("Accumulator", "accumulator", "true_accumulator"),
            ]
            cols = st.columns(4)
            for (label, key, true_key), col in zip(targets, cols):
                r = result[key]
                col.markdown(
                    f'<div class="metric-card"><h4>{label}</h4>'
                    f'<p>{r["predicted_condition"]}</p>'
                    f'<p style="font-size:0.9rem; font-weight:400; color:#8A8778;">true: {sample[true_key]} · {r["confidence"]*100:.0f}% conf.</p></div>',
                    unsafe_allow_html=True
                )

            st.markdown("### Explainability")
            tabs = st.tabs(["Cooler", "Valve", "Pump", "Accumulator"])
            for tab, (label, key, _) in zip(tabs, targets):
                with tab:
                    shap_panel(result[key]["top_features"], f"{label} — why this prediction")
        else:
            st.error(f"API error: {response.text}")


# ==================== IMS ====================
elif page == "🔧 IMS Bearing RUL":
    st.title("IMS Bearing — Degradation & RUL")

    ims_tab1, ims_tab2 = st.tabs(["Stage 1: Degradation Detection", "Stage 2: RUL Regression"])

    with ims_tab1:
        samples1 = pd.read_csv(SAMPLE_DIR / "ims_stage1_samples.csv")
        feature_cols1 = [c for c in samples1.columns if c != "label"]

        selected_idx1 = st.selectbox(
            "Select a sample reading",
            options=samples1.index,
            format_func=lambda i: f"Sample {i} (true: {'Degrading' if samples1.loc[i, 'label']==1 else 'Healthy'})",
            key="ims1"
        )

        if st.button("Check Degradation Status", type="primary"):
            features = samples1.loc[selected_idx1, feature_cols1].tolist()
            response = requests.post(f"{API_BASE}/predict/ims/stage1", json={"features": features})

            if response.status_code == 200:
                result = response.json()
                col1, col2 = st.columns(2)
                metric_card("Status", "Degrading" if result["is_degrading"] else "Healthy", col1)
                metric_card("Probability", f"{result['degradation_probability']*100:.1f}%", col2)
                shap_panel(result["top_features"])
            else:
                st.error(f"API error: {response.text}")

    with ims_tab2:
        samples2 = pd.read_csv(SAMPLE_DIR / "ims_stage2_samples.csv")
        feature_cols2 = [c for c in samples2.columns if c not in ("rul_pct", "rul_hours")]

        selected_idx2 = st.selectbox(
            "Select a sample reading (degrading window only)",
            options=samples2.index,
            format_func=lambda i: f"Sample {i} (true RUL: {samples2.loc[i, 'rul_hours']:.1f}h)",
            key="ims2"
        )

        if st.button("Predict RUL %", type="primary"):
            features = samples2.loc[selected_idx2, feature_cols2].tolist()
            response = requests.post(f"{API_BASE}/predict/ims/stage2", json={"features": features})

            if response.status_code == 200:
                result = response.json()
                col1, col2 = st.columns(2)
                metric_card("Predicted RUL", f"{result['predicted_rul_pct']*100:.1f}%", col1)
                metric_card("True RUL", f"{samples2.loc[selected_idx2, 'rul_pct']*100:.1f}%", col2)
                st.caption(result["note"])
                shap_panel(result["top_features"])
            else:
                st.error(f"API error: {response.text}")