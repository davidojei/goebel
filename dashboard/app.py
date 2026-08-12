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
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #262624 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1F1E1D !important;
        border-right: 1px solid #3A3936;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, div,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h3 {
        color: #E8E6DF !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #8A8778 !important;
    }
    [data-baseweb="select"] > div {
        background-color: #33322F !important;
        border-color: #3A3936 !important;
        color: #E8E6DF !important;
    }
    [data-baseweb="select"] div { color: #E8E6DF !important; }
    [data-baseweb="popover"] { background-color: #33322F !important; }
    li[role="option"] {
        color: #E8E6DF !important;
        background-color: #33322F !important;
    }
    li[role="option"]:hover { background-color: #D97757 !important; }
    .metric-card {
        background-color: #33322F !important;
        border: 1px solid #3A3936;
        border-left: 4px solid #D97757;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card h4 { color: #8A8778 !important; font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-card p { color: #E8E6DF !important; font-size: 1.8rem; font-weight: 700; }
    .status-ok { color: #7FB08A !important; font-weight: 600; }
    .status-warn { color: #E0836F !important; font-weight: 600; }
    .stButton button {
        background-color: #33322F !important; color: #E8E6DF !important;
        border: 1px solid #3A3936 !important; border-radius: 8px;
        font-weight: 600; padding: 0.5rem 1.5rem;
    }
    .stButton button:hover {
        background-color: #D97757 !important; color: white !important; border-color: #D97757 !important;
    }
    .shap-panel {
        background-color: #33322F !important; border: 1px solid #3A3936; border-radius: 10px;
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
        plot_bgcolor="#33322F", paper_bgcolor="#33322F",
        xaxis=dict(title="Impact on prediction", gridcolor="#3A3936", color="#E8E6DF"),
        yaxis=dict(title="", color="#E8E6DF"),
        font=dict(color="#E8E6DF")
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
            "container": {"padding": "0", "background-color": "#1F1E1D"},
            "icon": {"color": "#8A8778", "font-size": "15px"},
            "nav-link": {
                "font-size": "14px", "color": "#E8E6DF", "text-align": "left",
                "margin": "2px 0px", "padding": "8px 8px",
                "border-radius": "8px", "--hover-color": "#33322F",
                "white-space": "nowrap", "justify-content": "flex-start",
            },
            "nav-link-selected": {"background-color": "#D97757", "color": "white", "font-weight": "600"},
        },
        key="main_nav"
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
    st.title("Turbine Engine: Remaining Useful Life")

    mode = st.radio("Mode", ["Manual Sample", "Live Feed"], horizontal=True)

    if mode == "Manual Sample":
        samples = pd.read_csv(SAMPLE_DIR / "turbine_samples.csv")
        feature_cols = [c for c in samples.columns if c != "true_RUL"]

        selected_idx = st.selectbox(
            "Select a sample engine reading",
            options=samples.index,
            format_func=lambda i: f"Sample {i} (true RUL: {samples.loc[i, 'true_RUL']:.0f} cycles)"
        )

        if st.button("Predict RUL", type="primary"):
            with st.spinner("Calling API..."):
                try:
                    features = samples.loc[selected_idx, feature_cols].tolist()
                    response = requests.post(f"{API_BASE}/predict/turbine", json={"features": features}, timeout=10)
                    if response.status_code == 200:
                        result = response.json()
                        col1, col2, col3 = st.columns(3)
                        metric_card("Predicted RUL", f"{result['predicted_rul']:.0f} cycles", col1)
                        metric_card("True RUL", f"{samples.loc[selected_idx, 'true_RUL']:.0f} cycles", col2)
                        error = abs(result['predicted_rul'] - samples.loc[selected_idx, 'true_RUL'])
                        metric_card("Absolute Error", f"{error:.1f} cycles", col3)
                        shap_panel(result["top_features"])
                    else:
                        st.error(f"API returned {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

    else:  # Live Feed mode
        import time

        trajectory = pd.read_csv(SAMPLE_DIR / "turbine_live_feed.csv")
        feature_cols = [c for c in trajectory.columns if c not in ("cycle", "true_RUL")]

        speed = st.slider("Playback speed (seconds per cycle)", 0.1, 2.0, 0.5)
        start = st.button("▶ Start Live Feed", type="primary")

        status_placeholder = st.empty()
        chart_placeholder = st.empty()
        metrics_placeholder = st.empty()
        shap_placeholder = st.empty()
        history_chart_placeholder = st.empty()

        if start:
            predicted_history = []
            true_history = []
            cycles_seen = []

            for i in range(len(trajectory)):
                row = trajectory.iloc[i]
                cycle = int(row["cycle"])
                true_rul = row["true_RUL"]

                status_placeholder.markdown(f"**Streaming cycle {cycle}** — engine reading arriving...")

                try:
                    features = row[feature_cols].tolist()
                    response = requests.post(f"{API_BASE}/predict/turbine", json={"features": features}, timeout=10)
                    if response.status_code == 200:
                        result = response.json()
                        predicted_history.append(result["predicted_rul"])
                        true_history.append(true_rul)
                        cycles_seen.append(cycle)

                        with metrics_placeholder.container():
                            col1, col2, col3 = st.columns(3)
                            metric_card("Cycle", str(cycle), col1)
                            metric_card("Predicted RUL", f"{result['predicted_rul']:.0f}", col2)
                            metric_card("True RUL", f"{true_rul:.0f}", col3)

                        with shap_placeholder.container():
                            shap_panel(result["top_features"])

                        # Running chart of predicted vs true RUL over the stream
                        hist_fig = go.Figure()
                        hist_fig.add_trace(go.Scatter(x=cycles_seen, y=true_history, name="True RUL", line=dict(color="#8A8778", width=2, dash="dot")))
                        hist_fig.add_trace(go.Scatter(x=cycles_seen, y=predicted_history, name="Predicted RUL", line=dict(color="#D97757", width=3)))
                        hist_fig.update_layout(
                            title=dict(text="RUL: Predicted vs Actual", font=dict(color="#E8E6DF", size=15)),
                            height=300, margin=dict(l=10, r=10, t=40, b=10),
                            plot_bgcolor="#33322F", paper_bgcolor="#33322F",
                            xaxis=dict(title="Cycle", gridcolor="#3A3936", color="#E8E6DF"),
                            yaxis=dict(title="RUL (cycles)", gridcolor="#3A3936", color="#E8E6DF", rangemode="tozero"),
                            font=dict(color="#E8E6DF"),
                            legend=dict(font=dict(color="#E8E6DF"), orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                       )
                        history_chart_placeholder.plotly_chart(hist_fig, use_container_width=True, key=f"hist_{cycle}")

                    time.sleep(speed)
                except Exception as e:
                    status_placeholder.error(f"Stream error at cycle {cycle}: {e}")
                    break

            status_placeholder.markdown(f"**Stream complete** — {len(trajectory)} cycles processed.")

# ==================== BEARING ====================
elif page == "Bearing Fault":
    st.title("Bearing: Fault Classification")

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
    st.title("Hydraulic System: Component Health")

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