"""
Shared SHAP explainability helpers, used across all three sub-systems.
Works with any tree-based model (XGBoost, RandomForest) trained in this project.
"""
import shap
import numpy as np
import pandas as pd


def get_explainer(model):
    return shap.TreeExplainer(model)


def top_features(explainer, X_row, feature_names, n=5, class_index=None):
    shap_values = explainer.shap_values(X_row)

    if isinstance(shap_values, list):
        values = shap_values[class_index][0] if class_index is not None else shap_values[0]
    elif shap_values.ndim == 3:
        values = shap_values[0, :, class_index] if class_index is not None else shap_values[0, :, 0]
    else:
        values = shap_values[0]

    feature_impact = pd.DataFrame({"feature": feature_names, "shap_value": values})
    feature_impact["abs_value"] = feature_impact["shap_value"].abs()
    top = feature_impact.sort_values("abs_value", ascending=False).head(n)

    return list(zip(top["feature"], top["shap_value"]))


# ==================== Plain-English feature descriptions ====================

TURBINE_SENSOR_MEANINGS = {
    "sensor_2": "LPC outlet temperature",
    "sensor_3": "HPC outlet temperature",
    "sensor_4": "LPT outlet temperature",
    "sensor_7": "HPC outlet pressure",
    "sensor_8": "physical fan speed",
    "sensor_9": "physical core speed",
    "sensor_11": "HPC outlet static pressure",
    "sensor_12": "fuel flow ratio",
    "sensor_13": "corrected fan speed",
    "sensor_14": "corrected core speed",
    "sensor_15": "bypass ratio",
    "sensor_17": "bleed enthalpy",
    "sensor_20": "HPT coolant bleed",
    "sensor_21": "LPT coolant bleed",
}

def humanize_turbine_feature(feature_name):
    for sensor, meaning in TURBINE_SENSOR_MEANINGS.items():
        if feature_name == sensor:
            return f"the current {meaning}"
        if feature_name == f"{sensor}_roll_mean":
            return f"the recent average {meaning}"
        if feature_name == f"{sensor}_baseline_drift":
            return f"how far the {meaning} has drifted from this engine's healthy starting point"
    if feature_name == "op_setting_1":
        return "the operating altitude setting"
    if feature_name == "op_setting_2":
        return "the operating Mach number setting"
    return feature_name.replace("_", " ")


BEARING_TERM_MEANINGS = {
    "rms": "the overall vibration intensity",
    "kurtosis": "how sharp and spiky the vibration impacts are",
    "BPFO": "vibration matching the outer-race fault frequency",
    "BPFI": "vibration matching the inner-race fault frequency",
    "BSF": "vibration matching the ball-fault frequency",
}

def humanize_bearing_feature(feature_name):
    if feature_name in ("rms", "kurtosis"):
        return BEARING_TERM_MEANINGS[feature_name]
    for term, meaning in BEARING_TERM_MEANINGS.items():
        if feature_name.startswith(term):
            harmonic = "1" if "_x1_" in feature_name else "2" if "_x2_" in feature_name else "3" if "_x3_" in feature_name else ""
            suffix = f" (harmonic {harmonic})" if harmonic else ""
            return meaning + suffix
    return feature_name.replace("_", " ")


FEATURE_DESCRIPTIONS = {
    # Hydraulic
    "CE_mean": "the average cooling efficiency",
    "CP_mean": "the average cooling power",
    "SE_mean": "the average pump efficiency factor",
    "SE_std": "how much the pump efficiency factor varies",
    "FS1_mean": "the average flow rate",
    "PS2_std": "how much pressure fluctuates during each cycle",
    "PS1_max_window_diff_std": "the strongest pressure pulsation detected in any part of the cycle",
    "TS1_mean": "the average temperature reading",
    "EPS1_mean": "the average motor power draw",
    # IMS (reuses "std"/"kurtosis" meanings from bearing, listed here for standalone use)
    "std": "how much the vibration signal varies",
    "mean_abs": "the average vibration magnitude",
    "peak_to_peak": "the difference between the strongest and weakest vibration moments",
    "bpfo_energy": "vibration energy matching the outer-race fault frequency",
    "bpfi_energy": "vibration energy matching the inner-race fault frequency",
    "bsf_energy": "vibration energy matching the ball-fault frequency",
}


def humanize_feature(feature_name, subsystem="hydraulic"):
    if subsystem == "turbine":
        return humanize_turbine_feature(feature_name)
    elif subsystem == "bearing":
        return humanize_bearing_feature(feature_name)
    else:
        return FEATURE_DESCRIPTIONS.get(feature_name, feature_name.replace("_", " "))


def explain_prediction_text(top_features_list, subsystem="hydraulic"):
    """
    Converts (feature, shap_value) pairs into full, readable sentences,
    for non-technical dashboard viewers.
    """
    lines = []
    for feature, value in top_features_list:
        plain_name = humanize_feature(feature, subsystem)
        direction = "pointed toward this result" if value > 0 else "pointed away from this result"
        strength = "strongly" if abs(value) > 0.1 else "moderately" if abs(value) > 0.03 else "slightly"
        lines.append(f"- {plain_name.capitalize()} {strength} {direction}")
    return "\n".join(lines)