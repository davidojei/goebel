"""
FastAPI backend serving all three Goebel models via separate endpoints.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List

from src.shared.shap_utils import get_explainer, top_features, explain_prediction_text

app = FastAPI(title="Goebel — Multi-Asset Predictive Maintenance Platform")


@app.get("/health")
def health():
    return {"status": "ok"}


# ---- Turbine RUL endpoint ----

class TurbineFeatures(BaseModel):
    features: List[float]  # ordered list matching the model's feature_cols

# Load once at startup, not per-request - loading a model on every call would be slow and wasteful
try:
    turbine_model = joblib.load("models/turbine/final_model.pkl")
    turbine_feature_names = joblib.load("models/turbine/feature_names.pkl")
    turbine_explainer = get_explainer(turbine_model)
except FileNotFoundError:
    turbine_model = None
    turbine_explainer = None


@app.post("/predict/turbine")
def predict_turbine(payload: TurbineFeatures):
    if turbine_model is None:
        raise HTTPException(status_code=503, detail="Turbine model not loaded — run training and save the model first")

    if len(payload.features) != len(turbine_feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(turbine_feature_names)} features, got {len(payload.features)}"
        )

    X_row = pd.DataFrame([payload.features], columns=turbine_feature_names)
    prediction = float(turbine_model.predict(X_row)[0])

    top_feats = top_features(turbine_explainer, X_row, turbine_feature_names, n=5)
    explanation = explain_prediction_text(top_feats)

    return {
        "predicted_rul": round(prediction, 2),
        "explanation": explanation,
        "top_features": [{"feature": f, "impact": round(v, 4)} for f, v in top_feats]
    }


# ---- Bearing fault classification endpoint ----

class BearingFeatures(BaseModel):
    features: List[float]

try:
    bearing_model = joblib.load("models/bearing/final_model.pkl")
    bearing_feature_names = joblib.load("models/bearing/feature_names.pkl")
    bearing_explainer = get_explainer(bearing_model)
except FileNotFoundError:
    bearing_model = None
    bearing_explainer = None


@app.post("/predict/bearing")
def predict_bearing(payload: BearingFeatures):
    if bearing_model is None:
        raise HTTPException(status_code=503, detail="Bearing model not loaded")

    if len(payload.features) != len(bearing_feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(bearing_feature_names)} features, got {len(payload.features)}"
        )

    X_row = pd.DataFrame([payload.features], columns=bearing_feature_names)
    prediction = bearing_model.predict(X_row)[0]
    probabilities = bearing_model.predict_proba(X_row)[0]
    predicted_class_index = list(bearing_model.classes_).index(prediction)

    top_feats = top_features(bearing_explainer, X_row, bearing_feature_names, n=5, class_index=predicted_class_index)
    explanation = explain_prediction_text(top_feats)

    return {
        "predicted_fault_type": str(prediction),
        "confidence": round(float(probabilities[predicted_class_index]), 4),
        "explanation": explanation,
        "top_features": [{"feature": f, "impact": round(float(v), 4)} for f, v in top_feats]
    }


# ---- Hydraulic 4-target endpoint ----

class HydraulicFeatures(BaseModel):
    cooler_features: List[float]
    valve_features: List[float]
    pump_features: List[float]
    accumulator_features: List[float]

hydraulic_models = {}
hydraulic_feature_names = {}
hydraulic_explainers = {}

for target in ["cooler", "valve", "pump", "accumulator"]:
    try:
        hydraulic_models[target] = joblib.load(f"models/hydraulic/{target}_model.pkl")
        hydraulic_feature_names[target] = joblib.load(f"models/hydraulic/{target}_features.pkl")
        hydraulic_explainers[target] = get_explainer(hydraulic_models[target])
    except FileNotFoundError:
        hydraulic_models[target] = None


@app.post("/predict/hydraulic")
def predict_hydraulic(payload: HydraulicFeatures):
    results = {}
    feature_payloads = {
        "cooler": payload.cooler_features,
        "valve": payload.valve_features,
        "pump": payload.pump_features,
        "accumulator": payload.accumulator_features,
    }

    for target, features in feature_payloads.items():
        if hydraulic_models[target] is None:
            results[target] = {"error": f"{target} model not loaded"}
            continue

        expected_names = hydraulic_feature_names[target]
        if len(features) != len(expected_names):
            results[target] = {"error": f"Expected {len(expected_names)} features, got {len(features)}"}
            continue

        X_row = pd.DataFrame([features], columns=expected_names)
        model = hydraulic_models[target]
        prediction = model.predict(X_row)[0]
        probabilities = model.predict_proba(X_row)[0]
        predicted_class_index = list(model.classes_).index(prediction)

        top_feats = top_features(hydraulic_explainers[target], X_row, expected_names, n=3, class_index=predicted_class_index)

        results[target] = {
            "predicted_condition": str(prediction),
            "confidence": round(float(probabilities[predicted_class_index]), 4),
            "top_features": [{"feature": f, "impact": round(float(v), 4)} for f, v in top_feats]
        }

    return results

