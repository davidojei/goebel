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
    explanation = explain_prediction_text(top_feats, subsystem="turbine")

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
    explanation = explain_prediction_text(top_feats, subsystem="bearing")

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
        explanation = explain_prediction_text(top_feats, subsystem="hydraulic")

        results[target] = {
            "predicted_condition": str(prediction),
            "confidence": round(float(probabilities[predicted_class_index]), 4),
            "explanation": explanation,
            "top_features": [{"feature": f, "impact": round(float(v), 4)} for f, v in top_feats]
        }

    return results


# ---- IMS Bearing RUL endpoints (two stages) ----

class IMSStage1Features(BaseModel):
    features: List[float]

class IMSStage2Features(BaseModel):
    features: List[float]

try:
    ims_stage1_model = joblib.load("models/ims/stage1_model.pkl")
    ims_stage1_features = joblib.load("models/ims/stage1_features.pkl")
    ims_stage1_threshold = joblib.load("models/ims/stage1_threshold.pkl")
    ims_stage1_explainer = get_explainer(ims_stage1_model)
except FileNotFoundError:
    ims_stage1_model = None

try:
    ims_stage2_model = joblib.load("models/ims/stage2_model.pkl")
    ims_stage2_features = joblib.load("models/ims/stage2_features.pkl")
    ims_stage2_explainer = get_explainer(ims_stage2_model)
except FileNotFoundError:
    ims_stage2_model = None


@app.post("/predict/ims/stage1")
def predict_ims_stage1(payload: IMSStage1Features):
    if ims_stage1_model is None:
        raise HTTPException(status_code=503, detail="IMS Stage 1 model not loaded")

    if len(payload.features) != len(ims_stage1_features):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(ims_stage1_features)} features, got {len(payload.features)}"
        )

    X_row = pd.DataFrame([payload.features], columns=ims_stage1_features)
    probability = float(ims_stage1_model.predict_proba(X_row)[0][1])
    is_degrading = probability >= ims_stage1_threshold

    top_feats = top_features(ims_stage1_explainer, X_row, ims_stage1_features, n=5, class_index=1)
    explanation = explain_prediction_text(top_feats, subsystem="ims")

    return {
        "is_degrading": bool(is_degrading),
        "degradation_probability": round(probability, 4),
        "threshold_used": round(ims_stage1_threshold, 4),
        "explanation": explanation,
        "top_features": [{"feature": f, "impact": round(float(v), 4)} for f, v in top_feats]
    }


@app.post("/predict/ims/stage2")
def predict_ims_stage2(payload: IMSStage2Features):
    if ims_stage2_model is None:
        raise HTTPException(status_code=503, detail="IMS Stage 2 model not loaded")

    if len(payload.features) != len(ims_stage2_features):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(ims_stage2_features)} features, got {len(payload.features)}"
        )

    X_row = pd.DataFrame([payload.features], columns=ims_stage2_features)
    predicted_pct = float(ims_stage2_model.predict(X_row)[0])

    top_feats = top_features(ims_stage2_explainer, X_row, ims_stage2_features, n=5)
    explanation = explain_prediction_text(top_feats, subsystem="ims")

    return {
        "predicted_rul_pct": round(predicted_pct, 4),
        "note": "This is RUL as a % of the bearing's degrading-window length, not raw hours — see README for why.",
        "explanation": explanation,
        "top_features": [{"feature": f, "impact": round(float(v), 4)} for f, v in top_feats]
    }