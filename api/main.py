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