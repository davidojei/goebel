"""
Shared SHAP explainability helpers, used across all three sub-systems.
Works with any tree-based model (XGBoost, RandomForest) trained in this project.
"""
import shap
import numpy as np
import pandas as pd


def get_explainer(model):
    """
    Returns a shap.TreeExplainer for any tree-based model used in this project
    (XGBRegressor, XGBClassifier, RandomForestClassifier, RandomForestRegressor).
    TreeExplainer is exact and fast for tree ensembles - no need for the slower,
    approximate KernelExplainer used for non-tree models.
    """
    return shap.TreeExplainer(model)


def top_features(explainer, X_row, feature_names, n=5, class_index=None):
    """
    Given a single row of input data, return the top N features driving
    that specific prediction, as (feature_name, shap_value) pairs.

    X_row: a single-row DataFrame or 2D array (1, n_features)
    class_index: for multiclass classifiers, which class's explanation to use
                 (e.g. the predicted class). None for regression models.
    """
    shap_values = explainer.shap_values(X_row)

    # Handle the different shapes SHAP returns depending on model type/version
    if isinstance(shap_values, list):
        # Older SHAP API: list of arrays, one per class
        values = shap_values[class_index][0] if class_index is not None else shap_values[0]
    elif shap_values.ndim == 3:
        # Newer SHAP API: (samples, features, classes)
        values = shap_values[0, :, class_index] if class_index is not None else shap_values[0, :, 0]
    else:
        # Regression: (samples, features)
        values = shap_values[0]

    feature_impact = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values
    })
    feature_impact["abs_value"] = feature_impact["shap_value"].abs()
    top = feature_impact.sort_values("abs_value", ascending=False).head(n)

    return list(zip(top["feature"], top["shap_value"]))


def explain_prediction_text(top_features_list):
    """
    Converts a list of (feature, shap_value) pairs into a plain-English string,
    for the dashboard's explainability panel.
    """
    lines = []
    for feature, value in top_features_list:
        direction = "increased" if value > 0 else "decreased"
        lines.append(f"- {feature} {direction} the prediction (impact: {value:+.4f})")
    return "\n".join(lines)