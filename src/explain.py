"""
src/explain.py
Explainable AI (XAI) Engine using SHAP (SHapley Additive exPlanations).
Provides global feature importance and local instance-level explainability for customer churn predictions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "churn_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw_churn_data.csv")
GLOBAL_IMPORTANCE_PATH = os.path.join(BASE_DIR, "data", "global_shap_importance.json")

_explainer_cache = None
_feature_names_cache = None

def get_transformed_feature_names(preprocessor, num_features, cat_features):
    """Extract human-readable feature names output by ColumnTransformer."""
    names = list(num_features)
    try:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_names = cat_encoder.get_feature_names_out(cat_features)
        names.extend(list(cat_names))
    except Exception:
        # Fallback if categories are not present or named differently
        pass
    return names

def format_feature_label(feat_name: str) -> str:
    """Translate technical feature names into clear business descriptions."""
    clean = feat_name.replace("cat__", "").replace("num__", "")
    mapping = {
        "tenure": "Customer Tenure (Months)",
        "MonthlyCharges": "Monthly Bill Amount",
        "TotalCharges": "Lifetime Spend Total",
        "Contract_Month-to-month": "Month-to-Month Contract",
        "Contract_One year": "1-Year Fixed Contract",
        "Contract_Two year": "2-Year Long-Term Contract",
        "InternetService_Fiber optic": "Fiber Optic Internet Plan",
        "InternetService_DSL": "DSL Broadband Service",
        "InternetService_No": "No Internet Service"
    }
    return mapping.get(clean, clean.replace("_", " "))

def get_explainer():
    """Load or initialize SHAP TreeExplainer on the active champion pipeline."""
    global _explainer_cache, _feature_names_cache
    if _explainer_cache is not None:
        return _explainer_cache, _feature_names_cache

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    loaded = joblib.load(MODEL_PATH)
    pipeline = loaded["pipeline"] if isinstance(loaded, dict) and "pipeline" in loaded else loaded

    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")

    # Sample baseline data for background distribution
    df = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else pd.DataFrame({
        "tenure": [12, 24, 48], "MonthlyCharges": [70.0, 50.0, 90.0], "TotalCharges": [840.0, 1200.0, 4320.0],
        "Contract": ["Month-to-month", "One year", "Two year"], "InternetService": ["Fiber optic", "DSL", "No"]
    })

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in ["Contract", "InternetService"] if c in df.columns]
    feature_cols = num_cols + cat_cols

    X_sample = df[feature_cols].dropna().head(100)
    X_transformed = preprocessor.transform(X_sample)

    feature_names = get_transformed_feature_names(preprocessor, num_cols, cat_cols)
    if len(feature_names) != X_transformed.shape[1]:
        feature_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]

    # Build TreeExplainer (pure tree path traversal runs in <10ms)
    try:
        explainer = shap.TreeExplainer(classifier)
    except Exception:
        # Fallback to general Explainer
        explainer = shap.Explainer(classifier.predict_proba, X_transformed[:10])

    _explainer_cache = explainer
    _feature_names_cache = feature_names
    return _explainer_cache, _feature_names_cache

def explain_prediction(customer_dict: dict) -> dict:
    """
    Compute local SHAP explanations for a single customer inference.
    Returns top factors pushing towards Churn vs Retention.
    """
    loaded = joblib.load(MODEL_PATH)
    pipeline = loaded["pipeline"] if isinstance(loaded, dict) and "pipeline" in loaded else loaded
    preprocessor = pipeline.named_steps["preprocessor"]

    explainer, feature_names = get_explainer()

    # Create 1-row DataFrame
    input_df = pd.DataFrame([customer_dict])
    X_transformed = preprocessor.transform(input_df)

    shap_values = explainer(X_transformed)

    # For binary classification, extract churn class (index 1)
    if len(shap_values.values.shape) == 3:
        # (samples, features, classes)
        sv = shap_values.values[0, :, 1]
        base_val = float(shap_values.base_values[0, 1]) if hasattr(shap_values.base_values, "__getitem__") else float(shap_values.base_values)
    else:
        sv = shap_values.values[0]
        base_val = float(shap_values.base_values[0]) if hasattr(shap_values.base_values, "__getitem__") else float(shap_values.base_values)

    drivers = []
    for name, val in zip(feature_names, sv):
        label = format_feature_label(name)
        impact_pct = f"{'+' if val >= 0 else ''}{round(float(val) * 100, 1)}%"
        drivers.append({
            "feature": label,
            "raw_feature": name,
            "shap_value": round(float(val), 4),
            "impact_pct": impact_pct,
            "direction": "RISK_INCREASE" if val > 0 else "RETENTION_FORCE"
        })

    # Sort drivers
    risk_drivers = sorted([d for d in drivers if d["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)
    retention_drivers = sorted([d for d in drivers if d["shap_value"] <= 0], key=lambda x: x["shap_value"])

    return {
        "base_expected_value": round(base_val, 4),
        "top_risk_drivers": risk_drivers[:3],
        "top_retention_drivers": retention_drivers[:3]
    }

def compute_global_importance() -> dict:
    """Compute and persist global SHAP feature importance across the dataset."""
    if not os.path.exists(MODEL_PATH):
        return {}

    loaded = joblib.load(MODEL_PATH)
    pipeline = loaded["pipeline"] if isinstance(loaded, dict) and "pipeline" in loaded else loaded
    preprocessor = pipeline.named_steps["preprocessor"]

    explainer, feature_names = get_explainer()

    df = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else pd.DataFrame()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in ["Contract", "InternetService"] if c in df.columns]
    feature_cols = num_cols + cat_cols

    X_sample = df[feature_cols].dropna().head(200)
    X_transformed = preprocessor.transform(X_sample)

    shap_vals = explainer(X_transformed)
    if len(shap_vals.values.shape) == 3:
        vals = np.abs(shap_vals.values[:, :, 1]).mean(axis=0)
    else:
        vals = np.abs(shap_vals.values).mean(axis=0)

    importance_list = []
    for name, val in zip(feature_names, vals):
        importance_list.append({
            "feature": format_feature_label(name),
            "mean_abs_shap": round(float(val), 4),
            "relative_influence": round(float(val / (np.sum(vals) + 1e-9)) * 100, 2)
        })

    importance_list.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

    result = {
        "method": "SHAP (TreeExplainer)",
        "sample_size": len(X_sample),
        "global_importance": importance_list
    }

    os.makedirs(os.path.dirname(GLOBAL_IMPORTANCE_PATH), exist_ok=True)
    with open(GLOBAL_IMPORTANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result

if __name__ == "__main__":
    print("[XAI] Computing global SHAP importance...")
    res = compute_global_importance()
    print(f"Top 3 Global Features: {[f['feature'] for f in res.get('global_importance', [])[:3]]}")
