"""
tests/test_final_year_features.py
Unit and integration tests for Final Year Capstone features:
- Multi-Model Benchmark Tournament (Logistic Regression, Random Forest, LightGBM, XGBoost)
- Explainable AI (SHAP TreeExplainer local and global attributions)
- Prescriptive Retention Engine & Customer Lifetime Value (CLV)
- Financial ROI Cost-Benefit Simulation
- MLOps Governance & Observability APIs
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app import app
from src.prescriptive import calculate_clv, recommend_prescriptive_strategy, simulate_financial_roi
from src.explain import explain_prediction

client = TestClient(app)

def test_clv_calculation():
    """Verify Customer Lifetime Value calculation logic across contract types."""
    clv_month = calculate_clv(tenure=6, monthly_charges=80.0, contract="Month-to-month")
    assert "clv_amount" in clv_month
    assert clv_month["clv_amount"] > 0
    assert clv_month["annual_spend"] == 960.0

    clv_two_year = calculate_clv(tenure=12, monthly_charges=80.0, contract="Two year")
    assert clv_two_year["clv_amount"] >= clv_month["clv_amount"]

def test_prescriptive_strategy_recommendations():
    """Verify prescriptive engine returns action plans with financial ROI."""
    customer = {
        "tenure": 2,
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic"
    }
    strategy = recommend_prescriptive_strategy(customer, probability=0.88, risk_level="HIGH")
    assert strategy["risk_level"] == "HIGH"
    assert "prescriptive_intervention" in strategy
    assert "economic_value" in strategy
    assert "Fiber" in strategy["prescriptive_intervention"]["campaign_name"] or "Contract" in strategy["prescriptive_intervention"]["campaign_name"]
    assert strategy["economic_value"]["net_roi"] > 0

def test_financial_roi_simulation():
    """Verify financial ROI simulator computes threshold comparison economics."""
    sim = simulate_financial_roi(customer_cohort_size=1000, churn_rate=0.25)
    assert "standard_baseline_threshold_0_50" in sim
    assert "recall_tuned_mlops_threshold" in sim
    assert "business_value_delivered" in sim
    assert sim["business_value_delivered"]["additional_churners_saved"] > 0
    assert sim["business_value_delivered"]["additional_net_profit_saved"] > 0

def test_predict_with_shap_and_prescriptive_action():
    """Verify POST /predict returns enriched SHAP explanations and prescriptive actions."""
    payload = {
        "tenure": 3,
        "MonthlyCharges": 85.0,
        "TotalCharges": 255.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic"
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] in [0, 1]
    assert "risk_level" in data
    assert "financial_impact_annual" in data
    # Enriched fields
    assert "customer_lifetime_value" in data
    assert "shap_explanations" in data
    if data["shap_explanations"]:
        assert "top_risk_drivers" in data["shap_explanations"]

def test_api_benchmark_endpoint():
    """Verify GET /api/benchmark returns multi-model leaderboard."""
    res = client.get("/api/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert "leaderboard" in data
    assert "champion_model" in data
    algorithms = [m["algorithm"] for m in data["leaderboard"]]
    assert any("Random_Forest" in a for a in algorithms)

def test_api_explain_global_endpoint():
    """Verify GET /api/explain/global returns global feature influence rankings."""
    res = client.get("/api/explain/global")
    assert res.status_code == 200
    data = res.json()
    assert "global_importance" in data
    assert len(data["global_importance"]) > 0

def test_api_roi_simulation_endpoint():
    """Verify POST /api/roi-simulation returns financial comparison."""
    res = client.post("/api/roi-simulation", json={"cohort_size": 1500, "churn_rate": 0.28})
    assert res.status_code == 200
    data = res.json()
    assert "business_value_delivered" in data
    assert "additional_net_profit_saved" in data["business_value_delivered"]

def test_api_governance_endpoint():
    """Verify GET /api/governance consolidates MLOps observability."""
    res = client.get("/api/governance")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OPERATIONAL"
    assert "data_drift_monitoring" in data
    assert "retraining_lifecycle" in data
    assert "model_champion" in data
