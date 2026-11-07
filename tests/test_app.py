import os
import io
import sys
import pytest
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app import app

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_dummy_model_for_tests():
    """Ensure churn_model.pkl exists prior to running API tests."""
    model_path = os.path.join(PROJECT_ROOT, "churn_model.pkl")
    if not os.path.exists(model_path):
        clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        dummy_df = pd.DataFrame({
            "tenure": [1, 12, 24, 36],
            "MonthlyCharges": [20.0, 50.0, 70.0, 100.0],
            "TotalCharges": [20.0, 600.0, 1680.0, 3600.0],
            "Contract_Code": [0, 1, 2, 0],
            "InternetService_Code": [2, 1, 0, 2]
        })
        dummy_y = [1, 0, 0, 1]
        clf.fit(dummy_df, dummy_y)
        joblib.dump(clf, model_path)
    yield

def test_root_dashboard_endpoint():
    """Test GET / returns HTML portal page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_api_stats_endpoint():
    """Test GET /api/stats returns dataset metadata."""
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_records" in data

def test_health_check_endpoint():
    """Test GET /health returns status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"

def test_predict_endpoint_valid_request():
    """Test POST /predict returns single user churn prediction."""
    payload = {
        "tenure": 12,
        "MonthlyCharges": 65.5,
        "TotalCharges": 786.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "risk_level" in data

def test_bulk_predict_csv_upload():
    """Test POST /predict/bulk handles CSV batch prediction uploads."""
    csv_data = "tenure,MonthlyCharges,TotalCharges,Contract,InternetService\n1,29.85,29.85,Month-to-month,DSL\n34,56.95,1889.5,One year,Fiber optic\n"
    files = {"file": ("test_batch.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
    response = client.post("/predict/bulk", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_records_processed"] == 2
    assert "sample_results" in data

def test_predict_endpoint_invalid_tenure_negative():
    """Test POST /predict rejects negative tenure with HTTP 422."""
    payload = {
        "tenure": -5,
        "MonthlyCharges": 50.0,
        "TotalCharges": 250.0,
        "Contract": "Month-to-month",
        "InternetService": "DSL"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_endpoint_invalid_categorical_input():
    """Test POST /predict rejects invalid contract type with HTTP 422."""
    payload = {
        "tenure": 12,
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0,
        "Contract": "Infinite Contract",  # Invalid
        "InternetService": "DSL"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_endpoint_missing_required_fields():
    """Test POST /predict rejects request missing required MonthlyCharges with HTTP 422."""
    payload = {
        "tenure": 12,
        "TotalCharges": 600.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_decision_threshold_logic():
    """Test that prediction equals 1 when probability >= decision_threshold."""
    payload = {
        "tenure": 1,
        "MonthlyCharges": 110.0,
        "TotalCharges": 110.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "decision_threshold" in data
    thresh = data["decision_threshold"]
    prob = data["probability"]
    expected_pred = 1 if prob >= thresh else 0
    assert data["prediction"] == expected_pred

def test_model_loading_and_artifact_integrity():
    """Test that model artifact loads properly and contains pipeline steps."""
    from src.app import get_model, get_threshold
    pipeline = get_model()
    threshold = get_threshold()
    assert pipeline is not None
    assert 0.0 <= threshold <= 1.0
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")

def test_risk_segments_endpoint():
    """Test GET /api/risk-segments returns calculated statistics based on probabilities."""
    response = client.get("/api/risk-segments")
    if response.status_code == 200:
        data = response.json()
        assert "period" in data
        assert "segments" in data
        assert "distribution_percentages" in data
        assert "High Risk" in data["segments"]
        assert "Medium Risk" in data["segments"]
        assert "Low Risk" in data["segments"]

def test_api_metrics_endpoint():
    """Test GET /api/metrics returns live operational and inference metrics."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "total_predictions" in data
    assert "predictions_by_risk" in data
    assert "average_latency_ms" in data
    assert "model_decision_threshold" in data






def test_what_if_counterfactual_endpoint():
    """Test POST /api/what-if correctly computes counterfactual churn probability reduction and economics."""
    from src.app import simulate_what_if_counterfactual, WhatIfRequest
    req = WhatIfRequest(
        tenure=6,
        MonthlyCharges=85.0,
        TotalCharges=510.0,
        Contract="Month-to-month",
        InternetService="Fiber optic",
        proposed_contract="One year",
        discount_monthly=15.0,
        add_tech_support=True
    )
    data = simulate_what_if_counterfactual(req).model_dump()
    assert "baseline_probability" in data
    assert "counterfactual_probability" in data
    assert "risk_delta_percent" in data
    assert "net_economic_benefit" in data
    assert "voucher_code" in data
    assert "outreach_preview" in data
    assert data["monthly_charges_simulated"] == 70.0
    assert data["counterfactual_probability"] <= data["baseline_probability"]
    assert "TELCO-SAVE15-1YR" in data["voucher_code"]
