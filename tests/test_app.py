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
