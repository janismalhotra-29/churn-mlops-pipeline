import os
import sys
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train import create_sklearn_pipeline, find_optimal_threshold
from src.drift import detect_drift
from src.retrain import retrain_pipeline

def test_sklearn_pipeline_creation_and_fit():
    """Verify Scikit-Learn Pipeline creates ColumnTransformer and trains without errors."""
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = ["Contract", "InternetService"]
    pipeline = create_sklearn_pipeline(num_cols, cat_cols, n_estimators=10, max_depth=3)
    
    df = pd.DataFrame({
        "tenure": [1, 12, 24, 48],
        "MonthlyCharges": [20.0, 50.0, 70.0, 95.0],
        "TotalCharges": [20.0, 600.0, 1680.0, 4560.0],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "InternetService": ["DSL", "Fiber optic", "No", "Fiber optic"]
    })
    y = np.array([1, 0, 0, 1])
    
    pipeline.fit(df, y)
    preds = pipeline.predict(df)
    probas = pipeline.predict_proba(df)
    
    assert len(preds) == 4
    assert probas.shape == (4, 2)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

def test_threshold_tuning_recall_priority():
    """Verify optimal threshold calculation prioritizes recall."""
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    y_proba = np.array([0.85, 0.20, 0.70, 0.60, 0.15, 0.90, 0.40, 0.65])
    thresh = find_optimal_threshold(y_true, y_proba, target_recall=0.75)
    
    assert 0.0 < thresh < 1.0
    preds = (y_proba >= thresh).astype(int)
    rec = np.sum((preds == 1) & (y_true == 1)) / np.sum(y_true == 1)
    assert rec >= 0.75

def test_drift_detection_no_drift():
    """Verify that identical data distributions do not report false drift."""
    baseline = pd.DataFrame({
        "tenure": np.random.normal(30, 10, 100),
        "MonthlyCharges": np.random.normal(65, 20, 100),
        "TotalCharges": np.random.normal(1800, 500, 100)
    })
    current = baseline.copy()
    report = detect_drift(current, baseline)
    assert report["overall_drift_detected"] is False
    assert report["status"] == "stable"
    assert len(report["drifted_features"]) == 0

def test_drift_detection_triggered_on_shift():
    """Verify that heavily shifted features trigger data drift alert."""
    baseline = pd.DataFrame({
        "tenure": np.random.normal(30, 5, 100),
        "MonthlyCharges": np.random.normal(65, 5, 100),
        "TotalCharges": np.random.normal(1800, 50, 100)
    })
    current = pd.DataFrame({
        "tenure": np.random.normal(5, 2, 100),       # heavy shift
        "MonthlyCharges": np.random.normal(130, 10, 100), # heavy shift
        "TotalCharges": np.random.normal(3500, 100, 100)
    })
    report = detect_drift(current, baseline)
    assert report["overall_drift_detected"] is True
    assert report["status"] == "drift_detected"
    assert report["retraining_recommended"] is True

def test_retraining_execution_and_metrics():
    """Verify automated retraining script runs benchmark and generates report."""
    record = retrain_pipeline()
    assert "timestamp" in record
    assert "candidate_metrics" in record
    assert "promoted" in record
    assert "reason" in record
    assert "roc_auc" in record["candidate_metrics"]
    assert "recall" in record["candidate_metrics"]
