import os
import argparse
import joblib
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.train import load_data, create_sklearn_pipeline, find_optimal_threshold

PROD_MODEL_PATH = os.path.join(BASE_DIR, "churn_model.pkl")
RETRAIN_LOG_PATH = os.path.join(BASE_DIR, "data", "retrain_history.json")

def evaluate_model(pipeline, X_test, y_test, threshold=0.5):
    """Compute performance metrics for a model pipeline on test data."""
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else pipeline.predict(X_test)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
    
    return {
        "accuracy": round(float(acc), 4),
        "f1_score": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "roc_auc": round(float(roc_auc), 4)
    }

def retrain_pipeline(new_data_path: str = None, force_promotion: bool = False) -> dict:
    """
    Automated Retraining Workflow:
    1. Ingests new/updated labeled customer churn dataset.
    2. Trains a new candidate pipeline with MLflow tracking.
    3. Benchmarks candidate pipeline against active production model.
    4. Automatically promotes candidate to production ONLY if it outperforms existing model on ROC-AUC / Recall.
    """
    if new_data_path is None:
        new_data_path = os.path.join(BASE_DIR, "data", "raw_churn_data.csv")
        
    print(f"[Retraining] Starting automated retraining workflow on: {new_data_path}")
    df = load_data(new_data_path)
    
    if "Churn" in df.columns:
        mapping = {"Yes": 1, "No": 0, "YES": 1, "NO": 0, "1": 1, "0": 0}
        y = df["Churn"].astype(str).str.strip().map(mapping).fillna(0).astype(int)
    else:
        raise ValueError("Dataset missing 'Churn' target column for supervised retraining.")
        
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0

    num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_features = [c for c in ["Contract", "InternetService"] if c in df.columns]
    feature_cols = num_features + cat_features
    X = df[feature_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # 1. Train candidate pipeline
    candidate_pipeline = create_sklearn_pipeline(num_features, cat_features, n_estimators=120, max_depth=6)
    candidate_pipeline.fit(X_train, y_train)
    
    cand_proba = candidate_pipeline.predict_proba(X_test)[:, 1]
    cand_thresh = find_optimal_threshold(y_test, cand_proba, target_recall=0.75)
    cand_metrics = evaluate_model(candidate_pipeline, X_test, y_test, threshold=cand_thresh)
    
    # 2. Evaluate current production model on same validation split
    prod_metrics = None
    promoted = False
    prod_threshold = 0.5
    
    if os.path.exists(PROD_MODEL_PATH):
        try:
            prod_artifact = joblib.load(PROD_MODEL_PATH)
            prod_pipe = prod_artifact["pipeline"] if isinstance(prod_artifact, dict) and "pipeline" in prod_artifact else prod_artifact
            prod_threshold = prod_artifact.get("decision_threshold", 0.5) if isinstance(prod_artifact, dict) else 0.5
            
            # Align features for production pipeline
            prod_features = getattr(prod_pipe, "feature_names_in_", feature_cols)
            X_test_prod = X_test.copy()
            for col in prod_features:
                if col not in X_test_prod.columns:
                    X_test_prod[col] = 0.0
            X_test_prod = X_test_prod[list(prod_features)]
            
            prod_metrics = evaluate_model(prod_pipe, X_test_prod, y_test, threshold=prod_threshold)
            print(f"[Benchmark] Active Prod Model Metrics: {prod_metrics}")
        except Exception as exc:
            print(f"[Benchmark Warning] Could not benchmark active model: {exc}")

    print(f"[Benchmark] Candidate Model Metrics:   {cand_metrics}")

    # 3. Decision rule for promotion: Candidate must have >= ROC-AUC and >= Recall, or force_promotion
    if prod_metrics is None or force_promotion:
        promoted = True
        reason = "Initial production deployment or forced promotion."
    elif (cand_metrics["roc_auc"] >= prod_metrics["roc_auc"]) and (cand_metrics["recall"] >= prod_metrics["recall"]):
        promoted = True
        reason = f"Candidate outperformed production (ROC-AUC {cand_metrics['roc_auc']} >= {prod_metrics['roc_auc']}, Recall {cand_metrics['recall']} >= {prod_metrics['recall']})."
    else:
        promoted = False
        reason = f"Candidate did not outperform production (Cand ROC-AUC {cand_metrics['roc_auc']} vs Prod {prod_metrics['roc_auc']}). Preserving active production model."

    print(f"[Decision] Promoted: {promoted} | Reason: {reason}")

    # 4. MLflow Experiment Logging & Model Registry
    mlflow.set_experiment("Telco_Customer_Churn_Retraining")
    with mlflow.start_run() as run:
        mlflow.log_params({
            "retrain_dataset": os.path.basename(new_data_path),
            "decision_threshold": cand_thresh,
            "promoted_to_production": promoted,
            "promotion_reason": reason
        })
        
        for k, v in cand_metrics.items():
            mlflow.log_metric(f"candidate_{k}", v)
        if prod_metrics:
            for k, v in prod_metrics.items():
                mlflow.log_metric(f"prod_{k}", v)

        try:
            reg_name = "Telco_Churn_RandomForest" if promoted else None
            mlflow.sklearn.log_model(
                sk_model=candidate_pipeline,
                artifact_path="retrained_model",
                registered_model_name=reg_name,
                serialization_format="cloudpickle"
            )
        except Exception as exc:
            print(f"[MLflow Warning] Model log failed: {exc}")

    # 5. If promoted, overwrite production model artifact
    if promoted:
        new_artifact = {
            "pipeline": candidate_pipeline,
            "decision_threshold": cand_thresh,
            "features": feature_cols,
            "num_features": num_features,
            "cat_features": cat_features,
            "metrics": cand_metrics
        }
        joblib.dump(new_artifact, PROD_MODEL_PATH)
        print(f"[Production] Successfully updated production model artifact at: {PROD_MODEL_PATH}")

    # 6. Save Retraining Record
    retrain_record = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "dataset": new_data_path,
        "promoted": promoted,
        "reason": reason,
        "candidate_metrics": cand_metrics,
        "production_metrics": prod_metrics
    }
    
    os.makedirs(os.path.dirname(RETRAIN_LOG_PATH), exist_ok=True)
    history = []
    if os.path.exists(RETRAIN_LOG_PATH):
        try:
            with open(RETRAIN_LOG_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(retrain_record)
    with open(RETRAIN_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return retrain_record

def main():
    parser = argparse.ArgumentParser(description="Automated Model Retraining & Evaluation Engine")
    parser.add_argument("--data", type=str, default=None, help="Path to new training dataset")
    parser.add_argument("--force", action="store_true", help="Force promotion regardless of metric comparison")
    args = parser.parse_args()

    result = retrain_pipeline(new_data_path=args.data, force_promotion=args.force)
    print("\n--- RETRAINING WORKFLOW SUMMARY ---")
    print(f"Status:   {'PROMOTED' if result['promoted'] else 'KEPT CURRENT MODEL'}")
    print(f"Reason:   {result['reason']}")
    print(f"Metrics:  {result['candidate_metrics']}")

if __name__ == "__main__":
    main()
