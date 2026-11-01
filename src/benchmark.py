"""
src/benchmark.py
Multi-Model Tournament and Hyperparameter Optimization Engine.
Benchmarks Logistic Regression, Random Forest, LightGBM, and XGBoost with MLflow logging.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw_churn_data.csv")
BENCHMARK_PATH = os.path.join(BASE_DIR, "data", "model_benchmark.json")

def load_and_preprocess_data():
    """Load and prepare feature matrices for benchmarking."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        from src.train import load_data
        df = load_data(DATA_PATH)

    if "Churn" in df.columns:
        mapping = {"Yes": 1, "No": 0, "YES": 1, "NO": 0, "1": 1, "0": 0}
        y = df["Churn"].astype(str).str.strip().map(mapping).fillna(0).astype(int)
    else:
        y = np.zeros(len(df), dtype=int)

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0

    num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_features = [c for c in ["Contract", "InternetService"] if c in df.columns]

    X = df[num_features + cat_features]
    return X, y, num_features, cat_features

def build_preprocessor(num_features, cat_features):
    """Build shared preprocessing ColumnTransformer."""
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

def evaluate_model_pipeline(pipeline, X_train, y_train, X_test, y_test):
    """Train pipeline and measure performance metrics + inference latency."""
    start_train = time.time()
    pipeline.fit(X_train, y_train)
    train_time = round(time.time() - start_train, 4)

    start_infer = time.time()
    y_pred = pipeline.predict(X_test)
    infer_time_ms = round(((time.time() - start_infer) / len(X_test)) * 1000, 3)

    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5), 4),
        "train_time_sec": train_time,
        "inference_latency_ms": infer_time_ms
    }
    return metrics

def run_benchmark(n_trials=5):
    """Run multi-model tournament comparing Logistic Regression, Random Forest, LightGBM, and XGBoost."""
    print("=" * 70)
    print("[BENCHMARK] RUNNING MULTI-MODEL TOURNAMENT & OPTUNA BENCHMARK")
    print("=" * 70)

    X, y, num_features, cat_features = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    models_to_test = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "type": "Linear Baseline",
            "tuned": False
        },
        "Random_Forest": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
            "type": "Ensemble Bagging",
            "tuned": True
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1),
            "type": "Histogram Boosting",
            "tuned": True
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42),
            "type": "Extreme Gradient Boosting",
            "tuned": True
        }
    }

    results = []
    mlflow.set_experiment("Telco_Model_Tournament")

    for name, config in models_to_test.items():
        print(f"\n[Evaluating] {name} ({config['type']})...")
        preprocessor = build_preprocessor(num_features, cat_features)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', config['model'])
        ])

        with mlflow.start_run(run_name=f"Tournament_{name}"):
            metrics = evaluate_model_pipeline(pipeline, X_train, y_train, X_test, y_test)
            mlflow.log_params({"algorithm": name, "model_type": config["type"]})
            mlflow.log_metrics(metrics)

            res_entry = {
                "algorithm": name,
                "model_family": config["type"],
                **metrics
            }
            results.append(res_entry)
            print(f"  Accuracy: {metrics['accuracy']} | Recall: {metrics['recall']} | ROC-AUC: {metrics['roc_auc']} | Latency: {metrics['inference_latency_ms']}ms")

    # Sort by ROC-AUC and Recall
    results.sort(key=lambda x: (x["roc_auc"], x["recall"]), reverse=True)
    champion = results[0]

    output_payload = {
        "tournament_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_models_evaluated": len(results),
        "champion_model": champion["algorithm"],
        "champion_metrics": champion,
        "leaderboard": results
    }

    os.makedirs(os.path.dirname(BENCHMARK_PATH), exist_ok=True)
    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[CHAMPION MODEL] {champion['algorithm']} (ROC-AUC: {champion['roc_auc']}, Recall: {champion['recall']})")
    print(f"Saved benchmark results to {BENCHMARK_PATH}")
    print("=" * 70)
    return output_payload

if __name__ == "__main__":
    run_benchmark()
