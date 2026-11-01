import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
)

def load_data(data_path: str) -> pd.DataFrame:
    """Load Telco Customer Churn dataset from CSV, with a fallback dummy generator if missing."""
    if os.path.exists(data_path):
        print(f"[Ingestion] Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print(f"[Ingestion] Warning: '{data_path}' not found. Generating dummy Telco dataset...")
        np.random.seed(42)
        n_samples = 500
        tenure = np.random.randint(1, 72, size=n_samples)
        monthly_charges = np.random.uniform(18.0, 120.0, size=n_samples)
        total_charges = tenure * monthly_charges + np.random.normal(0, 15, size=n_samples)
        contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples)
        internet = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples)
        churn = np.random.choice(["Yes", "No"], size=n_samples, p=[0.27, 0.73])
        df = pd.DataFrame({
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "InternetService": internet,
            "Churn": churn
        })
    return df

def create_sklearn_pipeline(num_features, cat_features, n_estimators=100, max_depth=5, random_state=42):
    """Build a reproducible Scikit-Learn Pipeline combining ColumnTransformer preprocessing and RandomForest Classifier."""
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    transformers = [('num', num_transformer, num_features)]
    if cat_features:
        transformers.append(('cat', cat_transformer, cat_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        ))
    ])
    return pipeline

def find_optimal_threshold(y_true, y_proba, target_recall=0.75):
    """Threshold Tuning: Find optimal decision threshold prioritizing Recall (minimizing false negatives)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    optimal_threshold = 0.50
    for prec, rec, thresh in zip(precisions[:-1], recalls[:-1], thresholds):
        if rec >= target_recall:
            optimal_threshold = float(thresh)
    return round(float(optimal_threshold), 4)

def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw_churn_data.csv")
    model_output_path = os.path.join(base_dir, "churn_model.pkl")
    
    df = load_data(data_path)
    
    # Clean & encode target variable
    if "Churn" in df.columns:
        mapping = {"Yes": 1, "No": 0, "YES": 1, "NO": 0, "1": 1, "0": 0}
        y = df["Churn"].astype(str).str.strip().map(mapping).fillna(0).astype(int)
    else:
        y = np.zeros(len(df), dtype=int)
        
    # Standardize numeric columns
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            if col == "TotalCharges" and "tenure" in df.columns and "MonthlyCharges" in df.columns:
                df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"]
            else:
                df[col] = 0.0

    num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_features = [c for c in ["Contract", "InternetService"] if c in df.columns]
    
    feature_cols = num_features + cat_features
    X = df[feature_cols]
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Hyperparameters
    n_estimators = 100
    max_depth = 5
    random_state = 42
    
    # Construct Scikit-Learn Pipeline
    pipeline = create_sklearn_pipeline(num_features, cat_features, n_estimators, max_depth, random_state)
    
    # MLflow Experiment & Tracking Setup
    mlflow.set_experiment("Telco_Customer_Churn")
    
    with mlflow.start_run() as run:
        print(f"[MLflow] Started run ID: {run.info.run_id}")
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
        
        # Threshold Tuning
        optimal_threshold = find_optimal_threshold(y_test, y_proba, target_recall=0.75)
        
        print(f"[Evaluation] Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | ROC-AUC: {roc_auc:.4f}")
        print(f"[Threshold Tuning] Optimal threshold prioritizing Recall (>=0.75): {optimal_threshold}")
        
        # Log Hyperparameters & Metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("decision_threshold", optimal_threshold)
        
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_score", float(f1))
        mlflow.log_metric("precision", float(prec))
        mlflow.log_metric("recall", float(rec))
        mlflow.log_metric("roc_auc", float(roc_auc))
        
        # Log Scikit-Learn Model Artifact & Model Registry
        try:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="churn_model",
                registered_model_name="Telco_Churn_RandomForest",
                serialization_format="cloudpickle"
            )
        except Exception as exc:
            print(f"[MLflow Warning] Model registry log skipped: {exc}")
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="churn_model",
                serialization_format="cloudpickle"
            )
        
        # Save local binary artifact bundle
        model_artifact = {
            "pipeline": pipeline,
            "decision_threshold": optimal_threshold,
            "features": feature_cols,
            "num_features": num_features,
            "cat_features": cat_features
        }
        
        joblib.dump(model_artifact, model_output_path)
        print(f"[Artifact] Saved Scikit-Learn Pipeline artifact to: {model_output_path}")

if __name__ == "__main__":
    train()
