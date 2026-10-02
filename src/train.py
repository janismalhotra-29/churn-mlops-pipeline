import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_data(data_path: str) -> pd.DataFrame:
    """Load Telco Customer Churn dataset from CSV, with a fallback dummy generator if missing."""
    if os.path.exists(data_path):
        print(f"[Ingestion] Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print(f"[Ingestion] Warning: '{data_path}' not found. Generating dummy Telco dataset...")
        np.random.seed(42)
        n_samples = 250
        tenure = np.random.randint(1, 72, size=n_samples)
        monthly_charges = np.random.uniform(18.0, 120.0, size=n_samples)
        total_charges = tenure * monthly_charges + np.random.normal(0, 15, size=n_samples)
        churn = np.random.choice(["Yes", "No"], size=n_samples, p=[0.27, 0.73])
        df = pd.DataFrame({
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn
        })
    return df

def preprocess_data(df: pd.DataFrame):
    """Preprocess Telco dataset features: handle missing values in TotalCharges and binary encode Churn."""
    df = df.copy()
    
    # 1. Clean and convert TotalCharges to numeric, coercing non-numeric values to NaN
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        median_total = df["TotalCharges"].median()
        if pd.isna(median_total):
            median_total = 0.0
        df["TotalCharges"] = df["TotalCharges"].fillna(median_total)
    
    # 2. Binary-encode target variable 'Churn'
    if "Churn" in df.columns:
        if df["Churn"].dtype == object:
            mapping = {"Yes": 1, "No": 0, "YES": 1, "NO": 0, "1": 1, "0": 0}
            df["Churn"] = df["Churn"].astype(str).str.strip().map(mapping).fillna(0).astype(int)
    
    features = ["tenure", "MonthlyCharges", "TotalCharges"]
    X = df[features]
    y = df["Churn"]
    return X, y

def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw_churn_data.csv")
    model_output_path = os.path.join(base_dir, "churn_model.pkl")
    
    df = load_data(data_path)
    X, y = preprocess_data(df)
    
    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Hyperparameters
    n_estimators = 100
    max_depth = 5
    random_state = 42
    
    # MLflow Experiment Setup
    mlflow.set_experiment("Telco_Customer_Churn")
    
    with mlflow.start_run() as run:
        print(f"[MLflow] Started run ID: {run.info.run_id}")
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        # Calculate Evaluation Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        
        print(f"[Evaluation] Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        
        # Log Hyperparameters & Metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_score", float(f1))
        mlflow.log_metric("precision", float(prec))
        mlflow.log_metric("recall", float(rec))
        
        # Log MLflow Model Artifact
        mlflow.sklearn.log_model(clf, artifact_path="churn_model")
        
        # Save local binary artifact for FastAPI serving
        joblib.dump(clf, model_output_path)
        print(f"[Artifact] Model artifact saved to: {model_output_path}")

if __name__ == "__main__":
    train()
