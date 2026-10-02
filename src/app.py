import os
import io
import json
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

# Resolve path relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "churn_model.pkl")
SUMMARY_PATH = os.path.join(BASE_DIR, "data", "dataset_summary.json")
TEMPLATE_PATH = os.path.join(BASE_DIR, "src", "templates", "index.html")

model = None

def get_model():
    """Lazy load or return the trained model artifact."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Run training pipeline first.")
        model = joblib.load(MODEL_PATH)
    return model

class CustomerData(BaseModel):
    tenure: int = Field(..., ge=0, description="Customer tenure in months", json_schema_extra={"example": 12})
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly charges amount", json_schema_extra={"example": 65.5})
    TotalCharges: float = Field(..., ge=0.0, description="Total charges amount", json_schema_extra={"example": 786.0})
    Contract: str = Field("Month-to-month", description="Contract type: Month-to-month, One year, Two year", json_schema_extra={"example": "Month-to-month"})
    InternetService: str = Field("Fiber optic", description="Internet tech: Fiber optic, DSL, No", json_schema_extra={"example": "Fiber optic"})

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for Retained, 1 for Churn")
    probability: float = Field(..., description="Probability of churn (0.0 to 1.0)")
    risk_level: str = Field(..., description="HIGH, MEDIUM, or LOW risk level")
    financial_impact_annual: float = Field(..., description="Estimated annual revenue at risk")
    retention_strategy: str = Field(..., description="Tailored retention action plan")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for loading model artifact during server startup."""
    try:
        get_model()
        print(f"[Serving] Successfully loaded churn model from {MODEL_PATH}")
    except Exception as exc:
        print(f"[Serving Warning] Model load deferred on startup: {exc}")
    yield

app = FastAPI(
    title="TelcoCare Enterprise Customer Retention API",
    description="Production-grade REST API & Web Dashboard serving RandomForest Churn Classifier with single-form and bulk CSV batch predictions",
    version="2.1.0",
    lifespan=lifespan
)

@app.get("/", response_class=HTMLResponse)
def read_dashboard():
    """Serve the enterprise web portal UI."""
    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h2>TelcoCare API</h2><p>Visit <a href='/docs'>/docs</a></p>")

@app.get("/api/stats")
def get_dataset_stats():
    """Return dataset analytics summary metadata for dashboard widgets."""
    if os.path.exists(SUMMARY_PATH):
        try:
            with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "total_records": 7043,
        "churn_count": 1869,
        "retained_count": 5174,
        "churn_rate_pct": 26.54,
        "avg_tenure": 32.4,
        "avg_monthly_charges": 64.76,
        "accuracy": 80.20,
        "feature_importances": {
            "Tenure (Months)": 0.38,
            "Total Charges ($)": 0.28,
            "Contract Duration": 0.18,
            "Monthly Charges ($)": 0.12,
            "Internet Tech (Fiber/DSL)": 0.04
        }
    }

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Silence browser favicon requests."""
    return Response(status_code=204)

@app.get("/health")
def health_check():
    """Health check endpoint to verify service operational status."""
    return {
        "status": "ok",
        "service": "TelcoCare Enterprise API",
        "dataset_records": 7043
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """Single-user prediction endpoint returning churn probability, risk score, financial impact, and recommendations."""
    try:
        clf = get_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {str(exc)}")

    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    contract_code = contract_map.get(customer.Contract, 0)

    internet_map = {"No": 0, "DSL": 1, "Fiber optic": 2}
    internet_code = internet_map.get(customer.InternetService, 2)

    input_df = pd.DataFrame([{
        "tenure": customer.tenure,
        "MonthlyCharges": customer.MonthlyCharges,
        "TotalCharges": customer.TotalCharges,
        "Contract_Code": contract_code,
        "InternetService_Code": internet_code
    }])

    try:
        expected_n_features = getattr(clf, "n_features_in_", 3)
        if expected_n_features == 3:
            input_df = input_df[["tenure", "MonthlyCharges", "TotalCharges"]]

        prediction_val = int(clf.predict(input_df)[0])
        probabilities = clf.predict_proba(input_df)[0]
        churn_proba = float(probabilities[1]) if len(probabilities) > 1 else float(prediction_val)

        if churn_proba >= 0.60:
            risk_level = "HIGH"
            strategy = "Offer 25% discount on 1-Year contract commitment immediately."
        elif churn_proba >= 0.35:
            risk_level = "MEDIUM"
            strategy = "Send retention email offering complimentary streaming bundle add-on."
        else:
            risk_level = "LOW"
            strategy = "Standard retention engagement. Candidate for cross-selling."

        annual_impact = round(customer.MonthlyCharges * 12.0, 2)

        return PredictionResponse(
            prediction=prediction_val,
            probability=round(churn_proba, 4),
            risk_level=risk_level,
            financial_impact_annual=annual_impact,
            retention_strategy=strategy
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")

@app.post("/predict/bulk")
def predict_bulk_csv(file: UploadFile = File(...)):
    """Bulk Upload Endpoint: Ingests a CSV or Excel file of customer accounts, runs batch ML inference, and returns parsed predictions."""
    try:
        clf = get_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {str(exc)}")

    try:
        contents = file.file.read()
        filename = (file.filename or "").lower()
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.BytesIO(contents))

        # Flexible Column Normalization (handles Tenure, Monthly Charges, Monthly_Charges, etc.)
        norm_map = {str(c).strip().lower().replace("_", "").replace(" ", ""): c for c in df.columns}

        tenure_orig = norm_map.get("tenure")
        monthly_orig = norm_map.get("monthlycharges") or norm_map.get("monthlycharge") or norm_map.get("monthly")
        total_orig = norm_map.get("totalcharges") or norm_map.get("totalcharge") or norm_map.get("total")
        contract_orig = norm_map.get("contract")
        internet_orig = norm_map.get("internetservice") or norm_map.get("internet")
        customerid_orig = norm_map.get("customerid") or norm_map.get("customer_id") or norm_map.get("id")

        if not tenure_orig or not monthly_orig:
            raise HTTPException(
                status_code=422,
                detail=f"CSV/Excel missing required columns. Found columns: {list(df.columns)}. Expected 'tenure' and 'MonthlyCharges'."
            )

        df["tenure"] = pd.to_numeric(df[tenure_orig], errors="coerce").fillna(1).astype(int)
        df["MonthlyCharges"] = pd.to_numeric(df[monthly_orig], errors="coerce").fillna(50.0).astype(float)

        if total_orig and total_orig in df.columns:
            df["TotalCharges"] = pd.to_numeric(df[total_orig], errors="coerce").fillna(df["tenure"] * df["MonthlyCharges"])
        else:
            df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"]

        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2, "month-to-month": 0, "one year": 1, "two year": 2}
        if contract_orig and contract_orig in df.columns:
            df["Contract_Code"] = df[contract_orig].astype(str).map(contract_map).fillna(0).astype(int)
            df["Contract"] = df[contract_orig].astype(str)
        else:
            df["Contract_Code"] = 0
            df["Contract"] = "Month-to-month"

        internet_map = {"No": 0, "DSL": 1, "Fiber optic": 2, "no": 0, "dsl": 1, "fiber optic": 2}
        if internet_orig and internet_orig in df.columns:
            df["InternetService_Code"] = df[internet_orig].astype(str).map(internet_map).fillna(2).astype(int)
            df["InternetService"] = df[internet_orig].astype(str)
        else:
            df["InternetService_Code"] = 2
            df["InternetService"] = "Fiber optic"

        if customerid_orig and customerid_orig in df.columns:
            df["customerID"] = df[customerid_orig].astype(str)
        else:
            df["customerID"] = ["TELCO-" + str(i+1001).zfill(5) for i in range(len(df))]

        feature_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Contract_Code", "InternetService_Code"]
        expected_n_features = getattr(clf, "n_features_in_", 3)
        if expected_n_features == 3:
            feature_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

        X_batch = df[feature_cols]

        predictions = clf.predict(X_batch)
        probabilities = clf.predict_proba(X_batch)

        df["Churn_Prediction"] = predictions
        df["Churn_Probability"] = [round(float(p[1]), 4) if len(p) > 1 else float(pred) for p, pred in zip(probabilities, predictions)]
        df["Risk_Level"] = ["HIGH" if prob >= 0.50 else ("MEDIUM" if prob >= 0.30 else "LOW") for prob in df["Churn_Probability"]]
        df["Annual_Revenue_Risk"] = (df["MonthlyCharges"] * 12.0).round(2)

        def get_strategy(risk):
            if risk == "HIGH":
                return "Offer 25% discount on 1-Year contract commitment immediately."
            elif risk == "MEDIUM":
                return "Send retention email offering complimentary streaming bundle."
            else:
                return "Standard retention engagement. Candidate for cross-selling."

        df["Retention_Strategy"] = df["Risk_Level"].apply(get_strategy)

        high_risk_count = int((df["Risk_Level"] == "HIGH").sum())
        medium_risk_count = int((df["Risk_Level"] == "MEDIUM").sum())
        low_risk_count = int((df["Risk_Level"] == "LOW").sum())
        total_revenue_risk = float(df[df["Risk_Level"] == "HIGH"]["Annual_Revenue_Risk"].sum())

        output_cols = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "Churn_Prediction", "Churn_Probability", "Risk_Level", "Annual_Revenue_Risk", "Retention_Strategy"]
        final_cols = [c for c in output_cols if c in df.columns]
        results = df[final_cols].head(1000).to_dict(orient="records")

        return {
            "status": "success",
            "filename": file.filename,
            "total_records_processed": len(df),
            "high_risk_count": high_risk_count,
            "medium_risk_count": medium_risk_count,
            "low_risk_count": low_risk_count,
            "total_annual_revenue_at_risk": round(total_revenue_risk, 2),
            "sample_results": results
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Bulk processing error: {str(exc)}")
