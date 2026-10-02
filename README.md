# Production End-to-End Customer Churn MLOps Pipeline 🚀

An enterprise-ready MLOps framework designed to predict customer churn using the Telco Customer Churn dataset format. The repository integrates data preprocessing, model training with MLflow experiment tracking, REST API serving via FastAPI, unit testing with Pytest, Docker containerization, and automated CI/CD using GitHub Actions.

---

## 🏗 System Architecture

```
                       +------------------------+
                       |  data/raw_churn_data   |
                       +-----------+------------+
                                   |
                                   v
                       +------------------------+
                       |      src/train.py      |
                       +-----------+------------+
                                   |
         +-------------------------+-------------------------+
         |                                                   |
         v                                                   v
+------------------+                               +--------------------+
|  churn_model.pkl |                               | MLflow Experiment  |
+--------+---------+                               |   ("mlruns/")      |
         |                                         +--------------------+
         v
+------------------+
|    src/app.py    |  <--- REST Requests (GET /health, POST /predict)
|    (FastAPI)     |
+--------+---------+
         |
         v
+------------------+
|   Docker Image   |  <--- Containerized Microservice (Port 8000)
+------------------+
```

---

## 📂 Repository Structure

```
churn-mlops-pipeline/
├── .github/
│   └── workflows/
│       └── main.yml        # GitHub Actions CI/CD pipeline
├── data/
│   └── raw_churn_data.csv  # Telco Customer Churn sample dataset
├── src/
│   ├── train.py            # Data ingestion, preprocessing, training & MLflow logging
│   └── app.py              # FastAPI model serving REST service
├── tests/
│   └── test_app.py         # Pytest API endpoint test suite
├── Dockerfile              # Container definition (Python 3.10-slim)
├── requirements.txt        # Pinned project dependencies
├── .gitignore              # Ignored files (venv, mlruns, models, caches)
└── README.md               # Pipeline documentation & user guide
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup

Clone the repository and set up a Python 3.10 virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Model Training & Experiment Tracking

Run the training pipeline to preprocess data, fit the `RandomForestClassifier`, log parameters/metrics to MLflow, and export `churn_model.pkl`:

```bash
python src/train.py
```

#### Launching the MLflow UI

To inspect experiment runs, hyperparameters, accuracy, F1 scores, and model artifacts:

```bash
mlflow ui
```

Navigate to `http://127.0.0.1:5000` in your web browser to access the dashboard under the experiment **"Telco_Customer_Churn"**.

---

### 3. API Serving & Testing

#### Running Local Unit Tests

Run the test suite using `pytest`:

```bash
pytest tests/ -v
```

#### Launching the REST API Server

Start the Uvicorn ASGI server locally:

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

Access the interactive OpenAPI documentation at: `http://localhost:8000/docs`

---

## 🔌 API Endpoint Usage Examples

### 1. Health Check (`GET /health`)

```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "ok",
  "service": "MLOps Churn API"
}
```

---

### 2. Predict Customer Churn (`POST /predict`)

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "tenure": 2,
           "MonthlyCharges": 85.5,
           "TotalCharges": 171.0
         }'
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.7852,
  "risk_level": "HIGH"
}
```

---

## 🐳 Docker Deployment

### 1. Build Container Image

```bash
docker build -t churn-mlops-api:latest .
```

### 2. Run Container Microservice

```bash
docker run -d -p 8000:8000 --name churn-api churn-mlops-api:latest
```

Verify service status:

```bash
curl http://localhost:8000/health
```

---

## 🔄 GitHub Actions CI/CD Pipeline

The `.github/workflows/main.yml` pipeline automatically triggers on pushes to `main`:

1. **Environment Initialization**: Sets up Python 3.10 environment and updates `pip`.
2. **Dependency Installation**: Installs all required ML & Web dependencies from `requirements.txt`.
3. **Automated Training**: Executes `python src/train.py` to produce model artifacts.
4. **Automated Unit Testing**: Executes `pytest` to validate API contracts and inference.
5. **Container Packaging**: Builds the `churn-mlops-api:latest` Docker image.
