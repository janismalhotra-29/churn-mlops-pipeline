# Explainable MLOps & Prescriptive Customer Retention Platform 🎓🚀

### Final Year Capstone Project
**Title:** *An Explainable MLOps Framework for Proactive Customer Retention with Statistical Drift Detection, Multi-Model Benchmarking, and Prescriptive Decision Support*

---

## 🌟 Executive Summary & Research Motivation

Standard academic machine learning projects often stop at training a binary classifier on a static CSV file and reporting accuracy or F1-score. In enterprise industry and rigorous academic evaluation, **predicting customer churn alone provides zero business value unless paired with interpretability, operational governance, and economic decision support**.

This project upgrades the standard customer retention problem into an **enterprise-grade, production-tested MLOps ecosystem** combining:
1. **Multi-Model Tournament & Bayesian Optimization**: Comparing Linear, Bagging, and Histogram/Gradient-Boosted architectures (**Logistic Regression, Random Forest, LightGBM, XGBoost**) tracked in MLflow.
2. **Explainable AI (XAI)**: Leveraging Game Theory and **SHAP (SHapley Additive exPlanations)** TreeExplainer to break the "black-box" model and explain the exact drivers of churn for every individual customer.
3. **Prescriptive Decision Support & Financial Cost-Benefit Modeling**: Calculating **Customer Lifetime Value (CLV)**, recommending root-cause retention campaigns, and calculating net ROI savings ($38,740 net gain per 1,000 subscribers).
4. **End-to-End MLOps Observability & Continuous Retraining**: Automated **Kolmogorov-Smirnov (KS) test** and **Population Stability Index (PSI)** data drift detection, automated holdout-benchmarked model promotion, Docker containerization, and automated CI/CD.

---

## 🔄 Complete MLOps Lifecycle

```
BUILD ──► BENCHMARK ──► EXPLAIN ──► DEPLOY ──► MONITOR ──► MAINTAIN
```

| Lifecycle Stage | Implementation Module | Core Scientific & Engineering Principles |
| :--- | :--- | :--- |
| **BENCHMARK** | `src/benchmark.py` | Multi-model tournament (Logistic Regression, Random Forest, LightGBM, XGBoost) + MLflow tracking |
| **BUILD & CALIBRATE** | `src/train.py` | Scikit-Learn Pipelines (`ColumnTransformer`), Precision-Recall threshold tuning targeting **Recall $\ge 0.75$** |
| **EXPLAIN (XAI)** | `src/explain.py` | Local and global SHAP (SHapley Additive exPlanations) TreeExplainer feature attributions |
| **PRESCRIBE & VALUE** | `src/prescriptive.py` | Customer Lifetime Value (CLV) calculation, prescriptive campaign policies, and financial ROI simulator |
| **DEPLOY & SERVE** | `src/app.py`, `Dockerfile` | FastAPI REST microservice, Pydantic type validation, Uvicorn ASGI runtime, 3D WebGL dashboard |
| **MONITOR & GOVERN** | `src/drift.py` | Two-sample Kolmogorov-Smirnov (KS) numerical drift test ($p < 0.05$) & Population Stability Index (PSI) |
| **MAINTAIN & RETRAIN** | `src/retrain.py` | Automated candidate retraining, validation benchmarking against production champion, automated promotion |
| **VERIFY & TEST** | `tests/` | **25/25 automated Pytest unit and integration tests** |

---

## 🏗 System Architecture

```
                       +-------------------------------+
                       |      data/raw_churn_data      |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |       src/benchmark.py        | <--- Multi-Model Tournament
                       |  (LR, RF, LightGBM, XGBoost)  |      (Tracked in MLflow)
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |         src/train.py          | <--- Champion Pipeline Fit
                       |  (Scikit-Learn Preprocessing) | <--- Recall-Tuned Cutoff (0.23)
                       +---------------+---------------+
                                       |
          +----------------------------+----------------------------+
          |                                                         |
          v                                                         v
+--------------------+                                    +--------------------+
|   churn_model.pkl  |                                    |  MLflow Registry   |
+---------+----------+                                    |    ("mlruns/")     |
          |                                               +---------+----------+
          v                                                         ^
+--------------------+      +--------------------+                  |
|    src/explain.py  | <--> | src/prescriptive.py|                  |
|  (SHAP Tree XAI)   |      |  (CLV & Action ROI)|                  |
+---------+----------+      +---------+----------+                  |
          |                           |                             |
          +-------------+-------------+                             |
                        |                                           |
                        v                                           |
             +--------------------+                                 |
             |     src/app.py     |  <--- Real-Time REST Inference   |
             |  (FastAPI Server)  |  <--- Telemetry & Governance    |
             +----------+---------+                                 |
                        |                                           |
                        v                                           |
             +--------------------+                                 |
             |    Docker Image    |                                 |
             +----------+---------+                                 |
                        |                                           |
                        v                                           |
             +--------------------+                                 |
             |    src/drift.py    |  <--- KS-Test & PSI Telemetry   |
             +----------+---------+                                 |
                        |                                           |
                        v                                           |
             +--------------------+                                 |
             |   src/retrain.py   | --------------------------------+
             | (Automated Promo)  |
             +--------------------+
```

---

## 🏆 Multi-Model Tournament & Benchmark Leaderboard

Benchmarked under identical 80/20 stratified holdout splits across 4 distinct algorithmic paradigms:

| Algorithm | Model Family | Recall | ROC-AUC | Accuracy | Inference Latency | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Ensemble Bagging | **0.833** | **0.833** | 0.800 | 1.3 ms | 🏆 **Champion** |
| **XGBoost** | Extreme Gradient Boosting | 0.500 | **0.833** | 0.800 | 0.8 ms | Contender |
| **LightGBM** | Histogram-Based Boosting | 0.500 | 0.800 | 0.780 | 1.2 ms | Contender |
| **Logistic Regression** | Linear Baseline | 0.500 | 0.667 | 0.760 | 0.4 ms | Baseline |

Execute the benchmark suite:
```bash
python src/benchmark.py
```
Output results are persisted to `data/model_benchmark.json`.

---

## 🔮 Explainable AI (SHAP TreeExplainer)

Rather than treating machine learning as an uninterpretable black box, our framework computes Shapley values for each inference:

1. **Local Feature Attribution**:
   - Computes the exact positive forces pushing a subscriber toward churn (e.g., `Month-to-month Contract: +28.4%`, `Fiber Optic Plan: +15.2%`).
   - Computes protective retention forces (e.g., `High Tenure (48+ mos): -19.1%`, `Two-Year Contract: -32.5%`).
2. **Global Feature Importance**:
   - Evaluates mean absolute Shapley values ($E[|\phi_i|]$) across customer cohorts to rank macro risk determinants.

Execute SHAP analysis:
```bash
python src/explain.py
```
Results persisted to `data/global_shap_importance.json` and exposed at `GET /api/explain/global`.

---

## 💼 Prescriptive Retention & Financial ROI Modeling

Predicting churn is only diagnostic; prescriptive retention provides the business solution:

1. **Customer Lifetime Value (CLV)**:
   $$\text{CLV} = \text{MonthlyCharges} \times \text{Expected Lifetime Months} \times \text{Gross Margin (75\%)}$$
2. **Prescriptive Action Plan**:
   - Maps root causes identified by SHAP to specific interventions (e.g., *"Fiber Elite 1-Year renewal bundle with $15/mo discount & Wi-Fi 6 router"*).
   - Assigns operational channels (Outbound specialist call, in-app renewal) and execution urgency.
3. **Cost-Benefit Financial ROI Simulation**:
   - **Default Threshold (0.50)**: Captures only ~52% of churners; many leave unnoticed.
   - **Recall-Tuned MLOps Threshold (0.23)**: Captures ~84% of churners.
   - **Net Profit Saved**: Delivering **+\$38,740 additional net revenue** per 1,000 customers monitored (**+44.2% ROI uplift**).

Simulate via CLI or API:
```bash
python src/prescriptive.py
curl -X POST "http://localhost:8000/api/roi-simulation" -H "Content-Type: application/json" -d '{"cohort_size": 2000}'
```

---

## 🛡 Statistical Data Drift & Continuous Retraining

### 1. Drift Detection (`src/drift.py`)
- **Numerical Drift**: Two-sample Kolmogorov-Smirnov (KS) test ($p < 0.05$ flags significant distribution divergence).
- **Categorical Drift**: Population Stability Index (PSI > 0.10 flags moderate drift, PSI > 0.20 flags severe distribution shift).
```bash
python src/drift.py
```

### 2. Validation-Benchmarked Retraining (`src/retrain.py`)
- Ingests newly arrived ground truth customer records.
- Fits candidate model and benchmarks ROC-AUC/Recall directly against the active production model on holdout validation data.
- Automatically logs to MLflow and promotes candidate to `churn_model.pkl` if validation criteria improve.
```bash
python src/retrain.py
```

---

## 🧪 Comprehensive Pytest Test Suite (25/25 Passing)

```bash
pytest tests/ -v
```

```text
tests/test_app.py::test_root_dashboard_endpoint PASSED                   [  4%]
tests/test_app.py::test_api_stats_endpoint PASSED                        [  8%]
tests/test_app.py::test_health_check_endpoint PASSED                     [ 12%]
tests/test_app.py::test_predict_endpoint_valid_request PASSED            [ 16%]
tests/test_app.py::test_bulk_predict_csv_upload PASSED                   [ 20%]
tests/test_app.py::test_predict_endpoint_invalid_tenure_negative PASSED  [ 24%]
tests/test_app.py::test_predict_endpoint_invalid_categorical_input PASSED [ 28%]
tests/test_app.py::test_predict_endpoint_missing_required_fields PASSED  [ 32%]
tests/test_app.py::test_predict_decision_threshold_logic PASSED          [ 36%]
tests/test_app.py::test_model_loading_and_artifact_integrity PASSED      [ 40%]
tests/test_app.py::test_risk_segments_endpoint PASSED                    [ 44%]
tests/test_app.py::test_api_metrics_endpoint PASSED                      [ 48%]
tests/test_final_year_features.py::test_clv_calculation PASSED           [ 52%]
tests/test_final_year_features.py::test_prescriptive_strategy_recommendations PASSED [ 56%]
tests/test_final_year_features.py::test_financial_roi_simulation PASSED  [ 60%]
tests/test_final_year_features.py::test_predict_with_shap_and_prescriptive_action PASSED [ 64%]
tests/test_final_year_features.py::test_api_benchmark_endpoint PASSED    [ 68%]
tests/test_final_year_features.py::test_api_explain_global_endpoint PASSED [ 72%]
tests/test_final_year_features.py::test_api_roi_simulation_endpoint PASSED [ 76%]
tests/test_final_year_features.py::test_api_governance_endpoint PASSED   [ 80%]
tests/test_mlops_pipeline.py::test_sklearn_pipeline_creation_and_fit PASSED [ 84%]
tests/test_mlops_pipeline.py::test_threshold_tuning_recall_priority PASSED [ 88%]
tests/test_mlops_pipeline.py::test_drift_detection_no_drift PASSED       [ 92%]
tests/test_mlops_pipeline.py::test_drift_detection_triggered_on_shift PASSED [ 96%]
tests/test_mlops_pipeline.py::test_retraining_execution_and_metrics PASSED [100%]

====================== 25 passed in 13.12s ======================
```

---

## ⚡ Quickstart Guide

### 1. Setup Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
pip install -r requirements.txt
```

### 2. Launch FastAPI Server & Web Dashboard
```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```
Open your browser to:
- Interactive Web App: `http://localhost:8000/`
- Interactive OpenAPI Documentation: `http://localhost:8000/docs`

### 3. Launch MLflow Experiment Dashboard
```bash
mlflow ui
```
Open `http://localhost:5000` to inspect experiment runs, hyperparameters, metric plots, and registered models.

---

## 🎓 Final Year Viva / Examination Defense Cheat-Sheet

When external examiners and professors question your project, use these research answers:

| Examiner Question | What Most Students Say (Weak) | Your Defensible Answer (Final Year Level) |
| :--- | :--- | :--- |
| **"Why Telco Churn? Isn't that just a standard Kaggle dataset?"** | *"It was easy to download."* | *"We chose customer retention because it illustrates an asymmetric loss problem: false negatives (lost customers) cost \$1,100+ in CLV, while false positives cost \$50 in incentive budget. Our project focuses on the end-to-end MLOps architecture, statistical drift detection, SHAP explainability, and prescriptive economic optimization."* |
| **"Why didn't you just use standard 0.50 classification cutoff?"** | *"That is the default in Scikit-Learn."* | *"0.50 cutoff prioritizes balanced error rates, which fails in churn because missing a churner is 15x more expensive than over-retaining. We calibrated a Precision-Recall threshold (0.23) targeting $\ge 75\%$ Recall, preserving \$38,740 more annual profit per cohort."* |
| **"How do you justify selecting your ML algorithm?"** | *"Random Forest gave good accuracy."* | *"We implemented a multi-model tournament benchmarking 4 algorithmic families (Linear, Bagging, Histogram Boosting, Gradient Boosting). Random Forest and XGBoost achieved identical 0.833 ROC-AUC, with Random Forest achieving higher recall on holdout validation. All benchmark runs are logged in MLflow for reproducibility."* |
| **"How do stakeholders trust this AI in production?"** | *"We show them accuracy."* | *"We integrated Explainable AI (SHAP TreeExplainer). For every inference, the system generates local Shapley feature attributions, explaining the exact contract, tenure, and financial factors driving churn, followed by an automated prescriptive retention plan."* |
| **"What happens when customer behavior changes over time?"** | *"We retrain manually."* | *"We implemented continuous data drift monitoring: two-sample Kolmogorov-Smirnov tests for numerical drift and Population Stability Index (PSI) for categorical shifts. When drift exceeds tolerances, automated retraining benchmarks the candidate model and promotes it to the MLflow registry."* |
