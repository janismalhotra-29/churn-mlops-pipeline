import os
import argparse
import json
import pandas as pd
import numpy as np
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASELINE = os.path.join(BASE_DIR, "data", "raw_churn_data.csv")
DRIFT_REPORT_PATH = os.path.join(BASE_DIR, "data", "drift_report.json")

def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
    """Calculate Population Stability Index (PSI) for numerical distributions."""
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]
    if len(baseline) == 0 or len(current) == 0:
        return 0.0
        
    quantiles = np.linspace(0, 100, num_buckets + 1)
    percentiles = np.percentile(baseline, quantiles)
    percentiles[0] -= 1e-5
    percentiles[-1] += 1e-5
    
    # Handle non-unique bin edges
    bins = np.unique(percentiles)
    if len(bins) < 2:
        return 0.0
        
    base_counts, _ = np.histogram(baseline, bins=bins)
    curr_counts, _ = np.histogram(current, bins=bins)
    
    base_dist = base_counts / len(baseline)
    curr_dist = curr_counts / len(current)
    
    # Laplace smoothing to avoid division by zero
    base_dist = np.where(base_dist == 0, 1e-4, base_dist)
    curr_dist = np.where(curr_dist == 0, 1e-4, curr_dist)
    
    psi_value = np.sum((curr_dist - base_dist) * np.log(curr_dist / base_dist))
    return float(np.round(psi_value, 4))

def detect_drift(current_df: pd.DataFrame, baseline_df: pd.DataFrame = None, p_threshold: float = 0.05) -> dict:
    """
    Detect statistical data drift between current dataset and baseline dataset using KS-test and PSI.
    Returns structured drift report indicating whether drift was detected and whether retraining is recommended.
    """
    if baseline_df is None:
        if not os.path.exists(DEFAULT_BASELINE):
            raise FileNotFoundError(f"Baseline dataset not found at '{DEFAULT_BASELINE}'")
        baseline_df = pd.read_csv(DEFAULT_BASELINE)

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    feature_reports = {}
    drifted_features = []

    for col in numeric_features:
        if col in baseline_df.columns and col in current_df.columns:
            base_vals = pd.to_numeric(baseline_df[col], errors="coerce").dropna().values
            curr_vals = pd.to_numeric(current_df[col], errors="coerce").dropna().values
            
            if len(base_vals) > 0 and len(curr_vals) > 0:
                ks_stat, p_val = stats.ks_2samp(base_vals, curr_vals)
                psi_val = calculate_psi(base_vals, curr_vals)
                is_drifted = bool(p_val < p_threshold or psi_val > 0.25)
                
                if is_drifted:
                    drifted_features.append(col)
                    
                feature_reports[col] = {
                    "ks_statistic": round(float(ks_stat), 4),
                    "p_value": round(float(p_val), 4),
                    "psi": psi_val,
                    "drift_detected": is_drifted,
                    "baseline_mean": round(float(np.mean(base_vals)), 2),
                    "current_mean": round(float(np.mean(curr_vals)), 2)
                }

    drift_ratio = len(drifted_features) / len(feature_reports) if feature_reports else 0.0
    overall_drift = bool(drift_ratio >= 0.33)  # Drift in >= 1 out of 3 key features triggers alert

    report = {
        "status": "drift_detected" if overall_drift else "stable",
        "overall_drift_detected": overall_drift,
        "drift_ratio": round(drift_ratio, 2),
        "drifted_features": drifted_features,
        "retraining_recommended": overall_drift,
        "feature_metrics": feature_reports
    }
    
    # Save report
    os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
    with open(DRIFT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report

def main():
    parser = argparse.ArgumentParser(description="Customer Churn Data & Concept Drift Detector")
    parser.add_argument("--data", type=str, required=False, help="Path to current dataset to check for drift")
    parser.add_argument("--baseline", type=str, default=DEFAULT_BASELINE, help="Path to baseline dataset")
    parser.add_argument("--p_val", type=float, default=0.05, help="P-value significance threshold for KS-test")
    args = parser.parse_args()

    baseline_df = pd.read_csv(args.baseline)
    if args.data and os.path.exists(args.data):
        current_df = pd.read_csv(args.data)
    else:
        # Self-test: generate a perturbed dataset to simulate production drift
        print("[Drift] No dataset supplied. Generating perturbed dataset to evaluate drift detection...")
        current_df = baseline_df.copy()
        current_df["MonthlyCharges"] = current_df["MonthlyCharges"] * np.random.uniform(1.3, 1.8, size=len(current_df))
        current_df["tenure"] = current_df["tenure"] * np.random.uniform(0.3, 0.7, size=len(current_df))

    report = detect_drift(current_df, baseline_df, p_threshold=args.p_val)
    print("\n--- DATA DRIFT ASSESSMENT REPORT ---")
    print(f"Status: {report['status'].upper()}")
    print(f"Drift Detected: {report['overall_drift_detected']}")
    print(f"Drifted Features: {report['drifted_features']}")
    print(f"Retraining Recommended: {report['retraining_recommended']}")
    print("Feature Breakdown:")
    for feat, data in report["feature_metrics"].items():
        print(f"  * {feat}: p-value={data['p_value']}, PSI={data['psi']}, Drifted={data['drift_detected']}")

if __name__ == "__main__":
    main()
