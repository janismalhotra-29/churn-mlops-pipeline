"""
src/prescriptive.py
Prescriptive Decision Engine & Financial ROI / Cost-Benefit Simulation.
Translates predictive churn probabilities into actionable retention interventions and economic value.
"""

import os
import json
import numpy as np

def calculate_clv(tenure: int, monthly_charges: float, contract: str, profit_margin: float = 0.75) -> dict:
    """
    Calculate estimated Customer Lifetime Value (CLV) based on contract stability and spending.
    """
    # Expected remaining lifetime in months
    if contract == "Two year":
        expected_remaining_months = max(18, 24 - (tenure % 24))
    elif contract == "One year":
        expected_remaining_months = max(9, 12 - (tenure % 12))
    else:  # Month-to-month
        expected_remaining_months = max(3.0, min(14.0, float(tenure) * 0.35 + 2.0))

    clv = round(float(monthly_charges) * expected_remaining_months * profit_margin, 2)
    annual_spend = round(float(monthly_charges) * 12.0, 2)

    return {
        "clv_amount": clv,
        "annual_spend": annual_spend,
        "expected_remaining_months": round(expected_remaining_months, 1)
    }

def recommend_prescriptive_strategy(customer: dict, probability: float, risk_level: str) -> dict:
    """
    Generate prescriptive business interventions tailored to the customer's specific risk drivers.
    """
    tenure = customer.get("tenure", 12)
    monthly_charges = customer.get("MonthlyCharges", 65.0)
    contract = customer.get("Contract", "Month-to-month")
    internet = customer.get("InternetService", "Fiber optic")

    clv_meta = calculate_clv(tenure, monthly_charges, contract)
    clv = clv_meta["clv_amount"]

    if risk_level == "HIGH":
        if contract == "Month-to-month" and internet == "Fiber optic":
            action = {
                "campaign_name": "Fiber Elite Contract Lock-In",
                "recommended_action": "Proactively offer 12-Month Contract with $15/mo discount + complimentary Premium Wi-Fi 6 Router upgrade.",
                "action_channel": "Dedicated Retention Specialist Outbound Call",
                "incentive_budget": 90.0,
                "projected_success_rate": 0.62,
                "urgency": "CRITICAL (Execute within 48h)"
            }
        elif contract == "Month-to-month":
            action = {
                "campaign_name": "Annual Commitment Bridge Offer",
                "recommended_action": "Offer first 3 months at 20% off upon switching from Month-to-month to 1-Year Contract.",
                "action_channel": "In-App Push + Email Campaign with 1-Click Renewal",
                "incentive_budget": 45.0,
                "projected_success_rate": 0.54,
                "urgency": "HIGH (Execute within 5 days)"
            }
        else:
            action = {
                "campaign_name": "VIP Concierge Care",
                "recommended_action": "Schedule comprehensive account review with technical performance check on broadband line.",
                "action_channel": "Account Manager Outreach",
                "incentive_budget": 30.0,
                "projected_success_rate": 0.48,
                "urgency": "HIGH (Execute within 7 days)"
            }
    elif risk_level == "MEDIUM":
        action = {
            "campaign_name": "Proactive Value Optimization",
            "recommended_action": "Send personalized digital check-in survey and offer complimentary speed boost or streaming add-on for 6 months.",
            "action_channel": "Automated Personalized Email & SMS",
            "incentive_budget": 20.0,
            "projected_success_rate": 0.72,
            "urgency": "MODERATE (Execute within 14 days)"
        }
    else:  # LOW
        action = {
            "campaign_name": "Loyalty Advocacy & Referral Program",
            "recommended_action": "Enroll in loyalty rewards tier with double referral points and automated satisfaction check.",
            "action_channel": "Standard Portal Notification",
            "incentive_budget": 5.0,
            "projected_success_rate": 0.95,
            "urgency": "ROUTINE"
        }

    # Calculate expected net financial ROI of the action
    projected_retained_value = round(clv * action["projected_success_rate"], 2)
    net_roi = round(projected_retained_value - action["incentive_budget"], 2)

    return {
        "risk_level": risk_level,
        "clv_metrics": clv_meta,
        "prescriptive_intervention": action,
        "economic_value": {
            "customer_clv": clv,
            "incentive_cost": action["incentive_budget"],
            "expected_retained_value": projected_retained_value,
            "net_roi": net_roi
        }
    }

def simulate_financial_roi(
    customer_cohort_size: int = 1000,
    churn_rate: float = 0.265,
    avg_annual_clv: float = 1100.0,
    intervention_cost: float = 60.0,
    intervention_success_rate: float = 0.55
) -> dict:
    """
    Simulate financial cost-benefit comparison between:
    1. Default Standard Threshold (0.50) -> Low Recall (~50%)
    2. Recall-Optimized MLOps Threshold (0.23) -> High Recall (~85%)
    """
    total_churners = int(customer_cohort_size * churn_rate)

    # Threshold 0.50 (Standard baseline)
    recall_standard = 0.52
    precision_standard = 0.70
    caught_standard = int(total_churners * recall_standard)
    false_positives_standard = int(caught_standard * ((1 - precision_standard) / precision_standard))
    retained_standard = int(caught_standard * intervention_success_rate)
    revenue_saved_standard = retained_standard * avg_annual_clv
    campaign_cost_standard = (caught_standard + false_positives_standard) * intervention_cost
    net_profit_standard = revenue_saved_standard - campaign_cost_standard

    # Threshold 0.23 (Tuned Recall-first MLOps policy)
    recall_tuned = 0.84
    precision_tuned = 0.54
    caught_tuned = int(total_churners * recall_tuned)
    false_positives_tuned = int(caught_tuned * ((1 - precision_tuned) / precision_tuned))
    retained_tuned = int(caught_tuned * intervention_success_rate)
    revenue_saved_tuned = retained_tuned * avg_annual_clv
    campaign_cost_tuned = (caught_tuned + false_positives_tuned) * intervention_cost
    net_profit_tuned = revenue_saved_tuned - campaign_cost_tuned

    additional_net_savings = net_profit_tuned - net_profit_standard
    roi_uplift_pct = round(((net_profit_tuned - net_profit_standard) / max(1.0, net_profit_standard)) * 100, 1)

    return {
        "cohort_size": customer_cohort_size,
        "true_churners_in_cohort": total_churners,
        "standard_baseline_threshold_0_50": {
            "recall": recall_standard,
            "churners_identified": caught_standard,
            "churners_missed": total_churners - caught_standard,
            "customers_retained": retained_standard,
            "gross_revenue_preserved": revenue_saved_standard,
            "campaign_cost": campaign_cost_standard,
            "net_financial_gain": net_profit_standard
        },
        "recall_tuned_mlops_threshold": {
            "threshold_value": 0.23,
            "recall": recall_tuned,
            "churners_identified": caught_tuned,
            "churners_missed": total_churners - caught_tuned,
            "customers_retained": retained_tuned,
            "gross_revenue_preserved": revenue_saved_tuned,
            "campaign_cost": campaign_cost_tuned,
            "net_financial_gain": net_profit_tuned
        },
        "business_value_delivered": {
            "additional_churners_saved": retained_tuned - retained_standard,
            "additional_net_profit_saved": additional_net_savings,
            "roi_uplift_percent": f"+{roi_uplift_pct}%"
        }
    }

if __name__ == "__main__":
    test_customer = {
        "tenure": 3,
        "MonthlyCharges": 85.0,
        "TotalCharges": 255.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic"
    }
    strat = recommend_prescriptive_strategy(test_customer, probability=0.82, risk_level="HIGH")
    print(json.dumps(strat, indent=2))
    sim = simulate_financial_roi()
    print(f"\nNet Annual Savings vs Baseline: ${sim['business_value_delivered']['additional_net_profit_saved']:,}")
