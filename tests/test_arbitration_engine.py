"""
Sahaay AI — Tests for Section 4.5 & Section 6: Arbitration & Orchestration Layer

Verifies every row of the decision table and priority rules:
1. Tier 1 Supremacy: Fraud anomaly overrides everything else.
2. Anti-Ramesh Safeguard (Tier 2): High financial distress suppresses all promotional credit.
3. Explicit Intent (Tier 3): Explicit user inquiries trigger affordability assessment.
4. Health Guidance (Tier 4): Watch-band customers receive cash flow / savings guidance.
5. Next Best Action (Tier 5): Stable customers receive suitable recommendations with SHAP reasons.
6. Do-Nothing Default (Tier 6): System defaults to no action when no clear benefit exists.
7. Full Audit Trail: Audit logs are populated for governance inspection.
"""

import os
import sys
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "engines"))
from arbitration_engine import (
    arbitrate,
    arbitrate_customer,
    PriorityTier,
    ActionType,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def test_tier_1_fraud_overrides_everything():
    fraud_out = {
        "has_anomaly": True,
        "max_anomaly_score": 0.95,
        "flagged_signals": ["new_device", "amount_burst_spike", "imps_channel"],
        "flagged_transactions": [{"txn_id": "t1", "amount": 45000}],
    }
    stress_out = {
        "band": "high_concern",
        "score": 0.85,
        "drivers": ["missed_or_delayed_emis"],
    }
    rec_out = {
        "path_used": "xgboost",
        "top_recommendations": ["mutual_fund_sip", "credit_card"],
        "recommended_products": [{"product_id": "credit_card", "is_suitable": True, "confidence": 0.95}],
    }

    res = arbitrate("cust_fraud_test", fraud_out, stress_out, rec_out)
    assert res["final_action"] == ActionType.SECURITY_CONFIRMATION_REQUIRED.value
    assert res["priority_tier_applied"] == PriorityTier.TIER_1_SAFETY_FRAUD.value
    assert len(res["suppressed_actions"]) > 0
    # Audit trail verifies Rule 1 triggered
    assert res["audit_trail"][0]["rule"] == "RULE_1_FRAUD_SECURITY_OVERRIDE"
    assert res["audit_trail"][0]["decision"] == "TRIGGERED"


def test_tier_2_stress_suppresses_predatory_credit():
    """Anti-Ramesh scenario: A stressed customer must NEVER receive promotional loans or cards."""
    fraud_out = {"has_anomaly": False, "max_anomaly_score": 0.15}
    stress_out = {
        "band": "high_concern",
        "score": 0.78,
        "drivers": ["missed_or_delayed_emis", "cash_buffer_depletion"],
    }
    rec_out = {
        "path_used": "xgboost",
        "top_recommendations": ["credit_card", "personal_loan", "financial_health_checkin"],
        "recommended_products": [
            {"product_id": "credit_card", "is_suitable": True, "confidence": 0.88},
            {"product_id": "financial_health_checkin", "is_suitable": True, "confidence": 0.90},
        ],
    }

    res = arbitrate("cust_stressed_test", fraud_out, stress_out, rec_out)
    assert res["final_action"] == ActionType.FINANCIAL_STRESS_INTERVENTION.value
    assert res["priority_tier_applied"] == PriorityTier.TIER_2_STRESS_SUPPORT.value
    assert res["deciding_engine"] == "stress_engine"

    # Verify credit was suppressed
    suppressed_types = [s["type"] for s in res["suppressed_actions"]]
    assert "UNSUITABLE_CREDIT_AND_INVESTMENT_PRODUCTS" in suppressed_types


def test_tier_3_customer_explicit_inquiry():
    fraud_out = {"has_anomaly": False, "max_anomaly_score": 0.10}
    stress_out = {"band": "stable", "score": 0.12}
    rec_out = {
        "path_used": "xgboost",
        "top_recommendations": ["personal_loan", "credit_card"],
        "recommended_products": [{"product_id": "personal_loan", "is_suitable": True, "confidence": 0.85}],
    }

    res = arbitrate(
        "cust_inquiry_test",
        fraud_out,
        stress_out,
        rec_out,
        customer_intent="loan_inquiry",
    )
    assert res["final_action"] == ActionType.AFFORDABILITY_ASSESSMENT_AND_OPTIONS.value
    assert res["priority_tier_applied"] == PriorityTier.TIER_3_CUSTOMER_REQUEST.value
    assert "anti_predatory_disclosure" in res["action_payload"]


def test_tier_4_watch_state_guidance():
    fraud_out = {"has_anomaly": False, "max_anomaly_score": 0.10}
    stress_out = {"band": "watch", "score": 0.32, "drivers": ["savings_rate_decline"]}
    rec_out = {"path_used": "xgboost", "top_recommendations": [], "recommended_products": []}

    res = arbitrate("cust_watch_test", fraud_out, stress_out, rec_out)
    assert res["final_action"] == ActionType.FINANCIAL_HEALTH_INSIGHT.value
    assert res["priority_tier_applied"] == PriorityTier.TIER_4_HEALTH_GUIDANCE.value


def test_tier_5_stable_customer_recommendation():
    fraud_out = {"has_anomaly": False, "max_anomaly_score": 0.10}
    stress_out = {"band": "stable", "score": 0.08}
    rec_out = {
        "path_used": "xgboost",
        "top_recommendations": ["mutual_fund_sip", "term_insurance"],
        "recommended_products": [
            {
                "product_id": "mutual_fund_sip",
                "suitability_score": 0.98,
                "confidence": 0.98,
                "is_suitable": True,
                "primary_reasons": ["healthy_savings_rate", "stable_income_level"],
            }
        ],
    }

    res = arbitrate("cust_stable_test", fraud_out, stress_out, rec_out)
    assert res["final_action"] == ActionType.SUITABLE_PRODUCT_RECOMMENDATION.value
    assert res["priority_tier_applied"] == PriorityTier.TIER_5_PRODUCT_RECOMMENDATION.value
    assert res["action_payload"]["product_id"] == "mutual_fund_sip"
    assert "healthy_savings_rate" in res["action_payload"]["primary_reasons"]


def test_tier_6_do_nothing_default():
    fraud_out = {"has_anomaly": False, "max_anomaly_score": 0.05}
    stress_out = {"band": "stable", "score": 0.05}
    rec_out = {"path_used": "xgboost", "top_recommendations": [], "recommended_products": []}

    res = arbitrate("cust_idle_test", fraud_out, stress_out, rec_out)
    assert res["final_action"] == ActionType.DO_NOTHING.value
    assert res["priority_tier_applied"] == PriorityTier.TIER_6_DO_NOTHING.value


def test_arbitrate_customer_end_to_end():
    gt = pd.read_csv(os.path.join(DATA_DIR, "ground_truth_labels.csv"))
    fraud_cust = gt[gt["has_fraud_event"]].iloc[0]["customer_id"]
    stressed_cust = gt[gt["stress_label"] == "high_concern"].iloc[0]["customer_id"]
    stable_cust = gt[(gt["stress_label"] == "stable") & (~gt["has_fraud_event"])].iloc[0]["customer_id"]

    res_f = arbitrate_customer(fraud_cust)
    assert res_f["priority_tier_applied"] == PriorityTier.TIER_1_SAFETY_FRAUD.value

    res_s = arbitrate_customer(stressed_cust)
    assert res_s["priority_tier_applied"] == PriorityTier.TIER_2_STRESS_SUPPORT.value

    res_st = arbitrate_customer(stable_cust)
    assert res_st["priority_tier_applied"] in (
        PriorityTier.TIER_5_PRODUCT_RECOMMENDATION.value,
        PriorityTier.TIER_6_DO_NOTHING.value,
    )


if __name__ == "__main__":
    test_tier_1_fraud_overrides_everything()
    print("PASS: test_tier_1_fraud_overrides_everything")
    test_tier_2_stress_suppresses_predatory_credit()
    print("PASS: test_tier_2_stress_suppresses_predatory_credit")
    test_tier_3_customer_explicit_inquiry()
    print("PASS: test_tier_3_customer_explicit_inquiry")
    test_tier_4_watch_state_guidance()
    print("PASS: test_tier_4_watch_state_guidance")
    test_tier_5_stable_customer_recommendation()
    print("PASS: test_tier_5_stable_customer_recommendation")
    test_tier_6_do_nothing_default()
    print("PASS: test_tier_6_do_nothing_default")
    test_arbitrate_customer_end_to_end()
    print("PASS: test_arbitrate_customer_end_to_end")
    print("\nAll Arbitration Engine tests PASSED!")
