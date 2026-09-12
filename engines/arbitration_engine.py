"""
Sahaay AI — Section 4.5 & Section 6: Arbitration & Orchestration Layer

The central governance innovation: A deterministic, fully auditable priority
rule hierarchy that decides what action a customer actually experiences across
potentially conflicting engine outputs (Safety/Fraud vs. Stress vs. Recommendations).

Priority Hierarchy (Strict Top-to-Bottom Order):
1. Tier 1: Safety & Fraud Protection (Highest priority — can NEVER be overridden)
2. Tier 2: Financial-Stress Support (Suppresses all promotional credit and upselling)
3. Tier 3: Customer-Requested Assistance (Answers explicit customer inquiries)
4. Tier 4: Financial-Health Guidance (Proactive budget/savings insights)
5. Tier 5: Next-Best-Action Product Recommendation (Suitability-first, non-predatory)
6. Tier 6: Do Nothing (Protects customer attention and prevents notification fatigue)
"""

from __future__ import annotations
import os
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_layer"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "engines"))

from stress_engine import compute_stress_profile
from fraud_engine import detect_customer_anomalies
from recommendation_engine import generate_recommendations

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class PriorityTier(str, Enum):
    TIER_1_SAFETY_FRAUD = "TIER_1_SAFETY_FRAUD"
    TIER_2_STRESS_SUPPORT = "TIER_2_STRESS_SUPPORT"
    TIER_3_CUSTOMER_REQUEST = "TIER_3_CUSTOMER_REQUEST"
    TIER_4_HEALTH_GUIDANCE = "TIER_4_HEALTH_GUIDANCE"
    TIER_5_PRODUCT_RECOMMENDATION = "TIER_5_PRODUCT_RECOMMENDATION"
    TIER_6_DO_NOTHING = "TIER_6_DO_NOTHING"


class ActionType(str, Enum):
    SECURITY_CONFIRMATION_REQUIRED = "SECURITY_CONFIRMATION_REQUIRED"
    FINANCIAL_STRESS_INTERVENTION = "FINANCIAL_STRESS_INTERVENTION"
    AFFORDABILITY_ASSESSMENT_AND_OPTIONS = "AFFORDABILITY_ASSESSMENT_AND_OPTIONS"
    FINANCIAL_HEALTH_INSIGHT = "FINANCIAL_HEALTH_INSIGHT"
    SUITABLE_PRODUCT_RECOMMENDATION = "SUITABLE_PRODUCT_RECOMMENDATION"
    DO_NOTHING = "DO_NOTHING"


def arbitrate(
    customer_id: str,
    fraud_output: Dict[str, Any],
    stress_output: Dict[str, Any],
    recommendation_output: Dict[str, Any],
    customer_intent: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Core arbitration function: Applies the priority decision table top-to-bottom.
    Auditable by inspection in under 30 seconds.
    """
    audit_trail: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []
    context = context or {}

    fraud_anomaly = bool(fraud_output.get("has_anomaly", False))
    max_fraud_score = float(fraud_output.get("max_anomaly_score", 0.0))
    stress_band = stress_output.get("band", "stable")
    stress_score = float(stress_output.get("score", 0.0))
    top_recommendations = recommendation_output.get("top_recommendations", [])

    # --------------------------------------------------------------------------
    # Rule 1: Tier 1 — Safety & Fraud Anomaly (Top Priority)
    # --------------------------------------------------------------------------
    if fraud_anomaly or max_fraud_score >= 0.70:
        audit_trail.append({
            "rule": "RULE_1_FRAUD_SECURITY_OVERRIDE",
            "tier": PriorityTier.TIER_1_SAFETY_FRAUD.value,
            "condition": f"has_anomaly={fraud_anomaly}, max_score={max_fraud_score}",
            "decision": "TRIGGERED",
        })

        if top_recommendations:
            suppressed.append({
                "type": "RECOMMENDATIONS",
                "items": top_recommendations,
                "reason": "Suppressed due to active security anomaly burst. Security outranks all promotional actions.",
            })
        if stress_band in ("concern", "high_concern"):
            suppressed.append({
                "type": "STRESS_SUPPORT_CONVERSATION",
                "reason": "Delayed in favor of immediate transactional security verification.",
            })

        flagged_signals = fraud_output.get("flagged_signals", ["unusual_device_or_location"])
        return {
            "customer_id": customer_id,
            "final_action": ActionType.SECURITY_CONFIRMATION_REQUIRED.value,
            "priority_tier_applied": PriorityTier.TIER_1_SAFETY_FRAUD.value,
            "deciding_engine": "fraud_engine",
            "action_headline": "Security Verification Needed",
            "action_description": "We detected rapid unusual transactions from a new device or unfamiliar location. Please confirm if this was you.",
            "action_payload": {
                "max_anomaly_score": max_fraud_score,
                "flagged_signals": flagged_signals,
                "flagged_transactions": fraud_output.get("flagged_transactions", []),
                "action_prompt": "Did you authorize these transfers?",
                "support_route": "/security/confirm",
            },
            "suppressed_actions": suppressed,
            "audit_trail": audit_trail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    audit_trail.append({
        "rule": "RULE_1_FRAUD_SECURITY_OVERRIDE",
        "tier": PriorityTier.TIER_1_SAFETY_FRAUD.value,
        "decision": "PASSED (No active fraud anomaly)",
    })

    # --------------------------------------------------------------------------
    # Rule 2: Tier 2 — Financial Distress (The Anti-Ramesh Safeguard)
    # --------------------------------------------------------------------------
    if stress_band in ("concern", "high_concern") or stress_score >= 0.45:
        audit_trail.append({
            "rule": "RULE_2_FINANCIAL_STRESS_INTERVENTION",
            "tier": PriorityTier.TIER_2_STRESS_SUPPORT.value,
            "condition": f"stress_band={stress_band}, score={stress_score}",
            "decision": "TRIGGERED",
        })

        # Hard constraint: Suppress all promotional credit, credit cards, loans, investments
        predatory_products = [p for p in top_recommendations if p in ("personal_loan", "credit_card", "mutual_fund_sip")]
        if predatory_products:
            suppressed.append({
                "type": "UNSUITABLE_CREDIT_AND_INVESTMENT_PRODUCTS",
                "items": predatory_products,
                "reason": "Ethical safeguard: High-risk credit and investment marketing is prohibited for financially stressed customers.",
            })

        drivers = stress_output.get("drivers", ["missed_or_delayed_emis"])
        return {
            "customer_id": customer_id,
            "final_action": ActionType.FINANCIAL_STRESS_INTERVENTION.value,
            "priority_tier_applied": PriorityTier.TIER_2_STRESS_SUPPORT.value,
            "deciding_engine": "stress_engine",
            "action_headline": "Financial Health Support & EMI Assistance",
            "action_description": "We noticed upcoming payment pressure and reduced cash flow. Let's look at restructuring your EMIs or reviewing relief options.",
            "action_payload": {
                "stress_band": stress_band,
                "stress_score": stress_score,
                "primary_drivers": drivers,
                "recommended_relief_options": [
                    "emi_restructuring",
                    "financial_health_checkin",
                    "grace_period_extension",
                ],
                "support_route": "/support/restructure",
            },
            "suppressed_actions": suppressed,
            "audit_trail": audit_trail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    audit_trail.append({
        "rule": "RULE_2_FINANCIAL_STRESS_INTERVENTION",
        "tier": PriorityTier.TIER_2_STRESS_SUPPORT.value,
        "decision": "PASSED (Customer financial health is stable)",
    })

    # --------------------------------------------------------------------------
    # Rule 3: Tier 3 — Customer Explicit Request / Intent
    # --------------------------------------------------------------------------
    if customer_intent in ("loan_inquiry", "credit_card_inquiry", "affordability_check"):
        audit_trail.append({
            "rule": "RULE_3_EXPLICIT_CUSTOMER_INQUIRY",
            "tier": PriorityTier.TIER_3_CUSTOMER_REQUEST.value,
            "condition": f"customer_intent={customer_intent}",
            "decision": "TRIGGERED",
        })

        return {
            "customer_id": customer_id,
            "final_action": ActionType.AFFORDABILITY_ASSESSMENT_AND_OPTIONS.value,
            "priority_tier_applied": PriorityTier.TIER_3_CUSTOMER_REQUEST.value,
            "deciding_engine": "conversational_intent_handler",
            "action_headline": "Affordability Review & Available Options",
            "action_description": "Here is an objective affordability calculation showing the total loan cost, APR, and realistic EMI breakdown for your request.",
            "action_payload": {
                "inquiry_type": customer_intent,
                "affordability_status": "APPROVED_SAFE_BAND" if stress_band == "stable" else "CAUTION_BORDERLINE",
                "suitable_options": top_recommendations,
                "anti_predatory_disclosure": "Total repayment schedule with zero hidden charges provided prior to application.",
            },
            "suppressed_actions": suppressed,
            "audit_trail": audit_trail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    audit_trail.append({
        "rule": "RULE_3_EXPLICIT_CUSTOMER_INQUIRY",
        "tier": PriorityTier.TIER_3_CUSTOMER_REQUEST.value,
        "decision": "PASSED (No explicit user product inquiry pending)",
    })

    # --------------------------------------------------------------------------
    # Rule 4: Tier 4 — Financial Health Guidance / Budget Insight
    # --------------------------------------------------------------------------
    if stress_band == "watch" or context.get("has_upcoming_emi_due"):
        audit_trail.append({
            "rule": "RULE_4_FINANCIAL_HEALTH_GUIDANCE",
            "tier": PriorityTier.TIER_4_HEALTH_GUIDANCE.value,
            "condition": f"stress_band={stress_band}",
            "decision": "TRIGGERED",
        })

        return {
            "customer_id": customer_id,
            "final_action": ActionType.FINANCIAL_HEALTH_INSIGHT.value,
            "priority_tier_applied": PriorityTier.TIER_4_HEALTH_GUIDANCE.value,
            "deciding_engine": "stress_engine",
            "action_headline": "Monthly Cash Flow & Savings Insight",
            "action_description": "Keep an eye on upcoming bills this week to maintain your positive savings cushion.",
            "action_payload": {
                "stress_band": "watch",
                "insight_type": "cash_flow_optimization",
                "suggested_actions": ["set_aside_emi_buffer", "review_discretionary_spend"],
            },
            "suppressed_actions": suppressed,
            "audit_trail": audit_trail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    audit_trail.append({
        "rule": "RULE_4_FINANCIAL_HEALTH_GUIDANCE",
        "tier": PriorityTier.TIER_4_HEALTH_GUIDANCE.value,
        "decision": "PASSED",
    })

    # --------------------------------------------------------------------------
    # Rule 5: Tier 5 — Next-Best-Action Product Recommendation
    # --------------------------------------------------------------------------
    suitable_recs = [
        r for r in recommendation_output.get("recommended_products", [])
        if r.get("is_suitable") and r.get("confidence", 0) >= 0.60
    ]

    if suitable_recs:
        top_rec = suitable_recs[0]
        audit_trail.append({
            "rule": "RULE_5_SUITABLE_RECOMMENDATION",
            "tier": PriorityTier.TIER_5_PRODUCT_RECOMMENDATION.value,
            "condition": f"top_product={top_rec['product_id']}, confidence={top_rec['confidence']}",
            "decision": "TRIGGERED",
        })

        return {
            "customer_id": customer_id,
            "final_action": ActionType.SUITABLE_PRODUCT_RECOMMENDATION.value,
            "priority_tier_applied": PriorityTier.TIER_5_PRODUCT_RECOMMENDATION.value,
            "deciding_engine": "recommendation_engine",
            "action_headline": f"Personalized Recommendation: {top_rec['product_id'].replace('_', ' ').title()}",
            "action_description": "Based on your verified income consistency and healthy savings track record, this product offers genuine utility.",
            "action_payload": {
                "product_id": top_rec["product_id"],
                "suitability_score": top_rec["suitability_score"],
                "path_used": recommendation_output.get("path_used", "xgboost"),
                "primary_reasons": top_rec["primary_reasons"],
                "all_suitable_products": [r["product_id"] for r in suitable_recs],
                "opt_out_available": True,
            },
            "suppressed_actions": suppressed,
            "audit_trail": audit_trail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # --------------------------------------------------------------------------
    # Rule 6: Tier 6 — Do Nothing (Respect User Attention)
    # --------------------------------------------------------------------------
    audit_trail.append({
        "rule": "RULE_6_DO_NOTHING_DEFAULT",
        "tier": PriorityTier.TIER_6_DO_NOTHING.value,
        "decision": "TRIGGERED",
    })

    return {
        "customer_id": customer_id,
        "final_action": ActionType.DO_NOTHING.value,
        "priority_tier_applied": PriorityTier.TIER_6_DO_NOTHING.value,
        "deciding_engine": "default_safeguard",
        "action_headline": "No Action Needed",
        "action_description": "Everything is on track. No unnecessary notifications or unsolicited offers pushed.",
        "action_payload": {},
        "suppressed_actions": suppressed,
        "audit_trail": audit_trail,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def arbitrate_customer(
    customer_id: str,
    customer_intent: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience orchestrator: Fetches profile and transaction history for a
    customer, queries all three engines, and computes the arbitrated outcome.
    """
    profile_path = os.path.join(DATA_DIR, "customer_profile.csv")
    if not os.path.exists(profile_path):
        raise FileNotFoundError("customer_profile.csv not found.")

    profiles = pd.read_csv(profile_path)
    cust_row = profiles[profiles["customer_id"] == customer_id]
    if cust_row.empty:
        raise ValueError(f"Unknown customer_id: {customer_id}")

    profile_dict = cust_row.iloc[0].to_dict()

    # Query the 3 intelligence engines
    fraud_res = detect_customer_anomalies(customer_id)
    stress_res = compute_stress_profile(profile_dict)
    rec_res = generate_recommendations(profile_dict)

    # Execute Arbitration
    outcome = arbitrate(
        customer_id=customer_id,
        fraud_output=fraud_res,
        stress_output=stress_res,
        recommendation_output=rec_res,
        customer_intent=customer_intent,
        context=context,
    )

    # Attach top-level convenience summaries for frontend telemetry
    cov = rec_res.get("data_coverage_score") or stress_res.get("data_coverage_score", 1.0)
    note = rec_res.get("data_coverage_note") or stress_res.get("data_coverage_note")
    conf = rec_res.get("overall_confidence") or stress_res.get("overall_confidence", {
        "overall_confidence_score": 0.88,
        "overall_confidence_band": "HIGH",
        "data_coverage_score": cov,
        "data_coverage_note": note,
    })

    outcome["path_used"] = rec_res.get("path_used", "xgboost")
    audit = outcome.get("audit_trail", [])
    outcome["decision_rule_used"] = audit[-1].get("rule", "RULE_GOVERNANCE") if audit else "RULE_GOVERNANCE"
    outcome["action_body"] = outcome.get("action_description", "")
    outcome["stress_summary"] = stress_res
    outcome["fraud_summary"] = fraud_res
    outcome["recommendation_summary"] = rec_res
    outcome["overall_confidence"] = conf
    outcome["data_coverage_score"] = cov
    outcome["data_coverage_note"] = note
    outcome["arbitration_audit_trail"] = [
        {
            "tier": int(step.get("tier", "5")[-1]) if step.get("tier") and step.get("tier")[-1].isdigit() else 5,
            "name": step.get("rule", "").replace("_", " ").title(),
            "passed": step.get("decision") == "TRIGGERED",
            "reason": step.get("condition", step.get("decision", "")),
        }
        for step in audit
    ]
    return outcome
