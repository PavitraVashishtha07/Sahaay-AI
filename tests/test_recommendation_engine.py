"""
Sahaay AI — Tests for Section 4.2: Recommendation Engine

Verifies:
1. Gig-worker stress test: Gig/irregular income customers are recommended
   flexible_micro_credit & recurring_deposit_small_ticket and not suppressed.
2. Financially stressed customers receive relief products (health checkin,
   restructuring) and avoid high-risk credit.
3. Thin-history / cold-start customers route to gmm_fallback.
4. path_used field is explicitly present in all outputs.
5. SHAP feature explanations are attached to XGBoost recommendations.
6. Pattern A experiment executes and documents lift/delta.
"""

import os
import sys
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "engines"))
from recommendation_engine import (
    generate_recommendations,
    route_customer,
    run_pattern_a_experiment,
    train_models,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _get_profiles_and_labels():
    cp_path = os.path.join(DATA_DIR, "customer_profile.csv")
    gt_path = os.path.join(DATA_DIR, "ground_truth_labels.csv")
    assert os.path.exists(cp_path) and os.path.exists(gt_path)
    cp = pd.read_csv(cp_path)
    gt = pd.read_csv(gt_path)
    return cp, gt


@pytest.fixture(scope="module")
def dataset():
    return _get_profiles_and_labels()


def test_gig_worker_stress_test(dataset):
    """Gig workers must receive micro-credit / small-ticket products, not be suppressed."""
    cp, _ = dataset
    gig_customers = cp[cp["archetype_ground_truth"] == "gig_irregular"].iloc[:50]

    for _, row in gig_customers.iterrows():
        recs = generate_recommendations(row)
        assert recs["path_used"] == "xgboost"
        top = recs["top_recommendations"]
        assert "flexible_micro_credit" in top or "recurring_deposit_small_ticket" in top, (
            f"Gig worker {row['customer_id']} did not receive flexible micro-credit or RD small-ticket. Got: {top}"
        )


def test_financially_stressed_customer_relief(dataset):
    """Stressed customers must receive relief recommendations, not high-risk credit."""
    cp, _ = dataset
    stressed_customers = cp[cp["archetype_ground_truth"] == "financially_stressed"].iloc[:50]

    for _, row in stressed_customers.iterrows():
        recs = generate_recommendations(row)
        top = recs["top_recommendations"]
        unsuitable = recs["unsuitable_products"]
        assert "financial_health_checkin" in top or "emi_restructuring" in top
        assert "mutual_fund_sip" in unsuitable or "credit_card" in unsuitable


def test_thin_history_routes_to_gmm_fallback():
    """A customer with <30 days history or marked thin-history must route to gmm_fallback."""
    thin_profile = {
        "customer_id": "cust_cold_start_999",
        "age": 26,
        "avg_monthly_income": 35000,
        "income_type_declared": "gig_irregular",
        "tier": "Tier 2",
        "gender": "Female",
        "is_thin_history": True,
        "income_credit_count": 1,
        "txn_velocity_last_7d": 1,
        "essential_spend_share": 0.55,
        "avg_monthly_savings_rate": 0.25,
    }
    recs = generate_recommendations(thin_profile)
    assert recs["path_used"] == "gmm_fallback"
    assert "demographic_peer_group_assignment" in recs["recommended_products"][0]["primary_reasons"]
    assert len(recs["top_recommendations"]) > 0


def test_path_used_field_present(dataset):
    cp, _ = dataset
    sample = cp.iloc[0]
    recs_xgb = generate_recommendations(sample)
    assert "path_used" in recs_xgb
    assert recs_xgb["path_used"] in ("xgboost", "gmm_fallback")


def test_shap_explanations_present(dataset):
    cp, _ = dataset
    sample = cp[cp["archetype_ground_truth"] == "stable_salaried"].iloc[0]
    recs = generate_recommendations(sample)
    assert recs["path_used"] == "xgboost"
    for prod in recs["recommended_products"]:
        assert len(prod["primary_reasons"]) > 0
        assert isinstance(prod["primary_reasons"], list)


def test_pattern_a_experiment_runs():
    exp_res = run_pattern_a_experiment()
    assert "summary" in exp_res
    assert "mean_f1_delta" in exp_res
    assert "detailed_comparison" in exp_res
    assert len(exp_res["detailed_comparison"]) == 8


if __name__ == "__main__":
    cp, gt = _get_profiles_and_labels()
    test_gig_worker_stress_test((cp, gt))
    print("PASS: test_gig_worker_stress_test")
    test_financially_stressed_customer_relief((cp, gt))
    print("PASS: test_financially_stressed_customer_relief")
    test_thin_history_routes_to_gmm_fallback()
    print("PASS: test_thin_history_routes_to_gmm_fallback")
    test_path_used_field_present((cp, gt))
    print("PASS: test_path_used_field_present")
    test_shap_explanations_present((cp, gt))
    print("PASS: test_shap_explanations_present")
    test_pattern_a_experiment_runs()
    print("PASS: test_pattern_a_experiment_runs")
    print("\nAll Recommendation Engine tests PASSED!")
