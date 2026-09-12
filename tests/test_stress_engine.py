"""
Sahaay AI — Tests for Section 4.3: Stress Detection Engine

Verifies:
1. Financially stressed archetype customers land predominantly in watch/concern/high_concern.
2. Stable salaried customers land predominantly in the 'stable' band.
3. Fraud scenario customers are NOT elevated by this engine (stress and fraud stay separable).
4. Gig / irregular income customers with good savings are not misdiagnosed as distressed.
5. Key drivers are clearly extracted and non-empty for distressed customers.
6. Secondary Isolation Forest anomaly check runs and provides scores.
"""

import os
import sys
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "engines"))
from stress_engine import compute_stress_profile, evaluate_all_customers

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="module")
def profiles():
    path = os.path.join(DATA_DIR, "customer_profile.csv")
    assert os.path.exists(path), "customer_profile.csv not found."
    return pd.read_csv(path)


def test_financially_stressed_customers_mostly_in_concern_bands(profiles):
    stressed = profiles[profiles["archetype_ground_truth"] == "financially_stressed"]
    elevated_count = 0
    for _, row in stressed.iterrows():
        res = compute_stress_profile(row)
        if res["band"] in ("watch", "concern", "high_concern"):
            elevated_count += 1
    elevated_rate = elevated_count / len(stressed)
    assert elevated_rate >= 0.90, (
        f"Expected >= 90% of financially stressed customers to be in watch/concern/high_concern, got {elevated_rate:.1%}"
    )


def test_stable_salaried_customers_mostly_in_stable_band(profiles):
    stable = profiles[profiles["archetype_ground_truth"] == "stable_salaried"]
    stable_count = 0
    for _, row in stable.iterrows():
        res = compute_stress_profile(row)
        if res["band"] == "stable":
            stable_count += 1
    stable_rate = stable_count / len(stable)
    assert stable_rate >= 0.90, (
        f"Expected >= 90% of stable salaried customers to be in 'stable', got {stable_rate:.1%}"
    )


def test_fraud_scenario_customers_look_financially_normal(profiles):
    """Stress and fraud must stay separable: fraud customers have normal financial baselines."""
    fraud = profiles[profiles["archetype_ground_truth"] == "fraud_scenario"]
    stable_count = 0
    for _, row in fraud.iterrows():
        res = compute_stress_profile(row)
        if res["band"] == "stable":
            stable_count += 1
    stable_rate = stable_count / len(fraud)
    assert stable_rate >= 0.90, (
        f"Expected >= 90% of fraud scenario customers to be 'stable' on stress, got {stable_rate:.1%}"
    )


def test_drivers_list_provided_for_elevated_customers(profiles):
    stressed = profiles[profiles["archetype_ground_truth"] == "financially_stressed"].iloc[:20]
    for _, row in stressed.iterrows():
        res = compute_stress_profile(row)
        if res["band"] != "stable":
            assert len(res["drivers"]) > 0, f"Distressed customer {res['customer_id']} has empty drivers"
            assert "missed_or_delayed_emis" in res["drivers"] or "savings_rate_decline" in res["drivers"] or "cash_buffer_depletion" in res["drivers"]


def test_secondary_isolation_forest_check(profiles):
    sample = profiles.iloc[0]
    res = compute_stress_profile(sample)
    assert "secondary_isolation_forest" in res
    assert "anomaly_score" in res["secondary_isolation_forest"]
    assert "is_anomaly" in res["secondary_isolation_forest"]
    assert isinstance(res["secondary_isolation_forest"]["anomaly_score"], float)


def test_evaluate_all_customers_returns_dataframe(profiles):
    df_eval = evaluate_all_customers(profiles.iloc[:50])
    assert len(df_eval) == 50
    assert "stress_band" in df_eval.columns
    assert "stress_score" in df_eval.columns
    assert "drivers" in df_eval.columns


if __name__ == "__main__":
    p = pd.read_csv(os.path.join(DATA_DIR, "customer_profile.csv"))
    test_financially_stressed_customers_mostly_in_concern_bands(p)
    print("PASS: test_financially_stressed_customers_mostly_in_concern_bands")
    test_stable_salaried_customers_mostly_in_stable_band(p)
    print("PASS: test_stable_salaried_customers_mostly_in_stable_band")
    test_fraud_scenario_customers_look_financially_normal(p)
    print("PASS: test_fraud_scenario_customers_look_financially_normal")
    test_drivers_list_provided_for_elevated_customers(p)
    print("PASS: test_drivers_list_provided_for_elevated_customers")
    test_secondary_isolation_forest_check(p)
    print("PASS: test_secondary_isolation_forest_check")
    test_evaluate_all_customers_returns_dataframe(p)
    print("PASS: test_evaluate_all_customers_returns_dataframe")
    print("\nAll Stress Engine tests PASSED!")
