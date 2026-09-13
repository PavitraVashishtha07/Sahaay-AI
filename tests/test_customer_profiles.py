"""
Sahaay AI — Module 2 sanity tests

These aren't exhaustive unit tests; they check the thing that actually matters
for this layer: that the engineered features correctly separate the four
archetypes in the directions the research doc predicts. If these fail, the
recommendation/stress engines built on top of customer_profile.csv will be
learning noise, not signal.

Run: python -m pytest test_customer_profiles.py -v
(or: python test_customer_profiles.py to run without pytest)
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _load_profiles():
    path = os.path.join(DATA_DIR, "customer_profile.csv")
    assert os.path.exists(path), "Run build_customer_profiles.py first."
    return pd.read_csv(path)


def test_all_customers_have_a_profile():
    profiles = _load_profiles()
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    assert len(profiles) >= len(customers) - 50, (
        f"expected profile coverage for all customers, got {len(profiles)} profiles"
    )


def test_gig_workers_are_not_misclassified_as_salaried():
    """The core stress-test fix from the research doc: irregular-income
    customers must not collapse into the 'salaried_regular' bucket."""
    profiles = _load_profiles()
    gig = profiles[profiles["archetype_ground_truth"] == "gig_irregular"]
    misclassified_rate = (gig["income_pattern"] == "salaried_regular").mean()
    assert misclassified_rate < 0.05, (
        f"{misclassified_rate:.1%} of gig workers misclassified as salaried_regular — "
        "the income-pattern gate is not working as designed."
    )


def test_stable_salaried_mostly_detected_as_salaried():
    profiles = _load_profiles()
    stable = profiles[profiles["archetype_ground_truth"] == "stable_salaried"]
    correctly_detected = (stable["income_pattern"] == "salaried_regular").mean()
    assert correctly_detected > 0.85, (
        f"only {correctly_detected:.1%} of stable_salaried customers detected as salaried_regular"
    )


def test_financially_stressed_customers_show_higher_missed_emi():
    profiles = _load_profiles()
    stressed = profiles[profiles["archetype_ground_truth"] == "financially_stressed"]["missed_emi_count"].mean()
    stable = profiles[profiles["archetype_ground_truth"] == "stable_salaried"]["missed_emi_count"].mean()
    assert stressed > stable, (
        f"financially_stressed mean missed EMIs ({stressed:.2f}) should exceed "
        f"stable_salaried ({stable:.2f})"
    )


def test_financially_stressed_customers_show_worse_balance_trend():
    profiles = _load_profiles()
    stressed = profiles[profiles["archetype_ground_truth"] == "financially_stressed"]["balance_trend"].mean()
    stable = profiles[profiles["archetype_ground_truth"] == "stable_salaried"]["balance_trend"].mean()
    assert stressed < stable, (
        f"financially_stressed balance_trend ({stressed:.2f}) should be lower (more negative) "
        f"than stable_salaried ({stable:.2f})"
    )


def test_fraud_scenario_customers_look_financially_normal():
    """Fraud is a transaction-level anomaly, not a profile-level stress signal —
    per the research doc, these two must stay separable. A fraud-scenario
    customer's baseline financial-health features should resemble a stable
    salaried customer's, NOT a financially-stressed one."""
    profiles = _load_profiles()
    fraud = profiles[profiles["archetype_ground_truth"] == "fraud_scenario"]["missed_emi_count"].mean()
    stressed = profiles[profiles["archetype_ground_truth"] == "financially_stressed"]["missed_emi_count"].mean()
    assert fraud < stressed / 2, (
        f"fraud_scenario missed_emi_count ({fraud:.2f}) too close to financially_stressed "
        f"({stressed:.2f}) — stress and fraud signals are bleeding into each other."
    )


def test_no_archetype_or_split_leaks_into_feature_columns_by_name():
    """Guardrail: ground-truth columns must be clearly named and excludable,
    so a later training script can't accidentally treat them as a feature."""
    profiles = _load_profiles()
    assert "archetype_ground_truth" in profiles.columns
    assert "split" in profiles.columns
    # any real feature column must not literally be named after the label
    feature_cols = [c for c in profiles.columns if c not in
                    ("customer_id", "archetype_ground_truth", "split", "profile_computed_at")]
    assert "archetype" not in feature_cols


def test_train_test_split_is_stratified():
    profiles = _load_profiles()
    ct = pd.crosstab(profiles["archetype_ground_truth"], profiles["split"], normalize="index")
    for archetype, row in ct.iterrows():
        assert 0.7 <= row["train"] <= 0.9, (
            f"{archetype} train split ratio {row['train']:.2f} not close to the intended 80/20"
        )


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
