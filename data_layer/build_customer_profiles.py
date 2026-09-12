"""
Sahaay AI — Module 2: Unified Customer Intelligence Layer

The "shared brain." This script reads raw transactions + EMI records (via the
Module 1 data layer) and computes ONE feature-engineered profile per customer.

Every downstream engine (recommendation, conversational, stress/fraud,
arbitration) reads ONLY from customer_profile.csv (plus raw transactions where
an engine genuinely needs transaction-level detail, e.g. fraud anomaly
detection) — never maintains its own private view of the customer. That rule
is the entire point of this layer; see Section 2 of the feature-breakdown doc.

Deliberately NOT a learned embedding — hand-engineered, explainable features,
because a few thousand synthetic customers is too little data to train a
meaningful embedding, and explainability is a judged criterion.

Run:
    python build_customer_profiles.py --out ../data/customer_profile.csv
"""

from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_layer"))
from aa_interface import DATA_DIR, ConsentError

pd.options.mode.chained_assignment = None


# --------------------------------------------------------------------------
# Per-customer feature computation
# --------------------------------------------------------------------------

def _to_df(records, date_cols=None):
    df = pd.DataFrame(records)
    if df.empty:
        return df
    if date_cols:
        for c in date_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
    return df


def compute_salary_consistency(txns: pd.DataFrame) -> dict:
    if txns.empty or "txn_type" not in txns.columns:
        return {
            "income_pattern": "unknown", "avg_income_per_credit": 0.0,
            "income_credit_count": 0, "income_interval_days_mean": np.nan,
            "income_interval_days_std": np.nan, "income_amount_cv": np.nan,
        }
    credits = txns[txns["txn_type"] == "credit"]
    if credits.empty:
        return {
            "income_pattern": "unknown", "avg_income_per_credit": 0.0,
            "income_credit_count": 0, "income_interval_days_mean": np.nan,
            "income_interval_days_std": np.nan, "income_amount_cv": np.nan,
        }

    credits = credits.sort_values("timestamp")
    intervals = credits["timestamp"].diff().dt.days.dropna()
    amount_mean = credits["amount"].mean()
    amount_std = credits["amount"].std(ddof=0) or 0.0
    amount_cv = (amount_std / amount_mean) if amount_mean else 0.0  # coefficient of variation

    interval_mean = intervals.mean() if not intervals.empty else np.nan
    interval_std = intervals.std(ddof=0) if not intervals.empty else np.nan

    # Salaried: ~monthly interval (25-35 days), low amount variability.
    # Gig/irregular: short, variable intervals, higher amount variability.
    if pd.notna(interval_mean) and 25 <= interval_mean <= 35 and amount_cv < 0.25:
        pattern = "salaried_regular"
    elif pd.notna(interval_mean) and interval_mean < 15:
        pattern = "gig_irregular"
    else:
        pattern = "mixed_or_business"

    return {
        "income_pattern": pattern,
        "avg_income_per_credit": round(amount_mean, 2),
        "income_credit_count": len(credits),
        "income_interval_days_mean": round(interval_mean, 2) if pd.notna(interval_mean) else None,
        "income_interval_days_std": round(interval_std, 2) if pd.notna(interval_std) else None,
        "income_amount_cv": round(amount_cv, 3),
    }


def compute_rolling_income_stability(txns: pd.DataFrame, window_days=30) -> dict:
    if txns.empty or "txn_type" not in txns.columns:
        return {"rolling_income_30d_mean": 0.0, "rolling_income_30d_std": 0.0}
    credits = txns[txns["txn_type"] == "credit"].sort_values("timestamp")
    if credits.empty:
        return {"rolling_income_30d_mean": 0.0, "rolling_income_30d_std": 0.0}

    last_date = credits["timestamp"].max()
    window_start = last_date - pd.Timedelta(days=window_days)
    recent = credits[credits["timestamp"] >= window_start]
    return {
        "rolling_income_30d_mean": round(recent["amount"].sum(), 2),
        "rolling_income_30d_std": round(recent["amount"].std(ddof=0), 2) if len(recent) > 1 else 0.0,
    }


def compute_savings_rate_and_trend(txns: pd.DataFrame) -> dict:
    if txns.empty or "txn_type" not in txns.columns or "timestamp" not in txns.columns:
        return {"avg_monthly_savings_rate": 0.0, "savings_rate_trend": 0.0, "balance_trend": 0.0}

    txns = txns.copy()
    txns["month"] = txns["timestamp"].dt.to_period("M")
    monthly = txns.groupby(["month", "txn_type"])["amount"].sum().unstack(fill_value=0)
    monthly = monthly.reindex(columns=["credit", "debit"], fill_value=0).sort_index()
    monthly["savings_rate"] = np.where(
        monthly["credit"] > 0, (monthly["credit"] - monthly["debit"]) / monthly["credit"], 0.0
    )
    monthly["running_balance"] = (monthly["credit"] - monthly["debit"]).cumsum()

    avg_rate = monthly["savings_rate"].mean()

    # simple linear trend (slope) over the sequence of monthly savings rates
    if len(monthly) >= 2:
        x = np.arange(len(monthly))
        y = monthly["savings_rate"].values
        slope = np.polyfit(x, y, 1)[0]
        bal_x = np.arange(len(monthly))
        bal_y = monthly["running_balance"].values
        bal_slope = np.polyfit(bal_x, bal_y, 1)[0]
    else:
        slope = 0.0
        bal_slope = 0.0

    return {
        "avg_monthly_savings_rate": round(float(avg_rate), 4),
        "savings_rate_trend": round(float(slope), 5),
        "balance_trend": round(float(bal_slope), 2),
    }


def compute_emi_features(emis: pd.DataFrame, avg_monthly_income: float) -> dict:
    if emis.empty:
        return {
            "has_active_loans": False, "num_active_loans": 0, "total_monthly_emi": 0.0,
            "emi_to_income_ratio": 0.0, "missed_emi_count": 0, "late_emi_count": 0,
            "emi_on_time_rate": 1.0, "max_days_late": 0,
        }

    total_monthly_emi = emis.groupby("loan_type")["amount_due"].mean().sum()
    missed = int((emis["status"] == "missed").sum())
    late = int((emis["status"] == "paid_late").sum())
    on_time = int((emis["status"] == "paid_on_time").sum())
    total_settled = missed + late + on_time
    on_time_rate = (on_time / total_settled) if total_settled else 1.0
    emi_to_income = (total_monthly_emi / avg_monthly_income) if avg_monthly_income else 0.0

    return {
        "has_active_loans": True,
        "num_active_loans": emis["loan_type"].nunique(),
        "total_monthly_emi": round(total_monthly_emi, 2),
        "emi_to_income_ratio": round(min(emi_to_income, 5.0), 3),  # capped to avoid outlier blowup
        "missed_emi_count": missed,
        "late_emi_count": late,
        "emi_on_time_rate": round(on_time_rate, 3),
        "max_days_late": int(emis["days_late"].max()) if "days_late" in emis else 0,
    }


def compute_spend_composition(txns: pd.DataFrame) -> dict:
    if txns.empty or "txn_type" not in txns.columns:
        return {
            "essential_spend_share": 0.0, "discretionary_spend_share": 0.0, "cash_withdrawal_share": 0.0
        }
    debits = txns[txns["txn_type"] == "debit"]
    total_debit = debits["amount"].sum()
    essential_cats = {"grocery", "utilities", "rent", "healthcare", "emi_payment"}
    discretionary_cats = {"entertainment", "ecommerce", "fuel"}

    essential_spend = debits[debits["merchant_category"].isin(essential_cats)]["amount"].sum()
    discretionary_spend = debits[debits["merchant_category"].isin(discretionary_cats)]["amount"].sum()
    cash_withdrawal = debits[debits["channel"] == "cash_withdrawal"]["amount"].sum()

    return {
        "essential_spend_share": round(essential_spend / total_debit, 3) if total_debit else 0.0,
        "discretionary_spend_share": round(discretionary_spend / total_debit, 3) if total_debit else 0.0,
        "cash_withdrawal_share": round(cash_withdrawal / total_debit, 3) if total_debit else 0.0,
    }


def compute_novelty_and_velocity(txns: pd.DataFrame) -> dict:
    if txns.empty or "timestamp" not in txns.columns or "is_new_device" not in txns.columns:
        return {
            "new_device_rate": 0.0, "new_beneficiary_rate": 0.0, "new_merchant_rate": 0.0,
            "txn_velocity_last_7d": 0, "num_distinct_devices": 0,
        }

    n = len(txns)
    last_date = txns["timestamp"].max()
    last_7d = txns[txns["timestamp"] >= last_date - pd.Timedelta(days=7)]

    return {
        "new_device_rate": round(txns["is_new_device"].mean(), 4),
        "new_beneficiary_rate": round(txns["is_new_beneficiary"].mean(), 4),
        "new_merchant_rate": round(txns["is_new_merchant"].mean(), 4),
        "txn_velocity_last_7d": int(len(last_7d)),
        "num_distinct_devices": int(txns["device_id"].nunique()),
    }


def build_profile_for_customer(customer: dict, txns: pd.DataFrame, emis: pd.DataFrame,
                                consent_status: str) -> dict:
    """
    Note: this still respects the consent boundary — the batch job below checks
    each customer's consent status (loaded once, in bulk) before building a
    profile from their data, rather than skipping the check for performance.
    A per-customer call to aa_interface.get_consented_data() would re-read all
    four CSVs from disk on every iteration (O(n^2) for a batch of n customers);
    here the same consent semantics are enforced but the tables are loaded and
    joined once, up front, which is the honest equivalent of a nightly batch
    feature-refresh job hitting a real database instead of flat files.
    """
    if consent_status != "ACTIVE":
        raise ConsentError(f"Consent not active for {customer['customer_id']} (status={consent_status})")

    txns = _to_df(txns.to_dict(orient="records"), date_cols=["timestamp"])
    emis = _to_df(emis.to_dict(orient="records"), date_cols=["due_date"])

    salary = compute_salary_consistency(txns) if not txns.empty else compute_salary_consistency(pd.DataFrame())
    rolling_income = compute_rolling_income_stability(txns) if not txns.empty else {"rolling_income_30d_mean": 0.0, "rolling_income_30d_std": 0.0}
    savings = compute_savings_rate_and_trend(txns)
    avg_monthly_income = (
        (txns[txns["txn_type"] == "credit"]["amount"].sum() /
         max(1, txns["timestamp"].dt.to_period("M").nunique()))
        if not txns.empty else 0.0
    )
    emi_feats = compute_emi_features(emis, avg_monthly_income)
    spend = compute_spend_composition(txns) if not txns.empty else {
        "essential_spend_share": 0.0, "discretionary_spend_share": 0.0, "cash_withdrawal_share": 0.0
    }
    novelty = compute_novelty_and_velocity(txns)

    # Compute data coverage and confidence metadata
    if txns.empty or len(txns) < 5:
        data_coverage_score = 0.45
        data_coverage_note = "Some accounts may not be connected through AA yet (thin history or partial consent)"
    elif (txns["timestamp"].max() - txns["timestamp"].min()).days < 30:
        data_coverage_score = 0.55
        data_coverage_note = "Some accounts may not be connected through AA yet (thin history or partial consent)"
    else:
        data_coverage_score = 1.0
        data_coverage_note = None

    data_quality_score = 1.0
    model_conf_hint = 0.85
    overall_conf_score = round(0.40 * data_coverage_score + 0.30 * data_quality_score + 0.30 * model_conf_hint, 4)
    if data_coverage_score < 0.60:
        overall_conf_score = min(overall_conf_score, 0.65)
    elif data_coverage_score < 0.70:
        overall_conf_score = min(overall_conf_score, 0.75)

    if overall_conf_score >= 0.80:
        overall_conf_band = "HIGH"
    elif overall_conf_score >= 0.60:
        overall_conf_band = "MEDIUM"
    else:
        overall_conf_band = "LOW"

    profile = {
        "customer_id": customer["customer_id"],
        "age": customer["age"],
        "gender": customer["gender"],
        "city": customer["city"],
        "state": customer["state"],
        "tier": customer["tier"],
        "language_preference": customer["language_preference"],
        "income_type_declared": customer["income_type"],  # from onboarding form
        "avg_monthly_income": round(avg_monthly_income, 2),
        **salary,
        **rolling_income,
        **savings,
        **emi_feats,
        **spend,
        **novelty,
        "data_coverage_score": data_coverage_score,
        "data_quality_score": data_quality_score,
        "overall_confidence_score": overall_conf_score,
        "overall_confidence_band": overall_conf_band,
        "data_coverage_note": data_coverage_note,
        "overall_confidence": {
            "overall_confidence_score": overall_conf_score,
            "overall_confidence_band": overall_conf_band,
            "data_coverage_score": data_coverage_score,
            "data_quality_score": data_quality_score,
            "model_confidence": model_conf_hint,
            "data_coverage_note": data_coverage_note,
        },
        "profile_computed_at": datetime.now(timezone.utc).isoformat(),
        # kept ONLY for downstream train/test split & eval — never a model feature
        "archetype_ground_truth": customer["archetype"],
        "split": customer["split"],
    }
    return profile


# --------------------------------------------------------------------------
# Batch build
# --------------------------------------------------------------------------

def build_all_profiles() -> pd.DataFrame:
    print("Loading raw tables once...")
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"), parse_dates=["timestamp"])
    emi_records = pd.read_csv(os.path.join(DATA_DIR, "emi_records.csv"), parse_dates=["due_date"])
    consents = pd.read_csv(os.path.join(DATA_DIR, "consent_artefacts.csv"))

    # latest consent status per customer (a customer could in principle have >1
    # consent artefact over time; take the most recently issued one)
    consent_status_by_customer = (
        consents.sort_values("validity_start")
        .groupby("customer_id")["status"].last()
        .to_dict()
    )

    print("Grouping transactions and EMI records by customer (one pass)...")
    txn_groups = dict(list(transactions.groupby("customer_id")))
    emi_groups = dict(list(emi_records.groupby("customer_id")))
    empty_txns = pd.DataFrame(columns=transactions.columns)
    empty_emis = pd.DataFrame(columns=emi_records.columns)

    profiles = []
    skipped = 0
    for i, customer in enumerate(customers.to_dict(orient="records")):
        cid = customer["customer_id"]
        status = consent_status_by_customer.get(cid)
        try:
            profile = build_profile_for_customer(
                customer,
                txn_groups.get(cid, empty_txns),
                emi_groups.get(cid, empty_emis),
                status,
            )
            profiles.append(profile)
        except ConsentError:
            skipped += 1
        if (i + 1) % 2000 == 0:
            print(f"  ...{i + 1}/{len(customers)} customers processed")

    if skipped:
        print(f"  [info] skipped {skipped} customers with inactive/missing consent")
    return pd.DataFrame(profiles)


def main():
    parser = argparse.ArgumentParser(description="Build the shared customer_profile table (Module 2)")
    parser.add_argument("--out", type=str, default="../data/customer_profile.csv")
    args = parser.parse_args()

    print("Building unified customer profiles from consented data...")
    df = build_all_profiles()
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} profiles -> {args.out}")
    print(f"columns ({len(df.columns)}): {list(df.columns)}")

    print("\nIncome pattern distribution (derived, not declared):")
    print(df["income_pattern"].value_counts())

    print("\nMean EMI-to-income ratio by declared archetype (ground truth, eval only):")
    print(df.groupby("archetype_ground_truth")["emi_to_income_ratio"].mean().round(3))

    print("\nMean stress-relevant signals by declared archetype:")
    print(df.groupby("archetype_ground_truth")[
        ["missed_emi_count", "savings_rate_trend", "balance_trend", "cash_withdrawal_share"]
    ].mean().round(4))


if __name__ == "__main__":
    main()
