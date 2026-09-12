"""
Sahaay AI — Section 4.3: Stress Detection Engine

Implements the transparent, explainable weighted-score approach for customer
financial distress detection, backed by a secondary IsolationForest anomaly
check as an unsupervised stretch capability.

Design decisions (per Technical Build Plan Section 4.3 & Research Doc Section 5):
1. Transparent & Auditable: Weights are explicit named constants at the top,
   not buried inside formulas or hidden inside a black-box model.
2. 0–1 Normalized Signals: Each stress dimension is scaled to [0, 1].
3. Clear Bucketing: stable / watch / concern / high_concern.
4. Explanatory Drivers: Identifies the exact signals elevating distress.
5. Dual Output: Outputs the deterministic weighted score and the unsupervised
   IsolationForest deterioration anomaly score side-by-side.
"""

from __future__ import annotations
import os
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# --------------------------------------------------------------------------
# Explicit Weight Constants (Auditable & Adjustable Design Decisions)
# --------------------------------------------------------------------------
WEIGHT_MISSED_EMI = 0.35
WEIGHT_SAVINGS_DECLINE = 0.20
WEIGHT_ESSENTIAL_SPEND = 0.15
WEIGHT_BALANCE_DECLINE = 0.15
WEIGHT_CASH_SPIKE = 0.10
WEIGHT_EMI_BURDEN = 0.05

# Stress Band Thresholds
THRESHOLD_WATCH = 0.25
THRESHOLD_CONCERN = 0.45
THRESHOLD_HIGH_CONCERN = 0.70

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# --------------------------------------------------------------------------
# Signal Normalization Helpers (0.0 to 1.0)
# --------------------------------------------------------------------------

def _compute_missed_emi_signal(row: Dict[str, Any]) -> float:
    """Evaluates missed EMIs, late EMIs, on-time rate, and days late."""
    has_loans = row.get("has_active_loans", False)
    if not has_loans or (isinstance(has_loans, str) and has_loans.lower() == "false"):
        return 0.0

    missed = float(row.get("missed_emi_count", 0))
    late = float(row.get("late_emi_count", 0))
    on_time_rate = float(row.get("emi_on_time_rate", 1.0))
    max_late = float(row.get("max_days_late", 0))

    if missed == 0 and late == 0:
        return 0.0

    # Strong penalty for missed EMIs (each missed adds ~0.4)
    # Moderate penalty for late payments and low on-time rate
    raw = (missed * 0.4) + (late * 0.15) + ((1.0 - on_time_rate) * 0.3)
    if max_late >= 30:
        raw += 0.2
    return float(np.clip(raw, 0.0, 1.0))


def _compute_savings_decline_signal(row: Dict[str, Any]) -> float:
    """Evaluates savings rate trend and average monthly savings rate."""
    savings_trend = float(row.get("savings_rate_trend", 0.0))
    avg_savings_rate = float(row.get("avg_monthly_savings_rate", 0.3))

    signal = 0.0
    # Negative savings trend indicates deterioration
    if savings_trend < 0:
        signal += min(0.6, abs(savings_trend) * 3.0)

    # Low or negative absolute savings rate
    if avg_savings_rate <= 0:
        signal += 0.4
    elif avg_savings_rate < 0.15:
        signal += (0.15 - avg_savings_rate) * 2.0

    return float(np.clip(signal, 0.0, 1.0))


def _compute_essential_spend_signal(row: Dict[str, Any]) -> float:
    """Evaluates the proportion of income consumed by essential expenses."""
    essential_share = float(row.get("essential_spend_share", 0.5))
    # Baseline normal essential spend is 40-60%. >75% indicates stress.
    if essential_share <= 0.60:
        return 0.0
    raw = (essential_share - 0.60) / 0.35  # scales 0.60 -> 0.0, 0.95 -> 1.0
    return float(np.clip(raw, 0.0, 1.0))


def _compute_balance_decline_signal(row: Dict[str, Any]) -> float:
    """Evaluates running balance trend / cash cushion depletion."""
    balance_trend = float(row.get("balance_trend", 25000.0))
    avg_income = float(row.get("avg_monthly_income", 45000.0))

    # Compare balance trend to monthly income cushion
    cushion_ratio = balance_trend / (avg_income + 1e-5)
    if cushion_ratio >= 0.5:
        return 0.0
    raw = (0.5 - cushion_ratio) / 0.4  # scales 0.5 -> 0.0, 0.1 -> 1.0
    return float(np.clip(raw, 0.0, 1.0))


def _compute_cash_spike_signal(row: Dict[str, Any]) -> float:
    """Evaluates sudden cash withdrawal / ATM spikes (frequent stress signal)."""
    cash_share = float(row.get("cash_withdrawal_share", 0.0))
    # Normal cash withdrawal share is <12%. Stressed jumps to 20-30%.
    if cash_share <= 0.15:
        return 0.0
    raw = (cash_share - 0.15) / 0.20  # scales 0.15 -> 0.0, 0.35 -> 1.0
    return float(np.clip(raw, 0.0, 1.0))


def _compute_emi_burden_signal(row: Dict[str, Any]) -> float:
    """Evaluates Debt-to-Income / EMI-to-Income burden ratio."""
    emi_ratio = float(row.get("emi_to_income_ratio", 0.0))
    if emi_ratio <= 0.35:
        return 0.0
    raw = (emi_ratio - 0.35) / 0.35  # scales 0.35 -> 0.0, 0.70 -> 1.0
    return float(np.clip(raw, 0.0, 1.0))


# --------------------------------------------------------------------------
# Secondary IsolationForest Model for Deterioration Patterns
# --------------------------------------------------------------------------

_ISO_FOREST_MODEL: Optional[IsolationForest] = None
_ISO_FEATURES = [
    "missed_emi_count",
    "late_emi_count",
    "savings_rate_trend",
    "avg_monthly_savings_rate",
    "essential_spend_share",
    "balance_trend",
    "cash_withdrawal_share",
    "emi_to_income_ratio",
]


def _get_or_train_isolation_forest(df: Optional[pd.DataFrame] = None) -> IsolationForest:
    global _ISO_FOREST_MODEL
    if _ISO_FOREST_MODEL is not None:
        return _ISO_FOREST_MODEL

    if df is None:
        profile_path = os.path.join(DATA_DIR, "customer_profile.csv")
        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"{profile_path} not found.")
        df = pd.read_csv(profile_path)

    X = df[_ISO_FEATURES].fillna(0.0)
    model = IsolationForest(n_estimators=100, contamination=0.20, random_state=42)
    model.fit(X)
    _ISO_FOREST_MODEL = model
    return _ISO_FOREST_MODEL


# --------------------------------------------------------------------------
# Main Stress Evaluation Functions
# --------------------------------------------------------------------------

def compute_stress_profile(row_or_dict: Union[pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes deterministic weighted distress score + drivers + secondary
    IsolationForest anomaly check for a single customer.
    """
    row = row_or_dict.to_dict() if isinstance(row_or_dict, pd.Series) else dict(row_or_dict)
    customer_id = row.get("customer_id", "unknown")

    # 1. Compute Individual Signals
    s_missed_emi = _compute_missed_emi_signal(row)
    s_savings_decline = _compute_savings_decline_signal(row)
    s_essential_spend = _compute_essential_spend_signal(row)
    s_balance_decline = _compute_balance_decline_signal(row)
    s_cash_spike = _compute_cash_spike_signal(row)
    s_emi_burden = _compute_emi_burden_signal(row)

    signals = {
        "missed_emi_signal": round(s_missed_emi, 4),
        "savings_decline_signal": round(s_savings_decline, 4),
        "essential_spend_signal": round(s_essential_spend, 4),
        "balance_decline_signal": round(s_balance_decline, 4),
        "cash_spike_signal": round(s_cash_spike, 4),
        "emi_burden_signal": round(s_emi_burden, 4),
    }

    # 2. Weighted Sum
    total_score = (
        s_missed_emi * WEIGHT_MISSED_EMI
        + s_savings_decline * WEIGHT_SAVINGS_DECLINE
        + s_essential_spend * WEIGHT_ESSENTIAL_SPEND
        + s_balance_decline * WEIGHT_BALANCE_DECLINE
        + s_cash_spike * WEIGHT_CASH_SPIKE
        + s_emi_burden * WEIGHT_EMI_BURDEN
    )
    total_score = float(np.clip(total_score, 0.0, 1.0))

    # 3. Determine Stress Band
    if total_score >= THRESHOLD_HIGH_CONCERN:
        band = "high_concern"
    elif total_score >= THRESHOLD_CONCERN:
        band = "concern"
    elif total_score >= THRESHOLD_WATCH:
        band = "watch"
    else:
        band = "stable"

    # 4. Identify Primary Drivers (contributing > 0.03 weighted)
    driver_map = {
        "missed_emi_signal": ("missed_or_delayed_emis", s_missed_emi * WEIGHT_MISSED_EMI),
        "savings_decline_signal": ("savings_rate_decline", s_savings_decline * WEIGHT_SAVINGS_DECLINE),
        "essential_spend_signal": ("rising_essential_expense_share", s_essential_spend * WEIGHT_ESSENTIAL_SPEND),
        "balance_decline_signal": ("cash_buffer_depletion", s_balance_decline * WEIGHT_BALANCE_DECLINE),
        "cash_spike_signal": ("unusual_cash_withdrawals", s_cash_spike * WEIGHT_CASH_SPIKE),
        "emi_burden_signal": ("high_emi_burden_ratio", s_emi_burden * WEIGHT_EMI_BURDEN),
    }
    sorted_drivers = sorted(driver_map.values(), key=lambda x: x[1], reverse=True)
    drivers = [name for name, contrib in sorted_drivers if contrib > 0.03]
    if not drivers and band != "stable":
        drivers = [sorted_drivers[0][0]]

    # 5. Secondary Isolation Forest Check
    try:
        iso_model = _get_or_train_isolation_forest()
        feat_df = pd.DataFrame([{col: float(row.get(col, 0.0)) for col in _ISO_FEATURES}])
        raw_iso = float(iso_model.score_samples(feat_df)[0])
        iso_anomaly_score = float(np.clip(-raw_iso, 0.0, 1.0))
        iso_is_anomaly = bool(iso_model.predict(feat_df)[0] == -1)
    except Exception:
        iso_anomaly_score = 0.0
        iso_is_anomaly = False

    cov = float(row.get("data_coverage_score", 1.0))
    qual = float(row.get("data_quality_score", 1.0))
    note = row.get("data_coverage_note")
    model_conf = 0.90 if band != "watch" else 0.80
    overall_conf_score = round((0.40 * cov) + (0.30 * qual) + (0.30 * model_conf), 4)
    if cov < 0.60:
        overall_conf_score = min(overall_conf_score, 0.65)
    elif cov < 0.70:
        overall_conf_score = min(overall_conf_score, 0.75)

    overall_conf = {
        "overall_confidence_score": overall_conf_score,
        "overall_confidence_band": "HIGH" if overall_conf_score >= 0.80 else ("MEDIUM" if overall_conf_score >= 0.60 else "LOW"),
        "data_coverage_score": cov,
        "data_quality_score": qual,
        "model_confidence": model_conf,
        "data_coverage_note": note,
    }

    return {
        "customer_id": customer_id,
        "band": band,
        "score": round(total_score, 4),
        "drivers": drivers,
        "signals": signals,
        "weights_used": {
            "missed_emi": WEIGHT_MISSED_EMI,
            "savings_decline": WEIGHT_SAVINGS_DECLINE,
            "essential_spend": WEIGHT_ESSENTIAL_SPEND,
            "balance_decline": WEIGHT_BALANCE_DECLINE,
            "cash_spike": WEIGHT_CASH_SPIKE,
            "emi_burden": WEIGHT_EMI_BURDEN,
        },
        "secondary_isolation_forest": {
            "anomaly_score": round(iso_anomaly_score, 4),
            "is_anomaly": iso_is_anomaly,
        },
        "overall_confidence": overall_conf,
        "data_coverage_score": cov,
        "data_coverage_note": note,
    }


def evaluate_all_customers(profile_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Evaluates stress scores across the entire customer profile dataset."""
    if profile_df is None:
        profile_path = os.path.join(DATA_DIR, "customer_profile.csv")
        profile_df = pd.read_csv(profile_path)

    results = []
    for _, row in profile_df.iterrows():
        res = compute_stress_profile(row)
        results.append({
            "customer_id": res["customer_id"],
            "stress_band": res["band"],
            "stress_score": res["score"],
            "drivers": ",".join(res["drivers"]),
            "iso_anomaly_score": res["secondary_isolation_forest"]["anomaly_score"],
            "iso_is_anomaly": res["secondary_isolation_forest"]["is_anomaly"],
        })
    return pd.DataFrame(results)
