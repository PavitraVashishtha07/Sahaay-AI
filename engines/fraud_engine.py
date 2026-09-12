"""
Sahaay AI — Section 4.4: Fraud & Anomaly Detection Engine

Implements an unsupervised Isolation Forest anomaly detection engine operating
on transaction-level behavioral novelty signals.

Design decisions (per Technical Build Plan Section 4.4 & Research Doc Section 6):
1. Unsupervised by Design: Trained strictly on behavioral features without
   fitting on `is_fraud_label`. This reflects real-world banking where confirmed
   fraud labels are sparse, delayed, and noisy.
2. Novelty Signals: Evaluates sudden deviations from a customer's baseline:
   - Amount ratio to personal average transaction size
   - Device novelty (`is_new_device`)
   - Beneficiary novelty (`is_new_beneficiary`)
   - Merchant novelty (`is_new_merchant`)
   - Unusual transaction hour (e.g. late night / early morning)
   - Rapid transfer channel (`IMPS`)
3. Explainable Anomaly Framing: Never outputs an accusatory "this is fraud"
   verdict; returns an `anomaly_score` [0, 1] and specific `flagged_signals`
   (e.g., "new_device", "new_beneficiary", "amount_burst_spike").
4. Evaluation-Only Label Use: `is_fraud_label` is used strictly in
   `evaluate_model_performance()` to measure ROC-AUC and Recall after training.
"""

from __future__ import annotations
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_score, recall_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

FRAUD_FEATURES = [
    "amount_ratio",
    "is_new_device_f",
    "is_new_beneficiary_f",
    "is_new_merchant_f",
    "is_unusual_hour",
    "is_imps",
]

ANOMALY_THRESHOLD = 0.70

_FRAUD_MODEL: Optional[IsolationForest] = None
_CUSTOMER_BASELINES: Optional[Dict[str, float]] = None
_SCORE_MIN_MAX: Tuple[float, float] = (-0.75, -0.35)


def _compute_customer_baselines(transactions_df: pd.DataFrame) -> Dict[str, float]:
    """Computes baseline mean transaction amount per customer."""
    means = transactions_df.groupby("customer_id")["amount"].mean().to_dict()
    return means


def _get_customer_baselines() -> Dict[str, float]:
    global _CUSTOMER_BASELINES
    if _CUSTOMER_BASELINES is not None:
        return _CUSTOMER_BASELINES

    tx_path = os.path.join(DATA_DIR, "transactions.csv")
    if os.path.exists(tx_path):
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for chunk in pd.read_csv(tx_path, chunksize=50000, usecols=["customer_id", "amount"]):
            grp = chunk.groupby("customer_id")["amount"].agg(["sum", "count"])
            for cid, row in grp.iterrows():
                cid_str = str(cid)
                sums[cid_str] = sums.get(cid_str, 0.0) + float(row["sum"])
                counts[cid_str] = counts.get(cid_str, 0) + int(row["count"])
        _CUSTOMER_BASELINES = {cid: sums[cid] / counts[cid] for cid in sums if counts[cid] > 0}
    else:
        _CUSTOMER_BASELINES = {}
    return _CUSTOMER_BASELINES


def extract_features(
    df: pd.DataFrame,
    baselines: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Transforms raw transaction records into anomaly detection features.
    Strictly excludes is_fraud_label from output feature columns.
    """
    if baselines is None:
        baselines = _get_customer_baselines()

    tx = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(tx["timestamp"]):
        tx["timestamp"] = pd.to_datetime(tx["timestamp"])

    cust_means = tx["customer_id"].map(baselines).fillna(tx["amount"].mean())
    tx["amount_ratio"] = tx["amount"] / (cust_means + 1e-5)
    tx["hour"] = tx["timestamp"].dt.hour
    tx["is_unusual_hour"] = ((tx["hour"] >= 23) | (tx["hour"] <= 5)).astype(float)
    tx["is_new_device_f"] = tx["is_new_device"].astype(float)
    tx["is_new_beneficiary_f"] = tx["is_new_beneficiary"].astype(float)
    tx["is_new_merchant_f"] = tx["is_new_merchant"].astype(float)
    tx["is_imps"] = (tx["channel"].astype(str).str.lower() == "imps").astype(float)

    return tx[FRAUD_FEATURES]


def get_or_train_fraud_model(
    sample_size: int = 50000,
    force_retrain: bool = False,
) -> IsolationForest:
    """
    Trains an unsupervised IsolationForest on a representative sample of
    transactions. Does NOT use is_fraud_label during training.
    """
    global _FRAUD_MODEL, _SCORE_MIN_MAX
    if _FRAUD_MODEL is not None and not force_retrain:
        return _FRAUD_MODEL

    tx_path = os.path.join(DATA_DIR, "transactions.csv")
    if not os.path.exists(tx_path):
        raise FileNotFoundError(f"{tx_path} not found.")

    # Evenly sample rows across the full dataset without loading all rows into RAM
    tx_sample = pd.read_csv(tx_path, skiprows=lambda i: i > 0 and (i % 18 != 0))
    X_train = extract_features(tx_sample)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train)

    raw_scores = -model.score_samples(X_train)
    _SCORE_MIN_MAX = (float(raw_scores.min()), float(raw_scores.max()))
    _FRAUD_MODEL = model
    return _FRAUD_MODEL


def _normalize_score(raw_score: float) -> float:
    min_s, max_s = _SCORE_MIN_MAX
    norm = (raw_score - min_s) / (max_s - min_s + 1e-5)
    return float(np.clip(norm, 0.0, 1.0))


def _identify_flagged_signals(feat_row: Dict[str, Any]) -> List[str]:
    signals = []
    if feat_row.get("is_new_device_f", 0) > 0.5:
        signals.append("new_device")
    if feat_row.get("is_new_beneficiary_f", 0) > 0.5:
        signals.append("new_beneficiary")
    if feat_row.get("is_new_merchant_f", 0) > 0.5:
        signals.append("new_merchant")
    if feat_row.get("amount_ratio", 1.0) >= 2.5:
        signals.append("amount_burst_spike")
    if feat_row.get("is_imps", 0) > 0.5:
        signals.append("rapid_transfer_channel_imps")
    if feat_row.get("is_unusual_hour", 0) > 0.5:
        signals.append("unusual_hour")
    return signals


def score_transaction(
    txn: Dict[str, Any],
    customer_baseline_amount: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluates a single transaction dictionary for behavioral anomaly.
    Returns anomaly score [0.0 - 1.0], anomaly flag, and flagged signals.
    """
    model = get_or_train_fraud_model()
    single_df = pd.DataFrame([txn])

    baselines = None
    if customer_baseline_amount is not None:
        baselines = {txn.get("customer_id", "default"): customer_baseline_amount}

    feats = extract_features(single_df, baselines=baselines)
    raw_score = float(-model.score_samples(feats)[0])
    anomaly_score = _normalize_score(raw_score)
    is_anomaly = bool(anomaly_score >= ANOMALY_THRESHOLD)

    feat_dict = feats.iloc[0].to_dict()
    flagged = _identify_flagged_signals(feat_dict)

    return {
        "transaction_id": txn.get("transaction_id", "unknown"),
        "customer_id": txn.get("customer_id", "unknown"),
        "anomaly_score": round(anomaly_score, 4),
        "is_anomaly": is_anomaly,
        "flagged_signals": flagged,
        "amount": txn.get("amount", 0.0),
        "channel": txn.get("channel", "unknown"),
    }


def detect_customer_anomalies(
    customer_id: str,
    transactions_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Scans a customer's transaction history to detect any anomalous burst activity.
    """
    if transactions_df is None:
        tx_path = os.path.join(DATA_DIR, "transactions.csv")
        if not os.path.exists(tx_path):
            customer_txns = pd.DataFrame()
        else:
            chunks = []
            for chunk in pd.read_csv(tx_path, chunksize=50000):
                c = chunk[chunk["customer_id"] == customer_id]
                if not c.empty:
                    chunks.append(c)
            customer_txns = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    else:
        customer_txns = transactions_df[transactions_df["customer_id"] == customer_id]

    if customer_txns.empty:
        return {
            "customer_id": customer_id,
            "has_anomaly": False,
            "max_anomaly_score": 0.0,
            "anomalous_transactions_count": 0,
            "flagged_signals": [],
            "flagged_transactions": [],
        }

    model = get_or_train_fraud_model()
    feats = extract_features(customer_txns)
    raw_scores = -model.score_samples(feats)
    norm_scores = [_normalize_score(s) for s in raw_scores]

    anomalous_txns = []
    all_flags = set()
    for idx, (_, row) in enumerate(customer_txns.iterrows()):
        score = norm_scores[idx]
        if score >= ANOMALY_THRESHOLD:
            feat_dict = feats.iloc[idx].to_dict()
            flags = _identify_flagged_signals(feat_dict)
            all_flags.update(flags)
            anomalous_txns.append({
                "transaction_id": row.get("transaction_id"),
                "timestamp": str(row.get("timestamp")),
                "amount": row.get("amount"),
                "anomaly_score": round(score, 4),
                "flagged_signals": flags,
            })

    max_score = max(norm_scores) if norm_scores else 0.0

    cov = 0.45 if len(customer_txns) < 5 else 1.0
    qual = 1.0
    note = "Some accounts may not be connected through AA yet (thin history or partial consent)" if cov < 0.70 else None
    model_conf = 0.85
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
        "has_anomaly": len(anomalous_txns) > 0,
        "max_anomaly_score": round(max_score, 4),
        "anomalous_transactions_count": len(anomalous_txns),
        "flagged_signals": sorted(list(all_flags)),
        "flagged_transactions": anomalous_txns,
        "overall_confidence": overall_conf,
        "data_coverage_score": cov,
        "data_coverage_note": note,
    }


def evaluate_model_performance(sample_size: int = 50000) -> Dict[str, float]:
    """
    Evaluates the unsupervised model against evaluation-only ground truth labels.
    """
    tx_path = os.path.join(DATA_DIR, "transactions.csv")
    tx = pd.read_csv(tx_path)
    if len(tx) > sample_size:
        # Include all fraud transactions + random sample of normal transactions
        fraud = tx[tx["is_fraud_label"]]
        normal = tx[~tx["is_fraud_label"]].sample(n=sample_size - len(fraud), random_state=42)
        eval_df = pd.concat([fraud, normal]).sample(frac=1.0, random_state=42)
    else:
        eval_df = tx

    model = get_or_train_fraud_model()
    feats = extract_features(eval_df)
    raw_scores = -model.score_samples(feats)
    norm_scores = np.array([_normalize_score(s) for s in raw_scores])
    preds = (norm_scores >= ANOMALY_THRESHOLD).astype(int)
    y_true = eval_df["is_fraud_label"].astype(int).values

    auc = roc_auc_score(y_true, norm_scores)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)

    return {
        "roc_auc": round(float(auc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "total_evaluated": len(eval_df),
        "fraud_count": int(y_true.sum()),
    }
