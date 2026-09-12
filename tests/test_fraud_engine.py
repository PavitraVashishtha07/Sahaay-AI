"""
Sahaay AI — Tests for Section 4.4: Fraud & Anomaly Detection Engine

Verifies:
1. Fraud-burst transactions score in the top percentile of anomaly scores.
2. Normal transactions from stable salaried customers score low on anomaly.
3. Feature extraction strictly excludes is_fraud_label (unsupervised training).
4. Individual novelty signals (new_device, new_beneficiary, etc.) are accurately identified.
5. Customer-level anomaly scanning separates fraud-scenario customers from normal ones.
6. Unsupervised model evaluation achieves high ROC-AUC without training on labels.
"""

import os
import sys
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "engines"))
from fraud_engine import (
    FRAUD_FEATURES,
    extract_features,
    get_or_train_fraud_model,
    score_transaction,
    detect_customer_anomalies,
    evaluate_model_performance,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _get_sample_transactions():
    path = os.path.join(DATA_DIR, "transactions.csv")
    assert os.path.exists(path), "transactions.csv not found."
    df = pd.read_csv(path)
    fraud = df[df["is_fraud_label"]].sample(n=500, random_state=42)
    normal = df[~df["is_fraud_label"]].sample(n=2000, random_state=42)
    return pd.concat([fraud, normal]).reset_index(drop=True)


@pytest.fixture(scope="module")
def sample_transactions():
    return _get_sample_transactions()


def test_unsupervised_training_does_not_use_labels():
    """Verify that is_fraud_label is never part of FRAUD_FEATURES."""
    assert "is_fraud_label" not in FRAUD_FEATURES
    dummy_df = pd.DataFrame([{
        "customer_id": "c1",
        "timestamp": "2026-09-10 12:00:00",
        "amount": 5000,
        "is_new_device": True,
        "is_new_beneficiary": False,
        "is_new_merchant": False,
        "channel": "upi",
        "is_fraud_label": True,
    }])
    feats = extract_features(dummy_df)
    assert "is_fraud_label" not in feats.columns
    assert list(feats.columns) == FRAUD_FEATURES


def test_fraud_burst_transactions_score_high_anomaly(sample_transactions):
    fraud_txns = sample_transactions[sample_transactions["is_fraud_label"]]
    model = get_or_train_fraud_model()
    feats = extract_features(fraud_txns)
    raw_scores = -model.score_samples(feats)

    high_score_count = 0
    for _, row in fraud_txns.iterrows():
        res = score_transaction(row.to_dict())
        if res["anomaly_score"] >= 0.70:
            high_score_count += 1

    high_score_rate = high_score_count / len(fraud_txns)
    assert high_score_rate >= 0.90, (
        f"Expected >= 90% of fraud transactions to score >= 0.70 anomaly, got {high_score_rate:.1%}"
    )


def test_stable_normal_transactions_score_low(sample_transactions):
    normal_txns = sample_transactions[~sample_transactions["is_fraud_label"]]
    low_score_count = 0
    for _, row in normal_txns.iloc[:300].iterrows():
        res = score_transaction(row.to_dict())
        if res["anomaly_score"] < 0.50:
            low_score_count += 1

    low_score_rate = low_score_count / 300
    assert low_score_rate >= 0.90, (
        f"Expected >= 90% of normal transactions to score < 0.50 anomaly, got {low_score_rate:.1%}"
    )


def test_signal_flags_populated_correctly():
    fraud_like_txn = {
        "transaction_id": "txn_test_1",
        "customer_id": "cust_test_1",
        "timestamp": "2026-09-10 03:30:00",  # unusual hour
        "amount": 45000,                      # burst spike
        "is_new_device": True,
        "is_new_beneficiary": True,
        "is_new_merchant": True,
        "channel": "imps",
    }
    res = score_transaction(fraud_like_txn, customer_baseline_amount=3000)
    assert res["is_anomaly"] is True
    assert "new_device" in res["flagged_signals"]
    assert "new_beneficiary" in res["flagged_signals"]
    assert "amount_burst_spike" in res["flagged_signals"]
    assert "rapid_transfer_channel_imps" in res["flagged_signals"]
    assert "unusual_hour" in res["flagged_signals"]


def test_customer_level_anomaly_detection():
    gt = pd.read_csv(os.path.join(DATA_DIR, "ground_truth_labels.csv"))
    fraud_cust = gt[gt["has_fraud_event"]].iloc[0]["customer_id"]
    stable_cust = gt[~gt["has_fraud_event"] & (gt["stress_label"] == "stable")].iloc[0]["customer_id"]

    res_fraud = detect_customer_anomalies(fraud_cust)
    assert res_fraud["has_anomaly"] is True
    assert res_fraud["anomalous_transactions_count"] >= 3
    assert len(res_fraud["flagged_signals"]) > 0

    res_stable = detect_customer_anomalies(stable_cust)
    assert res_stable["has_anomaly"] is False
    assert res_stable["anomalous_transactions_count"] == 0


def test_model_evaluation_metrics():
    metrics = evaluate_model_performance(sample_size=10000)
    assert metrics["roc_auc"] >= 0.90, f"Expected ROC-AUC >= 0.90, got {metrics['roc_auc']}"
    assert metrics["recall"] >= 0.85, f"Expected Recall >= 0.85, got {metrics['recall']}"


if __name__ == "__main__":
    st = _get_sample_transactions()
    test_unsupervised_training_does_not_use_labels()
    print("PASS: test_unsupervised_training_does_not_use_labels")
    test_fraud_burst_transactions_score_high_anomaly(st)
    print("PASS: test_fraud_burst_transactions_score_high_anomaly")
    test_stable_normal_transactions_score_low(st)
    print("PASS: test_stable_normal_transactions_score_low")
    test_signal_flags_populated_correctly()
    print("PASS: test_signal_flags_populated_correctly")
    test_customer_level_anomaly_detection()
    print("PASS: test_customer_level_anomaly_detection")
    test_model_evaluation_metrics()
    print("PASS: test_model_evaluation_metrics")
    print("\nAll Fraud Engine tests PASSED!")
