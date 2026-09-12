"""
Sahaay AI — Tests for Security & Risk Gateway Layer (security/gateway.py)

Verifies:
1. Schema validation rejects/quarantines malformed records and negative transactions.
2. Source validation checks FIP bank name against authorized bank whitelist.
3. Missing / thin history customer gets lowered data_coverage_score and capped overall confidence.
4. An ingested transaction with >50x spike is quarantined and excluded from profile calculation.
5. Real persona contrast: Kavita Rao (cust_cold_start_new) vs Arjun Sharma (cust_86838bd208)
   shows demonstrable confidence difference (Medium/Low vs High) with data_coverage_note attached.
"""

import os
import sys
import pytest
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "security"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_layer"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "engines"))

from gateway import (
    validate_schema,
    validate_sources,
    compute_data_coverage,
    check_anomalies_on_ingest,
    compute_overall_confidence,
    process_secure_customer_pipeline,
    SecurityValidationError,
)
from aa_interface import get_active_consent_id_for_customer


def test_schema_validation_catches_negative_amount_and_malformed_records():
    """Verify malformed records and negative transactions are quarantined."""
    payload = {
        "customer": {
            "customer_id": "cust_test_001",
            "name": "Test User",
            "age": 30,
            "gender": "M",
            "income_type": "salaried",
            "tier": "Tier-1",
            "city": "Mumbai",
            "state": "Maharashtra",
        },
        "accounts": [
            {"account_id": "acc_001", "bank_name": "HDFC Bank", "opening_balance": 10000.0},
            {"account_id": "", "bank_name": "State Bank of India"},  # malformed
        ],
        "transactions": [
            {"transaction_id": "txn_001", "amount": 500.0, "timestamp": "2026-08-01 10:00", "txn_type": "debit"},
            {"transaction_id": "txn_002", "amount": -1500.0, "timestamp": "2026-08-02 11:00", "txn_type": "debit"},  # negative
        ],
        "emi_records": [],
        "consent": {"fip_name": "HDFC Bank", "status": "ACTIVE"},
    }

    is_valid, errors, clean = validate_schema(payload)
    assert is_valid is False
    assert len(errors) == 2
    assert len(clean["accounts"]) == 1
    assert len(clean["transactions"]) == 1
    assert clean["transactions"][0]["transaction_id"] == "txn_001"


def test_source_validation_bank_whitelist():
    """Verify source validation permits scheduled banks and flags unknown banks."""
    valid_payload = {
        "consent": {"fip_name": "HDFC Bank"},
        "accounts": [{"account_id": "acc_1", "bank_name": "ICICI Bank"}],
    }
    ok, errs = validate_sources(valid_payload)
    assert ok is True
    assert len(errs) == 0

    invalid_payload = {
        "consent": {"fip_name": "Shady Offshore Bank Ltd"},
        "accounts": [{"account_id": "acc_2", "bank_name": "Unknown PayCo"}],
    }
    bad_ok, bad_errs = validate_sources(invalid_payload)
    assert bad_ok is False
    assert len(bad_errs) == 2


def test_anomaly_on_ingest_quarantines_50x_spike():
    """Verify a >50x spike is quarantined and not included in clean transaction set."""
    normal_amounts = [100.0, 150.0, 120.0, 180.0, 200.0, 110.0]  # mean ~143
    txns = [
        {"transaction_id": f"txn_{i}", "amount": amt, "timestamp": f"2026-08-0{i+1} 10:00", "txn_type": "debit"}
        for i, amt in enumerate(normal_amounts)
    ]
    # Add a 50x spike (10,000 > 50 * 143 = 7,150)
    txns.append({
        "transaction_id": "txn_spike",
        "amount": 25000.0,
        "timestamp": "2026-08-08 12:00",
        "txn_type": "debit",
    })

    payload = {
        "customer": {"customer_id": "cust_spike_test"},
        "transactions": txns,
    }

    clean_txns, quarantined = check_anomalies_on_ingest(payload)
    assert len(quarantined) == 1
    assert quarantined[0]["transaction_id"] == "txn_spike"
    assert len(clean_txns) == len(normal_amounts)
    assert "txn_spike" not in [t["transaction_id"] for t in clean_txns]


def test_coverage_calculation_and_capping():
    """Verify thin history lowers data_coverage_score and attaches explanatory note."""
    # Full coverage customer (accounts, long transaction span, liabilities, complete kyc)
    full_payload = {
        "customer": {"name": "Arjun", "age": 30, "gender": "M", "income_type": "salaried", "tier": "Tier-1", "city": "Bangalore", "state": "Karnataka"},
        "accounts": [{"account_id": "acc_1", "bank_name": "HDFC Bank"}],
        "transactions": [
            {"timestamp": "2026-01-01 10:00"}
        ] * 10 + [
            {"timestamp": "2026-06-01 10:00"}
        ] * 10,
        "emi_records": [{"emi_id": "emi_1", "amount_due": 5000.0}, {"emi_id": "emi_2", "amount_due": 5000.0}, {"emi_id": "emi_3", "amount_due": 5000.0}],
    }
    cov_full, note_full, comps_full = compute_data_coverage(full_payload)
    assert cov_full >= 0.90
    assert note_full is None

    # Thin coverage customer (0 transactions, unverified liabilities)
    thin_payload = {
        "customer": {"name": "Kavita", "age": 24, "gender": "F", "income_type": "salaried", "tier": "Tier-2", "city": "Pune", "state": "Maharashtra"},
        "accounts": [{"account_id": "acc_cold", "bank_name": "HDFC Bank"}],
        "transactions": [],
        "emi_records": [],
    }
    cov_thin, note_thin, comps_thin = compute_data_coverage(thin_payload)
    assert cov_thin < 0.70
    assert note_thin is not None
    assert "Some accounts may not be connected through AA yet" in note_thin


def test_real_persona_confidence_contrast():
    """
    Validates Scenario B from Section 6 of the research document using real personas:
    Arjun Sharma (cust_86838bd208): Full history -> HIGH overall confidence (>0.85).
    Kavita Rao (cust_cold_start_new): Thin history (<30d) -> Capped overall confidence (<=0.70)
    with data_coverage_note attached.
    """
    arjun_cid = "cust_86838bd208"
    arjun_consent = get_active_consent_id_for_customer(arjun_cid)
    r_arjun = process_secure_customer_pipeline(arjun_cid, arjun_consent)

    kavita_cid = "cust_cold_start_new"
    kavita_consent = get_active_consent_id_for_customer(kavita_cid)
    r_kavita = process_secure_customer_pipeline(kavita_cid, kavita_consent)

    arjun_conf = r_arjun["gateway_metadata"]["overall_confidence"]
    kavita_conf = r_kavita["gateway_metadata"]["overall_confidence"]

    # Arjun: High confidence and full coverage
    assert arjun_conf["overall_confidence_band"] == "HIGH"
    assert arjun_conf["overall_confidence_score"] >= 0.85
    assert arjun_conf["data_coverage_score"] >= 0.90
    assert arjun_conf["data_coverage_note"] is None

    # Kavita: Materially lower confidence due to coverage constraint (Scenario B)
    assert kavita_conf["overall_confidence_score"] < arjun_conf["overall_confidence_score"]
    assert kavita_conf["data_coverage_score"] < 0.70
    assert kavita_conf["overall_confidence_band"] in ("MEDIUM", "LOW")
    assert kavita_conf["data_coverage_note"] is not None
    assert "Some accounts may not be connected through AA yet" in kavita_conf["data_coverage_note"]
