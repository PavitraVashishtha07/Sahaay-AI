"""
Sahaay AI — Security & Risk Gateway Layer (security/gateway.py)

Sits strictly between data ingestion (aa_interface.py) and customer intelligence
(build_customer_profiles.py / engines). Ensures that nothing reaches an AI engine
or profile table unvalidated.

Pipeline Execution Order:
1. Encrypted Ingestion
2. Schema Validation
3. Source Validation (Bank Whitelist)
4. Coverage Check (data_coverage_score calculation)
5. Anomaly-on-Ingest Check (Quarantine >50x spikes, >100x income jumps)
6. Data Minimization (Transient raw payload, feature + quality block output)
7. Feature Extraction (Clean records only)
8. Confidence Scoring (f(coverage, quality, model_confidence))
"""

from __future__ import annotations
import json
import os
import sys
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_layer"))
from aa_interface import DATA_DIR, get_consented_data, ConsentError
from build_customer_profiles import build_profile_for_customer
from schema import (
    Customer,
    Account,
    Transaction,
    EMIRecord,
    ConsentArtefact,
    IncomeType,
    Archetype,
    TransactionType,
    TransactionChannel,
    MerchantCategory,
    EMIStatus,
)

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
QUARANTINE_LOG_PATH = os.path.join(LOGS_DIR, "quarantine.jsonl")

# Whitelist of recognized Indian Financial Information Providers (FIPs)
RECOGNIZED_FIP_BANKS = {
    "hdfc bank",
    "state bank of india",
    "sbi",
    "icici bank",
    "axis bank",
    "kotak mahindra bank",
    "punjab national bank",
    "bank of baroda",
    "canara bank",
    "union bank of india",
    "federal bank",
    "idfc first bank",
    "indusind bank",
    "yes bank",
    "sahaay mock bank",
}


class SecurityValidationError(Exception):
    """Raised when incoming data fails schema, source, or security validation."""


def _ensure_logs_dir():
    os.makedirs(LOGS_DIR, exist_ok=True)


def _log_quarantine_record(record: dict, reason: str, customer_id: str):
    """Appends quarantined anomalous or malformed record to logs/quarantine.jsonl."""
    _ensure_logs_dir()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "customer_id": customer_id,
        "reason": reason,
        "record": record,
    }
    with open(QUARANTINE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# --------------------------------------------------------------------------
# 1. Encrypted Ingestion & Transport Abstraction
# --------------------------------------------------------------------------

def encrypted_ingest(customer_id: str, consent_id: str) -> dict:
    """
    Step 1: Secure Ingestion Wrapper.
    Simulates TLS 1.3 in-transit decryption and verifies consent validity
    via Account Aggregator boundary (aa_interface.py).
    """
    if not customer_id or not consent_id:
        raise SecurityValidationError("Missing customer_id or consent_id for secure ingestion.")
    
    # Calls AA interface which checks consent status, expiry, and customer existence
    raw_payload = get_consented_data(customer_id, consent_id)
    return raw_payload


# --------------------------------------------------------------------------
# 2. Schema Validation
# --------------------------------------------------------------------------

def validate_schema(payload: dict) -> Tuple[bool, List[str], dict]:
    """
    Step 2: Validates incoming records against data_layer/schema.py.
    Quarantines malformed items and returns clean payload.
    """
    errors = []
    customer = payload.get("customer", {})
    accounts = payload.get("accounts", [])
    transactions = payload.get("transactions", [])
    emi_records = payload.get("emi_records", [])
    consent = payload.get("consent", {})

    # Validate Customer schema
    required_cust_fields = ["customer_id", "name", "age", "gender", "income_type", "tier"]
    for f in required_cust_fields:
        if f not in customer or customer[f] is None:
            errors.append(f"Customer missing required field: {f}")

    # Validate Accounts
    valid_accounts = []
    for acc in accounts:
        if not acc.get("account_id") or not acc.get("bank_name"):
            errors.append(f"Malformed account record: {acc.get('account_id')}")
            _log_quarantine_record(acc, "Malformed Account Schema", customer.get("customer_id", "unknown"))
        else:
            valid_accounts.append(acc)

    # Validate Transactions
    valid_transactions = []
    for txn in transactions:
        try:
            amt = float(txn.get("amount", 0))
            if amt < 0:
                errors.append(f"Negative transaction amount: {txn.get('transaction_id')}")
                _log_quarantine_record(txn, "Negative Amount Schema Violation", customer.get("customer_id", "unknown"))
                continue
            if not txn.get("transaction_id") or not txn.get("timestamp"):
                errors.append(f"Malformed transaction missing id/timestamp: {txn.get('transaction_id')}")
                _log_quarantine_record(txn, "Missing ID/Timestamp", customer.get("customer_id", "unknown"))
                continue
            valid_transactions.append(txn)
        except (ValueError, TypeError):
            errors.append(f"Invalid transaction amount data type: {txn.get('transaction_id')}")
            _log_quarantine_record(txn, "Invalid Amount Type", customer.get("customer_id", "unknown"))

    # Validate EMIs
    valid_emis = []
    for emi in emi_records:
        if not emi.get("emi_id") or not emi.get("amount_due") is not None:
            errors.append(f"Malformed EMI record: {emi.get('emi_id')}")
            _log_quarantine_record(emi, "Malformed EMI Schema", customer.get("customer_id", "unknown"))
        else:
            valid_emis.append(emi)

    clean_payload = {
        "customer": customer,
        "accounts": valid_accounts,
        "transactions": valid_transactions,
        "emi_records": valid_emis,
        "consent": consent,
    }
    is_valid = len(errors) == 0
    return is_valid, errors, clean_payload


# --------------------------------------------------------------------------
# 3. Source Validation (Bank Whitelist)
# --------------------------------------------------------------------------

def validate_sources(payload: dict) -> Tuple[bool, List[str]]:
    """
    Step 3: Validates FIP and Bank source names against recognized scheduled bank whitelist.
    """
    errors = []
    consent = payload.get("consent", {})
    fip_name = str(consent.get("fip_name", "")).strip().lower()

    if fip_name and fip_name not in RECOGNIZED_FIP_BANKS:
        errors.append(f"Unrecognized FIP provider: '{consent.get('fip_name')}' not in authorized bank whitelist.")

    for acc in payload.get("accounts", []):
        bank = str(acc.get("bank_name", "")).strip().lower()
        if bank and bank not in RECOGNIZED_FIP_BANKS:
            errors.append(f"Unrecognized Bank Name: '{acc.get('bank_name')}' for account {acc.get('account_id')}.")

    return len(errors) == 0, errors


# --------------------------------------------------------------------------
# 4. Coverage Check
# --------------------------------------------------------------------------

def compute_data_coverage(payload: dict) -> Tuple[float, Optional[str], dict]:
    """
    Step 4: Computes data_coverage_score (0.0 to 1.0).
    Evaluates presence and depth of accounts, transactions, liabilities, and KYC.
    Never treats missing data as zero!
    
    If score < 0.70, attaches data_coverage_note.
    """
    customer = payload.get("customer", {})
    accounts = payload.get("accounts", [])
    transactions = payload.get("transactions", [])
    emi_records = payload.get("emi_records", [])

    coverage_components = {}

    # 1. Accounts linked (20%)
    has_accounts = len(accounts) > 0
    coverage_components["accounts_present"] = 0.20 if has_accounts else 0.0

    # 2. Transactions history depth & volume (35%)
    if len(transactions) == 0:
        txn_score = 0.0
    else:
        timestamps = [pd.to_datetime(t["timestamp"]) for t in transactions if "timestamp" in t]
        if timestamps:
            span_days = (max(timestamps) - min(timestamps)).days
            txn_count = len(transactions)
            if span_days >= 90 and txn_count >= 15:
                txn_score = 0.35
            elif span_days >= 30 and txn_count >= 5:
                txn_score = 0.22
            else:
                txn_score = 0.08  # Thin history (<30 days or few txns)
        else:
            txn_score = 0.05
    coverage_components["transaction_history_depth"] = round(txn_score, 3)

    # 3. Liabilities / EMI record coverage (25%)
    if len(emi_records) >= 3:
        emi_score = 0.25
    elif len(emi_records) > 0:
        emi_score = 0.18
    elif has_accounts and len(transactions) > 10:
        # Full AA consent confirmed no active loans
        emi_score = 0.25
    else:
        # Cold start / unverified liabilities
        emi_score = 0.05
    coverage_components["liabilities_coverage"] = round(emi_score, 3)

    # 4. KYC & Demographic completeness (20%)
    kyc_keys = ["name", "age", "gender", "income_type", "tier", "city", "state"]
    filled_kyc = sum(1 for k in kyc_keys if customer.get(k) is not None and str(customer.get(k)).strip() != "")
    kyc_score = 0.20 * (filled_kyc / len(kyc_keys))
    coverage_components["kyc_completeness"] = round(kyc_score, 3)

    data_coverage_score = round(sum(coverage_components.values()), 3)
    data_coverage_score = max(0.10, min(1.0, data_coverage_score))

    data_coverage_note = None
    if data_coverage_score < 0.70:
        data_coverage_note = "Some accounts may not be connected through AA yet (thin history or partial consent)"

    return data_coverage_score, data_coverage_note, coverage_components


# --------------------------------------------------------------------------
# 5. Anomaly / Fraud-on-Ingest Check
# --------------------------------------------------------------------------

def check_anomalies_on_ingest(payload: dict) -> Tuple[List[dict], List[dict]]:
    """
    Step 5: Flags implausible values before feature computation.
    - Transaction spike > 50x customer's own historical average.
    - Income credit jump > 100x customer's average credit amount.
    - Quarantines flagged transactions, excluding them from customer_profile.csv.
    """
    customer_id = payload.get("customer", {}).get("customer_id", "unknown")
    transactions = payload.get("transactions", [])
    if not transactions:
        return [], []

    df_txns = pd.DataFrame(transactions)
    if df_txns.empty or len(df_txns) < 3:
        return transactions, []

    df_txns["amount"] = pd.to_numeric(df_txns["amount"], errors="coerce").fillna(0.0)
    mean_amt = df_txns["amount"].mean()
    credits = df_txns[df_txns["txn_type"] == "credit"]
    mean_credit = credits["amount"].mean() if not credits.empty else mean_amt

    clean_txns = []
    quarantined = []

    for txn in transactions:
        amt = float(txn.get("amount", 0))
        txn_type = txn.get("txn_type")
        is_anom = False
        reason = ""

        # Check 1: 50x average transaction spike
        if mean_amt > 0 and amt > 50 * mean_amt:
            is_anom = True
            reason = f"Transaction amount ₹{amt:,.2f} is >50x customer historical average (₹{mean_amt:,.2f})"

        # Check 2: 100x income credit spike
        if txn_type == "credit" and mean_credit > 0 and amt > 100 * mean_credit:
            is_anom = True
            reason = f"Income credit amount ₹{amt:,.2f} is >100x customer historical credit average (₹{mean_credit:,.2f})"

        if is_anom:
            quarantined.append(txn)
            _log_quarantine_record(txn, reason, customer_id)
        else:
            clean_txns.append(txn)

    return clean_txns, quarantined


# --------------------------------------------------------------------------
# 8. Confidence Scoring Calculation
# --------------------------------------------------------------------------

def compute_overall_confidence(
    data_coverage_score: float,
    data_quality_score: float,
    model_confidence: float = 0.85,
    coverage_note: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Step 8: Computes unified confidence:
    overall_confidence = 0.40 * coverage + 0.30 * quality + 0.30 * model_confidence
    
    Returns confidence score, band (HIGH, MEDIUM, LOW), and explanatory note.
    """
    cov = max(0.0, min(1.0, float(data_coverage_score)))
    qual = max(0.0, min(1.0, float(data_quality_score)))
    mod = max(0.0, min(1.0, float(model_confidence)))

    score = round((0.40 * cov) + (0.30 * qual) + (0.30 * mod), 4)

    # Apply strict coverage cap: if coverage is thin (<0.60), overall confidence cannot exceed 0.65
    if cov < 0.60:
        score = min(score, 0.65)
    elif cov < 0.70:
        score = min(score, 0.75)

    if score >= 0.80:
        band = "HIGH"
    elif score >= 0.60:
        band = "MEDIUM"
    else:
        band = "LOW"

    return {
        "overall_confidence_score": score,
        "overall_confidence_band": band,
        "data_coverage_score": cov,
        "data_quality_score": qual,
        "model_confidence": mod,
        "data_coverage_note": coverage_note,
    }


# --------------------------------------------------------------------------
# Main Security Gateway Pipeline
# --------------------------------------------------------------------------

def process_secure_customer_pipeline(
    customer_id: str,
    consent_id: str,
    model_confidence_hint: float = 0.85,
) -> Dict[str, Any]:
    """
    Executes the complete 8-step Security & Risk Gateway pipeline:
    1. Encrypted Ingestion
    2. Schema Validation
    3. Source Validation
    4. Coverage Check
    5. Anomaly-on-Ingest Check (Quarantine)
    6. Data Minimization
    7. Feature Extraction (Clean data only)
    8. Confidence Scoring
    """
    # 1. Encrypted Ingestion
    raw_payload = encrypted_ingest(customer_id, consent_id)

    # 2. Schema Validation
    schema_ok, schema_errors, clean_payload = validate_schema(raw_payload)
    if not schema_ok and len(clean_payload["transactions"]) == 0 and len(clean_payload["accounts"]) == 0:
        raise SecurityValidationError(f"Critical schema failures: {schema_errors}")

    # 3. Source Validation
    source_ok, source_errors = validate_sources(clean_payload)
    if not source_ok:
        raise SecurityValidationError(f"Source validation rejected: {source_errors}")

    # 4. Coverage Check
    coverage_score, coverage_note, coverage_components = compute_data_coverage(clean_payload)

    # 5. Anomaly-on-Ingest Check
    clean_txns, quarantined_txns = check_anomalies_on_ingest(clean_payload)

    # Calculate Data Quality Score (penalized if malformed or quarantined records existed)
    total_raw_txns = len(raw_payload.get("transactions", []))
    quarantine_count = len(quarantined_txns)
    quality_ratio = 1.0 - (quarantine_count / max(1, total_raw_txns)) if total_raw_txns else 1.0
    data_quality_score = round(max(0.20, quality_ratio if schema_ok else quality_ratio * 0.85), 3)

    # 6. Data Minimization & 7. Feature Extraction
    # Transform validated clean records into DataFrames for feature computation
    cust_dict = clean_payload["customer"]
    clean_txns_df = pd.DataFrame(clean_txns)
    if not clean_txns_df.empty:
        clean_txns_df["timestamp"] = pd.to_datetime(clean_txns_df["timestamp"])
    clean_emis_df = pd.DataFrame(clean_payload["emi_records"])
    if not clean_emis_df.empty and "due_date" in clean_emis_df:
        clean_emis_df["due_date"] = pd.to_datetime(clean_emis_df["due_date"])

    consent_status = clean_payload.get("consent", {}).get("status", "ACTIVE")
    profile = build_profile_for_customer(cust_dict, clean_txns_df, clean_emis_df, consent_status)

    # 8. Confidence Scoring
    confidence_meta = compute_overall_confidence(
        data_coverage_score=coverage_score,
        data_quality_score=data_quality_score,
        model_confidence=model_confidence_hint,
        coverage_note=coverage_note,
    )

    # Attach confidence and coverage metadata to profile
    profile.update(confidence_meta)

    return {
        "customer_id": customer_id,
        "profile": profile,
        "gateway_metadata": {
            "schema_validated": True,
            "source_validated": True,
            "data_coverage_score": coverage_score,
            "data_coverage_note": coverage_note,
            "data_quality_score": data_quality_score,
            "quarantined_transactions_count": quarantine_count,
            "overall_confidence": confidence_meta,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        },
    }
