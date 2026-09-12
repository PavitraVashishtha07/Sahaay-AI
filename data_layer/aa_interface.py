"""
Sahaay AI — Account Aggregator Interface

This is the ONE module boundary between "where data comes from" and everything
built on top of it. Every downstream component (feature engineering, engines,
API routes) should only ever call get_consented_data() — never read the CSVs
directly.

Today: reads from the synthetic dataset in ../data/*.csv
Tomorrow: this function's internals change to call a real AA participant SDK
(e.g. Setu, Anumati) — nothing outside this file should need to change.

This also means: revoking consent, expiring consent, or a missing/invalid
consent_id are handled here, once, consistently — not scattered across the app.
"""

from __future__ import annotations
import os
from datetime import date
from typing import Optional

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def ensure_data_files_extracted():
    """Ensures all compressed .csv.gz files in data/ are decompressed on cloud boot."""
    if not os.path.exists(DATA_DIR):
        return
    import gzip, shutil
    for item in os.listdir(DATA_DIR):
        if item.endswith(".csv.gz"):
            csv_name = item[:-3]  # strip .gz -> .csv
            csv_path = os.path.join(DATA_DIR, csv_name)
            gz_path = os.path.join(DATA_DIR, item)
            if not os.path.exists(csv_path):
                with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)


# Run check on module load
ensure_data_files_extracted()


class ConsentError(Exception):
    """Raised when a request is made without valid, active consent."""


def _load(table: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{table}.csv")
    gz_path = os.path.join(DATA_DIR, f"{table}.csv.gz")

    # If uncompressed CSV is missing in container/deployment, auto-extract from .csv.gz
    if not os.path.exists(path) and os.path.exists(gz_path):
        import gzip, shutil
        with gzip.open(gz_path, "rb") as f_in, open(path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run generate_synthetic_data.py first."
        )
    return pd.read_csv(path)




def _check_consent(customer_id: str, consent_id: str) -> dict:
    consents = _load("consent_artefacts")
    row = consents[
        (consents["customer_id"] == customer_id) & (consents["consent_id"] == consent_id)
    ]
    if row.empty:
        raise ConsentError(f"No consent artefact found for customer={customer_id}, consent={consent_id}")

    consent = row.iloc[0].to_dict()
    if consent["status"] != "ACTIVE":
        raise ConsentError(f"Consent {consent_id} is not active (status={consent['status']})")

    validity_val = consent.get("validity_end") if pd.notna(consent.get("validity_end")) else consent.get("expires_at")
    if pd.notna(validity_val):
        validity_end = pd.to_datetime(validity_val).date()
        if validity_end < date.today():
            raise ConsentError(f"Consent {consent_id} expired on {validity_end}")

    return consent


def get_consented_data(customer_id: str, consent_id: str) -> dict:
    """
    The single entry point every other component should use.

    Returns a dict with keys: customer, accounts, transactions, emi_records, consent.
    Raises ConsentError if consent is missing, revoked, or expired.

    NOTE: does not return ground_truth_labels — those are evaluation-only and
    would never exist in a real AA response. Use get_ground_truth() explicitly
    and only in evaluation code.
    """
    consent = _check_consent(customer_id, consent_id)

    customers = _load("customers")
    accounts = _load("accounts")
    transactions = _load("transactions")
    emi_records = _load("emi_records")

    customer_row = customers[customers["customer_id"] == customer_id]
    if customer_row.empty:
        raise ValueError(f"Unknown customer_id: {customer_id}")

    return {
        "customer": customer_row.iloc[0].to_dict(),
        "accounts": accounts[accounts["customer_id"] == customer_id].to_dict(orient="records"),
        "transactions": transactions[transactions["customer_id"] == customer_id].to_dict(orient="records"),
        "emi_records": emi_records[emi_records["customer_id"] == customer_id].to_dict(orient="records"),
        "consent": consent,
    }


def get_all_customer_ids() -> list:
    """Convenience for batch jobs (e.g. nightly feature refresh) — a real AA
    integration would replace this with a portfolio/consent-registry query."""
    return _load("customers")["customer_id"].tolist()


def get_active_consent_id_for_customer(customer_id: str) -> Optional[str]:
    """Convenience lookup for the prototype only — a real system would already
    know the consent_id from its own consent-management flow."""
    consents = _load("consent_artefacts")
    row = consents[(consents["customer_id"] == customer_id) & (consents["status"] == "ACTIVE")]
    if row.empty:
        return None
    return row.iloc[0]["consent_id"]


def get_ground_truth(customer_id: str) -> dict:
    """Evaluation-only. Never call this from a production decision path."""
    labels = _load("ground_truth_labels")
    row = labels[labels["customer_id"] == customer_id]
    if row.empty:
        raise ValueError(f"No ground truth for customer_id: {customer_id}")
    return row.iloc[0].to_dict()


def revoke_consent(customer_id: str, consent_id: str) -> None:
    """
    Simulates a customer revoking consent mid-session (one of the adversarial
    scenarios called out in the research doc). Any component calling
    get_consented_data() after this will correctly receive a ConsentError.
    """
    path = os.path.join(DATA_DIR, "consent_artefacts.csv")
    consents = pd.read_csv(path)
    mask = (consents["customer_id"] == customer_id) & (consents["consent_id"] == consent_id)
    if not mask.any():
        raise ValueError("No matching consent artefact to revoke.")
    consents.loc[mask, "status"] = "REVOKED"
    consents.to_csv(path, index=False)
