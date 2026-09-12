"""
Sahaay AI — Tests for RBAC Role Enforcement & Immutable Audit Logging (Risk 5)

Verifies:
1. /customers/{id}/data (raw) is accessible ONLY to db_admin (403 for chatbot/user).
2. /customers/{id}/chat is accessible to chatbot/user roles.
3. /admin/audit-log is accessible ONLY to admin (403 for non-admin).
4. Every call to /arbitrate, /chat, and /consent/revoke generates an audit log entry.
5. Revoking consent records an audit entry and updates consent status immediately.
"""

import json
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))
from main import app, AUDIT_LOG_PATH

client = TestClient(app)

CUSTOMER_ID = "cust_86838bd208"
CONSENT_ID = "consent_ea76cede3b"


def test_rbac_raw_data_blocks_chatbot_role():
    """Verify chatbot role cannot access raw unmasked financial records."""
    res = client.get(
        f"/customers/{CUSTOMER_ID}/data?consent_id={CONSENT_ID}",
        headers={"X-User-Role": "chatbot"},
    )
    assert res.status_code == 403
    assert "db_admin role required" in res.json()["detail"]


def test_rbac_raw_data_blocks_standard_user_role():
    """Verify standard user role cannot access raw database dumps."""
    res = client.get(
        f"/customers/{CUSTOMER_ID}/data?consent_id={CONSENT_ID}",
        headers={"X-User-Role": "user"},
    )
    assert res.status_code == 403
    assert "db_admin role required" in res.json()["detail"]


def test_rbac_raw_data_permits_db_admin():
    """Verify db_admin role successfully accesses raw consented records."""
    res = client.get(
        f"/customers/{CUSTOMER_ID}/data?consent_id={CONSENT_ID}",
        headers={"X-User-Role": "db_admin"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "customer" in data
    assert "accounts" in data
    assert "transactions" in data
    assert "emi_records" in data


def test_rbac_audit_log_requires_admin():
    """Verify non-admin roles receive 403 on /admin/audit-log."""
    res_user = client.get("/admin/audit-log", headers={"X-User-Role": "user"})
    assert res_user.status_code == 403

    res_bot = client.get("/admin/audit-log", headers={"X-User-Role": "chatbot"})
    assert res_bot.status_code == 403

    res_admin = client.get("/admin/audit-log", headers={"X-User-Role": "admin"})
    assert res_admin.status_code == 200
    assert "audit_logs" in res_admin.json()


def test_audit_trail_recorded_on_chat_and_arbitrate():
    """Verify chat and arbitrate endpoints append real-time immutable audit records."""
    # Send chat
    res_chat = client.post(
        f"/customers/{CUSTOMER_ID}/chat",
        json={"message": "What is my next EMI payment?", "language": "en"},
        headers={"X-User-Role": "user"},
    )
    assert res_chat.status_code == 200

    # Send arbitrate
    res_arb = client.get(
        f"/customers/{CUSTOMER_ID}/arbitrate",
        headers={"X-User-Role": "backend"},
    )
    assert res_arb.status_code == 200

    # Inspect audit log via admin endpoint
    res_log = client.get("/admin/audit-log", headers={"X-User-Role": "admin"})
    assert res_log.status_code == 200
    logs = res_log.json()["audit_logs"]
    assert len(logs) >= 2

    # Check endpoints in recent logs
    endpoints = [l["endpoint"] for l in logs]
    assert any("/chat" in ep for ep in endpoints)
    assert any("/arbitrate" in ep for ep in endpoints)


def test_revoke_consent_audit_and_status_reflection():
    """Verify consent revocation writes an audit line and invalidates active consent lookup."""
    target_cust = "cust_9d6cacdc98"
    # Look up active consent
    res_con = client.get(f"/customers/{target_cust}/consent")
    assert res_con.status_code == 200
    cid = res_con.json()["consent_id"]

    try:
        # Revoke
        res_rev = client.post(
            "/consent/revoke",
            json={"customer_id": target_cust, "consent_id": cid},
            headers={"X-User-Role": "user"},
        )
        assert res_rev.status_code == 200
        assert res_rev.json()["status"] == "revoked"

        # Verify audit log recorded revocation
        res_log = client.get("/admin/audit-log", headers={"X-User-Role": "admin"})
        logs = res_log.json()["audit_logs"]
        rev_entry = next((l for l in logs if l["endpoint"] == "/consent/revoke" and l["customer_id"] == target_cust), None)
        assert rev_entry is not None
        assert f"Consent revoked: {cid}" in rev_entry["decision_or_outcome"]

        # Verify GET /consent reflects revocation (404 no active consent)
        res_con_after = client.get(f"/customers/{target_cust}/consent")
        assert res_con_after.status_code == 404
    finally:
        # Restore consent in CSV for test repeatability
        import pandas as pd
        con_path = os.path.join(os.path.dirname(__file__), "..", "data", "consent_artefacts.csv")
        df_con = pd.read_csv(con_path)
        df_con.loc[(df_con["customer_id"] == target_cust) & (df_con["consent_id"] == cid), "status"] = "ACTIVE"
        df_con.to_csv(con_path, index=False)
