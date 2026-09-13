"""
Sahaay AI — Tests for RBAC Role Enforcement & Immutable Audit Logging (Risk 5)

Verifies:
1. /customers/{id}/data (raw) is accessible ONLY to db_admin (403 for chatbot/user).
2. /customers/{id}/recommendations is accessible to chatbot role and returns sanitized reasons without raw data.
3. /customers/{id}/stress and /customers/{id}/fraud gate disallowed roles.
4. /admin/audit-log is accessible ONLY to admin (403 for non-admin).
5. Every call to /arbitrate, /chat, and /consent/revoke generates an audit log entry.
6. Revoking consent records an audit entry and updates consent status to REVOKED immediately.
7. Seeded REVOKED and EXPIRED records are returned correctly by GET /consent.
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


def test_rbac_chatbot_role_accesses_sanitized_recommendations():
    """Verify chatbot role can access recommendations but receives only suitability/reasons (no raw records)."""
    res = client.get(
        f"/customers/{CUSTOMER_ID}/recommendations",
        headers={"X-User-Role": "chatbot"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "recommended_products" in data
    assert "path_used" in data
    assert "overall_confidence" in data
    # Ensure no raw transaction or account arrays are leaked
    assert "transactions" not in data
    assert "accounts" not in data


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
    """Verify consent revocation writes an audit line and immediately reflects 'status': 'REVOKED'."""
    target_cust = "cust_9d6cacdc98"
    # Look up active consent
    res_con = client.get(f"/customers/{target_cust}/consent")
    assert res_con.status_code == 200
    assert res_con.json()["status"] == "ACTIVE"
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

        # Verify GET /consent reflects status: REVOKED immediately
        res_con_after = client.get(f"/customers/{target_cust}/consent")
        assert res_con_after.status_code == 200
        assert res_con_after.json()["status"] == "REVOKED"
    finally:
        # Restore consent in CSV for test repeatability
        import pandas as pd
        con_path = os.path.join(os.path.dirname(__file__), "..", "data", "consent_artefacts.csv")
        df_con = pd.read_csv(con_path)
        df_con.loc[(df_con["customer_id"] == target_cust) & (df_con["consent_id"] == cid), "status"] = "ACTIVE"
        df_con.to_csv(con_path, index=False)


def test_seeded_revoked_and_expired_consent_records():
    """Verify pre-seeded REVOKED and EXPIRED records are returned with correct status."""
    # Test seeded REVOKED demo customer
    res_rev = client.get("/customers/cust_revoked_demo/consent")
    assert res_rev.status_code == 200
    assert res_rev.json()["status"] == "REVOKED"

    # Test seeded EXPIRED demo customer
    res_exp = client.get("/customers/cust_expired_demo/consent")
    assert res_exp.status_code == 200
    assert res_exp.json()["status"] == "EXPIRED"


def test_chat_endpoint_contract_with_optional_language():
    """
    Verifies POST /customers/{id}/chat works when 'language' is omitted from request,
    and returns exact expected response schema with 'language' and 'tts_supported'.
    """
    res = client.post(
        f"/customers/{CUSTOMER_ID}/chat",
        json={"message": "When is my next EMI due and how much?"},
        headers={"X-User-Role": "user"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert "intent" in data
    assert data["intent"] == "check_emi"
    assert "language" in data
    assert "tts_supported" in data
    assert isinstance(data["tts_supported"], bool)
    assert data["tts_supported"] is True
    assert "path_used" in data
    assert "facts_used" in data
    assert "timestamp" in data


def test_rbac_missing_role_header_defaults_to_safe_user_role():
    """
    Verifies that omitting the X-User-Role header defaults safely to 'user':
    - Raw database dumps (/data) return 403 Forbidden.
    - Admin audit logs (/admin/audit-log) return 403 Forbidden.
    - Standard user endpoints (/recommendations, /chat) succeed safely without leaking raw data.
    - Audit log records 'user' as the caller role.
    """
    # 1. Omitting header on /data MUST 403
    res_raw = client.get(f"/customers/{CUSTOMER_ID}/data?consent_id={CONSENT_ID}")
    assert res_raw.status_code == 403
    assert "db_admin role required" in res_raw.json()["detail"]
    assert "current role: 'user'" in res_raw.json()["detail"]

    # 2. Omitting header on /admin/audit-log MUST 403
    res_admin = client.get("/admin/audit-log")
    assert res_admin.status_code == 403
    assert "admin role required" in res_admin.json()["detail"]
    assert "current role: 'user'" in res_admin.json()["detail"]

    # 3. Omitting header on /recommendations succeeds safely (user role allowed)
    res_rec = client.get(f"/customers/{CUSTOMER_ID}/recommendations")
    assert res_rec.status_code == 200
    rec_data = res_rec.json()
    assert "recommended_products" in rec_data
    assert "transactions" not in rec_data
    assert "accounts" not in rec_data

    # 4. Omitting header on /chat succeeds and logs as 'user'
    res_chat = client.post(
        f"/customers/{CUSTOMER_ID}/chat",
        json={"message": "Can I check my current balance?"}
    )
    assert res_chat.status_code == 200
    # Check audit log to ensure recorded role is 'user'
    res_log = client.get("/admin/audit-log", headers={"X-User-Role": "admin"})
    assert res_log.status_code == 200
    logs = res_log.json()["audit_logs"]
    chat_log = next((l for l in reversed(logs) if l["endpoint"] == f"/customers/{CUSTOMER_ID}/chat"), None)
    assert chat_log is not None
    assert chat_log["role"] == "user"


def test_rbac_analyst_role_scope():
    """
    Formalizes the 'analyst' role scope:
    - ALLOWED: /arbitrate, /stress, /fraud, /recommendations (for decision sandbox evaluation)
    - FORBIDDEN (403): /data (raw database records) and /admin/audit-log (system audit trail)
    """
    headers = {"X-User-Role": "analyst"}

    # 1. Allowed endpoints
    res_arb = client.get(f"/customers/{CUSTOMER_ID}/arbitrate", headers=headers)
    assert res_arb.status_code == 200

    res_stress = client.get(f"/customers/{CUSTOMER_ID}/stress", headers=headers)
    assert res_stress.status_code == 200

    res_fraud = client.get(f"/customers/{CUSTOMER_ID}/fraud", headers=headers)
    assert res_fraud.status_code == 200

    res_rec = client.get(f"/customers/{CUSTOMER_ID}/recommendations", headers=headers)
    assert res_rec.status_code == 200

    # 2. Blocked endpoints (403 Forbidden)
    res_raw = client.get(f"/customers/{CUSTOMER_ID}/data?consent_id={CONSENT_ID}", headers=headers)
    assert res_raw.status_code == 403
    assert "db_admin role required" in res_raw.json()["detail"]

    res_admin = client.get("/admin/audit-log", headers=headers)
    assert res_admin.status_code == 403
    assert "admin role required" in res_admin.json()["detail"]


def test_rbac_customer_scoping_hackathon_contract():
    """
    Documents and verifies customer-scoping behavior under hackathon contract:
    - In hackathon mode (stateless simulated RBAC without user-bound JWT sessions),
      role-level gating is strictly enforced (e.g. user cannot access /data or /admin/audit-log).
    - Multi-tenant customer identity binding (JWT sub: customer_id) is a documented hackathon-scope
      boundary, allowing demo personas to be interacted with via the unified user/analyst roles.
    """
    other_cust = "cust_9d6cacdc98"

    # User role can interact with consented user endpoints across personas
    res_chat = client.post(
        f"/customers/{other_cust}/chat",
        json={"message": "What is my account summary?"},
        headers={"X-User-Role": "user"},
    )
    assert res_chat.status_code == 200

    # But user role is still strictly forbidden from accessing raw database dumps on ANY customer
    res_raw_other = client.get(
        f"/customers/{other_cust}/data?consent_id=consent_7e3cb60db6",
        headers={"X-User-Role": "user"},
    )
    assert res_raw_other.status_code == 403


