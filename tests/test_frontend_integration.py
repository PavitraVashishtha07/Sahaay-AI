import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))

from app.main import app

client = TestClient(app)


def test_canonical_html_routes():
    """Verify all canonical frontend routes return HTTP 200 with text/html content-type."""
    canonical_routes = [
        "/",
        "/dashboard",
        "/chat",
        "/onboarding/name",
        "/onboarding/income",
        "/onboarding/purpose",
        "/onboarding/confirm",
        "/onboarding/completed",
        "/recommendation/detail",
        "/profile",
        "/cross-cutting-states",
    ]

    for route in canonical_routes:
        resp = client.get(route)
        assert resp.status_code == 200, f"Route {route} failed with status {resp.status_code}"
        assert "text/html" in resp.headers.get("content-type", ""), f"Route {route} did not return HTML"
        assert len(resp.text) > 200, f"Route {route} returned empty or too small HTML content"


def test_legacy_continuity_redirects():
    """Verify legacy paths cleanly redirect via 307 to canonical destinations without route multiplication."""
    redirect_checks = [
        ("/app", "/"),
        ("/app.html", "/"),
        ("/index.html", "/"),
        ("/dashboard.html", "/dashboard"),
        ("/chat.html", "/chat"),
        ("/profile-settings.html", "/profile"),
        ("/recommendation", "/recommendation/detail"),
        ("/recommendation.html", "/recommendation/detail"),
        ("/onboarding", "/onboarding/name"),
        ("/onboarding.html", "/onboarding/name"),
        ("/states.html", "/cross-cutting-states"),
    ]

    for legacy_path, target_path in redirect_checks:
        resp = client.get(legacy_path, follow_redirects=False)
        assert resp.status_code == 307, f"Legacy path {legacy_path} did not return 307 redirect"
        assert resp.headers.get("location") == target_path


def test_frontend_security_and_traversal_rejection():
    """Verify that unlisted pages, retired paths, and path traversal attempts are safely rejected with 404."""
    invalid_pages = ["/stitch/privacy", "/stitch/chat", "/non_existent_page", "/../main.py", "/etc/passwd", "/arbitrary_test"]
    for path in invalid_pages:
        resp = client.get(path)
        assert resp.status_code == 404, f"Unlisted/retired path '{path}' should return 404, got {resp.status_code}"


def test_onboarding_thin_aliases_and_state_machine():
    """Verify onboarding thin aliases call underlying tested logic and enforce state machine sequentially."""
    test_cust = "cust_test_onboarding_seq_99"

    # Step 0: Initial state check via alias is START
    r_init = client.get(f"/onboarding/{test_cust}/state")
    assert r_init.status_code == 200
    state_data = r_init.json()
    assert state_data.get("current_step") == "START"

    # Step 1: Start onboarding -> moves to COLLECT_NAME
    r_start = client.post(f"/onboarding/{test_cust}/step", json={"input": "start"})
    assert r_start.status_code == 200
    assert r_start.json().get("next_step") == "COLLECT_NAME"

    # Step 2: Provide Name -> moves to COLLECT_INCOME_TYPE
    r_step1 = client.post(f"/onboarding/{test_cust}/step", json={"input": "Priya Sharma"})
    assert r_step1.status_code == 200
    res1 = r_step1.json()
    assert res1.get("next_step") == "COLLECT_INCOME_TYPE"
    assert res1.get("session", {}).get("collected_data", {}).get("full_name") == "Priya Sharma"

    # Step 3: Provide Income -> moves to COLLECT_PURPOSE
    r_step2 = client.post(f"/onboarding/{test_cust}/step", json={"input": "salaried"})
    assert r_step2.status_code == 200
    res2 = r_step2.json()
    assert res2.get("next_step") == "COLLECT_PURPOSE"

    # Step 4: Verify Canonical Endpoint (/customers/{id}/onboarding-state) sees the same state
    r_canon = client.get(f"/customers/{test_cust}/onboarding-state")
    assert r_canon.status_code == 200
    assert r_canon.json().get("current_step") == "COLLECT_PURPOSE"


def test_rbac_security_gateway_preserved_for_frontend():
    """Confirm that the Security Gateway & RBAC rules cannot be bypassed from frontend routes."""
    customer_id = "cust_86838bd208"
    consent_id = "consent_ea76cede3b"

    # 1. Raw financial data requires db_admin role
    r_forbidden = client.get(f"/customers/{customer_id}/data?consent_id={consent_id}")
    assert r_forbidden.status_code == 403
    assert "db_admin role required" in r_forbidden.json().get("detail", "")

    # Authorized with db_admin header
    r_allowed = client.get(
        f"/customers/{customer_id}/data?consent_id={consent_id}",
        headers={"X-User-Role": "db_admin"}
    )
    assert r_allowed.status_code == 200
    assert "customer" in r_allowed.json()
    assert "accounts" in r_allowed.json()

    # 2. Audit log requires admin role
    r_audit_forbidden = client.get("/admin/audit-log")
    assert r_audit_forbidden.status_code == 403
    assert "admin role required" in r_audit_forbidden.json().get("detail", "")

    r_audit_allowed = client.get("/admin/audit-log", headers={"X-User-Role": "admin"})
    assert r_audit_allowed.status_code == 200
    assert "audit_logs" in r_audit_allowed.json()


def test_chat_contract_compliance():
    """Verify that POST /customers/{id}/chat returns expected payload schema."""
    resp = client.post(
        "/customers/cust_86838bd208/chat",
        json={"message": "Meri agli EMI kitni hai?"}
    )
    assert resp.status_code == 200
    data = resp.json()

    # Confirm required contract keys
    assert "reply" in data and isinstance(data["reply"], str)
    assert "intent" in data and isinstance(data["intent"], str)
    assert "language" in data and isinstance(data["language"], str)
    assert "tts_supported" in data and isinstance(data["tts_supported"], bool)
    assert "confidence" in data and isinstance(data["confidence"], (int, float))
    assert "facts_used" in data and isinstance(data["facts_used"], list)
