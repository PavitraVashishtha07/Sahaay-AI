import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    print("[PASS] /health works")

def test_personas():
    resp = client.get("/personas")
    assert resp.status_code == 200
    personas = resp.json()["personas"]
    assert len(personas) == 4
    print(f"[PASS] /personas works: {len(personas)} personas returned")

def test_dashboard():
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "Sahaay AI" in resp.text
    print("[PASS] /dashboard works")

def test_chat_stable_salaried():
    resp = client.post("/customers/cust_86838bd208/chat", json={"message": "When is my next EMI due?", "language": "en"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "check_emi"
    assert "reply" in data
    print(f"[PASS] /chat check_emi works: {data['reply']}")

def test_chat_hindi():
    resp = client.post("/customers/cust_86838bd208/chat", json={"message": "मेरी ईएमआई कितनी है?", "language": "hi"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "check_emi"
    print(f"[PASS] /chat hindi works: {data['reply']}")

def test_chat_out_of_scope():
    resp = client.post("/customers/cust_86838bd208/chat", json={"message": "What is quantum computing?", "language": "en"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "request_human_help"
    print(f"[PASS] /chat out-of-scope fallback works: {data['reply']}")

def test_onboarding():
    resp = client.get("/customers/cust_86838bd208/onboarding-state")
    assert resp.status_code == 200
    state = resp.json()
    print(f"[PASS] /onboarding-state works: step={state['current_step']}")

    resp_step = client.post("/customers/cust_86838bd208/onboarding-step", json={"input": "Arjun Sharma"})
    assert resp_step.status_code == 200
    print(f"[PASS] /onboarding-step works: next_step={resp_step.json()['next_step']}")

def test_all_4_personas_arbitrate():
    personas = client.get("/personas").json()["personas"]
    for p in personas:
        cid = p["customer_id"]
        resp = client.get(f"/customers/{cid}/arbitrate")
        assert resp.status_code == 200
        arb = resp.json()
        print(f"[PASS] Persona '{p['id']}' ({cid}): Tier={arb['priority_tier_applied']}, Action={arb['final_action']}, Path={arb['path_used']}")

if __name__ == "__main__":
    test_health()
    test_personas()
    test_dashboard()
    test_chat_stable_salaried()
    test_chat_hindi()
    test_chat_out_of_scope()
    test_onboarding()
    test_all_4_personas_arbitrate()
    print("\nALL API ENDPOINTS VERIFIED SUCCESSFULLY!")
