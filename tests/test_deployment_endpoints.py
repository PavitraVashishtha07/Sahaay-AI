import os
import sys
import time
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    t0 = time.time()
    r = client.get("/health")
    latency_ms = (time.time() - t0) * 1000
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    assert "Sahaay AI" in data.get("service", "")
    print(f"\n[PASS] GET /health -> Status: 200, Latency: {latency_ms:.1f}ms, Response: {data}")

def test_personas():
    t0 = time.time()
    r = client.get("/personas")
    latency_ms = (time.time() - t0) * 1000
    assert r.status_code == 200
    personas = r.json().get("personas", [])
    assert len(personas) == 4
    persona_names = [p["name"] for p in personas]
    assert "Arjun Sharma" in persona_names
    assert "Kavita Rao" in persona_names
    print(f"\n[PASS] GET /personas -> Status: 200, Latency: {latency_ms:.1f}ms, Personas: {persona_names}")

def test_arbitrate_arjun():
    t0 = time.time()
    r = client.get("/customers/cust_86838bd208/arbitrate")
    latency_ms = (time.time() - t0) * 1000
    assert r.status_code == 200
    data = r.json()
    assert data["final_action"] == "SUITABLE_PRODUCT_RECOMMENDATION"
    path_used = data.get("path_used") or data.get("action_payload", {}).get("path_used")
    assert path_used == "xgboost"
    suitable = data.get("action_payload", {}).get("all_suitable_products", [])
    assert len(suitable) > 0
    print(f"\n[PASS] GET /customers/cust_86838bd208/arbitrate (Arjun) -> Action: {data['final_action']}, Tier: {data['priority_tier_applied']}, Path: {path_used}, Suitable: {suitable}")

def test_arbitrate_kavita_cold_start():
    t0 = time.time()
    r = client.get("/customers/cust_cold_start_new/arbitrate")
    latency_ms = (time.time() - t0) * 1000
    assert r.status_code == 200
    data = r.json()
    assert data["final_action"] == "SUITABLE_PRODUCT_RECOMMENDATION"
    path_used = data.get("path_used") or data.get("action_payload", {}).get("path_used")
    assert path_used == "gmm_fallback"
    suitable = data.get("action_payload", {}).get("all_suitable_products", [])
    print(f"\n[PASS] GET /customers/cust_cold_start_new/arbitrate (Kavita) -> Action: {data['final_action']}, Tier: {data['priority_tier_applied']}, Path: {path_used}, Suitable: {suitable}")

def test_chat_multilingual_end_to_end():
    # 1. Hindi query
    t0 = time.time()
    r1 = client.post("/customers/cust_86838bd208/chat", json={"message": "Meri agli EMI kitni hai?"})
    latency_ms1 = (time.time() - t0) * 1000
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["intent"] == "check_emi"
    assert len(d1["reply"]) > 0
    print(f"\n[PASS] POST /customers/cust_86838bd208/chat (Hindi) -> Intent: {d1['intent']}, Latency: {latency_ms1:.1f}ms")

    # 2. English Product Recommendation query
    t0 = time.time()
    r2 = client.post("/customers/cust_86838bd208/chat", json={"message": "Recommend some investment products for me."})
    latency_ms2 = (time.time() - t0) * 1000
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["intent"] == "ask_about_product"
    assert len(d2["reply"]) > 0
    print(f"\n[PASS] POST /customers/cust_86838bd208/chat (English) -> Intent: {d2['intent']}, Latency: {latency_ms2:.1f}ms")
