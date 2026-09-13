import time
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_localhost():
    print("=" * 80)
    print("SAHAAY AI — LOCALHOST VERIFICATION SUITE")
    print(f"Target: {BASE_URL}")
    print("=" * 80)

    # 1. Health check
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/health")
    lat = (time.time() - t0) * 1000
    print(f"GET /health -> Status: {r.status_code} | Latency: {lat:.1f}ms | Response: {r.json()}")
    assert r.status_code == 200

    # 2. Canonical Frontend HTML Routes
    print("\n--- 2. CANONICAL FRONTEND HTML ROUTES ---")
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
        t0 = time.time()
        r = requests.get(f"{BASE_URL}{route}")
        lat = (time.time() - t0) * 1000
        is_html = "text/html" in r.headers.get("content-type", "")
        print(f"  {route:<26} -> Status: {r.status_code} | Latency: {lat:6.1f}ms | HTML: {'YES' if is_html else 'NO'} | Len: {len(r.text)}")
        assert r.status_code == 200
        assert is_html

    # 3. 307 Redirects
    print("\n--- 3. LEGACY 307 REDIRECTS ---")
    redirects = [
        ("/app", "/"),
        ("/onboarding", "/onboarding/name"),
        ("/recommendation", "/recommendation/detail"),
    ]
    for src, dst in redirects:
        r = requests.get(f"{BASE_URL}{src}", allow_redirects=False)
        loc = r.headers.get("location")
        print(f"  {src:<20} -> Status: {r.status_code} | Location: {loc}")
        assert r.status_code == 307
        assert loc == dst

    # 4. Personas End-to-End on Localhost
    print("\n--- 4. PERSONAS ARBITRATION ON LOCALHOST ---")
    personas = [
        ("Arjun Sharma (Stable Salaried)", "cust_86838bd208", "SUITABLE_PRODUCT_RECOMMENDATION", "xgboost"),
        ("Ravi Patel (Gig Irregular)", "cust_9d6cacdc98", "SUITABLE_PRODUCT_RECOMMENDATION", "xgboost"),
        ("Deepak Verma (Financially Stressed)", "cust_ffb3c15320", "FINANCIAL_STRESS_INTERVENTION", "xgboost"),
        ("Kavita Rao (Cold Start)", "cust_cold_start_new", "SUITABLE_PRODUCT_RECOMMENDATION", "gmm_fallback"),
    ]

    for label, cid, expected_action, expected_path in personas:
        t0 = time.time()
        r = requests.get(f"{BASE_URL}/customers/{cid}/arbitrate", headers={"X-User-Role": "analyst"})
        lat = (time.time() - t0) * 1000
        data = r.json()
        action = data.get("final_action")
        path_used = data.get("path_used") or data.get("action_payload", {}).get("path_used")
        print(f"  {label:<38} -> Status: {r.status_code} | Latency: {lat:6.1f}ms | Action: {action} | Path: {path_used}")
        assert r.status_code == 200
        assert action == expected_action
        assert path_used == expected_path

    # 5. Live Local Chat Test
    print("\n--- 5. LIVE CHAT ON LOCALHOST ---")
    t0 = time.time()
    r = requests.post(
        f"{BASE_URL}/customers/cust_86838bd208/chat",
        json={"message": "When is my next EMI due and how much?"},
        headers={"X-User-Role": "user"}
    )
    lat = (time.time() - t0) * 1000
    cdata = r.json()
    reply_ascii = cdata.get('reply', '').encode('ascii', errors='replace').decode('ascii')
    print(f"  POST /customers/cust_86838bd208/chat -> Status: {r.status_code} | Latency: {lat:6.1f}ms")
    print(f"    Intent: {cdata.get('intent')} | Language: {cdata.get('language')} | TTS: {cdata.get('tts_supported')}")
    print(f"    Reply: {reply_ascii}")
    assert r.status_code == 200

    print("\n" + "=" * 80)
    print("ALL LOCALHOST TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_localhost()
