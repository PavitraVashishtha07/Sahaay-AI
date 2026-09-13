"""
Sahaay AI — Comprehensive Live Deployment Verification Script
Tests all canonical frontend routes, redirects, Stitch whitelisting,
RBAC endpoints, and all 4 personas on https://sahaay-ai-euyo.onrender.com
"""

import time
import json
import urllib.request
import urllib.error

RENDER_URL = "https://sahaay-ai-euyo.onrender.com"

def make_request(path, method="GET", headers=None, data=None, follow_redirects=True):
    url = f"{RENDER_URL}{path}"
    req_headers = {"User-Agent": "Sahaay-Auditor/2.0"}
    if headers:
        req_headers.update(headers)
    body = json.dumps(data).encode("utf-8") if data else None
    if data:
        req_headers["Content-Type"] = "application/json"
    
    class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            if not follow_redirects:
                return None
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(NoRedirectHandler) if not follow_redirects else urllib.request.build_opener()
    req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    
    start = time.time()
    try:
        with opener.open(req, timeout=60) as resp:
            elapsed = time.time() - start
            res_body = resp.read().decode("utf-8", errors="ignore")
            try:
                parsed = json.loads(res_body)
            except Exception:
                parsed = res_body
            return {"status": resp.status, "elapsed_ms": round(elapsed * 1000, 2), "data": parsed, "headers": dict(resp.headers)}
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start
        err_body = e.read().decode("utf-8", errors="ignore")
        try:
            parsed = json.loads(err_body)
        except Exception:
            parsed = err_body
        return {"status": e.code, "elapsed_ms": round(elapsed * 1000, 2), "data": parsed, "headers": dict(e.headers)}
    except Exception as e:
        elapsed = time.time() - start
        return {"status": 0, "elapsed_ms": round(elapsed * 1000, 2), "error": str(e)}

def wait_for_deployment(max_retries=15, interval=15):
    print(f"Checking deployment status at {RENDER_URL}/health ...")
    for i in range(max_retries):
        res = make_request("/health")
        if res.get("status") == 200:
            print(f"Render instance is ONLINE (Latency: {res['elapsed_ms']}ms)")
            return True
        print(f"Attempt {i+1}/{max_retries}: Status {res.get('status')} - waiting {interval}s for Render build...")
        time.sleep(interval)
    return False

if __name__ == "__main__":
    print("=" * 80)
    print("SAHAAY AI — LIVE DEPLOYMENT VERIFICATION SUITE")
    print(f"Target: {RENDER_URL}")
    print("=" * 80)

    is_online = wait_for_deployment()
    if not is_online:
        print("ERROR: Service did not become ready in time.")
        exit(1)

    # 1. Cold-Start Latency Measurement
    print("\n--- 1. COLD-START / ASSET LATENCY CHECK ---")
    latency_res = make_request("/")
    print(f"GET / -> Status: {latency_res['status']} | Latency: {latency_res['elapsed_ms']} ms | Length: {len(str(latency_res.get('data', '')))} chars")

    # 2. Canonical Frontend HTML Routes
    print("\n--- 2. CANONICAL FRONTEND HTML ROUTES (HTTP 200 + HTML Content) ---")
    canonical_paths = [
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
    all_routes_ok = True
    for path in canonical_paths:
        r = make_request(path)
        is_html = "<!DOCTYPE html>" in str(r.get("data", "")) or "<html" in str(r.get("data", ""))
        status_ok = r["status"] == 200 and is_html
        if not status_ok:
            all_routes_ok = False
        print(f"  {path:<26} -> Status: {r['status']} | Latency: {r['elapsed_ms']:>6.1f}ms | HTML: {'YES' if is_html else 'NO'}")

    # 3. Stitch Dynamic Whitelisting & Path-Traversal Security
    print("\n--- 3. STITCH WHITELISTING & PATH-TRAVERSAL SECURITY ---")
    stitch_checks = [
        ("/stitch/chat", [200], "Valid Stitch chat frame"),
        ("/stitch/privacy", [200], "Valid Stitch privacy frame"),
        ("/stitch/onboarding-name", [200], "Valid Stitch onboarding frame"),
        ("/stitch/unlisted_page_forbidden", [404], "Unlisted page must return 404"),
        ("/stitch/../../etc/passwd", [400, 404], "Path traversal attempt rejected with 400 or 404"),
    ]
    for path, exp_statuses, desc in stitch_checks:
        r = make_request(path)
        matched = r["status"] in exp_statuses
        print(f"  {path:<35} -> Status: {r['status']} (Expected {exp_statuses}) | Match: {'PASS' if matched else 'FAIL'} ({desc})")

    # 4. Legacy Continuity 307 Redirects
    print("\n--- 4. LEGACY CONTINUITY 307 REDIRECTS ---")
    redirects = [
        ("/app", 307, "/"),
        ("/onboarding", 307, "/onboarding/name"),
        ("/recommendation", 307, "/recommendation/detail"),
    ]
    for path, exp_code, exp_target in redirects:
        r = make_request(path, follow_redirects=False)
        loc = r.get("headers", {}).get("location", "")
        print(f"  {path:<20} -> Status: {r['status']} (Expected {exp_code}) | Location: '{loc}'")

    # 5. Live Persona End-to-End Verification (All 4 Personas)
    print("\n--- 5. LIVE PERSONAS END-TO-END ARBITRATION & CONFIDENCE ---")
    personas = [
        ("Arjun Sharma (Stable Salaried)", "cust_86838bd208", "Tier 5 / RECOMMEND_PRODUCTS", "HIGH (0.95)"),
        ("Ravi Patel (Gig Irregular)", "cust_9d6cacdc98", "Tier 5 / RECOMMEND_PRODUCTS", "HIGH"),
        ("Deepak Verma (Financially Stressed)", "cust_ffb3c15320", "Tier 2 / STRESS_INTERVENTION", "N/A (Stress Relief)"),
        ("Kavita Rao (Cold Start Thin History)", "cust_cold_start_new", "Tier 5 / gmm_fallback", "MEDIUM (0.50)"),
    ]
    for name, cid, exp_tier, exp_conf in personas:
        r = make_request(f"/customers/{cid}/arbitrate", headers={"X-User-Role": "analyst"})
        data = r.get("data", {})
        action = data.get("final_action")
        path_used = data.get("path_used")
        conf_band = data.get("overall_confidence", {}).get("overall_confidence_band")
        conf_score = data.get("overall_confidence", {}).get("overall_confidence_score")
        cov_score = data.get("overall_confidence", {}).get("data_coverage_score")
        print(f"\n  Persona: {name} ({cid})")
        print(f"    Status: {r['status']} | Latency: {r['elapsed_ms']}ms")
        print(f"    Action: {action} | Path: {path_used}")
        print(f"    Confidence: {conf_band} ({conf_score}) | Data Coverage: {cov_score}")
        if cid == "cust_ffb3c15320":
            suppressed = data.get("suppressed_products", [])
            print(f"    Suppressed Predatory Products: {suppressed}")

    # 6. Live Grounded Multilingual Chat Interaction
    print("\n--- 6. LIVE GROUNDED MULTILINGUAL CHAT API ---")
    chat_res = make_request(
        "/customers/cust_86838bd208/chat",
        method="POST",
        headers={"X-User-Role": "chatbot"},
        data={"message": "When is my next EMI due and how much?"}
    )
    print(f"POST /customers/cust_86838bd208/chat -> Status: {chat_res['status']} | Latency: {chat_res['elapsed_ms']}ms")
    if chat_res['status'] == 200:
        cdata = chat_res.get("data", {})
        reply_safe = str(cdata.get('reply')).encode('ascii', errors='replace').decode('ascii')
        print(f"  Reply: {reply_safe}")
        print(f"  Intent: {cdata.get('intent')} | Language: {cdata.get('language')} | TTS Supported: {cdata.get('tts_supported')}")
        print(f"  Facts Used: {cdata.get('facts_used')}")

    print("\n" + "=" * 80)
    print("LIVE DEPLOYMENT VERIFICATION COMPLETE!")
    print("=" * 80)

