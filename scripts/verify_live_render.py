"""
Live Verification Script for Sahaay AI Deployed Backend
Tests all 7 audit items against https://sahaay-ai-euyo.onrender.com
"""

import time
import json
import urllib.request
import urllib.error

RENDER_URL = "https://sahaay-ai-euyo.onrender.com"

def make_request(path, method="GET", headers=None, data=None):
    url = f"{RENDER_URL}{path}"
    req_headers = {"User-Agent": "Sahaay-Auditor/1.0"}
    if headers:
        req_headers.update(headers)
    body = json.dumps(data).encode("utf-8") if data else None
    if data:
        req_headers["Content-Type"] = "application/json"
    
    req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            elapsed = time.time() - start
            res_body = resp.read().decode("utf-8")
            try:
                parsed = json.loads(res_body)
            except Exception:
                parsed = res_body
            return {"status": resp.status, "elapsed_ms": round(elapsed * 1000, 2), "data": parsed}
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start
        err_body = e.read().decode("utf-8")
        try:
            parsed = json.loads(err_body)
        except Exception:
            parsed = err_body
        return {"status": e.code, "elapsed_ms": round(elapsed * 1000, 2), "data": parsed}
    except Exception as e:
        elapsed = time.time() - start
        return {"status": 0, "elapsed_ms": round(elapsed * 1000, 2), "error": str(e)}

print(f"Connecting to live Render backend: {RENDER_URL}\n" + "="*70)

# Item 7: Demo Latency / Cold-Start Check
print("\n--- ITEM 7: DEMO LATENCY / HEALTH CHECK ---")
health_res = make_request("/health")
print(f"GET /health -> Status: {health_res['status']} | Latency: {health_res['elapsed_ms']} ms")
print(f"Response: {json.dumps(health_res.get('data'), indent=2)}")

# Item 1: RBAC Granularity
print("\n--- ITEM 1: RBAC GRANULARITY ---")
# 1a. Chatbot role on raw data (must return 403)
raw_bot = make_request("/customers/cust_86838bd208/data?consent_id=consent_ea76cede3b", headers={"X-User-Role": "chatbot"})
print(f"GET /customers/cust_86838bd208/data (X-User-Role: chatbot) -> Status: {raw_bot['status']} (Expected 403)")
print(f"Response: {json.dumps(raw_bot.get('data'), indent=2)}")

# 1b. Chatbot role on recommendations (must succeed 200 with sanitized data)
rec_bot = make_request("/customers/cust_86838bd208/recommendations", headers={"X-User-Role": "chatbot"})
print(f"\nGET /customers/cust_86838bd208/recommendations (X-User-Role: chatbot) -> Status: {rec_bot['status']} (Expected 200)")
if rec_bot['status'] == 200:
    data = rec_bot['data']
    print(f"path_used: {data.get('path_used')}")
    print(f"top_recommendations: {data.get('top_recommendations')}")
    print(f"recommended_products count: {len(data.get('recommended_products', []))}")
    print(f"sample reason codes: {data.get('recommended_products', [{}])[0].get('primary_reasons')}")
    print(f"has raw accounts: {'accounts' in data} | has raw transactions: {'transactions' in data}")

# 1c. db_admin role on raw data (must succeed 200)
raw_admin = make_request("/customers/cust_86838bd208/data?consent_id=consent_ea76cede3b", headers={"X-User-Role": "db_admin"})
print(f"\nGET /customers/cust_86838bd208/data (X-User-Role: db_admin) -> Status: {raw_admin['status']} (Expected 200)")
if raw_admin['status'] == 200:
    print(f"contains keys: {list(raw_admin['data'].keys())}")

# Item 2: GET /admin/audit-log endpoint
print("\n--- ITEM 2: GET /admin/audit-log ENDPOINT ---")
# 2a. Non-admin access (must return 403)
log_user = make_request("/admin/audit-log?limit=5", headers={"X-User-Role": "user"})
print(f"GET /admin/audit-log (X-User-Role: user) -> Status: {log_user['status']} (Expected 403)")
print(f"Response: {json.dumps(log_user.get('data'), indent=2)}")

# 2b. Admin access (must return 200 with logs)
log_admin = make_request("/admin/audit-log?limit=5", headers={"X-User-Role": "admin"})
print(f"\nGET /admin/audit-log (X-User-Role: admin) -> Status: {log_admin['status']} (Expected 200)")
print(f"Total entries: {log_admin.get('data', {}).get('total_entries')}")
print(f"Recent log preview:\n{json.dumps(log_admin.get('data', {}).get('audit_logs', [])[-2:], indent=2)}")

# Item 3: Consent Revocation -> Audit Trail Link
print("\n--- ITEM 3: CONSENT REVOCATION -> AUDIT TRAIL LINK ---")
test_cust = "cust_9d6cacdc98"
# 3a. Check consent before
con_before = make_request(f"/customers/{test_cust}/consent")
print(f"GET /customers/{test_cust}/consent (Before Revocation) -> Status: {con_before['status']}")
print(f"Response: {json.dumps(con_before.get('data'), indent=2)}")

cid = con_before.get('data', {}).get('consent_id', 'consent_7e3cb60db6')

# 3b. Call POST /consent/revoke
rev_res = make_request("/consent/revoke", method="POST", headers={"X-User-Role": "user"}, data={"customer_id": test_cust, "consent_id": cid})
print(f"\nPOST /consent/revoke -> Status: {rev_res['status']}")
print(f"Response: {json.dumps(rev_res.get('data'), indent=2)}")

# 3c. Check consent after (must reflect REVOKED)
con_after = make_request(f"/customers/{test_cust}/consent")
print(f"\nGET /customers/{test_cust}/consent (After Revocation) -> Status: {con_after['status']}")
print(f"Response: {json.dumps(con_after.get('data'), indent=2)}")

# 3d. Check audit log for revocation entry
log_after = make_request("/admin/audit-log?limit=5", headers={"X-User-Role": "admin"})
rev_entry = next((l for l in log_after.get('data', {}).get('audit_logs', []) if l.get('endpoint') == '/consent/revoke' and l.get('customer_id') == test_cust), None)
print(f"\nAudit Log Entry for Revocation:\n{json.dumps(rev_entry, indent=2)}")

# Item 4: Seeded REVOKED / EXPIRED Consent Rows
print("\n--- ITEM 4: SEEDED REVOKED / EXPIRED CONSENT ROWS ---")
demo_rev = make_request("/customers/cust_revoked_demo/consent")
print(f"GET /customers/cust_revoked_demo/consent -> Status: {demo_rev['status']}")
print(f"Response: {json.dumps(demo_rev.get('data'), indent=2)}")

demo_exp = make_request("/customers/cust_expired_demo/consent")
print(f"\nGET /customers/cust_expired_demo/consent -> Status: {demo_exp['status']}")
print(f"Response: {json.dumps(demo_exp.get('data'), indent=2)}")

seeded_rev = make_request("/customers/cust_18e35ec06a/consent")
print(f"\nGET /customers/cust_18e35ec06a/consent (Seeded Revoked in dataset) -> Status: {seeded_rev['status']}")
print(f"Response: {json.dumps(seeded_rev.get('data'), indent=2)}")

# Item 5: Confidence-Score Contrast (Arjun vs Kavita)
print("\n--- ITEM 5: CONFIDENCE-SCORE CONTRAST (ARJUN VS KAVITA) ---")
arb_arjun = make_request("/customers/cust_86838bd208/arbitrate")
arb_kavita = make_request("/customers/cust_cold_start_new/arbitrate")

print("\n--- ARJUN (cust_86838bd208 - Full History Stable Salaried) ---")
print(f"Status: {arb_arjun['status']}")
if arb_arjun['status'] == 200:
    conf_a = arb_arjun['data'].get('overall_confidence', {})
    print(f"path_used: {arb_arjun['data'].get('path_used')}")
    print(f"final_action: {arb_arjun['data'].get('final_action')}")
    print(f"overall_confidence_score: {conf_a.get('overall_confidence_score')}")
    print(f"overall_confidence_band: {conf_a.get('overall_confidence_band')}")
    print(f"data_coverage_score: {conf_a.get('data_coverage_score')}")
    print(f"data_coverage_note: {conf_a.get('data_coverage_note') or arb_arjun['data'].get('data_coverage_note')}")

print("\n--- KAVITA (cust_cold_start_new - Thin History Cold-Start) ---")
print(f"Status: {arb_kavita['status']}")
if arb_kavita['status'] == 200:
    conf_k = arb_kavita['data'].get('overall_confidence', {})
    print(f"path_used: {arb_kavita['data'].get('path_used')}")
    print(f"final_action: {arb_kavita['data'].get('final_action')}")
    print(f"overall_confidence_score: {conf_k.get('overall_confidence_score')}")
    print(f"overall_confidence_band: {conf_k.get('overall_confidence_band')}")
    print(f"data_coverage_score: {conf_k.get('data_coverage_score')}")
    print(f"data_coverage_note: {conf_k.get('data_coverage_note') or arb_kavita['data'].get('data_coverage_note')}")
