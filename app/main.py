"""
Sahaay AI — Core Application & Intelligence API

Exposes the full Sahaay AI stack over REST endpoints:
- Consent-gated Account Aggregator Data Layer (Module 1)
- Unified Customer Intelligence Profiles (Module 2)
- Stress Detection Engine (Section 4.3)
- Fraud & Behavioral Anomaly Engine (Section 4.4)
- Recommendation Engine with SHAP reasons & GMM cold-start fallback (Section 4.2)
- Arbitration & Governance Orchestration Engine (Section 4.5 & Section 6)
"""

from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_layer"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "engines"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "security"))

from aa_interface import (
    DATA_DIR,
    get_consented_data,
    get_all_customer_ids,
    get_active_consent_id_for_customer,
    revoke_consent,
    ConsentError,
)
from stress_engine import compute_stress_profile
from fraud_engine import detect_customer_anomalies
from recommendation_engine import generate_recommendations
from arbitration_engine import arbitrate_customer, arbitrate
from conversational_engine import (
    process_customer_message,
    onboarding_manager,
    extract_intent,
)

_PROFILE_PATH = os.path.join(DATA_DIR, "customer_profile.csv")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
AUDIT_LOG_PATH = os.path.join(LOGS_DIR, "audit.jsonl")


def get_role(request: Request, role_param: Optional[str] = None) -> str:
    """Extracts role from X-User-Role header or query param. Defaults to 'user'."""
    hdr = request.headers.get("X-User-Role") or request.headers.get("x-user-role")
    if hdr:
        return hdr.strip().lower()
    if role_param:
        return role_param.strip().lower()
    return "user"


def _sanitize_json_payload(obj: Any) -> Any:
    """Recursively replaces NaN, Infinity, and NaT with None so responses serialize cleanly."""
    if isinstance(obj, float):
        if pd.isna(obj) or pd.isna(obj) or obj != obj:
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_json_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json_payload(v) for v in obj]
    if pd.isna(obj):
        return None
    return obj


def log_audit_trail(endpoint: str, customer_id: str, role: str, outcome: Any):
    """Appends an immutable entry to logs/audit.jsonl."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "customer_id": customer_id,
        "role": role,
        "decision_or_outcome": outcome,
    }
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


app = FastAPI(
    title="Sahaay AI — Core Intelligence & Conversational API",
    description="Ethical, consent-gated financial companion with unified customer intelligence, deterministic arbitration, and grounded multilingual conversational layer.",
    version="1.0.0",
)

# Enable CORS for frontend access (supports wildcard or comma-separated origins via ALLOWED_ORIGINS)
raw_origins = os.getenv("ALLOWED_ORIGINS", "*").strip()
if raw_origins == "*" or not raw_origins:
    allowed_origins = ["*"]
else:
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True if allowed_origins != ["*"] else False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Pre-warms ML models and extracts dataset on server startup."""
    try:
        from aa_interface import ensure_data_files_extracted
        ensure_data_files_extracted()
    except Exception:
        pass
    try:
        from recommendation_engine import train_models
        train_models()
    except Exception:
        pass
    try:
        from fraud_engine import get_or_train_fraud_model
        get_or_train_fraud_model()
    except Exception:
        pass
    try:
        import gc
        gc.collect()
    except Exception:
        pass



class RevokeRequest(BaseModel):
    customer_id: str
    consent_id: str


class ArbitrateRequest(BaseModel):
    customer_intent: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = None


class OnboardingStepRequest(BaseModel):
    input: str


PERSONAS_METADATA = [
    {
        "id": "stable_salaried",
        "name": "Arjun Sharma",
        "title": "Stable Salaried IT Professional",
        "customer_id": "cust_86838bd208",
        "archetype": "stable_salaried",
        "income": "₹75,000 / month",
        "expected_tier": "Tier 5 (Proactive Optimization)",
        "expected_path": "xgboost",
        "key_signals": "Predictable monthly salary credits, 28% savings rate, zero missed EMIs.",
        "expected_action": "RECOMMEND_PRODUCTS (Mutual Fund SIP, Term Insurance)",
    },
    {
        "id": "gig_irregular",
        "name": "Ravi Patel",
        "title": "Gig Economy Driver / Partner",
        "customer_id": "cust_9d6cacdc98",
        "archetype": "gig_irregular",
        "income": "₹32,000 / month (Variable)",
        "expected_tier": "Tier 5 (Proactive Optimization)",
        "expected_path": "xgboost",
        "key_signals": "Daily/weekly micro-credits, fluctuating cash buffer, active bike loan.",
        "expected_action": "RECOMMEND_PRODUCTS (Flexible Micro-Credit, Small-Ticket RD)",
    },
    {
        "id": "financially_stressed",
        "name": "Deepak Verma",
        "title": "Distressed MSME Proprietor",
        "customer_id": "cust_ffb3c15320",
        "archetype": "financially_stressed",
        "income": "₹32,500 / month (Declining)",
        "expected_tier": "Tier 2 (Stress Intervention)",
        "expected_path": "deterministic_stress_relief",
        "key_signals": "3 missed EMIs, -₹18k 30-day cash outflow trend, high debt service ratio.",
        "expected_action": "STRESS_INTERVENTION (Suppresses credit cards/loans, offers EMI Restructuring)",
    },
    {
        "id": "cold_start",
        "name": "Kavita Rao",
        "title": "New-to-Bank / Thin History Customer",
        "customer_id": "cust_cold_start_new",
        "archetype": "thin_history",
        "income": "₹28,000 / month (Early Window)",
        "expected_tier": "Tier 5 / Tier 6 (Demographic GMM Fallback)",
        "expected_path": "gmm_fallback",
        "key_signals": "<30 days Account Aggregator transaction history, unsupervised clustering routing.",
        "expected_action": "GMM Fallback Safe Products (Recurring Deposit Small Ticket, Health Checkin)",
    },
]




@app.get("/health")
def health():
    return {"status": "ok", "service": "Sahaay AI Core Intelligence API"}


@app.get("/customers")
def list_customers():
    return {"customer_ids": get_all_customer_ids()}


@app.get("/customers/{customer_id}/consent")
def get_consent(customer_id: str):
    """
    Module 1 & Security Gateway: Returns active or latest consent artefact status
    for a given customer (ACTIVE, REVOKED, or EXPIRED).
    """
    from aa_interface import _load
    from datetime import date
    try:
        consents = _load("consent_artefacts")
        rows = consents[consents["customer_id"] == customer_id]
        if rows.empty:
            raise HTTPException(status_code=404, detail=f"No consent artefact found for customer '{customer_id}'.")

        latest = rows.iloc[-1].to_dict()
        status = str(latest.get("status", "ACTIVE")).upper()
        validity_val = latest.get("validity_end") if pd.notna(latest.get("validity_end")) else latest.get("expires_at")
        if pd.notna(validity_val) and status == "ACTIVE":
            try:
                validity_end = pd.to_datetime(validity_val).date()
                if validity_end < date.today():
                    status = "EXPIRED"
            except Exception:
                pass

        return _sanitize_json_payload({
            "customer_id": customer_id,
            "consent_id": latest.get("consent_id"),
            "status": status,
            "validity_start": str(latest.get("validity_start")) if pd.notna(latest.get("validity_start")) else None,
            "validity_end": str(validity_val) if pd.notna(validity_val) else None,
            "fip_name": latest.get("fip_name"),
            "fiu_name": latest.get("fiu_name"),
            "purpose": latest.get("purpose"),
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/customers/{customer_id}/data")
def get_data(customer_id: str, consent_id: str, request: Request, role: Optional[str] = None):
    """
    Module 1 & Security Gateway: Returns raw customer + accounts + transactions + EMI records.
    Strictly restricted to db_admin role per security access matrix. Chatbot/user roles receive 403.
    """
    current_role = get_role(request, role)
    if current_role != "db_admin":
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: db_admin role required for raw financial data (current role: '{current_role}').",
        )
    try:
        raw_data = get_consented_data(customer_id, consent_id)
        return _sanitize_json_payload(raw_data)
    except ConsentError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/customers/{customer_id}/profile")
def get_profile(customer_id: str, request: Request, role: Optional[str] = None):
    """
    Module 2: Returns the single shared customer_profile row (38 features),
    with internal ground truth stripped. Accessible to user, analyst, backend, admin.
    """
    if not os.path.exists(_PROFILE_PATH):
        raise HTTPException(status_code=503, detail="customer_profile.csv not built yet.")
    profiles = pd.read_csv(_PROFILE_PATH)
    row = profiles[profiles["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="No profile found for this customer.")
    record = row.iloc[0].to_dict()
    record.pop("archetype_ground_truth", None)
    return _sanitize_json_payload(record)


@app.get("/customers/{customer_id}/stress")
def get_stress(customer_id: str, request: Request, role: Optional[str] = None):
    """
    Section 4.3 & Security Gateway: Returns transparent stress score, stress band,
    causal drivers, and secondary isolation forest anomaly check.
    Accessible to admin, backend, analyst, user, and db_admin roles.
    """
    current_role = get_role(request, role)
    allowed_roles = {"admin", "backend", "analyst", "user", "db_admin"}
    if current_role not in allowed_roles:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: role '{current_role}' is not authorized to access stress engine records.",
        )
    if not os.path.exists(_PROFILE_PATH):
        raise HTTPException(status_code=503, detail="customer_profile.csv not built yet.")
    profiles = pd.read_csv(_PROFILE_PATH)
    row = profiles[profiles["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="No profile found for this customer.")
    return _sanitize_json_payload(compute_stress_profile(row.iloc[0].to_dict()))


@app.get("/customers/{customer_id}/fraud")
def get_fraud(customer_id: str, request: Request, role: Optional[str] = None):
    """
    Section 4.4 & Security Gateway: Returns transaction anomaly detection results,
    max score, and novelty signal flags.
    Accessible to admin, backend, analyst, user, and db_admin roles.
    """
    current_role = get_role(request, role)
    allowed_roles = {"admin", "backend", "analyst", "user", "db_admin"}
    if current_role not in allowed_roles:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: role '{current_role}' is not authorized to access fraud anomaly records.",
        )
    try:
        return _sanitize_json_payload(detect_customer_anomalies(customer_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/customers/{customer_id}/recommendations")
def get_recommendations(customer_id: str, request: Request, role: Optional[str] = None):
    """
    Section 4.2 & Security Gateway: Returns product suitability recommendations with SHAP reason
    codes and explicit path_used ('xgboost' or 'gmm_fallback').
    Accessible to chatbot, user, analyst, backend, admin. Returns strictly sanitized suitability
    and reason-code metadata without unmasked raw transaction records.
    """
    current_role = get_role(request, role)
    allowed_roles = {"chatbot", "user", "analyst", "backend", "admin", "db_admin"}
    if current_role not in allowed_roles:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: role '{current_role}' is not authorized to access recommendations.",
        )
    if not os.path.exists(_PROFILE_PATH):
        raise HTTPException(status_code=503, detail="customer_profile.csv not built yet.")
    profiles = pd.read_csv(_PROFILE_PATH)
    row = profiles[profiles["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="No profile found for this customer.")
    return _sanitize_json_payload(generate_recommendations(row.iloc[0].to_dict()))


@app.get("/customers/{customer_id}/arbitrate")
def get_arbitration(
    customer_id: str,
    request: Request,
    intent: Optional[str] = Query(None, description="Optional customer explicit intent (e.g. 'loan_inquiry')"),
    role: Optional[str] = None,
):
    """
    Section 4.5 & Security Gateway: Runs the complete arbitration governance hierarchy across
    all three engines and records an immutable audit log.
    """
    try:
        res = arbitrate_customer(customer_id, customer_intent=intent)
        current_role = get_role(request, role)
        log_audit_trail(
            endpoint=f"/customers/{customer_id}/arbitrate",
            customer_id=customer_id,
            role=current_role,
            outcome=res.get("final_action", "ARBITRATED"),
        )
        return _sanitize_json_payload(res)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/customers/{customer_id}/arbitrate")
def post_arbitration(customer_id: str, req: ArbitrateRequest, request: Request, role: Optional[str] = None):
    """
    Section 4.5: POST variant allowing explicit customer intent and context payloads.
    """
    try:
        res = arbitrate_customer(customer_id, customer_intent=req.customer_intent, context=req.context)
        current_role = get_role(request, role)
        log_audit_trail(
            endpoint=f"/customers/{customer_id}/arbitrate",
            customer_id=customer_id,
            role=current_role,
            outcome=res.get("final_action", "ARBITRATED"),
        )
        return _sanitize_json_payload(res)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/consent/revoke")
def revoke(req: RevokeRequest, request: Request, role: Optional[str] = None):
    """Simulates a customer revoking consent mid-session and records audit log."""
    try:
        revoke_consent(req.customer_id, req.consent_id)
        current_role = get_role(request, role)
        log_audit_trail(
            endpoint="/consent/revoke",
            customer_id=req.customer_id,
            role=current_role,
            outcome=f"Consent revoked: {req.consent_id}",
        )
        return {"status": "revoked", "customer_id": req.customer_id, "consent_id": req.consent_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/personas")
def get_personas():
    """
    Returns the 4 pre-configured personas for the demo interface.
    Each persona maps to a real customer scenario in the dataset.
    """
    return {"personas": PERSONAS_METADATA}


@app.post("/customers/{customer_id}/chat")
def chat(customer_id: str, req: ChatRequest, request: Request, role: Optional[str] = None):
    """
    Section 5 & Security Gateway: Natural language customer interaction layer.
    Extracts structured intent via Gemini Flash / fallback, pulls verified backend facts,
    and returns a non-hallucinated response with audit logging.
    """
    try:
        res = process_customer_message(customer_id, req.message, language=req.language)
        current_role = get_role(request, role)
        log_audit_trail(
            endpoint=f"/customers/{customer_id}/chat",
            customer_id=customer_id,
            role=current_role,
            outcome={"intent": res["intent"], "reply_preview": res["reply"][:80]},
        )
        return {
            "reply": res["reply"],
            "intent": res["intent"],
            "language": res["language"],
            "tts_supported": res.get("tts_supported", True),
            "path_used": res["path_used"],
            "facts_used": res.get("facts_used_keys", ["verified_backend_records"]),
            "confidence": res["confidence"],
            "timestamp": res["timestamp"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/admin/audit-log")
def get_audit_log(request: Request, limit: int = Query(50, ge=1, le=1000), role: Optional[str] = None):
    """
    Security Gateway (Risk 5): Protected administrative audit trail.
    Accessible only by admin role.
    """
    current_role = get_role(request, role)
    if current_role != "admin":
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: admin role required for audit logs (current role: '{current_role}').",
        )

    if not os.path.exists(AUDIT_LOG_PATH):
        return {"audit_logs": [], "total_entries": 0}

    entries = []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return {"audit_logs": entries[-limit:], "total_entries": len(entries)}


@app.get("/customers/{customer_id}/onboarding-state")
def get_onboarding_state(customer_id: str):
    """
    Section 5: Returns the current state-machine step for a customer
    going through the onboarding / KYC flow.
    """
    return onboarding_manager.get_state(customer_id)


@app.post("/customers/{customer_id}/onboarding-step")
def advance_onboarding(customer_id: str, req: OnboardingStepRequest):
    """
    Section 5: Advances the onboarding state machine sequentially.
    """
    res = onboarding_manager.advance_step(customer_id, req.input)
    return res


# ==============================================================================
# ONBOARDING ENDPOINT ALIASES (Thin forwarding to Section 5 handlers)
# ==============================================================================
@app.get("/onboarding/{customer_id}/state", include_in_schema=False)
def get_onboarding_state_alias(customer_id: str):
    """Thin alias for /customers/{customer_id}/onboarding-state."""
    return get_onboarding_state(customer_id)


@app.post("/onboarding/{customer_id}/step", include_in_schema=False)
def advance_onboarding_alias(customer_id: str, req: OnboardingStepRequest):
    """Thin alias for /customers/{customer_id}/onboarding-step."""
    return advance_onboarding(customer_id, req)


# ==============================================================================
# PRODUCTION FRONTEND SERVING & CANONICAL ROUTE MOUNTING
# ==============================================================================
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "public")
_CSS_DIR = os.path.join(_FRONTEND_DIR, "css")

if os.path.exists(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")
if os.path.exists(_CSS_DIR):
    app.mount("/css", StaticFiles(directory=_CSS_DIR), name="css")


def serve_html(filename: str) -> HTMLResponse:
    """Helper to serve static HTML pages safely from the frontend public directory."""
    filepath = os.path.join(_FRONTEND_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    raise HTTPException(status_code=404, detail=f"Page '{filename}' not found.")


# Canonical Page Routes
@app.get("/", response_class=HTMLResponse)
def page_home():
    """Customer Portal Homepage (app.html)."""
    return serve_html("app.html")


@app.get("/dashboard", response_class=HTMLResponse)
def page_dashboard():
    """Decision Sandbox & Intelligence Telemetry Dashboard (dashboard.html)."""
    return serve_html("dashboard.html")


@app.get("/chat", response_class=HTMLResponse)
def page_chat():
    """Vernacular Conversational AI Assistant (chat-assistant.html)."""
    return serve_html("chat-assistant.html")


@app.get("/onboarding/name", response_class=HTMLResponse)
def page_onboarding_name():
    """Onboarding Step 1: Collect Name (onboarding.html)."""
    return serve_html("onboarding.html")


@app.get("/onboarding/income", response_class=HTMLResponse)
def page_onboarding_income():
    """Onboarding Step 2: Income Type (onboarding-income.html)."""
    return serve_html("onboarding-income.html")


@app.get("/onboarding/purpose", response_class=HTMLResponse)
def page_onboarding_purpose():
    """Onboarding Step 3: Purpose Selection (onboarding-purpose.html)."""
    return serve_html("onboarding-purpose.html")


@app.get("/onboarding/confirm", response_class=HTMLResponse)
def page_onboarding_confirm():
    """Onboarding Step 4: Confirmation & Summary (onboarding-confirm.html)."""
    return serve_html("onboarding-confirm.html")


@app.get("/onboarding/completed", response_class=HTMLResponse)
def page_onboarding_completed():
    """Onboarding Step 5: Completion & Celebration (onboarding-completed.html)."""
    return serve_html("onboarding-completed.html")


@app.get("/recommendation/detail", response_class=HTMLResponse)
def page_recommendation_detail():
    """Recommendation Deep-Dive View (recommendation-detail.html)."""
    return serve_html("recommendation-detail.html")


@app.get("/profile", response_class=HTMLResponse)
def page_profile():
    """Customer Profile & 10-Language Settings (profile.html)."""
    return serve_html("profile.html")


@app.get("/cross-cutting-states", response_class=HTMLResponse)
def page_cross_cutting_states():
    """Cross-cutting UI states: Skeleton & gentle error recovery (cross-cutting-states.html)."""
    return serve_html("cross-cutting-states.html")


# Whitelisted Stitch UI Reference Frames
STITCH_PAGE_WHITELIST: Dict[str, str] = {
    "privacy": "stitch_privacy_consent.html",
    "chat": "stitch_chat_assistant.html",
    "cross-cutting": "stitch_cross_cutting_states.html",
    "onboarding-completed": "stitch_onboarding_completed.html",
    "onboarding-confirm": "stitch_onboarding_confirm.html",
    "onboarding-income": "stitch_onboarding_income.html",
    "onboarding-name": "stitch_onboarding_name.html",
    "onboarding-purpose": "stitch_onboarding_purpose.html",
    "profile": "stitch_profile_settings.html",
    "recommendation": "stitch_recommendation_detail.html",
    "master": "stitch.html",
}


@app.get("/stitch/{page}", response_class=HTMLResponse)
def page_stitch_whitelisted(page: str):
    """Serves 1:1 Stitch design system frames validated against an explicit whitelist."""
    normalized_key = page.lower().replace(".html", "").replace("_", "-")
    target_file = STITCH_PAGE_WHITELIST.get(normalized_key)
    if not target_file:
        raise HTTPException(status_code=404, detail=f"Stitch design frame '{page}' not found or not in allowed list.")
    return serve_html(target_file)


# Legacy continuity redirects (clean redirects, not duplicate handlers)
@app.get("/app", include_in_schema=False)
@app.get("/app.html", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def redirect_app():
    return RedirectResponse(url="/", status_code=307)


@app.get("/dashboard.html", include_in_schema=False)
def redirect_dashboard_html():
    return RedirectResponse(url="/dashboard", status_code=307)


@app.get("/chat.html", include_in_schema=False)
@app.get("/chat-assistant.html", include_in_schema=False)
def redirect_chat_html():
    return RedirectResponse(url="/chat", status_code=307)


@app.get("/profile.html", include_in_schema=False)
@app.get("/profile-settings.html", include_in_schema=False)
def redirect_profile_html():
    return RedirectResponse(url="/profile", status_code=307)


@app.get("/recommendation", include_in_schema=False)
@app.get("/recommendation.html", include_in_schema=False)
@app.get("/recommendation-detail.html", include_in_schema=False)
def redirect_recommendation():
    return RedirectResponse(url="/recommendation/detail", status_code=307)


@app.get("/onboarding", include_in_schema=False)
@app.get("/onboarding.html", include_in_schema=False)
@app.get("/onboarding-name.html", include_in_schema=False)
def redirect_onboarding():
    return RedirectResponse(url="/onboarding/name", status_code=307)


@app.get("/onboarding-income.html", include_in_schema=False)
def redirect_onboarding_income_html():
    return RedirectResponse(url="/onboarding/income", status_code=307)


@app.get("/onboarding-purpose.html", include_in_schema=False)
def redirect_onboarding_purpose_html():
    return RedirectResponse(url="/onboarding/purpose", status_code=307)


@app.get("/onboarding-confirm.html", include_in_schema=False)
def redirect_onboarding_confirm_html():
    return RedirectResponse(url="/onboarding/confirm", status_code=307)


@app.get("/onboarding-completed.html", include_in_schema=False)
def redirect_onboarding_completed_html():
    return RedirectResponse(url="/onboarding/completed", status_code=307)


@app.get("/cross-cutting-states.html", include_in_schema=False)
@app.get("/states.html", include_in_schema=False)
def redirect_states_html():
    return RedirectResponse(url="/cross-cutting-states", status_code=307)

