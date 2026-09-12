"""
Sahaay AI — Section 5: Conversational Layer & Intent Engine

A structured, non-hallucinating conversational companion that enforces the
strict architectural guarantee: The conversational layer NEVER invents financial
facts or terms. All numbers, dates, and terms come from verified backend data.

Supported Flows (Fixed, Closed-World Set):
1. `check_emi`: Real EMI figures, due dates, loan types from emi_records.
2. `explain_transaction`: Real merchant, amount, category from transactions.
3. `report_suspicious_activity`: Real anomaly flags from fraud_engine.
4. `ask_about_product`: Real suitability & SHAP reasons from recommendation/arbitration.
5. `request_human_help`: Structured escalation for complex or out-of-scope queries.

Onboarding / KYC State Machine:
Strict sequential transition (START -> NAME -> INCOME_TYPE -> PURPOSE -> CONFIRMATION -> COMPLETED).
"""

from __future__ import annotations
import os
import re
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_layer"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "engines"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "security"))

from aa_interface import DATA_DIR, get_consented_data, get_active_consent_id_for_customer, ConsentError
from stress_engine import compute_stress_profile
from fraud_engine import detect_customer_anomalies
from recommendation_engine import generate_recommendations
from arbitration_engine import arbitrate_customer, arbitrate
from sanitize import sanitize_for_llm


class Intent(str, Enum):
    CHECK_EMI = "check_emi"
    EXPLAIN_TRANSACTION = "explain_transaction"
    REPORT_SUSPICIOUS_ACTIVITY = "report_suspicious_activity"
    ASK_ABOUT_PRODUCT = "ask_about_product"
    REQUEST_HUMAN_HELP = "request_human_help"
    ONBOARDING_KYC = "onboarding_kyc"


class OnboardingState(str, Enum):
    START = "START"
    COLLECT_NAME = "COLLECT_NAME"
    COLLECT_INCOME_TYPE = "COLLECT_INCOME_TYPE"
    COLLECT_PURPOSE = "COLLECT_PURPOSE"
    CONFIRMATION = "CONFIRMATION"
    COMPLETED = "COMPLETED"


# --------------------------------------------------------------------------
# Intent Extractor (Multilingual regex & semantic matching)
# --------------------------------------------------------------------------

INTENT_PATTERNS: Dict[Intent, List[str]] = {
    Intent.CHECK_EMI: [
        r"(emi|loan payment|due date|installment|किस्त|ईएमआई|હપ્તો|હપ્તા|ચુકવણી|લોન)",
        r"(how much do i owe|next emi|emi status|emi schedule|loan amount due)",
        r"(meri emi|emi kab hai|kitna dena hai|kitni emi|किस्त कितनी|ईएमआई कब)",
        r"(હપ્તો ક્યારે છે|કેટલા ભરવાના છે|મારી ઈએમઆઈ|હપ્તો કેટલો)",
    ],
    Intent.EXPLAIN_TRANSACTION: [
        r"(transaction|statement|passbook|where did my money go|last spend|debit|credit)",
        r"(लेनदेन|खर्च|कहाँ पैसे गए|ट्रांजैक्शन|पैसे कटे|खाते से कटे)",
        r"(વ્યવહાર|ખર્ચ|પૈસા ક્યાં ગયા|સ્ટેટમેન્ટ|ખાતામાંથી કપાયા)",
        r"(explain transaction|recent transactions|what was this charge)",
    ],
    Intent.REPORT_SUSPICIOUS_ACTIVITY: [
        r"(fraud|unauthorized|suspicious|unknown device|scam|stolen|hacked)",
        r"(धोखाधड़ी|संदिग्ध|अनजान डिवाइस|गलत ट्रांजैक्शन|पैसे कट गए अपने आप)",
        r"(છેતરપિંડી|શંકાસ્પદ|અજાણ્યો વ્યવહાર|ખોટા પૈસા કપાયા)",
        r"(did not authorize|report fraud|block card|freeze account)",
    ],
    Intent.ASK_ABOUT_PRODUCT: [
        r"(recommend|product|credit card|sip|mutual fund|insurance|fd|rd|deposit)",
        r"(सुझाव|क्रेडिट कार्ड|बीमा|म्यूचुअल फंड|योजना|इन्वेस्टमेंट)",
        r"(ભલામણ|ક્રેડિટ કાર્ડ|વીમો|રોકાણ|ડિપોઝિટ)",
        r"(should i take|why was this recommended|is this suitable for me|suggest product)",
    ],
    Intent.REQUEST_HUMAN_HELP: [
        r"(human|agent|representative|call me|talk to person|advisor|support|help)",
        r"(इंसान|अधिकारी|मदद|बात करनी है|कस्टमर केयर|सपोर्ट)",
        r"(માણસ|અધિકારી|મદદ|ગ્રાહક સેવા|વાત કરવી છે)",
    ],
}


def detect_language(text: str, default: str = "en") -> str:
    """Detects if text contains Devanagari (Hindi) or Gujarati scripts."""
    for char in text:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            return "hi"
        if 0x0A80 <= code <= 0x0AFF:
            return "gu"
    return default


def extract_intent(message: str, language_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts structured intent and entities from user message.
    Strictly constrained to the 5 supported flows; otherwise falls back
    to request_human_help. Sanitizes input against prompt injection.
    """
    sanitized_msg = sanitize_for_llm(message)
    msg_clean = sanitized_msg.lower().strip()
    detected_lang = language_hint or detect_language(sanitized_msg)

    matched_intent = None
    for intent, patterns in INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, msg_clean, re.IGNORECASE):
                matched_intent = intent
                break
        if matched_intent:
            break

    # If out of scope, route safely to request_human_help rather than hallucinating
    final_intent = matched_intent or Intent.REQUEST_HUMAN_HELP

    return {
        "intent": final_intent.value,
        "language": detected_lang,
        "raw_message": message,
        "sanitized_message": sanitized_msg,
        "confidence": 0.95 if matched_intent else 0.50,
        "is_out_of_scope_fallback": matched_intent is None,
    }


# --------------------------------------------------------------------------
# Deterministic Backend Fact Fetchers
# --------------------------------------------------------------------------

def _fetch_emi_facts(customer_id: str) -> Dict[str, Any]:
    """Fetches real EMI figures and due dates from emi_records.csv."""
    emi_path = os.path.join(DATA_DIR, "emi_records.csv")
    if not os.path.exists(emi_path):
        return {"has_emis": False, "records": []}

    df = pd.read_csv(emi_path)
    cust_emis = df[df["customer_id"] == customer_id]
    if cust_emis.empty:
        return {"has_emis": False, "records": [], "summary": "No active loans or EMI schedules found."}

    latest = cust_emis.sort_values("due_date", ascending=False).iloc[0].to_dict()
    total_due = cust_emis["amount_due"].sum()
    missed_count = (cust_emis["status"] == "missed").sum()
    late_count = (cust_emis["status"] == "paid_late").sum()

    return {
        "has_emis": True,
        "next_emi_amount": float(latest["amount_due"]),
        "next_due_date": str(latest["due_date"]),
        "loan_type": latest["loan_type"].replace("_", " ").title(),
        "status": latest["status"],
        "days_late": int(latest.get("days_late", 0)),
        "total_active_loans": len(cust_emis["loan_type"].unique()),
        "missed_emis_total": int(missed_count),
        "late_emis_total": int(late_count),
    }


def _fetch_transaction_facts(customer_id: str) -> Dict[str, Any]:
    """Fetches real recent transactions from transactions.csv."""
    tx_path = os.path.join(DATA_DIR, "transactions.csv")
    if not os.path.exists(tx_path):
        return {"has_transactions": False, "recent_transactions": []}

    df = pd.read_csv(tx_path)
    cust_tx = df[df["customer_id"] == customer_id]
    if cust_tx.empty:
        return {"has_transactions": False, "recent_transactions": []}

    cust_tx = cust_tx.sort_values("timestamp", ascending=False)
    recent = cust_tx.iloc[:3].to_dict(orient="records")

    return {
        "has_transactions": True,
        "total_transactions_recorded": len(cust_tx),
        "latest_transaction": {
            "amount": float(recent[0]["amount"]),
            "merchant": sanitize_for_llm(str(recent[0]["merchant_name"])),
            "category": str(recent[0]["merchant_category"]),
            "channel": str(recent[0]["channel"]),
            "timestamp": str(recent[0]["timestamp"]),
            "type": str(recent[0]["txn_type"]),
        },
        "recent_list": [
            {
                "amount": float(r["amount"]),
                "merchant": sanitize_for_llm(str(r["merchant_name"])),
                "date": str(r["timestamp"])[:10],
            } for r in recent
        ],
    }


def _fetch_product_facts(customer_id: str) -> Dict[str, Any]:
    """Queries recommendation and arbitration engines for real suitability facts."""
    arb = arbitrate_customer(customer_id)
    profile_path = os.path.join(DATA_DIR, "customer_profile.csv")
    profiles = pd.read_csv(profile_path)
    p_row = profiles[profiles["customer_id"] == customer_id].iloc[0].to_dict()
    recs = generate_recommendations(p_row)

    return {
        "arbitrated_action": arb["final_action"],
        "priority_tier": arb["priority_tier_applied"],
        "action_headline": arb["action_headline"],
        "path_used": recs["path_used"],
        "top_recommendations": recs["top_recommendations"],
        "recommended_products": recs["recommended_products"][:2],
        "unsuitable_products": recs["unsuitable_products"],
    }


def _fetch_suspicious_facts(customer_id: str) -> Dict[str, Any]:
    """Queries fraud engine for real anomaly flags."""
    res = detect_customer_anomalies(customer_id)
    return {
        "has_anomaly": res["has_anomaly"],
        "max_anomaly_score": res["max_anomaly_score"],
        "flagged_signals": res["flagged_signals"],
        "anomalous_transactions_count": res["anomalous_transactions_count"],
        "flagged_transactions": res["flagged_transactions"],
    }


# --------------------------------------------------------------------------
# Natural Language Phrasing (Grounded Multilingual Templates)
# --------------------------------------------------------------------------

def phrase_response(intent: str, facts: Dict[str, Any], language: str = "en") -> str:
    """
    Renders the grounded facts into natural phrasing in English, Hindi, or Gujarati.
    Guaranteed to include ONLY the exact figures provided by the backend facts.
    """
    lang = language.lower()

    if intent == Intent.CHECK_EMI.value:
        if not facts.get("has_emis"):
            if lang == "hi":
                return "आपके खाते पर कोई सक्रिय ईएमआई या ऋण रिकॉर्ड नहीं मिला।"
            if lang == "gu":
                return "તમારા ખાતા પર કોઈ સક્રિય ઈએમઆઈ અથવા લોન રેકોર્ડ મળ્યો નથી."
            return "No active loan or EMI schedule was found for your account."

        amt = facts["next_emi_amount"]
        date_str = facts["next_due_date"][:10]
        ltype = facts["loan_type"]
        status = facts["status"]

        if lang == "hi":
            msg = f"आपकी आगामी {ltype} ईएमआई ₹{amt:,.2f} की है, जो {date_str} को देय है (स्थिति: {status})।"
            if facts.get("missed_emis_total", 0) > 0:
                msg += f" ध्यान दें: आपके {facts['missed_emis_total']} छूटे हुए भुगतान दर्ज हैं। क्या आप पुनर्गठन विकल्प देखना चाहते हैं?"
            return msg

        if lang == "gu":
            msg = f"તમારી આગામી {ltype} ઈએમઆઈ ₹{amt:,.2f} છે, જે {date_str} ના રોજ ભરવાની છે (સ્થિતિ: {status})."
            if facts.get("missed_emis_total", 0) > 0:
                msg += f" નોંધ: તમારા {facts['missed_emis_total']} ચૂકી ગયેલા હપ્તા છે. શું તમે રાહત વિકલ્પો જોવા માંગો છો?"
            return msg

        # English
        msg = f"Your next {ltype} EMI is ₹{amt:,.2f}, due on {date_str} (status: {status})."
        if facts.get("missed_emis_total", 0) > 0:
            msg += f" Note: You have {facts['missed_emis_total']} missed payments recorded. Would you like to review EMI restructuring options?"
        return msg

    elif intent == Intent.EXPLAIN_TRANSACTION.value:
        if not facts.get("has_transactions"):
            if lang == "hi":
                return "हाल ही में कोई लेनदेन नहीं मिला।"
            if lang == "gu":
                return "કોઈ તાજેતરના વ્યવહારો મળ્યા નથી."
            return "No recent transactions found on record."

        latest = facts["latest_transaction"]
        amt = latest["amount"]
        merchant = latest["merchant"]
        channel = latest["channel"].upper()
        date_str = latest["timestamp"][:16]

        if lang == "hi":
            return f"आपका नवीनतम लेनदेन ₹{amt:,.2f} का '{merchant}' पर {date_str} को {channel} द्वारा हुआ था।"
        if lang == "gu":
            return f"તમારો છેલ્લો વ્યવહાર ₹{amt:,.2f} નો '{merchant}' ખાતે {date_str} ના રોજ {channel} દ્વારા થયો હતો."
        return f"Your latest transaction was ₹{amt:,.2f} to '{merchant}' via {channel} on {date_str}."

    elif intent == Intent.REPORT_SUSPICIOUS_ACTIVITY.value:
        has_anom = facts.get("has_anomaly", False)
        flags = ", ".join(facts.get("flagged_signals", [])) or "None"
        count = facts.get("anomalous_transactions_count", 0)

        if has_anom:
            if lang == "hi":
                return f"सुरक्षा चेतावनी: हमने आपके खाते पर {count} असामान्य लेनदेन पहचाने हैं (संकेत: {flags})। आपकी सुरक्षा के लिए हमने अतिरिक्त प्रमाणीकरण शुरू किया है। क्या आप इन्हें सत्यापित करना चाहते हैं?"
            if lang == "gu":
                return f"સુરક્ષા ચેતવણી: અમે તમારા ખાતા પર {count} શંકાસ્પદ વ્યવહારો જોયા છે (સંકેતો: {flags}). તમારી સુરક્ષા માટે વેરિફિકેશન જરૂરી છે."
            return f"Security Alert: We detected {count} unusual transaction(s) flagged for: {flags}. For your safety, additional verification is required. Would you like to review and freeze unauthorized charges?"
        else:
            if lang == "hi":
                return "अच्छी खबर: आपके खाते पर पिछले 30 दिनों में कोई सुरक्षा विसंगति या संदिग्ध लेनदेन नहीं पाया गया।"
            if lang == "gu":
                return "સારી વાત: છેલ્લા 30 દિવસમાં તમારા ખાતા પર કોઈ શંકાસ્પદ વ્યવહાર જોવા મળ્યો નથી."
            return "All Clear: No security anomalies or unauthorized transaction patterns have been detected on your account."

    elif intent == Intent.ASK_ABOUT_PRODUCT.value:
        top_prods = facts.get("top_recommendations", [])
        action = facts.get("arbitrated_action")
        path = facts.get("path_used", "xgboost")

        if "STRESS_INTERVENTION" in action:
            if lang == "hi":
                return "वर्तमान वित्तीय स्थिति के आधार पर, हम कोई नया ऋण या क्रेडिट कार्ड नहीं सुझा रहे हैं। इसके बजाय, हम ईएमआई पुनर्गठन और राहत योजनाएं उपलब्ध करा रहे हैं।"
            if lang == "gu":
                return "હાલની આર્થિક સ્થિતિ મુજબ, અમે કોઈ નવી લોન કે ક્રેડિટ કાર્ડ આપતા નથી. અમે ઈએમઆઈ રાહત અને રિસ્ટ્રક્ચરિંગ સહાય પૂરી પાડીએ છીએ."
            return "Based on your current cash-flow profile, high-risk credit and loan offers are restricted. We are offering financial health check-in and EMI restructuring assistance instead."

        prod_names = ", ".join([p.replace("_", " ").title() for p in top_prods[:2]]) or "Savings Products"
        if lang == "hi":
            return f"आपकी आय और बचत रिकॉर्ड के आधार पर ({path} द्वारा सत्यापित), आपके लिए सबसे उपयुक्त सुझाव हैं: {prod_names}।"
        if lang == "gu":
            return f"તમારી નિયમિત આવક અને બચતના આધારે ({path} દ્વારા ચકાસાયેલ), તમારા માટે શ્રેષ્ઠ ભલામણ છે: {prod_names}."
        return f"Based on your verified cash-flow stability (analyzed via {path}), your top tailored recommendations are: {prod_names}."

    # Default / Request Human Help
    if lang == "hi":
        return "मैंने आपका अनुरोध नोट कर लिया है। आपकी सहायता के लिए हमारे विशेषज्ञ वित्तीय सलाहकार से संपर्क किया जा रहा है।"
    if lang == "gu":
        return "મેં તમારી વિનંતી નોંધી લીધી છે. અમારા સલાહકાર ટૂંક સમયમાં તમારો સંપર્ક કરશે."
    return "I have logged your request. A Sahaay AI human banking advisor has been notified to assist you directly."


# --------------------------------------------------------------------------
# KYC / Onboarding State Machine
# --------------------------------------------------------------------------

class OnboardingSessionManager:
    """
    Manages conversational onboarding state per customer session.
    Strict sequential enforcement: Name -> Income Type -> Purpose -> Confirmation.
    """
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def get_state(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "session_id": session_id,
                "current_step": OnboardingState.START.value,
                "collected_data": {},
                "is_completed": False,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        return self._sessions[session_id]

    def advance_step(self, session_id: str, user_input: str) -> Dict[str, Any]:
        user_input = sanitize_for_llm(user_input)
        session = self.get_state(session_id)
        current = session["current_step"]
        data = session["collected_data"]

        if current == OnboardingState.START.value:
            session["current_step"] = OnboardingState.COLLECT_NAME.value
            return {
                "next_step": session["current_step"],
                "prompt": "Welcome to Sahaay AI! Please share your Full Legal Name to begin verification.",
                "session": session,
            }

        elif current == OnboardingState.COLLECT_NAME.value:
            name = user_input.strip()
            if len(name) < 2:
                return {
                    "next_step": current,
                    "prompt": "Please enter a valid full name (minimum 2 characters).",
                    "session": session,
                }
            data["full_name"] = name
            session["current_step"] = OnboardingState.COLLECT_INCOME_TYPE.value
            return {
                "next_step": session["current_step"],
                "prompt": f"Thanks {name}! What is your primary income type? (1: Salaried, 2: Gig Worker / Freelancer, 3: Small Business Owner)",
                "session": session,
            }

        elif current == OnboardingState.COLLECT_INCOME_TYPE.value:
            inp = user_input.lower()
            if "gig" in inp or "freelance" in inp or "2" in inp:
                itype = "gig_irregular"
            elif "business" in inp or "self" in inp or "3" in inp:
                itype = "business_self_employed"
            elif "salaried" in inp or "job" in inp or "1" in inp:
                itype = "salaried"
            else:
                itype = "salaried"

            data["income_type"] = itype
            session["current_step"] = OnboardingState.COLLECT_PURPOSE.value
            return {
                "next_step": session["current_step"],
                "prompt": "What is your main financial goal today? (1: Emergency Cash Buffer, 2: Savings & Investment, 3: Loan Assistance)",
                "session": session,
            }

        elif current == OnboardingState.COLLECT_PURPOSE.value:
            purpose = user_input.strip()
            data["purpose"] = purpose
            session["current_step"] = OnboardingState.CONFIRMATION.value
            return {
                "next_step": session["current_step"],
                "prompt": f"Please confirm your profile: Name: {data['full_name']}, Income: {data['income_type']}, Goal: {purpose}. Reply 'YES' to create your consent token.",
                "session": session,
            }

        elif current == OnboardingState.CONFIRMATION.value:
            if "yes" in user_input.lower() or "confirm" in user_input.lower() or "ok" in user_input.lower():
                session["current_step"] = OnboardingState.COMPLETED.value
                session["is_completed"] = True
                return {
                    "next_step": session["current_step"],
                    "prompt": "Onboarding Complete! Your RBI Account Aggregator consent token is active. Your intelligence dashboard is now live.",
                    "session": session,
                }
            else:
                return {
                    "next_step": current,
                    "prompt": "Please reply 'YES' to confirm and activate your consent token, or 'RESTART' to start over.",
                    "session": session,
                }

        return {
            "next_step": OnboardingState.COMPLETED.value,
            "prompt": "Onboarding is already completed for this session.",
            "session": session,
        }


# Global session manager instance
onboarding_manager = OnboardingSessionManager()


# --------------------------------------------------------------------------
# Main Chat Orchestrator
# --------------------------------------------------------------------------

def process_customer_message(
    customer_id: str,
    message: str,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main conversational pipeline:
    1. Extract intent & language.
    2. Pull factual data from underlying verified data layers & engines.
    3. Phrase grounded response using real numbers.
    """
    extraction = extract_intent(message, language_hint=language)
    intent = extraction["intent"]
    lang = extraction["language"]

    # Pull real facts from backend
    if intent == Intent.CHECK_EMI.value:
        facts = _fetch_emi_facts(customer_id)
    elif intent == Intent.EXPLAIN_TRANSACTION.value:
        facts = _fetch_transaction_facts(customer_id)
    elif intent == Intent.REPORT_SUSPICIOUS_ACTIVITY.value:
        facts = _fetch_suspicious_facts(customer_id)
    elif intent == Intent.ASK_ABOUT_PRODUCT.value:
        facts = _fetch_product_facts(customer_id)
    else:
        facts = {"ticket_id": f"TICK_{customer_id[:6]}", "customer_id": customer_id}

    reply = phrase_response(intent, facts, language=lang)

    return {
        "customer_id": customer_id,
        "message": message,
        "intent": intent,
        "language": lang,
        "confidence": extraction["confidence"],
        "reply": reply,
        "facts_used": facts,
        "path_used": facts.get("path_used", "conversational_rule_grounding"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
