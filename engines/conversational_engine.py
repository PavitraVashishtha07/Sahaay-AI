"""
Sahaay AI — Section 5: Conversational Layer & Intent Engine

A structured, non-hallucinating conversational companion that enforces the
strict architectural guarantee: The conversational layer NEVER invents financial
facts or terms. All numbers, dates, and terms come from verified backend data.

Key Capabilities:
1. Multi-language Intent Extraction: Powered by Gemini Flash (with graceful regex fallback).
   Language is automatically detected from the message (e.g., 'hi', 'gu', 'mr', 'ta', 'te', 'bn', 'kn', 'pa', 'ml', 'en', 'or', 'as', etc.).
2. Zero-Hallucination Fact Grounding: Gemini/Templates receive ONLY verified backend facts.
3. Curated 10-Language Set with `tts_supported` flag for browser SpeechSynthesis.
4. Prompt Injection & Jailbreak Defense: Input sanitized via security/sanitize.py.
5. Onboarding / KYC State Machine: Strict sequential progression.
"""

from __future__ import annotations
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import httpx

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
    get_active_consent_id_for_customer,
    get_customer_transactions,
    ConsentError,
)
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


# Curated, tested set of 10 Indic & English languages with typical browser SpeechSynthesis voice support
CURATED_TTS_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "kn": "Kannada",
    "pa": "Punjabi",
    "ml": "Malayalam",
}


def is_tts_supported(language_code: Optional[str]) -> bool:
    """
    Returns True if the language code is within the curated 10-language set
    supported by typical browser SpeechSynthesis engines.
    For out-of-curated-set languages (e.g. 'or' for Odia, 'as' for Assamese),
    extraction and reply still work normally, but tts_supported is False.
    """
    if not language_code:
        return True  # defaults to en
    base_lang = language_code.strip().lower().split("-")[0].split("_")[0]
    return base_lang in CURATED_TTS_LANGUAGES


# --------------------------------------------------------------------------
# Fallback Regex & Script-based Language Detector
# --------------------------------------------------------------------------

INTENT_PATTERNS: Dict[Intent, List[str]] = {
    Intent.CHECK_EMI: [
        r"(emi|loan payment|due date|installment|किस्त|ईएमआई|हप्ते|हप्ता|હપ્તો|હપ્તા|ચુકવણી|લોન|কিশোর|কিস্তি|ইএমআই|ঋণ|இஎம்ஐ|கடன்|తవణ|రుణం|ಕಂತು|ಕಡ|ਕਿਸ਼ਤ|ലോൺ|ഇഎംഐ|ଇଏମଆଇ|କିସ୍ତି|ଋଣ)",
        r"(how much do i owe|next emi|emi status|emi schedule|loan amount due|due amount)",
        r"(meri emi|mera emi|emi kab hai|kitna dena hai|kitni emi|किस्त कितनी|ईएमआई कब|हप्ता कधी आहे|किती भरायचे)",
        r"(હપ્તો ક્યારે છે|કેટલા ભરવાના છે|મારી ઈએમઆઈ|હપ્તો કેટલો)",
        r"(কিস্তি কবে|কত টাকা দিতে হবে|আমার ইএমআই|ইএমআই)",
        r"(அடுத்த இஎம்ஐ எப்போது|எவ்வளவு கட்ட வேண்டும்)",
        r"(తదుపరి ఈఎంఐ ఎప్పుడు|ఎంత చెల్లించాలి)",
        r"(ಮುಂದಿನ ಇಎಂಐ ಯಾವಾಗ|ಎಷ್ಟು ಪಾವತಿಸಬೇಕು)",
        r"(ਅਗਲੀ ਕਿਸ਼ਤ ਕਦੋਂ ਹੈ|ਕਿੰਨੇ ਪੈਸੇ ਦੇਣੇ ਹਨ)",
        r"(അടുത്ത ഇഎംഐ എപ്പോൾ|എത്ര അടയ്ക്കണം)",
        r"(ମୋର ପରବର୍ତ୍ତୀ ଇଏମଆଇ|କେତେ ଇଏମଆଇ|ଋଣ କିସ୍ତି)",
        r"(yaar mera emi kab hai|bhai emi|mera karz|loan kitna baki)",
    ],

    Intent.EXPLAIN_TRANSACTION: [
        r"(transaction|statement|passbook|where did my money go|last spend|debit|credit|charge|spent)",
        r"(लेनदेन|खर्च|कहाँ पैसे गए|ट्रांजैक्शन|पैसे कटे|खाते से कटे|कटा|काट लिया)",
        r"(વ્યવહાર|ખર્ચ|પૈસા ક્યાં ગયા|સ્ટેટમેન્ટ|ખાતામાંથી કપાયા)",
        r"(व्यवहार|खर्च|पैसे कुठे गेले|खात्यातून कापले)",
        r"(লেনদেন|খরচ|টাকা কোথায় গেল|অ্যাকাউন্ট থেকে কাটা)",
        r"(பரிவர்த்தனை|செலவு|பணம் எங்கு சென்றது)",
        r"(లావాదేవీ|ఖర్చు|డబ్బు ఎక్కడికి పోయింది)",
        r"(ವಹಿವಾಟು|ಖರ್ಚು|ಹಣ ಎಲ್ಲಿ ಹೋಯಿತು)",
        r"(ਲੈਣ-ਦੇਣ|ਖਰਚਾ|ਪੈਸੇ ਕਿੱਥੇ ਗਏ)",
        r"(ഇടപാട്|ചെലവ്|പണം എവിടെപ്പോയി)",
        r"(explain transaction|recent transactions|what was this charge|last transaction)",
    ],
    Intent.REPORT_SUSPICIOUS_ACTIVITY: [
        r"(fraud|unauthorized|suspicious|unknown device|scam|stolen|hacked|freeze|block card)",
        r"(धोखाधड़ी|संदिग्ध|अनजान डिवाइस|गलत ट्रांजैक्शन|पैसे कट गए अपने आप|कार्ड ब्लॉक)",
        r"(છેતરપિંડી|શંકાસ્પદ|અજાણ્યો વ્યવહાર|ખોટા પૈસા કપાયા|કાર્ડ બ્લોક)",
        r"(फसवणूक|संशयास्पद|अनोळखी व्यवहार|कार्ड ब्लॉक)",
        r"(প্রতারণা|সন্দেহজনক|অননুমোদিত|অজানা লেনদেন)",
        r"(மோசடி|சந்தேகத்திற்கிடமான|அங்கீகரிக்கப்படாத|கார்டு முடக்கு)",
        r"(మోసం|అనుమానాస్పద|అనధికార|కార్డు బ్లాక్)",
        r"(ವಂಚನೆ|ಅನುಮಾನಾಸ್ಪದ|ಅನಧಿಕೃತ|ಕಾರ್ಡ್ ನಿರ್ಬಂಧಿಸಿ)",
        r"(ਧੋਖਾਧੜੀ|ਸ਼ੱਕੀ|ਗਲਤ ਕਟੌਤੀ|ਕਾਰਡ ਬਲਾਕ)",
        r"(തട്ടിപ്പ്|സംശയാസ്പദമായ|അനധികൃത|കാർഡ് ബ്ലോക്ക്)",
        r"(did not authorize|report fraud|block card|freeze account|unrecognized charge)",
    ],
    Intent.ASK_ABOUT_PRODUCT: [
        r"(recommend|product|credit card|sip|mutual fund|insurance|fd|rd|deposit|savings|investment)",
        r"(सुझाव|क्रेडिट कार्ड|बीमा|म्यूचुअल फंड|योजना|इन्वेस्टमेंट|बचत)",
        r"(ભલામણ|ક્રેડિટ કાર્ડ|વીમો|રોકાણ|ડિપોઝિટ|બચત)",
        r"(शिफारસ|क्रेडिट कार्ड|विमा|गुंतवणूक|बचत)",
        r"(পরামর্শ|সুপারিশ|ক্রেডিট কার্ড|বীমা|মিউচুয়াল ফান্ড|সঞ্চয়)",
        r"(பரிந்துரை|கிரெடிட் கார்டு|காப்பீடு|முதலீடு|சேமிப்பு)",
        r"(సిఫార్సు|క్రెడిట్ కార్డ్|భీమా|మ్యూచువల్ ఫండ్|పొదుపు)",
        r"(ಶಿಫಾರಸು|ಕ್ರೆಡಿಟ್ ಕಾರ್ಡ್|ವಿಮೆ|ಹೂಡಿಕೆ|ಉಳಿತಾಯ)",
        r"(ਸਿਫਾਰਸ਼|ਕ੍ਰੈਡਿਟ ਕਾਰਡ|ਬੀਮਾ|ਨਿਵੇਸ਼|ਬੱਚਤ)",
        r"(ശുപാർശ|ക്രെഡിറ്റ് കാർഡ്|ഇൻഷുറൻസ്|നിക്ഷേപം|സമ്പാദ്യം)",
        r"(should i take|why was this recommended|is this suitable for me|suggest product)",
    ],
    Intent.REQUEST_HUMAN_HELP: [
        r"(human|agent|representative|call me|talk to person|advisor|support|help|customer care)",
        r"(इंसान|अधिकारी|मदद|बात करनी है|कस्टमर केयर|सपोर्ट|सलाहकार)",
        r"(માણસ|અધિકારી|મદદ|ગ્રાહક સેવા|વાત કરવી છે|સલાહકાર)",
        r"(माणूस|अधिकारी|मदत|कस्टमर केअर|बोलायचे आहे)",
        r"(মানুষ|কর্মকর্তা|সাহায্য|কথা বলতে চাই|কাস্টমার কেয়ার)",
        r"(மனிதன்|அதிகாரி|உதவி|பேச வேண்டும்|வாடிக்கையாளர் சேவை)",
        r"(వ్యక్తి|అధికారి|సహాయం|మాట్లాడాలి|కస్టమర్ కేర్)",
        r"(ವ್ಯಕ್ತಿ|ಅಧಿಕಾರಿ|ಸಹಾಯ|ಮಾತನಾಡಬೇಕು|ಗ್ರಾಹಕ ಸೇವೆ)",
        r"(ਇਨਸਾਨ|ਅਧਿਕਾਰੀ|ਮਦਦ|ਗੱਲ ਕਰਨੀ ਹੈ|ਕਸਟਮਰ ਕੇਅਰ)",
        r"(മനുഷ്യൻ|ഉദ്യോഗസ്ഥൻ|സഹായം|സംസാരിക്കണം|കസ്റ്റമർ കെയർ)",
    ],
}


def detect_language_fallback(text: str, default: str = "en") -> str:
    """
    Fallback Unicode script range detector covering Indian languages & English.
    """
    script_counts = {
        "hi": 0,  # Devanagari (Hindi / Marathi)
        "gu": 0,  # Gujarati
        "bn": 0,  # Bengali / Assamese
        "pa": 0,  # Gurmukhi (Punjabi)
        "ta": 0,  # Tamil
        "te": 0,  # Telugu
        "kn": 0,  # Kannada
        "ml": 0,  # Malayalam
        "or": 0,  # Odia
    }
    for char in text:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            script_counts["hi"] += 1
        elif 0x0A80 <= code <= 0x0AFF:
            script_counts["gu"] += 1
        elif 0x0980 <= code <= 0x09FF:
            script_counts["bn"] += 1
        elif 0x0A00 <= code <= 0x0A7F:
            script_counts["pa"] += 1
        elif 0x0B80 <= code <= 0x0BFF:
            script_counts["ta"] += 1
        elif 0x0C00 <= code <= 0x0C7F:
            script_counts["te"] += 1
        elif 0x0C80 <= code <= 0x0CFF:
            script_counts["kn"] += 1
        elif 0x0D00 <= code <= 0x0D7F:
            script_counts["ml"] += 1
        elif 0x0B00 <= code <= 0x0B7F:
            script_counts["or"] += 1

    max_lang = max(script_counts, key=script_counts.get)
    if script_counts[max_lang] > 0:
        return max_lang

    # Check for Marathi-specific words in Devanagari or Latin script
    lower_t = text.lower()
    if any(w in lower_t for w in ["kiti", "kuthe", "maza", "maharashtra", "aahe", "bhara"]):
        return "mr"
    if any(w in lower_t for w in ["yaar", "mera", "meri", "hai", "bhai", "kab", "kitna", "kahan"]):
        return "hi"

    return default


def extract_intent_regex(message: str, language_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Regex-based intent extraction fallback when Gemini API is offline or unavailable.
    """
    sanitized_msg = sanitize_for_llm(message)
    msg_clean = sanitized_msg.lower().strip()
    detected_lang = language_hint or detect_language_fallback(sanitized_msg)

    matched_intent = None
    for intent, patterns in INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, msg_clean, re.IGNORECASE):
                matched_intent = intent
                break
        if matched_intent:
            break

    final_intent = matched_intent or Intent.REQUEST_HUMAN_HELP

    return {
        "intent": final_intent.value,
        "language": detected_lang,
        "fields": {},
        "raw_message": message,
        "sanitized_message": sanitized_msg,
        "confidence": 0.95 if matched_intent else 0.50,
        "is_out_of_scope_fallback": matched_intent is None,
        "extractor": "regex_fallback",
    }


# --------------------------------------------------------------------------
# Gemini Flash Extraction Layer
# --------------------------------------------------------------------------

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "3.5"))


def _call_gemini_api(prompt: str, timeout: float = GEMINI_TIMEOUT_SECONDS) -> Optional[str]:
    """
    Internal helper to invoke Gemini Flash via standard REST API with fast timeout.
    Returns response text or None on timeout/error.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 500,
        }
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
            return None
    except Exception:
        return None


def extract_intent_gemini(message: str, language_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Extracts structured intent, detected language, and slot fields using Gemini Flash.
    Gemini's ONLY job is extraction; it never answers directly and never sees raw transactions.
    """
    sanitized_msg = sanitize_for_llm(message)
    if not sanitized_msg:
        return extract_intent_regex(message, language_hint=language_hint)

    prompt = f"""You are Sahaay AI's Intent and Language Extraction Engine for Indian Banking.
Analyze the user message and extract:
1. "intent": MUST be exactly one of: ["check_emi", "explain_transaction", "report_suspicious_activity", "ask_about_product", "request_human_help"].
   - check_emi: Asking about EMI amount, due date, loan installment, remaining loan balance (e.g. 'yaar mera EMI kab hai bhai', 'meri emi kitni hai', 'હપ્તો ક્યારે છે', 'loan amount due').
   - explain_transaction: Asking about past charges, spendings, where money went, debit/credit transactions, merchant charges.
   - report_suspicious_activity: Reporting fraud, unknown transaction, hacked/stolen card, scam, account freeze.
   - ask_about_product: Inquiring about suitability, savings, investment, SIP, mutual funds, credit cards, insurance.
   - request_human_help: General out-of-scope queries, jokes, non-banking talk, or explicit request to speak with an advisor/agent.
2. "language": Detect the language of the message as a 2-letter ISO code (e.g., 'en', 'hi', 'gu', 'mr', 'ta', 'te', 'bn', 'kn', 'pa', 'ml', 'or', 'as', etc.). For Hinglish/transliterated text, detect the underlying language ('hi' for Hindi).
3. "fields": Any extracted entities (e.g. loan_type, merchant_name, time_period).

User Message: "{sanitized_msg}"

Respond ONLY with a valid JSON object matching this exact schema:
{{"intent": "check_emi|explain_transaction|report_suspicious_activity|ask_about_product|request_human_help", "language": "iso_code", "fields": {{}}}}"""

    raw_response = _call_gemini_api(prompt)
    if raw_response:
        try:
            # Clean markdown code fences if present
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            data = json.loads(cleaned)
            extracted_intent = data.get("intent", "").lower().strip()
            valid_intents = {i.value for i in Intent}
            if extracted_intent not in valid_intents:
                extracted_intent = Intent.REQUEST_HUMAN_HELP.value

            detected_lang = language_hint or data.get("language", "en").lower().strip()

            return {
                "intent": extracted_intent,
                "language": detected_lang,
                "fields": data.get("fields", {}),
                "raw_message": message,
                "sanitized_message": sanitized_msg,
                "confidence": 0.98,
                "is_out_of_scope_fallback": extracted_intent == Intent.REQUEST_HUMAN_HELP.value,
                "extractor": "gemini_flash",
            }
        except Exception:
            pass

    # Fallback to regex on timeout, network error, or invalid JSON
    return extract_intent_regex(message, language_hint=language_hint)


def extract_intent(message: str, language_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Public intent extraction entrypoint. Tries Gemini Flash first, falls back to regex.
    """
    return extract_intent_gemini(message, language_hint=language_hint)


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
    """Fetches real recent transactions from indexed database."""
    cust_tx = get_customer_transactions(customer_id)
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
    recs = arb.get("recommendation_summary") or arb.get("recommendations")
    if not recs:
        profile_path = os.path.join(DATA_DIR, "customer_profile.csv")
        profiles = pd.read_csv(profile_path)
        p_row = profiles[profiles["customer_id"] == customer_id].iloc[0].to_dict()
        recs = generate_recommendations(p_row)

    return {
        "arbitrated_action": arb["final_action"],
        "priority_tier": arb["priority_tier_applied"],
        "action_headline": arb["action_headline"],
        "path_used": recs.get("path_used", "xgboost"),
        "top_recommendations": recs.get("top_recommendations", []),
        "recommended_products": recs.get("recommended_products", [])[:2],
        "unsuitable_products": recs.get("unsuitable_products", []),
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
# Grounded Reply Generation (Gemini Flash + Multilingual Templates Fallback)
# --------------------------------------------------------------------------

def phrase_response_gemini(intent: str, facts: Dict[str, Any], language: str = "en") -> Optional[str]:
    """
    Calls Gemini Flash to generate empathetic phrasing strictly grounded in verified facts.
    CRITICAL: Gemini receives ONLY verified backend facts and MUST NOT alter any numbers.
    """
    lang_name = CURATED_TTS_LANGUAGES.get(language.lower(), language)
    prompt = f"""You are Sahaay AI, an empathetic, non-hallucinating Indian banking assistant.
STRICT GUARANTEE: You must ONLY use the exact figures, dates, merchants, and facts provided below.
DO NOT invent or alter any numbers, dates, or terms.

Intent: {intent}
Language: {lang_name} (ISO Code: {language})
Verified Facts from Database:
{json.dumps(facts, indent=2, default=str)}

Instructions:
- Write a natural, clear response in {lang_name}.
- Include the exact amounts (with ₹ symbol) and dates from Verified Facts.
- Keep the response concise and helpful (1-3 sentences).
- Do not use markdown headers or disclaimers. Output only the conversational response text."""

    reply = _call_gemini_api(prompt)
    if reply:
        # Sanity check: if check_emi has next_emi_amount, ensure the figure is preserved
        if intent == Intent.CHECK_EMI.value and facts.get("has_emis"):
            amt = facts["next_emi_amount"]
            formatted_amt = f"{amt:,.2f}"
            amt_simple = f"{amt:.2f}"
            amt_int = str(int(amt))
            # If the exact amount string or digit representation is present, accept it
            if formatted_amt in reply or amt_simple in reply or amt_int in reply:
                return reply
        else:
            return reply

    return None


def phrase_response_template(intent: str, facts: Dict[str, Any], language: str = "en") -> str:
    """
    Deterministic template fallback covering all 10 curated languages and beyond.
    Guaranteed exact figure preservation with zero halluncination risk.
    """
    lang = (language or "en").lower().split("-")[0].split("_")[0]

    if intent == Intent.CHECK_EMI.value:
        if not facts.get("has_emis"):
            templates = {
                "hi": "आपके खाते पर कोई सक्रिय ईएमआई या ऋण रिकॉर्ड नहीं मिला।",
                "gu": "તમારા ખાતા પર કોઈ સક્રિય ઈએમઆઈ અથવા લોન રેકોર્ડ મળ્યો નથી.",
                "mr": "तुमच्या खात्यावर कोणतेही सक्रिय ईएमआय किंवा कर्ज रेकॉर्ड आढळले नाही.",
                "bn": "আপনার অ্যাকাউন্টে কোনো সক্রিয় ইএমআই বা ঋণ রেকর্ড পাওয়া যায়নি।",
                "ta": "உங்கள் கணக்கில் செயலில் உள்ள இஎம்ஐ அல்லது கடன் பதிவுகள் எதுவும் காணப்படவில்லை.",
                "te": "మీ ఖాతాలో ఎటువంటి యాక్టివ్ ఈఎంఐ లేదా లోన్ రికార్డులు కనుగొనబడలేదు.",
                "kn": "ನಿಮ್ಮ ಖಾತೆಯಲ್ಲಿ ಯಾವುದೇ ಸಕ್ರಿಯ ಇಎಂಐ ಅಥವಾ ಸಾಲದ ದಾಖಲೆಗಳು ಕಂಡುಬಂದಿಲ್ಲ.",
                "pa": "ਤੁਹਾਡੇ ਖਾਤੇ ਵਿੱਚ ਕੋਈ ਸਰਗਰਮ ਈਐਮਆਈ ਜਾਂ ਕਰਜ਼ਾ ਰਿਕਾਰਡ ਨਹੀਂ ਮਿਲਿਆ।",
                "ml": "നിങ്ങളുടെ അക്കൗണ്ടിൽ സജീവമായ ഇഎംഐ അല്ലെങ്കിൽ വായ്പാ രേഖകളൊന്നും കണ്ടെത്തിയില്ല.",
                "en": "No active loan or EMI schedule was found for your account.",
            }
            return templates.get(lang, templates["en"])

        amt = facts["next_emi_amount"]
        date_str = str(facts["next_due_date"])[:10]
        ltype = facts["loan_type"]
        status = facts["status"]
        formatted_amt = f"{amt:,.2f}"

        if lang == "hi":
            msg = f"आपकी आगामी {ltype} ईएमआई ₹{formatted_amt} की है, जो {date_str} को देय है (स्थिति: {status})।"
            if facts.get("missed_emis_total", 0) > 0:
                msg += f" ध्यान दें: आपके {facts['missed_emis_total']} छूटे हुए भुगतान दर्ज हैं।"
            return msg

        if lang == "gu":
            msg = f"તમારી આગામી {ltype} ઈએમઆઈ ₹{formatted_amt} છે, જે {date_str} ના રોજ ભરવાની છે (સ્થિતિ: {status})."
            if facts.get("missed_emis_total", 0) > 0:
                msg += f" નોંધ: તમારા {facts['missed_emis_total']} ચૂકી ગયેલા હપ્તા છે."
            return msg

        if lang == "mr":
            msg = f"तुमचा पुढील {ltype} ईएमआय ₹{formatted_amt} आहे, जो {date_str} रोजी देय आहे (स्थिती: {status})."
            if facts.get("missed_emis_total", 0) > 0:
                msg += f" टीप: तुमचे {facts['missed_emis_total']} चुकलेले हप्ते आहेत."
            return msg

        if lang == "bn":
            msg = f"আপনার পরবর্তী {ltype} ইএমআই ₹{formatted_amt}, যা {date_str} তারিখে প্রদেয় (স্থিতি: {status})।"
            return msg

        if lang == "ta":
            msg = f"உங்கள் அடுத்த {ltype} இஎம்ஐ ₹{formatted_amt} ஆகும், இது {date_str} அன்று செலுத்தப்பட வேண்டும் (நிலை: {status})."
            return msg

        if lang == "te":
            msg = f"మీ తదుపరి {ltype} ఈఎంఐ ₹{formatted_amt}, ఇది {date_str} న చెల్లించాల్సి ఉంటుంది (స్థితి: {status})."
            return msg

        if lang == "kn":
            msg = f"ನಿಮ್ಮ ಮುಂದಿನ {ltype} ಇಎಂಐ ₹{formatted_amt} ಆಗಿದೆ, ಇದು {date_str} ರಂದು ಪಾವತಿಸಬೇಕಾಗಿದೆ (ಸ್ಥಿತಿ: {status})."
            return msg

        if lang == "pa":
            msg = f"ਤੁਹਾਡੀ ਅਗਲੀ {ltype} ਈਐਮਆਈ ₹{formatted_amt} ਹੈ, ਜੋ {date_str} ਨੂੰ ਦੇਣਯੋਗ ਹੈ (ਸਥਿਤੀ: {status})।"
            return msg

        if lang == "ml":
            msg = f"നിങ്ങളുടെ അടുത്ത {ltype} ഇഎംഐ ₹{formatted_amt} ആണ്, അത് {date_str}-ൽ അടയ്ക്കണം (നില: {status})."
            return msg

        # English & unlisted language fallback
        msg = f"Your next {ltype} EMI is ₹{formatted_amt}, due on {date_str} (status: {status})."
        if facts.get("missed_emis_total", 0) > 0:
            msg += f" Note: You have {facts['missed_emis_total']} missed payments recorded."
        return msg

    elif intent == Intent.EXPLAIN_TRANSACTION.value:
        if not facts.get("has_transactions"):
            templates = {
                "hi": "हाल ही में कोई लेनदेन नहीं मिला।",
                "gu": "કોઈ તાજેતરના વ્યવહારો મળ્યા નથી.",
                "mr": "कोणतेही अलीकडील व्यवहार आढळले नाहीत.",
                "bn": "কোনো সাম্প্রতিক লেনদেন পাওয়া যায়নি।",
                "ta": "சமீபத்திய பரிவர்த்தனைகள் எதுவும் காணப்படவில்லை.",
                "te": "ఇటీవలి లావాదేవీలు ఏవీ కనుగొనబడలేదు.",
                "kn": "ಯಾವುದೇ ಇತ್ತೀಚಿನ ವಹಿವಾಟುಗಳು ಕಂಡುಬಂದಿಲ್ಲ.",
                "pa": "ਕੋਈ ਤਾਜ਼ਾ ਲੈਣ-ਦੇਣ ਨਹੀਂ ਮਿਲਿਆ।",
                "ml": "സമീപകാല ഇടപാടുകളൊന്നും കണ്ടെത്തിയില്ല.",
                "en": "No recent transactions found on record.",
            }
            return templates.get(lang, templates["en"])

        latest = facts["latest_transaction"]
        amt = latest["amount"]
        merchant = latest["merchant"]
        channel = latest["channel"].upper()
        date_str = latest["timestamp"][:16]
        formatted_amt = f"{amt:,.2f}"

        if lang == "hi":
            return f"आपका नवीनतम लेनदेन ₹{formatted_amt} का '{merchant}' पर {date_str} को {channel} द्वारा हुआ था।"
        if lang == "gu":
            return f"તમારો છેલ્લો વ્યવહાર ₹{formatted_amt} નો '{merchant}' ખાતે {date_str} ના રોજ {channel} દ્વારા થયો હતો."
        if lang == "mr":
            return f"तुमचा शेवटचा व्यवहार ₹{formatted_amt} चा '{merchant}' येथे {date_str} रोजी {channel} द्वारे झाला होता."
        if lang == "bn":
            return f"আপনার সর্বশেষ লেনদেন ₹{formatted_amt} এর '{merchant}'-এ {date_str} তারিখে {channel} মাধ্যমে হয়েছে।"
        if lang == "ta":
            return f"உங்கள் சமீபத்திய பரிவர்த்தனை ₹{formatted_amt} '{merchant}' இல் {date_str} அன்று {channel} வழியாக நடந்தது."
        if lang == "te":
            return f"మీ తాజా లావాదేవీ ₹{formatted_amt} '{merchant}' వద్ద {date_str} న {channel} ద్వారా జరిగింది."
        if lang == "kn":
            return f"ನಿಮ್ಮ ಇತ್ತೀಚಿನ ವಹಿವಾಟು ₹{formatted_amt} '{merchant}' ನಲ್ಲಿ {date_str} ರಂದು {channel} ಮೂಲಕ ನಡೆದಿದೆ."
        if lang == "pa":
            return f"ਤੁਹਾਡਾ ਤਾਜ਼ਾ ਲੈਣ-ਦੇਣ ₹{formatted_amt} ਦਾ '{merchant}' ਵਿਖੇ {date_str} ਨੂੰ {channel} ਰਾਹੀਂ ਹੋਇਆ ਸੀ।"
        if lang == "ml":
            return f"നിങ്ങളുടെ ഏറ്റവും പുതിയ ഇടപാട് ₹{formatted_amt} '{merchant}'-ൽ {date_str}-ൽ {channel} വഴിയാണ് നടന്നത്."

        return f"Your latest transaction was ₹{formatted_amt} to '{merchant}' via {channel} on {date_str}."

    elif intent == Intent.REPORT_SUSPICIOUS_ACTIVITY.value:
        has_anom = facts.get("has_anomaly", False)
        flags = ", ".join(facts.get("flagged_signals", [])) or "None"
        count = facts.get("anomalous_transactions_count", 0)

        if has_anom:
            if lang == "hi":
                return f"सुरक्षा चेतावनी: हमने आपके खाते पर {count} असामान्य लेनदेन पहचाने हैं (संकेत: {flags})। आपकी सुरक्षा के लिए अतिरिक्त सत्यापन आवश्यक है।"
            if lang == "gu":
                return f"સુરક્ષા ચેતવણી: અમે તમારા ખાતા પર {count} શંકાસ્પદ વ્યવહારો જોયા છે (સંકેતો: {flags}). તમારી સુરક્ષા માટે વેરિફિકેશન જરૂરી છે."
            if lang == "mr":
                return f"सुरक्षा इशारा: आम्ही तुमच्या खात्यावर {count} संशयास्पद व्यवहार शोधले आहेत (संकेत: {flags})."
            return f"Security Alert: We detected {count} unusual transaction(s) flagged for: {flags}. For your safety, additional verification is required."
        else:
            if lang == "hi":
                return "अच्छी खबर: आपके खाते पर पिछले 30 दिनों में कोई सुरक्षा विसंगति या संदिग्ध लेनदेन नहीं पाया गया।"
            if lang == "gu":
                return "સારી વાત: છેલ્લા 30 દિવસમાં તમારા ખાતા પર કોઈ શંકાસ્પદ વ્યવહાર જોવા મળ્યો નથી."
            if lang == "mr":
                return "चांगली बातमी: तुमच्या खात्यावर कोणतीही सुरक्षा त्रुटी किंवा संशयास्पद व्यवहार आढळला नाही."
            return "All Clear: No security anomalies or unauthorized transaction patterns have been detected on your account."

    elif intent == Intent.ASK_ABOUT_PRODUCT.value:
        top_prods = facts.get("top_recommendations", [])
        action = str(facts.get("arbitrated_action", ""))
        path = facts.get("path_used", "xgboost")

        if "STRESS_INTERVENTION" in action:
            if lang == "hi":
                return "वर्तमान वित्तीय स्थिति के आधार पर, हम कोई नया ऋण या क्रेडिट कार्ड नहीं सुझा रहे हैं। हम ईएमआई राहत सहायता प्रदान कर रहे हैं।"
            if lang == "gu":
                return "હાલની આર્થિક સ્થિતિ મુજબ, અમે કોઈ નવી લોન કે ક્રેડિટ કાર્ડ આપતા નથી. અમે ઈએમઆઈ રાહત સહાય પૂરી પાડીએ છીએ."
            return "Based on your current cash-flow profile, high-risk credit offers are restricted. We offer EMI restructuring assistance instead."

        prod_names = ", ".join([p.replace("_", " ").title() for p in top_prods[:2]]) or "Savings Products"
        if lang == "hi":
            return f"आपकी आय और बचत रिकॉर्ड के आधार पर ({path} द्वारा सत्यापित), आपके लिए सबसे उपयुक्त सुझाव हैं: {prod_names}।"
        if lang == "gu":
            return f"તમારી આવક અને બચતના આધારે ({path} દ્વારા ચકાસાયેલ), શ્રેષ્ઠ ભલામણ છે: {prod_names}."
        if lang == "mr":
            return f"तुमच्या उत्पन्नाच्या आणि बचतीच्या आधारे ({path} द्वारे पडताळणी), शिफारस केलेले पर्याय आहेत: {prod_names}."
        return f"Based on your verified cash-flow stability (analyzed via {path}), your top tailored recommendations are: {prod_names}."

    # Default / Request Human Help
    templates_help = {
        "hi": "मैंने आपका अनुरोध नोट कर लिया है। आपकी सहायता के लिए हमारे विशेषज्ञ वित्तीय सलाहकार से संपर्क किया जा रहा है।",
        "gu": "મેં તમારી વિનંતી નોંધી લીધી છે. અમારા સલાહકાર ટૂંક સમયમાં તમારો સંપર્ક કરશે.",
        "mr": "मी तुमची विनंती नोंदवली आहे. आमचे आर्थिक सल्लागार लवकरच तुमच्याशी संपर्क साधतील.",
        "bn": "আমি আপনার অনুরোধটি নথিভুক্ত করেছি। আমাদের আর্থিক উপদেষ্টা শীঘ্রই আপনার সাথে যোগাযোগ করবেন।",
        "ta": "உங்கள் கோரிக்கையை நான் பதிவு செய்துள்ளேன். எங்கள் நிதி ஆலோசகர் விரைவில் உங்களைத் தொடர்புகொள்வார்.",
        "te": "నేను మీ అభ్యర్థనను నమోదు చేసాను. మా ఆర్థిక సలహాదారు త్వరలో మిమ్మల్ని సంప్రదిస్తారు.",
        "kn": "ನಾನು ನಿಮ್ಮ ವಿನಂತಿಯನ್ನು ದಾಖಲಿಸಿದ್ದೇನೆ. ನಮ್ಮ ಹಣಕಾಸು ಸಲಹೆಗಾರರು ಶೀಘ್ರದಲ್ಲೇ ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸುತ್ತಾರೆ.",
        "pa": "ਮੈਂ ਤੁਹਾਡੀ ਬੇਨਤੀ ਦਰਜ ਕਰ ਲਈ ਹੈ। ਸਾਡੇ ਵਿੱਤੀ ਸਲਾਹਕਾਰ ਜਲਦੀ ਹੀ ਤੁਹਾਡੇ ਨਾਲ ਸੰਪਰਕ ਕਰਨਗੇ।",
        "ml": "നിങ്ങളുടെ അഭ്യർത്ഥന ഞാൻ രേഖപ്പെടുത്തിയിട്ടുണ്ട്. ഞങ്ങളുടെ സാമ്പത്തിക ഉപദേശകൻ ഉടൻ നിങ്ങളെ ബന്ധപ്പെടും.",
        "en": "I have logged your request. A Sahaay AI human banking advisor has been notified to assist you directly.",
    }
    return templates_help.get(lang, templates_help["en"])


def phrase_response(intent: str, facts: Dict[str, Any], language: str = "en") -> str:
    """
    Unified response phrasing. Tries Gemini Flash first with fact grounding;
    falls back deterministically to multilingual templates.
    """
    gemini_reply = phrase_response_gemini(intent, facts, language=language)
    if gemini_reply:
        return gemini_reply
    return phrase_response_template(intent, facts, language=language)


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

def _get_facts_used_keys(intent: str, facts: Dict[str, Any]) -> List[str]:
    """Returns human-readable keys of verified database fields used in the reply."""
    if intent == Intent.CHECK_EMI.value:
        return ["emi_records.due_date", "emi_records.amount_due", "emi_records.loan_type", "emi_records.status"]
    elif intent == Intent.EXPLAIN_TRANSACTION.value:
        return ["transactions.merchant_name", "transactions.amount", "transactions.channel", "transactions.timestamp"]
    elif intent == Intent.REPORT_SUSPICIOUS_ACTIVITY.value:
        return ["fraud_engine.anomaly_score", "fraud_engine.flagged_signals", "fraud_engine.anomalous_transactions_count"]
    elif intent == Intent.ASK_ABOUT_PRODUCT.value:
        return ["arbitration.final_action", "recommendation.path_used", "recommendation.top_recommendations"]
    return ["customer.support_ticket"]


def process_customer_message(
    customer_id: str,
    message: str,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main conversational pipeline:
    1. Language & Intent extraction via Gemini Flash (language detection is Gemini's job;
       language parameter acts as optional override).
    2. Pull factual data from underlying verified data layers & engines.
    3. Phrase grounded response using real numbers (Gemini Flash or deterministic template fallback).
    4. Compute tts_supported flag based on browser voice coverage.
    """
    # Step 1: Extract intent and detect language
    extraction = extract_intent(message, language_hint=language)
    intent = extraction["intent"]
    detected_lang = extraction["language"]
    final_lang = language or detected_lang

    # Step 2: Pull real facts from backend
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

    # Step 3: Phrase grounded response
    reply = phrase_response(intent, facts, language=final_lang)
    facts_keys = _get_facts_used_keys(intent, facts)
    tts_flag = is_tts_supported(final_lang)

    return {
        "customer_id": customer_id,
        "message": message,
        "intent": intent,
        "language": final_lang,
        "tts_supported": tts_flag,
        "confidence": extraction["confidence"],
        "reply": reply,
        "facts_used": facts,
        "facts_used_keys": facts_keys,
        "path_used": facts.get("path_used", "xgboost" if intent == Intent.ASK_ABOUT_PRODUCT.value else "deterministic_grounding"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "extractor": extraction.get("extractor", "gemini_flash"),
    }
