"""
Tests for Sahaay AI Conversational Layer (Section 5)
Verifies:
1. Exact EMI figure preservation (no invented/hallucinated numbers).
2. Multi-language Intent Extraction (English, Hindi, Gujarati, Marathi, Tamil, Telugu, Bengali, Kannada, Punjabi, Malayalam).
3. Hinglish free-form colloquial phrasing ("yaar mera EMI kab hai bhai").
4. Out-of-curated-set language (Odia) extraction with tts_supported: False.
5. Gemini-failure fallback (mock timeout/error to confirm regex extraction).
6. Out-of-scope intent fallback to request_human_help.
7. Strict sequential progression in Onboarding / KYC State Machine.
8. Grounding across all 5 supported flows with real backend data layers.
"""

import os
import sys
from unittest.mock import patch
import pytest
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_layer"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "engines"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "security"))

from conversational_engine import (
    Intent,
    OnboardingState,
    OnboardingSessionManager,
    extract_intent,
    phrase_response,
    process_customer_message,
    is_tts_supported,
    CURATED_TTS_LANGUAGES,
    _fetch_emi_facts,
    _fetch_transaction_facts,
    _fetch_product_facts,
    _fetch_suspicious_facts,
)
from aa_interface import DATA_DIR


@pytest.fixture
def sample_customer_id():
    profiles_path = os.path.join(DATA_DIR, "customer_profile.csv")
    df = pd.read_csv(profiles_path)
    return str(df.iloc[0]["customer_id"])


@pytest.fixture
def customer_with_emi():
    emi_path = os.path.join(DATA_DIR, "emi_records.csv")
    df = pd.read_csv(emi_path)
    return str(df.iloc[0]["customer_id"])


# --------------------------------------------------------------------------
# 1. Exact EMI Preservation & Anti-Hallucination Guarantees
# --------------------------------------------------------------------------

def test_check_emi_returns_exact_figure(customer_with_emi):
    """Confirm check_emi returns the exact EMI figure from emi_records.csv."""
    emi_path = os.path.join(DATA_DIR, "emi_records.csv")
    df = pd.read_csv(emi_path)
    cust_emis = df[df["customer_id"] == customer_with_emi].sort_values("due_date", ascending=False)
    expected_amt = float(cust_emis.iloc[0]["amount_due"])
    expected_date = str(cust_emis.iloc[0]["due_date"])[:10]

    res = process_customer_message(customer_with_emi, "When is my next EMI due and how much?")
    assert res["intent"] == Intent.CHECK_EMI.value
    assert res["facts_used"]["has_emis"] is True
    assert res["facts_used"]["next_emi_amount"] == expected_amt
    assert res["tts_supported"] is True
    
    # Exact amount must be present in string reply
    formatted_amt = f"{expected_amt:,.2f}"
    assert formatted_amt in res["reply"]
    assert expected_date in res["reply"]


def test_check_emi_hindi_gujarati_exact_figures(customer_with_emi):
    """Test Hindi and Gujarati translations also contain exact figures."""
    emi_path = os.path.join(DATA_DIR, "emi_records.csv")
    df = pd.read_csv(emi_path)
    cust_emis = df[df["customer_id"] == customer_with_emi].sort_values("due_date", ascending=False)
    expected_amt = float(cust_emis.iloc[0]["amount_due"])
    formatted_amt = f"{expected_amt:,.2f}"

    res_hi = process_customer_message(customer_with_emi, "मेरी ईएमआई कितनी है?")
    assert res_hi["intent"] == Intent.CHECK_EMI.value
    assert res_hi["language"] == "hi"
    assert (formatted_amt in res_hi["reply"] or f"{expected_amt:.2f}" in res_hi["reply"] or str(int(expected_amt)) in res_hi["reply"])

    res_gu = process_customer_message(customer_with_emi, "મારો હપ્તો કેટલો છે?")
    assert res_gu["intent"] == Intent.CHECK_EMI.value
    assert res_gu["language"] == "gu"
    assert res_gu["tts_supported"] is True
    assert (formatted_amt in res_gu["reply"] or f"{expected_amt:.2f}" in res_gu["reply"] or str(int(expected_amt)) in res_gu["reply"])


# --------------------------------------------------------------------------
# 2. Hinglish Free-Form Colloquial Extraction & Exact Fact Grounding
# --------------------------------------------------------------------------

def test_hinglish_free_form_intent_and_exact_emi(customer_with_emi):
    """
    Verifies colloquial Hinglish ('yaar mera EMI kab hai bhai') extracts
    intent: 'check_emi' and injects the exact backend-verified EMI figure.
    """
    emi_path = os.path.join(DATA_DIR, "emi_records.csv")
    df = pd.read_csv(emi_path)
    cust_emis = df[df["customer_id"] == customer_with_emi].sort_values("due_date", ascending=False)
    expected_amt = float(cust_emis.iloc[0]["amount_due"])
    formatted_amt = f"{expected_amt:,.2f}"

    hinglish_queries = [
        "yaar mera EMI kab hai bhai",
        "bhai agla installment kitna dena hai",
        "meri emi kitni baki hai batao na",
    ]
    for q in hinglish_queries:
        res = process_customer_message(customer_with_emi, q)
        assert res["intent"] == Intent.CHECK_EMI.value
        assert res["tts_supported"] is True
        assert formatted_amt in res["reply"], f"Exact figure {formatted_amt} missing from reply: {res['reply']}"


# --------------------------------------------------------------------------
# 3. Curated 10-Language Set Tests (Intent + Fact-Grounded Phrasing)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "lang_code,message,expected_intent",
    [
        ("en", "How much is my next loan installment due?", Intent.CHECK_EMI.value),
        ("hi", "मेरी अगली किस्त कितनी और कब देय है?", Intent.CHECK_EMI.value),
        ("gu", "મારી આગામી લોનનો હપ્તો કેટલો છે?", Intent.CHECK_EMI.value),
        ("mr", "माझा पुढील कर्जाचा हप्ता किती आणि कधी आहे?", Intent.CHECK_EMI.value),
        ("bn", "আমার পরবর্তী ঋণের কিস্তি কত টাকা?", Intent.CHECK_EMI.value),
        ("ta", "எனது அடுத்த கடன் தவணை எவ்வளவு?", Intent.CHECK_EMI.value),
        ("te", "నా తదుపరి రుణ తవణ ఎంత మరియు ఎప్పుడు?", Intent.CHECK_EMI.value),
        ("kn", "ನನ್ನ ಮುಂದಿನ ಸಾಲದ ಕಂತು ಎಷ್ಟು?", Intent.CHECK_EMI.value),
        ("pa", "ਮੇਰੀ ਅਗਲੀ ਕਰਜ਼ੇ ਦੀ ਕਿਸ਼ਤ ਕਿੰਨੀ ਹੈ?", Intent.CHECK_EMI.value),
        ("ml", "എന്റെ അടുത്ത ലോൺ ഇഎംഐ എത്രയാണ്?", Intent.CHECK_EMI.value),
    ],
)
def test_curated_10_languages_end_to_end(customer_with_emi, lang_code, message, expected_intent):
    """
    Validates all 10 curated Indic/English languages:
    1. Intent is extracted accurately.
    2. Detected language or override matches expected.
    3. tts_supported is True.
    4. Exact verified backend numbers are included in the reply.
    """
    emi_path = os.path.join(DATA_DIR, "emi_records.csv")
    df = pd.read_csv(emi_path)
    cust_emis = df[df["customer_id"] == customer_with_emi].sort_values("due_date", ascending=False)
    expected_amt = float(cust_emis.iloc[0]["amount_due"])
    formatted_amt = f"{expected_amt:,.2f}"

    res = process_customer_message(customer_with_emi, message, language=lang_code)
    assert res["intent"] == expected_intent
    assert res["language"] == lang_code
    assert res["tts_supported"] is True
    assert formatted_amt in res["reply"]
    assert "facts_used_keys" in res
    assert "emi_records.amount_due" in res["facts_used_keys"]


# --------------------------------------------------------------------------
# 4. Out-of-Curated-Set Language Test (Odia / Assamese)
# --------------------------------------------------------------------------

def test_out_of_curated_set_language_attempts_extraction_with_tts_false(customer_with_emi):
    """
    For an out-of-curated-set language (e.g. Odia 'or' or Assamese 'as'):
    - Extraction & fact-grounded reply generation still attempt normally (not blocked).
    - tts_supported is correctly returned as False (since browser voices are typically unavailable).
    """
    odia_query = "ମୋର ପରବର୍ତ୍ତୀ ଇଏମଆଇ କେତେ?"
    res = process_customer_message(customer_with_emi, odia_query, language="or")
    
    assert res["intent"] == Intent.CHECK_EMI.value
    assert res["language"] == "or"
    assert res["tts_supported"] is False  # Must be False for out-of-curated-set
    assert "facts_used" in res


# --------------------------------------------------------------------------
# 5. Gemini API Failure & Timeout Fallback Test
# --------------------------------------------------------------------------

def test_gemini_api_failure_falls_back_to_regex(customer_with_emi):
    """
    Simulates a Gemini API timeout or connection failure.
    Confirms the engine gracefully falls back to deterministic regex extraction
    and template phrasing without crashing or breaking the chat response.
    """
    with patch("conversational_engine._call_gemini_api", return_value=None):
        res = process_customer_message(customer_with_emi, "Check my loan emi status please")
        assert res["intent"] == Intent.CHECK_EMI.value
        assert res["facts_used"]["has_emis"] is True
        assert res["tts_supported"] is True
        assert res["extractor"] == "regex_fallback"
        assert "₹" in res["reply"] or "due" in res["reply"].lower()


# --------------------------------------------------------------------------
# 6. Out-of-Scope Fallback to Request Human Help
# --------------------------------------------------------------------------

def test_out_of_scope_routes_to_human_help(sample_customer_id):
    """Out-of-scope messages must route to request_human_help rather than improvising."""
    weird_messages = [
        "What is the capital of France?",
        "Tell me a joke about robots",
        "Can you write a poem for my friend?",
        "How do I fix my washing machine?",
    ]
    for msg in weird_messages:
        res = process_customer_message(sample_customer_id, msg)
        assert res["intent"] == Intent.REQUEST_HUMAN_HELP.value
        assert "advisor" in res["reply"].lower() or "human" in res["reply"].lower() or "notified" in res["reply"].lower()


# --------------------------------------------------------------------------
# 7. Onboarding State Machine Sequential Transitions
# --------------------------------------------------------------------------

def test_onboarding_state_machine_cannot_skip_steps():
    """State machine must enforce strict progression: START -> NAME -> INCOME_TYPE -> PURPOSE -> CONFIRMATION -> COMPLETED."""
    mgr = OnboardingSessionManager()
    session_id = "test_session_123"

    # Initial state is START
    state = mgr.get_state(session_id)
    assert state["current_step"] == OnboardingState.START.value

    # Step 1: Advance to COLLECT_NAME
    step1 = mgr.advance_step(session_id, "hello")
    assert step1["next_step"] == OnboardingState.COLLECT_NAME.value

    # Validation: Empty/invalid name cannot advance
    bad_name = mgr.advance_step(session_id, "a")
    assert bad_name["next_step"] == OnboardingState.COLLECT_NAME.value

    # Step 2: Provide valid name -> advances to COLLECT_INCOME_TYPE
    step2 = mgr.advance_step(session_id, "Priya Sharma")
    assert step2["next_step"] == OnboardingState.COLLECT_INCOME_TYPE.value
    assert step2["session"]["collected_data"]["full_name"] == "Priya Sharma"

    # Step 3: Provide income type -> advances to COLLECT_PURPOSE
    step3 = mgr.advance_step(session_id, "gig worker")
    assert step3["next_step"] == OnboardingState.COLLECT_PURPOSE.value
    assert step3["session"]["collected_data"]["income_type"] == "gig_irregular"

    # Step 4: Provide purpose -> advances to CONFIRMATION
    step4 = mgr.advance_step(session_id, "Emergency cash buffer")
    assert step4["next_step"] == OnboardingState.CONFIRMATION.value
    assert step4["session"]["collected_data"]["purpose"] == "Emergency cash buffer"

    # Validation: Unconfirmed answer stays at CONFIRMATION
    not_confirmed = mgr.advance_step(session_id, "maybe later")
    assert not_confirmed["next_step"] == OnboardingState.CONFIRMATION.value
    assert not_confirmed["session"]["is_completed"] is False

    # Step 5: Confirm YES -> advances to COMPLETED
    step5 = mgr.advance_step(session_id, "YES")
    assert step5["next_step"] == OnboardingState.COMPLETED.value
    assert step5["session"]["is_completed"] is True


# --------------------------------------------------------------------------
# 8. Other Flows Grounding
# --------------------------------------------------------------------------

def test_explain_transaction_flow(sample_customer_id):
    """Verifies transaction explanation fetches real recent transaction amount and merchant."""
    res = process_customer_message(sample_customer_id, "Explain my recent transactions")
    assert res["intent"] == Intent.EXPLAIN_TRANSACTION.value
    assert "facts_used" in res
    if res["facts_used"]["has_transactions"]:
        latest = res["facts_used"]["latest_transaction"]
        amt_str = f"{latest['amount']:,.2f}"
        assert amt_str in res["reply"]
        assert latest["merchant"] in res["reply"]


def test_report_suspicious_activity_flow(sample_customer_id):
    """Verifies suspicious activity check grounds in fraud engine output."""
    res = process_customer_message(sample_customer_id, "I want to report suspicious activity on my card")
    assert res["intent"] == Intent.REPORT_SUSPICIOUS_ACTIVITY.value
    assert "facts_used" in res
    assert "has_anomaly" in res["facts_used"]


def test_ask_about_product_flow(sample_customer_id):
    """Verifies product inquiry grounds in recommendation/arbitration facts."""
    res = process_customer_message(sample_customer_id, "What is the best product or loan recommendation for me?")
    assert res["intent"] == Intent.ASK_ABOUT_PRODUCT.value
    assert "facts_used" in res
    assert "arbitrated_action" in res["facts_used"]
