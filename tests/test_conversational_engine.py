"""
Tests for Sahaay AI Conversational Layer (Section 5)
Verifies:
1. Exact EMI figure preservation (no invented/hallucinated numbers).
2. Out-of-scope intent fallback to request_human_help.
3. Strict sequential progression in Onboarding / KYC State Machine (no skipping steps).
4. Multilingual response phrasing (English, Hindi, Gujarati).
5. Grounding across all 5 supported flows with real backend data layers.
"""

import os
import sys
import pytest
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_layer"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "engines"))

from conversational_engine import (
    Intent,
    OnboardingState,
    OnboardingSessionManager,
    extract_intent,
    phrase_response,
    process_customer_message,
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

    res_hi = process_customer_message(customer_with_emi, "मेरी ईएमआई कितनी है?", language="hi")
    assert res_hi["intent"] == Intent.CHECK_EMI.value
    assert formatted_amt in res_hi["reply"]

    res_gu = process_customer_message(customer_with_emi, "મારો હપ્તો કેટલો છે?", language="gu")
    assert res_gu["intent"] == Intent.CHECK_EMI.value
    assert formatted_amt in res_gu["reply"]


# --------------------------------------------------------------------------
# 2. Out-of-Scope Fallback to Request Human Help
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
# 3. Onboarding State Machine Sequential Transitions
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
# 4. Other Flows Grounding
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
