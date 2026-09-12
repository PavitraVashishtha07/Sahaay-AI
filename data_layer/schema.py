"""
Sahaay AI — Data Layer Schema

Defines the shape of data as if it arrived through India's Account Aggregator (AA)
ecosystem: consented transactions, KYC, and bureau-style data. For the prototype,
this data is synthetically generated (see generate_synthetic_data.py) but the
schema mirrors what a real AA pull (via an AA participant like Setu / Anumati)
would deliver, so swapping the source later only touches aa_interface.py.

Archetypes covered (per the research doc):
  1. Stable salaried customer — consistent income, no missed payments
  2. Small-business / gig customer — irregular, seasonal receipts
  3. Financially stressed customer — income reduction, delayed EMIs
  4. Fraud scenario — new device, new beneficiary, unusual location, rapid transfers
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional


class IncomeType(str, Enum):
    SALARIED = "salaried"
    GIG_IRREGULAR = "gig_irregular"
    BUSINESS_SELF_EMPLOYED = "business_self_employed"


class Archetype(str, Enum):
    STABLE_SALARIED = "stable_salaried"
    GIG_IRREGULAR = "gig_irregular"
    FINANCIALLY_STRESSED = "financially_stressed"
    FRAUD_SCENARIO = "fraud_scenario"


class TransactionType(str, Enum):
    CREDIT = "credit"
    DEBIT = "debit"


class TransactionChannel(str, Enum):
    UPI = "upi"
    NEFT = "neft"
    IMPS = "imps"
    CARD = "card"
    CASH_WITHDRAWAL = "cash_withdrawal"
    AUTO_DEBIT = "auto_debit"


class MerchantCategory(str, Enum):
    SALARY = "salary"
    GIG_PAYOUT = "gig_payout"
    BUSINESS_RECEIPT = "business_receipt"
    GROCERY = "grocery"
    RENT = "rent"
    EMI_PAYMENT = "emi_payment"
    UTILITIES = "utilities"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    ENTERTAINMENT = "entertainment"
    INSURANCE_PREMIUM = "insurance_premium"
    INVESTMENT = "investment"
    TRANSFER_PEER = "transfer_peer"
    ECOMMERCE = "ecommerce"
    FUEL = "fuel"
    UNKNOWN_MERCHANT = "unknown_merchant"  # used for fraud-scenario novelty signals


class EMIStatus(str, Enum):
    PAID_ON_TIME = "paid_on_time"
    PAID_LATE = "paid_late"
    MISSED = "missed"
    UPCOMING = "upcoming"


@dataclass
class ConsentArtefact:
    """
    Mirrors the consent object RBI's AA Directions require: purpose-specific,
    time-bound, and explicit about what data is being shared with whom.
    """
    consent_id: str
    customer_id: str
    fip_name: str  # Financial Information Provider, e.g. "HDFC Bank"
    fiu_name: str = "Sahaay AI"  # Financial Information User (this platform)
    purpose: str = "Personalized recommendations, onboarding assistance, and early-warning support"
    data_requested: str = "12 months of transactions, account balance, EMI schedule"
    fetch_frequency: str = "daily"
    validity_start: date = field(default_factory=date.today)
    validity_end: Optional[date] = None
    status: str = "ACTIVE"  # ACTIVE | REVOKED | EXPIRED


@dataclass
class Customer:
    customer_id: str
    name: str
    age: int
    gender: str
    city: str
    state: str
    tier: str  # "Tier 1" | "Tier 2" | "Tier 3" | "Tier 4"
    language_preference: str
    income_type: IncomeType
    archetype: Archetype  # ground truth, not used as a model feature directly
    onboarding_date: date


@dataclass
class Account:
    account_id: str
    customer_id: str
    bank_name: str
    account_type: str  # "savings" | "current"
    opening_balance: float


@dataclass
class Transaction:
    transaction_id: str
    account_id: str
    customer_id: str
    timestamp: datetime
    amount: float
    txn_type: TransactionType
    channel: TransactionChannel
    merchant_category: MerchantCategory
    merchant_name: str
    city: str
    device_id: str
    beneficiary_id: Optional[str]
    is_new_device: bool = False
    is_new_beneficiary: bool = False
    is_new_merchant: bool = False
    # ground-truth label, not something a real system would receive up front
    is_fraud_label: bool = False


@dataclass
class EMIRecord:
    emi_id: str
    customer_id: str
    account_id: str
    loan_type: str  # "personal_loan" | "home_loan" | "vehicle_loan" | "credit_card"
    due_date: date
    amount_due: float
    status: EMIStatus
    days_late: int = 0


@dataclass
class GroundTruthLabels:
    """
    Kept separate from customer-facing tables. In a real system these would never
    exist in this form — they're here purely to evaluate prototype model quality
    against a known-correct answer, per the research doc's evaluation approach.
    """
    customer_id: str
    stress_label: str  # "stable" | "watch" | "concern" | "high_concern"
    suitable_products: list  # e.g. ["recurring_deposit", "credit_card"]
    unsuitable_products: list
    has_fraud_event: bool
