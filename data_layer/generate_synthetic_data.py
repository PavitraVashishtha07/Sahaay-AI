"""
Sahaay AI — Synthetic Data Generator

Generates an India-appropriate synthetic dataset shaped like data that would
arrive through the Account Aggregator pipe: customers, accounts, transactions,
EMI schedules, consent artefacts, and ground-truth labels for evaluation.

NOT real bank data. NOT PaySim. Built specifically to cover the four customer
archetypes called out in the research doc:
  1. Stable salaried       — consistent income, no missed payments
  2. Gig / irregular       — variable-sized, irregular-timing income
  3. Financially stressed  — income drop, delayed EMIs, rising essential spend
  4. Fraud scenario        — new device/beneficiary/location, rapid transfer burst

Run:
    python generate_synthetic_data.py --n-customers 300 --out-dir ../data
"""

from __future__ import annotations
import argparse
import random
import uuid
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from schema import (
    Archetype, IncomeType, TransactionType, TransactionChannel,
    MerchantCategory, EMIStatus,
)

fake = Faker("en_IN")
Faker.seed(42)
random.seed(42)
np.random.seed(42)

CITIES_BY_TIER = {
    "Tier 1": [("Mumbai", "Maharashtra"), ("Delhi", "Delhi"), ("Bengaluru", "Karnataka")],
    "Tier 2": [("Nagpur", "Maharashtra"), ("Indore", "Madhya Pradesh"), ("Coimbatore", "Tamil Nadu")],
    "Tier 3": [("Jhansi", "Uttar Pradesh"), ("Bilaspur", "Chhattisgarh"), ("Guna", "Madhya Pradesh")],
    "Tier 4": [("Chhindwara", "Madhya Pradesh"), ("Deoria", "Uttar Pradesh"), ("Barpeta", "Assam")],
}

LANGUAGES = ["Hindi", "English", "Marathi", "Tamil", "Telugu", "Gujarati", "Bengali", "Kannada"]

BANKS = ["State Bank of India", "HDFC Bank", "ICICI Bank", "Axis Bank", "Punjab National Bank", "Bank of Baroda"]

MERCHANTS = {
    MerchantCategory.GROCERY: ["BigBasket", "Local Kirana Store", "DMart", "More Supermarket"],
    MerchantCategory.RENT: ["Landlord Transfer"],
    MerchantCategory.UTILITIES: ["Electricity Board", "Gas Agency", "Broadband ISP"],
    MerchantCategory.HEALTHCARE: ["Apollo Pharmacy", "Local Clinic", "Diagnostic Lab"],
    MerchantCategory.EDUCATION: ["School Fee Portal", "Coaching Institute", "Online Course Platform"],
    MerchantCategory.ENTERTAINMENT: ["Netflix", "BookMyShow", "Local Cinema"],
    MerchantCategory.ECOMMERCE: ["Amazon", "Flipkart", "Myntra"],
    MerchantCategory.FUEL: ["Indian Oil", "HP Petrol Pump"],
    MerchantCategory.INVESTMENT: ["Zerodha", "Groww", "Post Office RD"],
    MerchantCategory.INSURANCE_PREMIUM: ["LIC Premium", "Star Health Insurance"],
    MerchantCategory.TRANSFER_PEER: ["Peer Transfer"],
    MerchantCategory.UNKNOWN_MERCHANT: ["Unrecognized Merchant XZ", "New Payee 9911"],
}

LOAN_TYPES = ["personal_loan", "vehicle_loan", "credit_card", "home_loan"]


def _rand_city_tier():
    tier = random.choices(["Tier 1", "Tier 2", "Tier 3", "Tier 4"], weights=[0.25, 0.3, 0.25, 0.2])[0]
    city, state = random.choice(CITIES_BY_TIER[tier])
    return city, state, tier


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _daterange_months_back(months: int):
    today = date.today()
    start = today - timedelta(days=months * 30)
    return start, today


# --------------------------------------------------------------------------
# Per-archetype transaction generators
# --------------------------------------------------------------------------

def gen_stable_salaried_transactions(customer_id, account_id, months=6):
    """Consistent monthly salary, predictable spend, no missed EMIs."""
    txns = []
    start, end = _daterange_months_back(months)
    monthly_salary = round(random.uniform(35000, 90000), 2)
    device_id = _new_id("dev")
    day = start
    month_idx = 0
    while day < end:
        salary_date = start + timedelta(days=30 * month_idx + random.randint(0, 1))
        if salary_date >= end:
            break
        txns.append(_make_txn(customer_id, account_id, salary_date, monthly_salary,
                               TransactionType.CREDIT, TransactionChannel.NEFT,
                               MerchantCategory.SALARY, "Employer Salary Credit", device_id))
        # regular monthly spend, proportion of salary
        for cat in [MerchantCategory.GROCERY, MerchantCategory.UTILITIES, MerchantCategory.RENT,
                    MerchantCategory.ENTERTAINMENT, MerchantCategory.FUEL]:
            amt = round(monthly_salary * random.uniform(0.03, 0.15), 2)
            spend_date = salary_date + timedelta(days=random.randint(1, 25))
            merchant = random.choice(MERCHANTS.get(cat, ["Generic Merchant"]))
            txns.append(_make_txn(customer_id, account_id, spend_date, amt,
                                   TransactionType.DEBIT, random.choice(list(TransactionChannel)),
                                   cat, merchant, device_id))
        # occasional saving / investment
        if random.random() < 0.6:
            amt = round(monthly_salary * random.uniform(0.05, 0.2), 2)
            inv_date = salary_date + timedelta(days=random.randint(2, 10))
            txns.append(_make_txn(customer_id, account_id, inv_date, amt,
                                   TransactionType.DEBIT, TransactionChannel.UPI,
                                   MerchantCategory.INVESTMENT, random.choice(MERCHANTS[MerchantCategory.INVESTMENT]),
                                   device_id))
        month_idx += 1
    return txns


def gen_gig_irregular_transactions(customer_id, account_id, months=6):
    """Frequent, small, irregularly-timed payouts instead of one monthly salary."""
    txns = []
    start, end = _daterange_months_back(months)
    device_id = _new_id("dev")
    day = start
    while day < end:
        # payout every 2-6 days, variable size
        day += timedelta(days=random.randint(2, 6))
        if day >= end:
            break
        amt = round(random.uniform(500, 4000), 2)
        txns.append(_make_txn(customer_id, account_id, day, amt,
                               TransactionType.CREDIT, TransactionChannel.UPI,
                               MerchantCategory.GIG_PAYOUT, "Gig Platform Payout", device_id))
        if random.random() < 0.5:
            spend_amt = round(amt * random.uniform(0.3, 0.9), 2)
            cat = random.choice([MerchantCategory.GROCERY, MerchantCategory.FUEL, MerchantCategory.UTILITIES])
            txns.append(_make_txn(customer_id, account_id, day + timedelta(days=1), spend_amt,
                                   TransactionType.DEBIT, TransactionChannel.UPI,
                                   cat, random.choice(MERCHANTS.get(cat, ["Generic Merchant"])), device_id))
    return txns


def gen_financially_stressed_transactions(customer_id, account_id, months=6):
    """Starts like stable salaried, then income drops and essential-expense share rises."""
    txns = []
    start, end = _daterange_months_back(months)
    monthly_salary = round(random.uniform(30000, 60000), 2)
    device_id = _new_id("dev")
    month_idx = 0
    decline_start_month = months // 2
    while True:
        salary_date = start + timedelta(days=30 * month_idx + random.randint(0, 2))
        if salary_date >= end:
            break
        # income shrinks and arrives later after the decline point
        if month_idx >= decline_start_month:
            shrink_factor = max(0.3, 1 - 0.15 * (month_idx - decline_start_month + 1))
            salary_date += timedelta(days=random.randint(3, 10))  # delayed credit
            amt = round(monthly_salary * shrink_factor, 2)
        else:
            amt = monthly_salary
        txns.append(_make_txn(customer_id, account_id, salary_date, amt,
                               TransactionType.CREDIT, TransactionChannel.NEFT,
                               MerchantCategory.SALARY, "Employer Salary Credit", device_id))
        # essential spend share rises after decline point
        essential_cats = [MerchantCategory.GROCERY, MerchantCategory.UTILITIES, MerchantCategory.HEALTHCARE, MerchantCategory.RENT]
        discretionary_cats = [MerchantCategory.ENTERTAINMENT, MerchantCategory.ECOMMERCE]
        for cat in essential_cats:
            frac = random.uniform(0.12, 0.22) if month_idx >= decline_start_month else random.uniform(0.05, 0.12)
            spend_amt = round(monthly_salary * frac, 2)
            spend_date = salary_date + timedelta(days=random.randint(1, 20))
            txns.append(_make_txn(customer_id, account_id, spend_date, spend_amt,
                                   TransactionType.DEBIT, random.choice(list(TransactionChannel)),
                                   cat, random.choice(MERCHANTS.get(cat, ["Generic Merchant"])), device_id))
        if month_idx < decline_start_month and random.random() < 0.4:
            cat = random.choice(discretionary_cats)
            spend_amt = round(monthly_salary * random.uniform(0.03, 0.08), 2)
            spend_date = salary_date + timedelta(days=random.randint(1, 20))
            txns.append(_make_txn(customer_id, account_id, spend_date, spend_amt,
                                   TransactionType.DEBIT, TransactionChannel.CARD,
                                   cat, random.choice(MERCHANTS.get(cat, ["Generic Merchant"])), device_id))
        # occasional cash withdrawal spike late in the decline (a stress signal)
        if month_idx >= decline_start_month and random.random() < 0.5:
            cash_amt = round(monthly_salary * random.uniform(0.1, 0.25), 2)
            cash_date = salary_date + timedelta(days=random.randint(2, 15))
            txns.append(_make_txn(customer_id, account_id, cash_date, cash_amt,
                                   TransactionType.DEBIT, TransactionChannel.CASH_WITHDRAWAL,
                                   MerchantCategory.UNKNOWN_MERCHANT, "ATM Withdrawal", device_id))
        month_idx += 1
    return txns


def gen_fraud_scenario_transactions(customer_id, account_id, months=6):
    """Normal baseline behavior, then an injected burst: new device, new beneficiary,
    unusual location, rapid sequence of transfers."""
    txns = gen_stable_salaried_transactions(customer_id, account_id, months=months)
    _, end = _daterange_months_back(months)
    fraud_device = _new_id("dev")
    fraud_city = "Unknown City"
    fraud_burst_start = end - timedelta(days=random.randint(2, 5))
    for i in range(random.randint(4, 7)):
        ts = fraud_burst_start + timedelta(hours=random.randint(0, 6) + i)
        amt = round(random.uniform(8000, 45000), 2)
        beneficiary_id = _new_id("benef")
        txn = _make_txn(customer_id, account_id, ts, amt,
                         TransactionType.DEBIT, TransactionChannel.IMPS,
                         MerchantCategory.UNKNOWN_MERCHANT, "Unrecognized Payee", fraud_device,
                         beneficiary_id=beneficiary_id, city_override=fraud_city)
        txn["is_new_device"] = True
        txn["is_new_beneficiary"] = True
        txn["is_new_merchant"] = True
        txn["is_fraud_label"] = True
        txns.append(txn)
    return txns


def _make_txn(customer_id, account_id, when, amount, txn_type, channel, category, merchant_name,
              device_id, beneficiary_id=None, city_override=None):
    if isinstance(when, date) and not isinstance(when, datetime):
        when = datetime(when.year, when.month, when.day, random.randint(6, 22), random.randint(0, 59))
    return {
        "transaction_id": _new_id("txn"),
        "account_id": account_id,
        "customer_id": customer_id,
        "timestamp": when,
        "amount": amount,
        "txn_type": txn_type.value,
        "channel": channel.value,
        "merchant_category": category.value,
        "merchant_name": merchant_name,
        "city": city_override or "Home City",
        "device_id": device_id,
        "beneficiary_id": beneficiary_id or _new_id("benef"),
        "is_new_device": False,
        "is_new_beneficiary": False,
        "is_new_merchant": False,
        "is_fraud_label": False,
    }


# --------------------------------------------------------------------------
# EMI schedule generator
# --------------------------------------------------------------------------

def gen_emi_records(customer_id, account_id, archetype: Archetype, months=6):
    records = []
    if archetype == Archetype.GIG_IRREGULAR and random.random() < 0.5:
        return records  # many gig workers have no formal EMI product yet
    n_loans = random.choice([1, 1, 2])
    start, end = _daterange_months_back(months)
    for _ in range(n_loans):
        loan_type = random.choice(LOAN_TYPES)
        amount_due = round(random.uniform(2000, 25000), 2)
        month_idx = 0
        while True:
            due_date = start + timedelta(days=30 * month_idx + random.randint(0, 3))
            if due_date >= end:
                break
            if archetype == Archetype.FINANCIALLY_STRESSED and month_idx >= months // 2:
                status = random.choices(
                    [EMIStatus.MISSED, EMIStatus.PAID_LATE, EMIStatus.PAID_ON_TIME],
                    weights=[0.35, 0.4, 0.25]
                )[0]
                days_late = random.randint(3, 20) if status == EMIStatus.PAID_LATE else (999 if status == EMIStatus.MISSED else 0)
            else:
                status = EMIStatus.PAID_ON_TIME
                days_late = 0
            records.append({
                "emi_id": _new_id("emi"),
                "customer_id": customer_id,
                "account_id": account_id,
                "loan_type": loan_type,
                "due_date": due_date,
                "amount_due": amount_due,
                "status": status.value,
                "days_late": days_late,
            })
            month_idx += 1
    return records


# --------------------------------------------------------------------------
# Ground truth label derivation
# --------------------------------------------------------------------------

def derive_ground_truth(customer_id, archetype: Archetype, emi_records):
    missed = sum(1 for e in emi_records if e["status"] == EMIStatus.MISSED.value)
    late = sum(1 for e in emi_records if e["status"] == EMIStatus.PAID_LATE.value)

    if archetype == Archetype.FINANCIALLY_STRESSED:
        stress_label = "high_concern" if missed >= 2 else "concern"
        suitable = ["financial_health_checkin", "emi_restructuring"]
        unsuitable = ["personal_loan", "credit_card", "investment_product"]
    elif archetype == Archetype.GIG_IRREGULAR:
        stress_label = "watch" if late > 0 else "stable"
        suitable = ["flexible_micro_credit", "recurring_deposit_small_ticket"]
        unsuitable = ["fixed_emi_personal_loan"]
    elif archetype == Archetype.FRAUD_SCENARIO:
        stress_label = "stable"
        suitable = ["recurring_deposit", "credit_card"]
        unsuitable = []
    else:  # stable salaried
        stress_label = "stable"
        suitable = ["recurring_deposit", "credit_card", "mutual_fund_sip", "term_insurance"]
        unsuitable = []

    return {
        "customer_id": customer_id,
        "stress_label": stress_label,
        "suitable_products": suitable,
        "unsuitable_products": unsuitable,
        "has_fraud_event": archetype == Archetype.FRAUD_SCENARIO,
    }


# --------------------------------------------------------------------------
# Main generation loop
# --------------------------------------------------------------------------

ARCHETYPE_WEIGHTS = {
    Archetype.STABLE_SALARIED: 0.45,
    Archetype.GIG_IRREGULAR: 0.25,
    Archetype.FINANCIALLY_STRESSED: 0.20,
    Archetype.FRAUD_SCENARIO: 0.10,
}

ARCHETYPE_GENERATORS = {
    Archetype.STABLE_SALARIED: gen_stable_salaried_transactions,
    Archetype.GIG_IRREGULAR: gen_gig_irregular_transactions,
    Archetype.FINANCIALLY_STRESSED: gen_financially_stressed_transactions,
    Archetype.FRAUD_SCENARIO: gen_fraud_scenario_transactions,
}

INCOME_TYPE_BY_ARCHETYPE = {
    Archetype.STABLE_SALARIED: IncomeType.SALARIED,
    Archetype.GIG_IRREGULAR: IncomeType.GIG_IRREGULAR,
    Archetype.FINANCIALLY_STRESSED: IncomeType.SALARIED,
    Archetype.FRAUD_SCENARIO: IncomeType.SALARIED,
}


def generate_dataset(n_customers: int, months: int = 6, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    customers, accounts, all_txns, all_emis, all_labels, all_consents = [], [], [], [], [], []

    archetypes_pool = random.choices(
        list(ARCHETYPE_WEIGHTS.keys()),
        weights=list(ARCHETYPE_WEIGHTS.values()),
        k=n_customers,
    )

    # Stratified 80/20 train/test split, assigned per archetype so the class
    # balance is preserved in both splits (used by the recommendation engine
    # and stress/fraud model training in later modules).
    split_assignment = {}
    by_archetype_indices = {a: [] for a in ARCHETYPE_WEIGHTS}
    for idx, a in enumerate(archetypes_pool):
        by_archetype_indices[a].append(idx)
    for a, idxs in by_archetype_indices.items():
        shuffled = idxs.copy()
        random.shuffle(shuffled)
        cutoff = int(len(shuffled) * 0.8)
        for j in shuffled[:cutoff]:
            split_assignment[j] = "train"
        for j in shuffled[cutoff:]:
            split_assignment[j] = "test"

    for i, archetype in enumerate(archetypes_pool):
        customer_id = _new_id("cust")
        city, state, tier = _rand_city_tier()
        onboarding_date = fake.date_between(start_date="-2y", end_date="-6M")

        customers.append({
            "customer_id": customer_id,
            "name": fake.name(),
            "age": random.randint(21, 60),
            "gender": random.choice(["Male", "Female", "Other"]),
            "city": city,
            "state": state,
            "tier": tier,
            "language_preference": random.choice(LANGUAGES),
            "income_type": INCOME_TYPE_BY_ARCHETYPE[archetype].value,
            "archetype": archetype.value,  # kept for eval only, never fed to models
            "onboarding_date": onboarding_date,
            "split": split_assignment[i],  # "train" | "test" — stratified by archetype
        })

        account_id = _new_id("acct")
        accounts.append({
            "account_id": account_id,
            "customer_id": customer_id,
            "bank_name": random.choice(BANKS),
            "account_type": "savings",
            "opening_balance": round(random.uniform(2000, 50000), 2),
        })

        txns = ARCHETYPE_GENERATORS[archetype](customer_id, account_id, months=months)
        all_txns.extend(txns)

        emis = gen_emi_records(customer_id, account_id, archetype, months=months)
        all_emis.extend(emis)

        all_labels.append(derive_ground_truth(customer_id, archetype, emis))

        all_consents.append({
            "consent_id": _new_id("consent"),
            "customer_id": customer_id,
            "fip_name": accounts[-1]["bank_name"],
            "fiu_name": "Sahaay AI",
            "purpose": "Personalized recommendations, onboarding assistance, and early-warning support",
            "data_requested": f"{months} months of transactions, account balance, EMI schedule",
            "fetch_frequency": "daily",
            "validity_start": date.today() - timedelta(days=1),
            "validity_end": date.today() + timedelta(days=364),
            "status": "ACTIVE",
        })

    return {
        "customers": pd.DataFrame(customers),
        "accounts": pd.DataFrame(accounts),
        "transactions": pd.DataFrame(all_txns).sort_values("timestamp").reset_index(drop=True),
        "emi_records": pd.DataFrame(all_emis),
        "ground_truth_labels": pd.DataFrame(all_labels),
        "consent_artefacts": pd.DataFrame(all_consents),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Sahaay AI synthetic dataset")
    parser.add_argument("--n-customers", type=int, default=10000)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--out-dir", type=str, default="../data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = generate_dataset(args.n_customers, months=args.months, seed=args.seed)

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    for name, df in dataset.items():
        path = os.path.join(args.out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"wrote {len(df):>6} rows -> {path}")

    print("\nArchetype distribution:")
    print(dataset["customers"]["archetype"].value_counts())


if __name__ == "__main__":
    main()
