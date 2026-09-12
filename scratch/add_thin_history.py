import pandas as pd
import os

data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
cid = "cust_cold_start_new"

# 1. customers.csv
cust_path = os.path.join(data_dir, "customers.csv")
cust_df = pd.read_csv(cust_path)
if cid not in cust_df["customer_id"].values:
    new_c = {
        "customer_id": cid,
        "first_name": "Kavita",
        "last_name": "Rao",
        "age": 24,
        "gender": "F",
        "city": "Pune",
        "state": "Maharashtra",
        "tier": "Tier-2",
        "language_preference": "en",
        "income_type_declared": "salaried",
        "onboarded_at": "2026-08-25",
        "archetype": "thin_history",
        "split": "test",
    }
    cust_df = pd.concat([cust_df, pd.DataFrame([new_c])], ignore_index=True)
    cust_df.to_csv(cust_path, index=False)
    print("Added to customers.csv")

# 2. consent_artefacts.csv
con_path = os.path.join(data_dir, "consent_artefacts.csv")
con_df = pd.read_csv(con_path)
if cid not in con_df["customer_id"].values:
    new_con = {
        "consent_id": "con_cold_001",
        "customer_id": cid,
        "status": "ACTIVE",
        "created_at": "2026-08-25T10:00:00Z",
        "expires_at": "2027-08-25T10:00:00Z",
        "data_types": "['profile', 'accounts', 'transactions', 'emi_records']",
        "purpose_code": "FINANCIAL_INTELLIGENCE",
    }
    con_df = pd.concat([con_df, pd.DataFrame([new_con])], ignore_index=True)
    con_df.to_csv(con_path, index=False)
    print("Added to consent_artefacts.csv")

# 3. accounts.csv
acc_path = os.path.join(data_dir, "accounts.csv")
acc_df = pd.read_csv(acc_path)
if cid not in acc_df["customer_id"].values:
    new_acc = {
        "account_id": "acc_cold_001",
        "customer_id": cid,
        "account_type": "savings",
        "current_balance": 18500.0,
        "currency": "INR",
        "opened_at": "2026-08-25T10:00:00Z",
    }
    acc_df = pd.concat([acc_df, pd.DataFrame([new_acc])], ignore_index=True)
    acc_df.to_csv(acc_path, index=False)
    print("Added to accounts.csv")

# 4. customer_profile.csv
prof_path = os.path.join(data_dir, "customer_profile.csv")
prof_df = pd.read_csv(prof_path)
if cid not in prof_df["customer_id"].values:
    new_prof = {col: 0.0 for col in prof_df.columns}
    new_prof.update({
        "customer_id": cid,
        "age": 24,
        "gender": "F",
        "city": "Pune",
        "state": "Maharashtra",
        "tier": "Tier-2",
        "language_preference": "en",
        "income_type_declared": "salaried",
        "avg_monthly_income": 28000.0,
        "income_pattern": "regular_monthly",
        "avg_income_per_credit": 28000.0,
        "income_credit_count": 1,
        "income_interval_days_mean": 30.0,
        "income_interval_days_std": 0.0,
        "income_amount_cv": 0.02,
        "rolling_income_30d_mean": 28000.0,
        "rolling_income_30d_std": 0.0,
        "avg_monthly_savings_rate": 0.25,
        "savings_rate_trend": 0.0,
        "balance_trend": 0.0,
        "has_active_loans": False,
        "num_active_loans": 0,
        "total_monthly_emi": 0.0,
        "emi_to_income_ratio": 0.0,
        "missed_emi_count": 0,
        "late_emi_count": 0,
        "emi_on_time_rate": 1.0,
        "max_days_late": 0,
        "essential_spend_share": 0.55,
        "discretionary_spend_share": 0.20,
        "cash_withdrawal_share": 0.10,
        "new_device_rate": 0.0,
        "new_beneficiary_rate": 0.0,
        "new_merchant_rate": 0.1,
        "txn_velocity_last_7d": 1,
        "num_distinct_devices": 1,
        "profile_computed_at": "2026-09-12T10:00:00Z",
        "archetype_ground_truth": "thin_history",
        "split": "test",
    })
    prof_df = pd.concat([prof_df, pd.DataFrame([new_prof])], ignore_index=True)
    prof_df.to_csv(prof_path, index=False)
    print("Added to customer_profile.csv")

print("Cold-start persona setup successfully!")
