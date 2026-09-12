"""
Seeds REVOKED and EXPIRED consent artefacts in data/consent_artefacts.csv
and adds demo customer profiles for instant persona demonstration.
"""
import pandas as pd
import os

data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
consent_path = os.path.join(data_dir, "consent_artefacts.csv")
cust_path = os.path.join(data_dir, "customers.csv")
acc_path = os.path.join(data_dir, "accounts.csv")
prof_path = os.path.join(data_dir, "customer_profile.csv")

# 1. Update Consent Artefacts
df = pd.read_csv(consent_path)

# Update rows 100 to 114 to REVOKED (15 rows)
revoked_custs = df.iloc[100:115]["customer_id"].tolist()
df.loc[100:114, "status"] = "REVOKED"

# Update rows 115 to 124 to EXPIRED with past date (10 rows)
expired_custs = df.iloc[115:125]["customer_id"].tolist()
df.loc[115:124, "status"] = "EXPIRED"
df.loc[115:124, "validity_end"] = "2025-05-15"

# Add dedicated demo rows
demo_rows = pd.DataFrame([
    {
        "consent_id": "con_demo_revoked_01",
        "customer_id": "cust_revoked_demo",
        "fip_name": "HDFC Bank",
        "fiu_name": "Sahaay AI",
        "purpose": "Personalized recommendations, onboarding assistance, and early-warning support",
        "data_requested": "12 months of transactions, account balance, EMI schedule",
        "fetch_frequency": "daily",
        "validity_start": "2026-01-01",
        "validity_end": "2027-01-01",
        "status": "REVOKED",
        "created_at": "2026-01-01T00:00:00Z",
        "expires_at": "2027-01-01T00:00:00Z",
        "data_types": "['profile', 'accounts', 'transactions', 'emi_records']",
        "purpose_code": "FINANCIAL_INTELLIGENCE"
    },
    {
        "consent_id": "con_demo_expired_01",
        "customer_id": "cust_expired_demo",
        "fip_name": "ICICI Bank",
        "fiu_name": "Sahaay AI",
        "purpose": "Personalized recommendations, onboarding assistance, and early-warning support",
        "data_requested": "12 months of transactions, account balance, EMI schedule",
        "fetch_frequency": "daily",
        "validity_start": "2024-01-01",
        "validity_end": "2025-01-01",
        "status": "EXPIRED",
        "created_at": "2024-01-01T00:00:00Z",
        "expires_at": "2025-01-01T00:00:00Z",
        "data_types": "['profile', 'accounts', 'transactions', 'emi_records']",
        "purpose_code": "FINANCIAL_INTELLIGENCE"
    }
])

df = df[~df["customer_id"].isin(["cust_revoked_demo", "cust_expired_demo"])]
df = pd.concat([df, demo_rows], ignore_index=True)
df.to_csv(consent_path, index=False)

# 2. Add Demo Customers to customers.csv
df_cust = pd.read_csv(cust_path)
df_cust = df_cust[~df_cust["customer_id"].isin(["cust_revoked_demo", "cust_expired_demo"])]
new_custs = pd.DataFrame([
    {
        "customer_id": "cust_revoked_demo",
        "name": "Sanjay Gupta (Revoked Consent)",
        "age": 42,
        "gender": "Male",
        "city": "Mumbai",
        "state": "Maharashtra",
        "tier": "Tier 1",
        "language_preference": "hi",
        "income_type_declared": "salaried",
        "archetype_ground_truth": "stable_salaried",
        "account_created_at": "2026-01-01",
        "split": "test",
        "first_name": "Sanjay",
        "last_name": "Gupta",
        "employment_type": "salaried",
        "created_at": "2026-01-01"
    },
    {
        "customer_id": "cust_expired_demo",
        "name": "Pooja Nair (Expired Consent)",
        "age": 35,
        "gender": "Female",
        "city": "Bengaluru",
        "state": "Karnataka",
        "tier": "Tier 1",
        "language_preference": "en",
        "income_type_declared": "salaried",
        "archetype_ground_truth": "stable_salaried",
        "account_created_at": "2024-01-01",
        "split": "test",
        "first_name": "Pooja",
        "last_name": "Nair",
        "employment_type": "salaried",
        "created_at": "2024-01-01"
    }
])
df_cust = pd.concat([df_cust, new_custs], ignore_index=True)
df_cust.to_csv(cust_path, index=False)

# 3. Add to accounts.csv
df_acc = pd.read_csv(acc_path)
df_acc = df_acc[~df_acc["customer_id"].isin(["cust_revoked_demo", "cust_expired_demo"])]
new_accs = pd.DataFrame([
    {
        "account_id": "acc_revoked_001",
        "customer_id": "cust_revoked_demo",
        "bank_name": "HDFC Bank",
        "account_type": "savings",
        "current_balance": 45000.0,
        "balance": 45000.0,
        "currency": "INR",
        "opened_date": "2026-01-01T00:00:00Z"
    },
    {
        "account_id": "acc_expired_001",
        "customer_id": "cust_expired_demo",
        "bank_name": "ICICI Bank",
        "account_type": "savings",
        "current_balance": 32000.0,
        "balance": 32000.0,
        "currency": "INR",
        "opened_date": "2024-01-01T00:00:00Z"
    }
])
df_acc = pd.concat([df_acc, new_accs], ignore_index=True)
df_acc.to_csv(acc_path, index=False)

# 4. Add to customer_profile.csv
df_prof = pd.read_csv(prof_path)
df_prof = df_prof[~df_prof["customer_id"].isin(["cust_revoked_demo", "cust_expired_demo"])]
# Clone Arjun's profile structure for demo customers
sample_row = df_prof[df_prof["customer_id"] == "cust_86838bd208"].iloc[0].to_dict()

row_rev = sample_row.copy()
row_rev["customer_id"] = "cust_revoked_demo"
row_rev["city"] = "Mumbai"
row_rev["state"] = "Maharashtra"

row_exp = sample_row.copy()
row_exp["customer_id"] = "cust_expired_demo"
row_exp["city"] = "Bengaluru"
row_exp["state"] = "Karnataka"

df_prof = pd.concat([df_prof, pd.DataFrame([row_rev, row_exp])], ignore_index=True)
df_prof.to_csv(prof_path, index=False)

print("\nFinal consent status breakdown:")
print(df["status"].value_counts())
print("\nSuccessfully synchronized customers, accounts, and customer_profiles.")
