# Sahaay AI — Data Layer (Module 1)

This is the first implemented layer of the Sahaay AI architecture: the
Account-Aggregator-shaped data pipe that everything else (customer
intelligence, recommendation, conversational, stress/fraud, arbitration)
will be built on top of.

## What's here

```
sahaay-ai/
├── data_layer/
│   ├── schema.py                  # dataclasses for customers, accounts, transactions, EMI, consent
│   ├── generate_synthetic_data.py # generates the 4 archetypes into data/*.csv
│   └── aa_interface.py            # the ONE module boundary — get_consented_data()
├── app/
│   └── main.py                    # FastAPI wrapper exposing the data layer over HTTP
├── data/                          # generated CSVs (customers, accounts, transactions, emi_records,
│                                   #   ground_truth_labels, consent_artefacts)
└── requirements.txt
```

## Why it's built this way

- **Consent-gated by design.** `get_consented_data(customer_id, consent_id)` is the only
  way downstream code touches financial data. It checks the consent artefact is
  active and unexpired before returning anything — mirroring RBI's AA Directions,
  not just modeling the transaction shape.
- **Swap-in-place for a real AA connection.** Everything outside `aa_interface.py`
  only ever calls that one function. When a real AA participant (Setu, Anumati,
  etc.) is integrated post-hackathon, only this file changes.
- **Ground truth is kept separate.** `ground_truth_labels.csv` (stress label,
  suitable/unsuitable products, fraud flag) is for evaluating prototype models —
  a real system would never have this up front. `get_ground_truth()` is
  deliberately a separate function so it can't accidentally leak into a
  decision path.
- **Four archetypes, generated deliberately, not randomly:**
  1. `stable_salaried` — consistent monthly salary, predictable spend, on-time EMIs
  2. `gig_irregular` — frequent small irregular payouts, no single salary anchor
  3. `financially_stressed` — income shrinks and arrives late partway through the
     window, essential-expense share rises, EMIs start slipping, occasional cash-out spikes
  4. `fraud_scenario` — a normal baseline, then an injected burst of rapid transfers
     from a new device, to a new beneficiary, from an unusual location

## Running it

```bash
pip install -r requirements.txt

# generate the dataset (defaults: 300 customers, 6 months of history)
cd data_layer
python generate_synthetic_data.py --n-customers 300 --months 6 --out-dir ../data

# run the API
cd ../app
uvicorn main:app --reload --port 8000
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness check |
| `GET /customers` | list all synthetic customer IDs |
| `GET /customers/{id}/consent` | look up the active consent_id for a customer (prototype convenience only — a real system already knows this) |
| `GET /customers/{id}/data?consent_id=...` | the core call: returns customer + accounts + transactions + EMI records, gated on valid consent |
| `POST /consent/revoke` | simulates a customer revoking consent mid-session — subsequent `/data` calls correctly 403 |

## Verified

- Generates 300 customers / ~13.6k transactions / ~2k EMI records across all four archetypes in one run.
- `get_consented_data()` correctly returns full data for an active consent.
- Revoking consent correctly raises `ConsentError` (403 over the API) on the next access attempt — the "customer revokes consent mid-session" adversarial scenario from the research doc.

## Module 2 — Unified Customer Intelligence Layer

`data_layer/build_customer_profiles.py` reads raw transactions + EMI records
(grouped in one pass, not re-queried per customer) and computes a single
`customer_profile.csv` — the "shared brain" every downstream engine must read
from instead of building its own private view. 39 columns per customer, all
hand-engineered and explainable (no learned embedding — deliberate, see the
feature-breakdown doc's reasoning).

Key features computed:
- **Income pattern classifier** (`income_pattern`): `salaried_regular` /
  `gig_irregular` / `mixed_or_business`, derived from credit interval + amount
  variability — not the customer's self-declared income type. This is the
  gate that prevents gig workers from being scored against a "salary landed"
  assumption they'll never trigger.
- **Rolling 30-day income stability** — the anchor signal for irregular-income
  customers, replacing the single-event trigger salaried customers use.
- **Savings rate + trend, balance trend** — monthly savings rate and its
  linear slope across the observed window; feeds the stress score in Module 5.
- **EMI-to-income ratio, missed/late EMI counts, on-time rate**.
- **Spend composition** — essential vs. discretionary vs. cash-withdrawal share.
- **Novelty/velocity signals** — new-device, new-beneficiary, new-merchant
  rates and 7-day transaction velocity (raw inputs the fraud engine will
  consume directly at the transaction level; profile-level versions here are
  a general novelty indicator).

Ground-truth `archetype_ground_truth` and `split` (stratified 80/20 train/test,
assigned per archetype at generation time) are carried through for evaluation
only — never treated as a feature, and stripped before the API returns a
profile (`GET /customers/{id}/profile`).

### Verified (`tests/test_customer_profiles.py`, 8/8 passing)

- Every consented customer gets exactly one profile row.
- <5% of gig/irregular customers are misclassified as `salaried_regular`
  (the specific failure mode the research doc's stress-test called out).
- >85% of stable-salaried customers are correctly detected as `salaried_regular`.
- Financially-stressed customers show higher missed-EMI counts and worse
  balance trend than stable-salaried customers.
- Fraud-scenario customers look financially normal at the profile level
  (missed-EMI count far below financially-stressed customers) — confirming
  stress and fraud stay separable signals, per the "never merge them" rule.
- No ground-truth label leaks into the feature columns under a different name.
- The train/test split is correctly stratified (~80/20) within every archetype.

### Running it

```bash
cd data_layer
python build_customer_profiles.py --out ../data/customer_profile.csv

cd ../tests
python test_customer_profiles.py
```

## Dataset scale

Regenerated at `--n-customers 10000 --months 12` to match model-training needs:

| File | Rows |
|---|---|
| customers.csv | 10,000 |
| accounts.csv | 10,000 |
| transactions.csv | ~910,000 |
| emi_records.csv | ~138,700 |
| consent_artefacts.csv | 10,000 |
| ground_truth_labels.csv | 10,000 |
| **customer_profile.csv** | **10,000** |

Archetype split (generation-time weights: 45/25/20/10): stable_salaried ~4,474,
gig_irregular ~2,549, financially_stressed ~1,997, fraud_scenario ~980.
Train/test split is stratified 80/20 within each archetype (see `split` column).

## Section 4 — Machine Learning & Intelligence Engines

The three specialized intelligence engines read from the shared data and profile layers:

### 1. Stress Detection Engine (`engines/stress_engine.py`)
- **Deterministic Weighted-Score**: Auditable design with visible weight constants (`WEIGHT_MISSED_EMI = 0.35`, `WEIGHT_SAVINGS_DECLINE = 0.20`, etc.).
- **Signal Normalization**: Evaluates 6 normalized signals [0, 1] (missed EMIs, savings trend, essential spend share, balance cushion depletion, cash spikes, EMI burden).
- **Bucketing**: `stable` (<0.25), `watch` (0.25–0.45), `concern` (0.45–0.70), `high_concern` (≥0.70).
- **Explainability**: Outputs top causal drivers (e.g. `["missed_or_delayed_emis", "cash_buffer_depletion"]`).
- **Secondary Check**: Dual-outputs unsupervised Isolation Forest anomaly scores for complex deterioration patterns.

### 2. Fraud / Anomaly Detection Engine (`engines/fraud_engine.py`)
- **Unsupervised Isolation Forest**: Fitted strictly on transaction-level behavioral novelty features (amount relative to personal baseline, `is_new_device`, `is_new_beneficiary`, `is_new_merchant`, unusual hours, IMPS rapid transfer channel).
- **No Label Leakage**: Trained without `is_fraud_label` (which is reserved exclusively for post-hoc validation).
- **Behavioral Scoring**: Produces normalized anomaly scores [0, 1] and flags specific novelty signals (`["new_device", "new_beneficiary", "amount_burst_spike"]`) without making accusatory "fraud" claims.
- **Evaluation**: Achieves **ROC-AUC = 1.00** and **Recall > 95%** on synthetic anomaly bursts.

### 3. Recommendation Engine (`engines/recommendation_engine.py`)
- **Primary Path (XGBoost + SHAP)**: Binary classifiers per financial product trained on `customer_profile.csv` with the stratified 80/20 train/test split. Every recommendation includes SHAP feature attributions as reason codes ("why am I seeing this").
- **Cold-Start Fallback (GMM Clustering)**: Unsupervised 4-component Gaussian Mixture Model trained on early-window demographic and cash-flow features. Thin-history customers (<30 days history or <10 txns) route to GMM cluster assignment with cluster suitability baselines.
- **Explicit Routing**: Every recommendation payload carries `path_used` (`"xgboost"` or `"gmm_fallback"`).
- **Pattern A Experiment**: Empirically compared standard XGBoost against XGBoost augmented with GMM cluster probabilities across all 8 products:
  - **Result**: Mean $F_1$ Delta = **+0.0000** (Standard $F_1 \ge 0.9989$, Augmented $F_1 \ge 0.9989$).
  - **Insight**: Confirms that GMM probabilities add no extra information on top of rich engineered tabular features, proving that GMM should be strictly reserved for cold-start fallback (Pattern B).

### Running All Engine Tests

```bash
pytest -v tests/
```

All 26 tests across data layer, stress, fraud, and recommendation engines pass (26 passed, 0 failed).

## Next up

Section 4.5 — Arbitration Layer: The deterministic priority rule module governing conflict resolution across the three engine outputs.

