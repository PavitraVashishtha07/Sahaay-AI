# Sahaay AI — Core Intelligence & Multilingual Financial Engine

[![Backend Tests](https://img.shields.io/badge/pytest-71%20passed-brightgreen.svg)](tests/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Sahaay AI is an ethical, consent-gated financial companion built for the Indian banking and Account Aggregator (AA) ecosystem. It unites deterministic financial arbitration, XGBoost suitability models with SHAP explainability, unsupervised behavioral fraud detection, and a multilingual Gemini Flash intent layer grounded in verified bank records.

---

## Architecture Overview

```
sahaay-ai/
├── data_layer/
│   ├── schema.py                  # Dataclasses for customers, accounts, txns, EMI, consent
│   ├── generate_synthetic_data.py # 10,000 customers / ~910k transactions dataset generator
│   ├── build_customer_profiles.py # 38-feature unified profile calculator
│   └── aa_interface.py            # Consent-gated AA pipe with cloud gzip auto-extraction
├── engines/
│   ├── stress_engine.py           # Module 4.3: Auditable weighted stress scoring & drivers
│   ├── fraud_engine.py            # Module 4.4: Unsupervised Isolation Forest anomaly detection
│   ├── recommendation_engine.py   # Module 4.2: Dual-path XGBoost + SHAP & GMM cold-start
│   ├── arbitration_engine.py      # Module 4.5: 6-Tier deterministic priority state machine
│   └── conversational_engine.py   # Module 5: Multilingual Gemini Flash extraction & factual reply
├── security/
│   └── sanitize.py                # Indic & multi-script prompt injection / delimiter sanitizer
├── app/
│   └── main.py                    # FastAPI REST API with dynamic CORS & RBAC security gateway
├── data/                          # Datasets (compressed transactions.csv.gz for cloud deployments)
├── reference/
│   └── voice_chat_demo.html       # Reference frontend implementation for Web Speech ASR/TTS
├── tests/                         # 71 comprehensive unit & integration tests
├── render.yaml                    # Render Blueprint configuration
├── railway.json                   # Railway Nixpacks deployment configuration
├── Procfile                       # Process configuration (uvicorn start command)
└── requirements.txt               # Pinned, production-compatible dependencies
```

---

## Cloud Deployment Guide (Render / Railway)

### 1. Render Deployment (Recommended)
1. Fork or push this repository to GitHub: `https://github.com/PavitraVashishtha07/Sahaay-AI.git`.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New +** -> **Web Service** (or **Blueprint**).
3. Connect your repository. Render automatically reads [`render.yaml`](file:///c:/Users/Pavitra/OneDrive/Desktop/HackOut/New%20folder/sahaay-ai/render.yaml) or you can manually configure:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. In the **Environment Variables** section, configure the required variables listed below.
5. Click **Deploy Web Service**.

### 2. Railway Deployment
1. In the [Railway Dashboard](https://railway.app/), click **New Project** -> **Deploy from GitHub repo**.
2. Railway will automatically detect [`railway.json`](file:///c:/Users/Pavitra/OneDrive/Desktop/HackOut/New%20folder/sahaay-ai/railway.json) / [`Procfile`](file:///c:/Users/Pavitra/OneDrive/Desktop/HackOut/New%20folder/sahaay-ai/Procfile).
3. Add environment variables under the **Variables** tab.

---

## Environment Variables & Configuration

All secrets and settings are injected via environment variables. **No API keys or sensitive credentials are hardcoded in the codebase.**

| Variable Name | Required? | Default Value | Purpose / Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Optional (Recommended) | *None* | Google Gemini API key for multilingual intent & entity extraction (`gemini-2.5-flash`). If omitted, Sahaay AI automatically operates using its built-in deterministic regex fallback engine with zero downtime. |
| `PORT` | Auto-set | `8000` | Port on which the Uvicorn ASGI server listens. Automatically provided by Render and Railway. |
| `ALLOWED_ORIGINS` | Optional | `*` | Comma-separated list of allowed frontend domains for CORS (e.g. `https://sahaay-frontend.vercel.app,http://localhost:3000`). Default `*` allows seamless teammate frontend connectivity. |
| `GEMINI_MODEL` | Optional | `gemini-2.5-flash` | Gemini model variant used for language detection and JSON slot extraction. |
| `GEMINI_TIMEOUT_SECONDS` | Optional | `3.5` | Fast timeout threshold for Gemini API requests before automatically falling back to regex extraction to preserve sub-second latency. |

---

## Dataset Size Strategy for Deployment

- **Full Scale Maintained**: Sahaay AI runs on the full **10,000-customer / ~910,000-transaction dataset** across all 4 archetypes (Stable Salaried, Gig Worker, Financially Stressed, Fraud Scenario).
- **GitHub 100MB Limit Handling**: `data/transactions.csv` (162.5MB uncompressed) is stored as `data/transactions.csv.gz` (43.3MB) in Git, well below GitHub's 100MB file limit.
- **Instant Cloud Auto-Extraction**: On container boot, [`data_layer/aa_interface.ensure_data_files_extracted()`](file:///c:/Users/Pavitra/OneDrive/Desktop/HackOut/New%20folder/sahaay-ai/data_layer/aa_interface.py) automatically unpacks `transactions.csv.gz` into `transactions.csv` in **~0.5 seconds**.
- **Container Limits**: The entire uncompressed dataset is ~185MB, comfortably within the 512MB RAM / 1GB–5GB disk limits of Render and Railway free tiers. **No downsampling or dataset truncation is needed.**

---

## Core API Endpoints

### 1. System Health & Metadata
- `GET /health` — Health check endpoint (returns `{"status": "ok"}`).
- `GET /personas` — Returns the 4 demo personas with archetype and customer IDs.
- `GET /customers` — List of all customer IDs in the system.

### 2. Customer Intelligence & Analytics
- `GET /customers/{id}/profile` — 38-feature unified profile with ground truth stripped.
- `GET /customers/{id}/stress` — Auditable weighted stress score, risk band, and causal drivers.
- `GET /customers/{id}/fraud` — Unsupervised Isolation Forest anomaly score and novelty flags.
- `GET /customers/{id}/recommendations` — XGBoost suitability predictions + SHAP reason codes (or GMM fallback).

### 3. Arbitration Governance Layer (Section 4.5)
- `GET /customers/{id}/arbitrate` — Runs the full 6-tier deterministic priority arbitration engine.
- `POST /customers/{id}/arbitrate` — POST variant supporting explicit customer intents and context payloads.

### 4. Grounded Multilingual Conversational Layer (Section 5)
- `POST /customers/{id}/chat` — Multi-language natural language conversation layer:
  - Supports 10 Indian languages: English (`en`), Hindi (`hi`), Gujarati (`gu`), Marathi (`mr`), Tamil (`ta`), Telugu (`te`), Bengali (`bn`), Kannada (`kn`), Punjabi (`pa`), Malayalam (`ml`).
  - Gemini Flash performs extraction only (intent, language, entities) — it never sees raw transaction data and never fabricates balances.
  - Responses are deterministically compiled from verified backend financial facts.
  - Returns `tts_supported: true` and speech synthesis metadata.

---

## Local Development & Testing

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the test suite (71 tests)
pytest -v

# 3. Start local development server
uvicorn app.main:app --reload --port 8000
```

---

## Note on Scaffold Dashboard
The `/dashboard` and `/app` routes in [`app/main.py`](file:///c:/Users/Pavitra/OneDrive/Desktop/HackOut/New%20folder/sahaay-ai/app/main.py) serve an internal HTML/JS verification scaffold for developer sanity checks. The real production frontend is maintained in a separate repository by the frontend team.
