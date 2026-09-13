# Sahaay AI — Frontend Reattachment & Integration Guide

This guide explains how to reattach and serve the Sahaay AI frontend after modifying or rebuilding the backend.

---

## 📁 Package Contents

```
frontend/
├── public/
│   ├── css/
│   │   └── sahaay-app.css                  # Shared styles, tokens, shimmer animations
│   ├── app.html                            # Vernacular Banking Customer Homepage
│   ├── chat-assistant.html                 # AI Chat Assistant (State 4A/4B/4C)
│   ├── chat.html                           # Chat Assistant alias
│   ├── cross-cutting-states.html           # Cross-cutting States: Skeleton Loading & Gentle Error
│   ├── dashboard.html                      # Interactive Intelligence & Decision Sandbox
│   ├── index.html                          # Root Decision Panel / Dashboard
│   ├── onboarding.html                     # Onboarding Step 1: Collect Name
│   ├── onboarding-name.html                # Onboarding Step 1 alias
│   ├── onboarding-income.html              # Onboarding Step 2: Income Type
│   ├── onboarding-purpose.html             # Onboarding Step 3: Purpose Selection
│   ├── onboarding-confirm.html             # Onboarding Step 4: Confirmation & Summary
│   ├── onboarding-completed.html           # Onboarding Step 5: Celebration State
│   ├── profile.html                        # My Profile & Settings (10 languages)
│   ├── profile-settings.html               # Profile alias
│   ├── recommendation-detail.html          # Recommendation Deep-Dive (Mutual Fund SIP)
│   ├── recommendation.html                 # Recommendation alias
│   ├── states.html                         # Component reference alias
│   ├── stitch.html                         # Stitch Design System Master Frame
│   ├── stitch_*.html                       # 1:1 Stitch Node Reference files
├── src/
│   ├── app/
│   │   └── page.tsx                        # Next.js App Router root page
│   ├── components/
│   │   ├── AIDecisionPanel.tsx             # React Decision Panel
│   │   ├── ArbitrationHero.tsx             # React Arbitration Component
│   │   ├── ChatAssistant.tsx               # React Chat Assistant Component
│   │   └── PersonaSelector.tsx             # React Persona Switcher
└── package.json                            # Next.js / Tailwind dependencies
```

---

## 🚀 Option 1: Reattach to FastAPI / Python Backend

To mount and serve these HTML files in a FastAPI backend:

```python
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()

# 1. Mount static directory
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "public")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Helper function
def serve_html(filename: str):
    path = os.path.join(FRONTEND_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Page not found</h2>", status_code=404)

# 2. Main Customer Routes
@app.get("/app", response_class=HTMLResponse)
@app.get("/customer", response_class=HTMLResponse)
def page_customer(): return serve_html("app.html")

@app.get("/onboarding", response_class=HTMLResponse)
@app.get("/onboarding/name", response_class=HTMLResponse)
def page_onboarding_name(): return serve_html("onboarding.html")

@app.get("/onboarding/income", response_class=HTMLResponse)
def page_onboarding_income(): return serve_html("onboarding-income.html")

@app.get("/onboarding/purpose", response_class=HTMLResponse)
def page_onboarding_purpose(): return serve_html("onboarding-purpose.html")

@app.get("/onboarding/confirm", response_class=HTMLResponse)
def page_onboarding_confirm(): return serve_html("onboarding-confirm.html")

@app.get("/onboarding/completed", response_class=HTMLResponse)
def page_onboarding_completed(): return serve_html("onboarding-completed.html")

@app.get("/recommendation/detail", response_class=HTMLResponse)
def page_recommendation(): return serve_html("recommendation-detail.html")

@app.get("/chat", response_class=HTMLResponse)
@app.get("/chat/assistant", response_class=HTMLResponse)
@app.get("/talk", response_class=HTMLResponse)
def page_chat(): return serve_html("chat-assistant.html")

@app.get("/profile", response_class=HTMLResponse)
@app.get("/settings", response_class=HTMLResponse)
def page_profile(): return serve_html("profile.html")

@app.get("/cross-cutting-states", response_class=HTMLResponse)
@app.get("/component-states", response_class=HTMLResponse)
def page_cross_cutting(): return serve_html("cross-cutting-states.html")

@app.get("/dashboard", response_class=HTMLResponse)
def page_dashboard(): return serve_html("dashboard.html")
```

---

## ⚡ Option 2: Reattach to Express / Node.js Backend

```javascript
const express = require('express');
const path = require('path');
const app = express();

const publicDir = path.join(__dirname, 'frontend/public');

// Serve static assets
app.use(express.static(publicDir));
app.use('/static', express.static(publicDir));

// Route handlers
app.get('/app', (req, res) => res.sendFile(path.join(publicDir, 'app.html')));
app.get('/onboarding', (req, res) => res.sendFile(path.join(publicDir, 'onboarding.html')));
app.get('/onboarding/income', (req, res) => res.sendFile(path.join(publicDir, 'onboarding-income.html')));
app.get('/onboarding/purpose', (req, res) => res.sendFile(path.join(publicDir, 'onboarding-purpose.html')));
app.get('/onboarding/confirm', (req, res) => res.sendFile(path.join(publicDir, 'onboarding-confirm.html')));
app.get('/onboarding/completed', (req, res) => res.sendFile(path.join(publicDir, 'onboarding-completed.html')));
app.get('/recommendation/detail', (req, res) => res.sendFile(path.join(publicDir, 'recommendation-detail.html')));
app.get('/chat/assistant', (req, res) => res.sendFile(path.join(publicDir, 'chat-assistant.html')));
app.get('/profile', (req, res) => res.sendFile(path.join(publicDir, 'profile.html')));
app.get('/cross-cutting-states', (req, res) => res.sendFile(path.join(publicDir, 'cross-cutting-states.html')));
app.get('/dashboard', (req, res) => res.sendFile(path.join(publicDir, 'dashboard.html')));

app.listen(8085, () => console.log('Server running on port 8085'));
```

---

## ⚛️ Option 3: Run as Standalone Next.js Application

If you prefer building upon the Next.js React codebase:

```bash
cd frontend
npm install
npm run dev
```
Runs at `http://localhost:3000`.

---

## 🔌 API Endpoints Expected by Frontend Pages

When implementing your new backend, these are the API contracts called by the frontend:

| Endpoint | Method | Used In | Purpose |
|---|---|---|---|
| `/personas` | `GET` | `dashboard.html` | Returns list of benchmark customer personas |
| `/customers/{id}/arbitrate` | `GET` | `dashboard.html` | Returns arbitration decision tier, recommendations & SHAP reasons |
| `/consent/revoke` | `POST` | `dashboard.html` | Revokes consent for customer ID |
| `/customers/{id}/chat` | `POST` | `dashboard.html`, `chat-assistant.html` | Sends user query to conversational AI engine |
| `/customers/{id}/onboarding-step` | `POST` | `dashboard.html` | Progresses onboarding state machine |
| `/onboarding/{id}/step` | `POST` | `onboarding-*.html` | Saves user response for current onboarding step |
