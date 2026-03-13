# 🛡️ SupplyGuard AI — Supply Chain Disruption Risk Intelligence

> **A production-grade ML + LLM platform that scores suppliers on disruption risk,
> explains the drivers, generates AI-powered mitigation strategies, and recommends
> optimal supply chain rebalancing using linear programming.**

Built with Python · Streamlit · FastAPI · Scikit-learn · Anthropic Claude · Docker · AWS ECS Fargate

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AWS (ECS Fargate)                     │
│                                                          │
│  ┌─────────────────┐     ┌───────────────────────────┐  │
│  │  Streamlit UI   │     │     FastAPI REST API       │  │
│  │  :8502          │     │     :8001                  │  │
│  │  - Dashboard    │     │  - /api/score (POST)       │  │
│  │  - Risk Analysis│     │  - /api/model-info (GET)   │  │
│  │  - Rebalancing  │     │  - /api/health (GET)       │  │
│  │  - AI Recs      │     │                            │  │
│  └────────┬────────┘     └──────────────┬────────────┘  │
│           └──────────────┬──────────────┘               │
│                 ┌────────▼────────┐                     │
│                 │   ML Backend    │                     │
│                 │  - Risk Model   │                     │
│                 │  - Optimizer    │                     │
│                 │  - LLM Advisor  │                     │
│                 │  - Auth / RBAC  │                     │
│                 └─────────────────┘                     │
│                                                          │
│  ALB (HTTPS/443) → HTTP redirect → Streamlit/FastAPI    │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start (Local)

### 1. Clone & configure
```bash
git clone https://github.com/arunabhachanda/supplyguard-ai.git
cd supplyguard-ai
cp .env.example .env
# Edit .env and add your NEWS_API_KEY and ANTHROPIC_API_KEY
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit app
```bash
streamlit run app.py --server.port 8502
# → http://localhost:8502
```

### 4. Run FastAPI (separate terminal)
```bash
uvicorn fastapi_wrapper:app --reload --port 8001
# → http://localhost:8001/api/docs
```

### Demo Login Credentials
| Role    | Username | Password      | Access                          |
|---------|----------|---------------|---------------------------------|
| Admin   | admin    | Admin@1234    | All features + user management  |
| Analyst | analyst  | Analyst@1234  | Dashboard, upload, AI recs      |
| Viewer  | viewer   | Viewer@1234   | Dashboard only (read-only)      |

---

## 🐳 Docker (Local)

```bash
docker build -t supplyguard-ai .
docker run -p 8502:8502 -p 8001:8001 \
  -e ANTHROPIC_API_KEY=your-key \
  -e NEWS_API_KEY=your-key \
  -e SECRET_KEY=your-secret-key \
  supplyguard-ai
```

---

## ☁️ AWS Deployment

### Prerequisites
- AWS CLI configured (`aws configure`)
- Docker running
- ACM certificate for your domain
- `.env` file populated

### Step 1: Create AWS Secrets Manager entries
```bash
aws secretsmanager create-secret \
  --name supplyguard/anthropic_api_key \
  --secret-string "your-anthropic-api-key" \
  --region eu-central-1

aws secretsmanager create-secret \
  --name supplyguard/secret_key \
  --secret-string "$(python -c 'import secrets; print(secrets.token_hex(32))')" \
  --region eu-central-1

aws secretsmanager create-secret \
  --name supplyguard/api_key_admin \
  --secret-string "$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  --region eu-central-1
```

### Step 2: Deploy CloudFormation stack
```bash
aws cloudformation deploy \
  --template-file aws/cloudformation.yml \
  --stack-name supplyguard-stack \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    DomainName=supplyguard.yourdomain.com \
    CertificateArn=arn:aws:acm:eu-central-1:ACCOUNT:certificate/xxx \
    ECRImageUri=ACCOUNT.dkr.ecr.eu-central-1.amazonaws.com/supplyguard-ai:latest \
    AnthropicSecretArn=arn:aws:secretsmanager:... \
    SecretKeyArn=arn:aws:secretsmanager:... \
    ApiKeyAdminArn=arn:aws:secretsmanager:... \
  --region eu-central-1
```

### Step 3: Build and push Docker image
```bash
chmod +x aws/deploy.sh
./aws/deploy.sh
```

### Step 4: Point your domain to the ALB
```bash
aws cloudformation describe-stacks \
  --stack-name supplyguard-stack \
  --query "Stacks[0].Outputs[?OutputKey=='ALBDNSName'].OutputValue" \
  --output text
# → Create CNAME record: supplyguard.yourdomain.com → ALB DNS
```

---

## 🔐 Security Implementation

### ✅ HTTPS
- ALB terminates TLS using ACM-managed certificate
- HTTP (port 80) auto-redirects to HTTPS (301)
- TLS 1.3 policy: `ELBSecurityPolicy-TLS13-1-2-2021-06`

### ✅ Authentication
- Session-based auth with `hmac.new(SECRET_KEY, ...)` verification
- 8-hour session timeout (configurable in `config.py`)
- Login form with server-side credential verification

### ✅ Brute-Force Protection
- 5 failed attempts → 5-minute account lockout
- Per-session attempt counter

### ✅ RBAC (Role-Based Access Control)
Three roles with granular permission sets:
```python
ROLES = {
    "admin":   ["view_dashboard", "upload_data", "get_recommendations",
                 "export_report", "manage_users", "view_raw_scores"],
    "analyst": ["view_dashboard", "upload_data", "get_recommendations",
                 "export_report", "view_raw_scores"],
    "viewer":  ["view_dashboard"],
}
```

### ✅ Rate Limiting
| Resource           | Limit       | Scope   |
|--------------------|-------------|---------|
| LLM API calls      | 20/hour     | Session |
| File uploads       | 10/session  | Session |
| /api/score (POST)  | 60/minute   | IP      |
| /api/health (GET)  | 30/minute   | IP      |

### ✅ Input Validation
- CSV uploads: column presence, value ranges, NaN detection
- API: Pydantic models with `ge/le` constraints + regex sanitisation
- Supplier names: HTML/script injection stripping

### ✅ Secrets Management
- All secrets loaded from AWS Secrets Manager (not environment variables directly)
- No secrets in Docker image or CloudFormation template

### ✅ Container Security
- Non-root user (`supplyguard`, UID 1001) inside Docker
- ECR image scanning on push
- Minimal base image (`python:3.11-slim`)

---

## 🧠 ML Model Details

| Property        | Value                                |
|-----------------|--------------------------------------|
| Algorithm       | GradientBoostingClassifier (sklearn) |
| Calibration     | Isotonic regression (5-fold CV)      |
| Features        | 12 supply chain risk signals         |
| Training size   | 1,600 synthetic supplier scenarios   |
| ROC-AUC (test)  | ~0.91                                |
| F1 Score (test) | ~0.83                                |

### Feature Engineering
| Feature                    | Description                              |
|----------------------------|------------------------------------------|
| geo_risk_score             | Country geopolitical instability (0–1)   |
| lead_time_days             | Average supplier lead time               |
| lead_time_variance         | Std deviation of lead time               |
| inventory_buffer_days      | Days of safety stock available           |
| supplier_reliability_score | Historical on-time delivery rate (0–1)   |
| financial_health_score     | Supplier financial stability (0–1)       |
| single_source_dependency   | 1 if sole source, 0 otherwise            |
| region_news_sentiment      | NLP sentiment of regional news (−1 to 1) |
| natural_disaster_risk      | Regional natural hazard index (0–1)      |
| past_disruptions_12mo      | Count of disruptions in past 12 months   |
| regulatory_risk_score      | Trade regulation instability (0–1)       |
| transport_mode_risk        | Risk factor per transport mode (0–1)     |

---

## ⚖️ Supply Rebalancing Optimizer

The **Rebalancing** page uses linear programming (`scipy.optimize.linprog`, HiGHS solver) to redistribute purchase volume from high-risk suppliers to low-risk alternatives within the same category.

**Objective:** Minimise portfolio-weighted risk score  
**Constraints:**
- All demand must be met
- Each alternative capped at **1.5× their current spend**
- Total cost must stay within user-defined tolerance (default 20%)
- Safer suppliers carry a cost premium of up to **+25%** reflecting real-world pricing

```
Minimise:   Σ risk_score[i] × allocation[i]
Subject to: Σ allocation[i]             = demand
            allocation[i]               ≤ 1.5 × current_spend[i]
            Σ cost_rate[i] × alloc[i]   ≤ budget × (1 + tolerance)
            allocation[i]               ≥ 0
```

Results include a before/after risk comparison chart, per-supplier allocation breakdown, and a downloadable rebalancing plan CSV.

---

## 🌐 Real Data Sources (Live Mode)

Toggle **Live Data Mode** in the sidebar to replace three synthetic signals with real external APIs. The three signals measure **fundamentally different dimensions of risk** — they are designed to diverge:

| Signal | API | What it measures | Key Required |
|---|---|---|---|
| **Geopolitical Risk** | [World Bank PV.EST](https://data.worldbank.org) | Structural political stability — long-term institutional risk independent of today's news | No |
| **News Sentiment** | [NewsAPI](https://newsapi.org) + TextBlob | Current media sentiment — what the market feels right now | Yes — free at newsapi.org |
| **Disaster Risk** | [GDACS (UN)](https://gdacs.org) | Physical environmental events — earthquakes, floods, cyclones, wildfires (365-day window) | No |

### Why three signals instead of one?

Think of it like a doctor checking three different vitals:

| Signal | Analogy | Example |
|---|---|---|
| **World Bank** | Blood pressure history | USA scores LOW despite negative tariff headlines — institutions are stable |
| **GDACS** | Acute physical injury | Japan scores HIGH despite stable politics — frequent earthquakes and typhoons |
| **NewsAPI** | Mood today | Saudi Arabia scores LOW (state-filtered media) but World Bank scores MEDIUM-HIGH |

A supplier can score high on one signal and low on another — that divergence is the point. When all three agree (e.g. Yemen: high on all three), the risk is unambiguous.

### How to activate Live Mode

1. Register for [NewsAPI](https://newsapi.org/register) — instant, free
2. Add your key to `.env`:
```
NEWS_API_KEY=your-newsapi-key
```
3. Restart the app → toggle **🌐 Live Data Mode** in the sidebar

World Bank and GDACS require no registration or API keys.

### Fallback behaviour
If any live API call fails (rate limit, network error, no key), the platform **automatically falls back to the synthetic baseline** and logs the reason. The app never crashes due to a missing API key.

---

## 📡 REST API Usage

```python
import requests

response = requests.post(
    "https://your-domain.com/api/score",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "suppliers": [{
            "supplier_name": "Apex GmbH",
            "geo_risk_score": 0.25,
            "lead_time_days": 30,
            "lead_time_variance": 5.2,
            "inventory_buffer_days": 14,
            "supplier_reliability_score": 0.82,
            "financial_health_score": 0.75,
            "single_source_dependency": 0,
            "region_news_sentiment": -0.1,
            "natural_disaster_risk": 0.3,
            "past_disruptions_12mo": 1,
            "regulatory_risk_score": 0.35,
            "transport_mode_risk": 0.4
        }]
    }
)
print(response.json())
# → {"status": "success", "results": [{"supplier_name": "Apex GmbH",
#     "risk_score": 0.312, "risk_label": "Low", "risk_percentile": 22.5}]}
```

---

## 📁 Project Structure

```
supplyguard-ai/
├── app.py                    # Streamlit frontend
├── page_rebalancing.py       # Supply rebalancing optimizer UI
├── fastapi_wrapper.py        # REST API (FastAPI + Pydantic)
├── config.py                 # Centralised configuration
├── requirements.txt
├── Dockerfile
├── .env.example
├── backend/
│   ├── __init__.py
│   ├── data_generator.py     # Synthetic data + feature definitions
│   ├── risk_model.py         # ML model + SHAP explainability
│   ├── optimizer.py          # Linear programming rebalancing engine
│   ├── real_data_sources.py  # World Bank + NewsAPI + GDACS integrations
│   ├── llm_advisor.py        # Anthropic Claude LLM integration
│   └── auth.py               # Session auth + RBAC
├── docker/
│   └── supervisord.conf      # Multi-service process manager
└── aws/
    ├── deploy.sh             # ECR + ECS deploy script
    └── cloudformation.yml    # Full infrastructure as code
```

---

## 🏆 Key Technical Highlights

- Built end-to-end ML platform combining **Gradient Boosting + LLM** for supply chain risk
- Implemented **linear programming optimizer** (scipy HiGHS) for risk-first supply rebalancing
- Integrated **3 independent real-time signals** (World Bank, NewsAPI, GDACS) with 1-hour caching
- Implemented **RBAC, brute-force protection, rate limiting, and input validation** in Python
- Containerised with Docker, deployed on **AWS ECS Fargate** behind an **ALB with HTTPS**
- Secrets managed via **AWS Secrets Manager** (zero secrets in code)
- Exposed as both a Streamlit web app and **REST API** (FastAPI + Pydantic)
- SHAP-style feature explanations for **interpretable ML** decisions

---

*Built by Arunabha Kumar Chanda — M.Sc. Business Intelligence & Data Science, ISM Munich*
