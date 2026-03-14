# 🛡️ SupplyGuard AI — Supply Chain Disruption Risk Intelligence

> **A production-grade ML + LLM web platform that scores suppliers on disruption risk,
> explains the drivers, generates AI-powered mitigation strategies, and recommends
> optimal supply chain rebalancing using linear programming.**

Built with Python · Streamlit · FastAPI · Scikit-learn · FinBERT · Anthropic Claude · SciPy · Docker · AWS ECS Fargate

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
│                 │  - FinBERT NLP  │                     │
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
# Edit .env — add NEWS_API_KEY, ANTHROPIC_API_KEY, HF_FINBERT_REPO
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
  -e HF_FINBERT_REPO=arunabhachanda/supplychain-finbert \
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
| Task            | Binary classification (disruption risk) |
| Output          | Calibrated probability [0,1] → High / Medium / Low |
| Features        | 12 supply chain risk signals         |
| Training size   | 1,280 rows (80% of 1,600 synthetic)  |
| Test size       | 320 rows (20%)                       |
| ROC-AUC (test)  | ~0.91                                |
| F1 Score (test) | ~0.83                                |

### Feature Engineering
| Feature | Live Mode | Description |
|---|---|---|
| `geo_risk_score` | 🟢 World Bank PV.EST | Geopolitical instability [0–1] |
| `region_news_sentiment` | 🟢 NewsAPI + FinBERT | Headline polarity [−1 to +1] |
| `natural_disaster_risk` | 🟢 GDACS (UN) | Disaster event score [0–1] |
| `lead_time_days` | ⚪ Synthetic | Average supplier lead time |
| `lead_time_variance` | ⚪ Synthetic | Std deviation of lead times |
| `inventory_buffer_days` | ⚪ Synthetic | Safety stock in days |
| `supplier_reliability_score` | ⚪ Synthetic | On-time delivery rate [0–1] |
| `financial_health_score` | ⚪ Synthetic | Supplier solvency [0–1] |
| `single_source_dependency` | ⚪ Synthetic | 1 if sole source, 0 otherwise |
| `past_disruptions_12mo` | ⚪ Synthetic | Disruption count, past 12 months |
| `regulatory_risk_score` | ⚪ Synthetic | Trade regulation instability [0–1] |
| `transport_mode_risk` | ⚪ Synthetic | Risk by transport mode [0–1] |

🟢 = overwritten by live API in Live Mode · ⚪ = synthetic / user-uploaded CSV

---

## 🤗 FinBERT Sentiment Model

News sentiment is powered by a fine-tuned **SupplyChain FinBERT** model, replacing the original TextBlob lexicon approach.

| Property | Value |
|---|---|
| Base model | [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) |
| Fine-tuned on | ~40,600 samples (FinGPT financial sentiment + Twitter Financial News + 70 synthetic geopolitical headlines) |
| Task | 3-class sentiment: negative / neutral / positive |
| Strategy | Transfer learning — layers 0–9 frozen, layers 10–11 + head trainable |
| Test accuracy | 0.64 (2× better than 33% random baseline) |
| Negative class F1 | 0.79 (most critical class for risk detection) |
| HuggingFace Hub | [arunabhachanda/supplychain-finbert](https://huggingface.co/arunabhachanda/supplychain-finbert) |

### How sentiment feeds into risk scoring

```
NewsAPI headlines
      ↓
FinBERT → P(negative), P(neutral), P(positive) per article
      ↓
polarity = P(positive) - P(negative)  →  [-1.0, +1.0]
      ↓
region_news_sentiment updated in supplier DataFrame
      ↓
GradientBoosting re-scores all 50 suppliers
      ↓
Dashboard shows updated risk scores with LIVE badge
```

### To fine-tune the model yourself
```bash
python build_dataset.py        # downloads ~40K real headlines from HuggingFace
huggingface-cli login
python finetune_finbert.py --hf_repo YOUR_USERNAME/supplychain-finbert
```

---

## ⚖️ Supply Rebalancing Optimizer

The **Rebalancing** page uses linear programming (`scipy.optimize.linprog`, HiGHS solver) to redistribute purchase volume from high-risk suppliers to low-risk alternatives within the same category.

**Objective:** Minimise portfolio-weighted risk score
**Constraints:**
- All demand must be met
- Each alternative capped at **1.5× their current spend**
- Total cost must stay within user-defined tolerance (default 20%)
- Safer suppliers carry a cost premium of up to **+25%**

```
Minimise:   Σ risk_score[i] × allocation[i]
Subject to: Σ allocation[i]             = demand
            allocation[i]               ≤ 1.5 × current_spend[i]
            Σ cost_rate[i] × alloc[i]   ≤ budget × (1 + tolerance)
            allocation[i]               ≥ 0
```

---

## 🌐 Real Data Sources (Live Mode)

Toggle **Live Data Mode** in the sidebar to replace three synthetic signals with real external APIs:

| Signal | API | What it measures | Key Required |
|---|---|---|---|
| **Geopolitical Risk** | [World Bank PV.EST](https://data.worldbank.org) | Structural political stability | No |
| **News Sentiment** | [NewsAPI](https://newsapi.org) + FinBERT | Current media sentiment (fine-tuned BERT) | Yes — free at newsapi.org |
| **Disaster Risk** | [GDACS (UN)](https://gdacs.org) | Earthquakes, floods, cyclones, wildfires | No |

### Doctor analogy

| Signal | Analogy | Example |
|---|---|---|
| **World Bank** | Blood pressure history | USA LOW despite negative tariff headlines — institutions are stable |
| **GDACS** | Acute physical injury | Japan HIGH despite stable politics — frequent earthquakes and typhoons |
| **NewsAPI + FinBERT** | Mood today | Saudi Arabia LOW (state-filtered media) but World Bank MEDIUM-HIGH |

### How to activate Live Mode

Add to `.env`:
```
NEWS_API_KEY=your-newsapi-key
HF_FINBERT_REPO=arunabhachanda/supplychain-finbert
```
Then restart and toggle **🌐 Live Data Mode** in the sidebar. World Bank and GDACS need no keys.

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
├── build_dataset.py          # FinBERT training data builder (~40K real samples)
├── finetune_finbert.py       # FinBERT fine-tuning + HuggingFace Hub push
├── fix_push_readme.py        # HuggingFace model card updater
├── requirements.txt
├── Dockerfile
├── .env.example
├── data/
│   └── supplychain_finbert/  # Fine-tuned model weights (local cache)
├── backend/
│   ├── __init__.py
│   ├── data_generator.py     # Synthetic supplier data (50 display + 1,600 training rows)
│   ├── risk_model.py         # GradientBoosting + isotonic calibration
│   ├── sentiment_model.py    # FinBERT inference wrapper + TextBlob fallback
│   ├── optimizer.py          # LP rebalancing engine (scipy HiGHS)
│   ├── real_data_sources.py  # World Bank + NewsAPI/FinBERT + GDACS integrations
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

- **Full-stack ML web app** (Streamlit + FastAPI) built in direct response to real-world supply chain crises — Ukraine war, Red Sea disruptions, Iran sanctions
- **Transfer Learning + Fine-tuning**: adapted ProsusAI/FinBERT on ~40K real financial headlines for supply-chain NLP sentiment; model published to [HuggingFace Hub](https://huggingface.co/arunabhachanda/supplychain-finbert)
- **Binary classification** (GradientBoosting + isotonic calibration, ROC-AUC ~0.91) with 3-signal live data enrichment replacing 3 of 12 input features in real time
- **Linear programming optimizer** (scipy HiGHS) for risk-first supply rebalancing — up to 81% risk reduction within cost tolerance
- **3 independent real-time signals** (World Bank PV.EST, NewsAPI + FinBERT, UN GDACS) with 1-hour in-process caching and graceful fallback
- **RBAC, brute-force protection, rate limiting, input validation** — production-grade security layer
- Containerised with Docker, deployed on **AWS ECS Fargate** behind an **ALB with HTTPS/TLS 1.3**
- Secrets managed via **AWS Secrets Manager** — zero secrets in code, image, or CloudFormation
- **Anthropic Claude LLM** for structured JSON mitigation recommendations and executive portfolio briefs

---

*Built by Arunabha Kumar Chanda — M.Sc. Business Intelligence & Data Science, ISM Munich*
*GitHub: [arunabhachanda](https://github.com/arunabhachanda) · HuggingFace: [arunabhachanda](https://huggingface.co/arunabhachanda)*
