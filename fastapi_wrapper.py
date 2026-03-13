"""
FastAPI REST API wrapper for SupplyGuard AI.
Exposes the ML risk scoring model as a secured API endpoint.

Security features:
  - OAuth2 Bearer token authentication
  - RBAC enforcement
  - Rate limiting via slowapi
  - Pydantic input validation
  - HTTPS via AWS ALB (no SSL termination here)

Run: uvicorn fastapi_wrapper:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import os
import time
import secrets
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import pandas as pd

# Import our ML module
import sys
sys.path.insert(0, os.path.dirname(__file__))
from backend.risk_model import predict_risk, FEATURE_COLUMNS
from config import cfg

# ─────────────────────────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="SupplyGuard AI — Risk Scoring API",
    description=(
        "Supply Chain Disruption Risk Intelligence API.\n\n"
        "Scores suppliers on their disruption probability using a trained "
        "Gradient Boosting model with 12 supply chain features."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Trusted hosts (set to your ALB DNS or domain in production) ──
# app.add_middleware(TrustedHostMiddleware, allowed_hosts=["yourdomain.com", "*.yourdomain.com"])


# ─────────────────────────────────────────────────────────────────
# Auth (simple API-key Bearer for demo; replace with OAuth2/Cognito)
# ─────────────────────────────────────────────────────────────────
_API_KEYS = {
    # key → {role, name}
    os.getenv("API_KEY_ADMIN", "supplyguard-admin-key-change-me"): {
        "role": "admin", "name": "Admin Client"
    },
    os.getenv("API_KEY_ANALYST", "supplyguard-analyst-key-change-me"): {
        "role": "analyst", "name": "Analyst Client"
    },
    os.getenv("API_KEY_VIEWER", "supplyguard-viewer-key-change-me"): {
        "role": "viewer", "name": "Viewer Client"
    },
}

security = HTTPBearer()

ROLE_PERMISSIONS = {
    "admin":   ["score", "batch_score", "health", "docs"],
    "analyst": ["score", "batch_score", "health"],
    "viewer":  ["health"],
}


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate Bearer token and return user context."""
    token = credentials.credentials
    user = _API_KEYS.get(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_role(required_permission: str):
    """Dependency factory for RBAC."""
    def checker(user: dict = Depends(verify_token)):
        allowed = ROLE_PERMISSIONS.get(user["role"], [])
        if required_permission not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user['role']}' lacks '{required_permission}' permission.",
            )
        return user
    return checker


# ─────────────────────────────────────────────────────────────────
# Pydantic schemas (input validation)
# ─────────────────────────────────────────────────────────────────
class SupplierInput(BaseModel):
    supplier_name: str = Field(..., min_length=1, max_length=200, example="Apex GmbH")
    region: Optional[str] = Field(None, max_length=100, example="Western Europe")
    category: Optional[str] = Field(None, max_length=100, example="Electronics")
    transport_mode: Optional[str] = Field(None, max_length=50, example="Sea Freight")

    # ML features
    geo_risk_score: float = Field(..., ge=0.0, le=1.0, example=0.25)
    lead_time_days: float = Field(..., ge=1, le=365, example=30.0)
    lead_time_variance: float = Field(..., ge=0, le=200, example=5.2)
    inventory_buffer_days: float = Field(..., ge=0, le=365, example=14.0)
    supplier_reliability_score: float = Field(..., ge=0.0, le=1.0, example=0.82)
    financial_health_score: float = Field(..., ge=0.0, le=1.0, example=0.75)
    single_source_dependency: int = Field(..., ge=0, le=1, example=0)
    region_news_sentiment: float = Field(..., ge=-1.0, le=1.0, example=-0.1)
    natural_disaster_risk: float = Field(..., ge=0.0, le=1.0, example=0.3)
    past_disruptions_12mo: int = Field(..., ge=0, le=50, example=1)
    regulatory_risk_score: float = Field(..., ge=0.0, le=1.0, example=0.35)
    transport_mode_risk: float = Field(..., ge=0.0, le=1.0, example=0.4)

    @validator("supplier_name")
    def sanitize_name(cls, v):
        # Strip HTML/script injection
        import re
        v = re.sub(r"[<>\"'`]", "", v).strip()
        if not v:
            raise ValueError("supplier_name cannot be empty after sanitization")
        return v

    class Config:
        schema_extra = {
            "example": {
                "supplier_name": "Apex Components GmbH",
                "region": "Western Europe",
                "category": "Automotive Parts",
                "transport_mode": "Sea Freight",
                "geo_risk_score": 0.25,
                "lead_time_days": 30.0,
                "lead_time_variance": 5.2,
                "inventory_buffer_days": 14.0,
                "supplier_reliability_score": 0.82,
                "financial_health_score": 0.75,
                "single_source_dependency": 0,
                "region_news_sentiment": -0.1,
                "natural_disaster_risk": 0.3,
                "past_disruptions_12mo": 1,
                "regulatory_risk_score": 0.35,
                "transport_mode_risk": 0.4,
            }
        }


class ScoreRequest(BaseModel):
    suppliers: list[SupplierInput] = Field(..., min_items=1, max_items=500)


class SupplierResult(BaseModel):
    supplier_name: str
    risk_score: float
    risk_label: str
    risk_percentile: float


class ScoreResponse(BaseModel):
    status: str
    scored_at: str
    model_version: str
    count: int
    results: list[SupplierResult]


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["System"])
@limiter.limit("30/minute")
async def health(request: Request):
    """Public health check endpoint (no auth required)."""
    return {"status": "healthy", "service": "SupplyGuard AI", "version": cfg.app_version}


@app.post("/api/score", response_model=ScoreResponse, tags=["Risk Scoring"])
@limiter.limit("60/minute")
async def score_suppliers(
    request: Request,
    payload: ScoreRequest,
    user: dict = Depends(require_role("score")),
):
    """
    Score one or more suppliers for disruption risk.

    **Required role:** analyst or admin

    Returns risk_score (0–1), risk_label (Low/Medium/High), 
    and risk_percentile relative to the batch.
    """
    # Convert to DataFrame
    records = [s.dict() for s in payload.suppliers]
    df = pd.DataFrame(records)

    # Score
    try:
        scored = predict_risk(df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    results = [
        SupplierResult(
            supplier_name=row["supplier_name"],
            risk_score=float(row["risk_score"]),
            risk_label=str(row["risk_label"]),
            risk_percentile=float(row["risk_percentile"]),
        )
        for _, row in scored.iterrows()
    ]

    from datetime import datetime, timezone
    return ScoreResponse(
        status="success",
        scored_at=datetime.now(timezone.utc).isoformat(),
        model_version=cfg.app_version,
        count=len(results),
        results=results,
    )


@app.get("/api/model-info", tags=["System"])
@limiter.limit("10/minute")
async def model_info(
    request: Request,
    user: dict = Depends(require_role("score")),
):
    """Return model performance metrics and feature list."""
    from backend.risk_model import get_model_metrics
    metrics = get_model_metrics()
    return {
        "model": "GradientBoostingClassifier + Isotonic Calibration",
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
        "thresholds": {
            "low": "< 0.35",
            "medium": "0.35 – 0.65",
            "high": "> 0.65",
        },
    }


# ─────────────────────────────────────────────────────────────────
# Dev runner
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_wrapper:app", host="0.0.0.0", port=8001, reload=True)
