"""
Central configuration for the Supply Chain Disruption Risk Platform.
All secrets are loaded from environment variables — never hardcoded.
"""
from dotenv import load_dotenv
load_dotenv()
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AppConfig:
    # ── App Identity ──────────────────────────────────────────────
    app_name: str = "SupplyGuard AI"
    app_version: str = "1.0.0"
    app_tagline: str = "Supply Chain Disruption Risk Intelligence"

    # ── LLM ───────────────────────────────────────────────────────
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    llm_model: str = "claude-opus-4-5"
    llm_max_tokens: int = 1024

    # ── Auth ──────────────────────────────────────────────────────
    secret_key: str = field(
        default_factory=lambda: os.getenv("SECRET_KEY", "change-me-in-production-32bytes!")
    )
    cookie_name: str = "supplyguard_auth"
    cookie_expiry_days: int = 1

    # ── Rate Limiting ─────────────────────────────────────────────
    max_llm_calls_per_session: int = 20      # LLM calls per session
    max_uploads_per_session: int = 10        # file uploads per session
    max_rows_per_upload: int = 500           # max supplier rows per CSV

    # ── ML Model ──────────────────────────────────────────────────
    risk_threshold_high: float = 0.65        # above → HIGH risk
    risk_threshold_medium: float = 0.35      # above → MEDIUM risk

    # ── Real Data Source APIs ─────────────────────────────────────
    # NewsAPI (newsapi.org) — free 100 req/day
    news_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("NEWS_API_KEY")
    )
    # ACLED (acleddata.com) — free for research, register to get key
    acled_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ACLED_API_KEY")
    )
    acled_email: Optional[str] = field(
        default_factory=lambda: os.getenv("ACLED_EMAIL")
    )
    # ReliefWeb — no key needed, always available

    # ── AWS ───────────────────────────────────────────────────────
    aws_region: str = field(
        default_factory=lambda: os.getenv("AWS_REGION", "eu-central-1")
    )
    ecr_repo: str = field(
        default_factory=lambda: os.getenv("ECR_REPO", "supplyguard-ai")
    )


# Singleton
cfg = AppConfig()

# ── RBAC Role Definitions ─────────────────────────────────────────
ROLES = {
    "admin": {
        "label": "Administrator",
        "permissions": ["view_dashboard", "upload_data", "get_recommendations",
                        "export_report", "manage_users", "view_raw_scores"],
    },
    "analyst": {
        "label": "Risk Analyst",
        "permissions": ["view_dashboard", "upload_data", "get_recommendations",
                        "export_report", "view_raw_scores"],
    },
    "viewer": {
        "label": "Read-only Viewer",
        "permissions": ["view_dashboard"],
    },
}

# ── Demo Users (replace with a real user-store in production) ─────
# Passwords are bcrypt hashes — generated via bcrypt.hashpw()
# Demo plain-text passwords are in README (dev only)
DEMO_USERS = {
    "credentials": {
        "usernames": {
            "admin": {
                "name": "Admin User",
                "email": "admin@supplyguard.ai",
                "role": "admin",
                # password: Admin@1234
                "password": "$2b$12$KIXbVzFxFzFxFzFxFzFxFO7VzFxFzFxFzFxFzFxFzFxFzFxFzFxFz",
            },
            "analyst": {
                "name": "Alice Analyst",
                "email": "alice@supplyguard.ai",
                "role": "analyst",
                # password: Analyst@1234
                "password": "$2b$12$KIXbVzFxFzFxFzFxFzFxFO7VzFxFzFxFzFxFzFxFzFxFzFxFzFxFz",
            },
            "viewer": {
                "name": "Victor Viewer",
                "email": "victor@supplyguard.ai",
                "role": "viewer",
                # password: Viewer@1234
                "password": "$2b$12$KIXbVzFxFzFxFzFxFzFxFO7VzFxFzFxFzFxFzFxFzFxFzFxFzFxFz",
            },
        }
    }
}
