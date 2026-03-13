"""
LLM-powered supply chain risk advisor.
Uses Anthropic's API to generate structured mitigation recommendations.
Includes session-level rate limiting.
"""
from __future__ import annotations

import json
import time
from typing import Optional
import streamlit as st

from config import cfg

# ─────────────────────────────────────────────────────────────────
# Rate limiter (session-scoped)
# ─────────────────────────────────────────────────────────────────
RATE_LIMIT_KEY = "_llm_call_count"
RATE_LIMIT_RESET_KEY = "_llm_reset_ts"
SESSION_WINDOW_SECONDS = 3600   # 1 hour window


def _check_rate_limit() -> tuple[bool, str]:
    """Returns (allowed, message)."""
    now = time.time()
    reset_ts = st.session_state.get(RATE_LIMIT_RESET_KEY, 0)

    if now - reset_ts > SESSION_WINDOW_SECONDS:
        st.session_state[RATE_LIMIT_KEY] = 0
        st.session_state[RATE_LIMIT_RESET_KEY] = now

    count = st.session_state.get(RATE_LIMIT_KEY, 0)
    if count >= cfg.max_llm_calls_per_session:
        remaining_seconds = int(SESSION_WINDOW_SECONDS - (now - reset_ts))
        return False, (
            f"Rate limit reached ({cfg.max_llm_calls_per_session} AI calls/hour). "
            f"Resets in {remaining_seconds // 60}m {remaining_seconds % 60}s."
        )

    st.session_state[RATE_LIMIT_KEY] = count + 1
    return True, ""


# ─────────────────────────────────────────────────────────────────
# LLM Call
# ─────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are SupplyGuard AI, an expert supply chain risk analyst. 
You provide concise, actionable, evidence-based risk mitigation strategies.
Respond ONLY with valid JSON matching the exact schema provided. 
No markdown, no preamble, no explanation outside the JSON."""

_RISK_PROMPT_TEMPLATE = """
Analyze this supplier's risk profile and return mitigation strategies.

Supplier: {supplier_name}
Region: {region}
Category: {category}
Risk Score: {risk_score} / 1.0 ({risk_label} Risk)
Transport Mode: {transport_mode}

Key Risk Drivers:
{risk_drivers}

Return ONLY this JSON schema:
{{
  "executive_summary": "<2-sentence plain-language risk summary>",
  "immediate_actions": [
    {{"action": "<action>", "priority": "Critical|High|Medium", "timeline": "<e.g. 1-2 weeks>", "cost_impact": "Low|Medium|High"}}
  ],
  "strategic_recommendations": [
    {{"recommendation": "<recommendation>", "benefit": "<expected benefit>", "effort": "Low|Medium|High"}}
  ],
  "kpi_to_monitor": ["<KPI 1>", "<KPI 2>", "<KPI 3>"],
  "risk_trend": "Improving|Stable|Deteriorating",
  "confidence": "High|Medium|Low"
}}

Provide 3 immediate_actions and 3 strategic_recommendations. Keep each item concise (under 30 words).
"""


def get_mitigation_recommendations(
    supplier_name: str,
    region: str,
    category: str,
    risk_score: float,
    risk_label: str,
    transport_mode: str,
    top_risk_drivers: list[dict],
) -> Optional[dict]:
    """
    Call the Anthropic LLM API and return structured mitigation strategies.

    Returns None if rate-limited, API key missing, or API error.
    Posts st.error / st.warning directly for UI feedback.
    """
    # Rate limit check
    allowed, msg = _check_rate_limit()
    if not allowed:
        st.warning(f"⏱️ {msg}")
        return None

    # API key check
    if not cfg.anthropic_api_key:
        st.error("🔑 ANTHROPIC_API_KEY not set. Add it to your .env file.")
        return _fallback_recommendations(supplier_name, risk_label)

    # Format risk drivers for prompt
    drivers_text = "\n".join(
        f"  - {d['label']}: importance={d['importance']:.3f} ({d['direction']})"
        for d in top_risk_drivers[:5]
    )

    prompt = _RISK_PROMPT_TEMPLATE.format(
        supplier_name=supplier_name,
        region=region,
        category=category,
        risk_score=round(risk_score, 3),
        risk_label=risk_label,
        transport_mode=transport_mode,
        risk_drivers=drivers_text,
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        message = client.messages.create(
            model=cfg.llm_model,
            max_tokens=cfg.llm_max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        # Strip any accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)

    except json.JSONDecodeError as e:
        st.error(f"LLM returned malformed JSON: {e}")
        return _fallback_recommendations(supplier_name, risk_label)
    except Exception as e:
        st.error(f"LLM API error: {e}")
        return _fallback_recommendations(supplier_name, risk_label)


def _fallback_recommendations(supplier_name: str, risk_label: str) -> dict:
    """Static fallback when API is unavailable."""
    return {
        "executive_summary": (
            f"{supplier_name} presents {risk_label.lower()} disruption risk based on "
            "current operational and geopolitical indicators. Immediate review recommended."
        ),
        "immediate_actions": [
            {"action": "Audit current inventory buffer adequacy", "priority": "High",
             "timeline": "1 week", "cost_impact": "Low"},
            {"action": "Contact supplier for business continuity plan review", "priority": "High",
             "timeline": "2 weeks", "cost_impact": "Low"},
            {"action": "Identify and pre-qualify alternative suppliers", "priority": "Critical",
             "timeline": "3-4 weeks", "cost_impact": "Medium"},
        ],
        "strategic_recommendations": [
            {"recommendation": "Implement dual-sourcing strategy for critical components",
             "benefit": "Reduces single-source dependency risk by 50%+", "effort": "High"},
            {"recommendation": "Establish safety stock policy linked to supplier risk score",
             "benefit": "Buffers against lead time disruptions", "effort": "Medium"},
            {"recommendation": "Deploy real-time supplier financial health monitoring",
             "benefit": "Early warning of supplier insolvency", "effort": "Medium"},
        ],
        "kpi_to_monitor": [
            "On-Time Delivery Rate", "Lead Time Variance", "Inventory Cover Days"
        ],
        "risk_trend": "Stable",
        "confidence": "Medium",
    }


def get_portfolio_summary(
    high_risk_count: int,
    medium_risk_count: int,
    low_risk_count: int,
    total_spend_at_risk: float,
    top_risky_regions: list[str],
) -> Optional[dict]:
    """Generate an executive portfolio-level risk narrative."""
    allowed, msg = _check_rate_limit()
    if not allowed:
        st.warning(f"⏱️ {msg}")
        return None

    if not cfg.anthropic_api_key:
        return _fallback_portfolio(high_risk_count, medium_risk_count, total_spend_at_risk)

    prompt = f"""
Supply chain portfolio risk summary for an executive briefing.

Portfolio Stats:
- High Risk Suppliers: {high_risk_count}
- Medium Risk Suppliers: {medium_risk_count}
- Low Risk Suppliers: {low_risk_count}
- Estimated Annual Spend at High Risk: ${total_spend_at_risk:,.0f}
- Top At-Risk Regions: {', '.join(top_risky_regions)}

Return ONLY this JSON:
{{
  "headline": "<One powerful headline sentence for executive>",
  "portfolio_health": "Critical|At Risk|Moderate|Healthy",
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"],
  "board_recommendation": "<One paragraph board-level recommendation>",
  "30_day_priorities": ["<priority 1>", "<priority 2>", "<priority 3>"]
}}
"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        message = client.messages.create(
            model=cfg.llm_model,
            max_tokens=800,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        st.error(f"LLM API error (portfolio): {e}")
        return _fallback_portfolio(high_risk_count, medium_risk_count, total_spend_at_risk)


def _fallback_portfolio(high: int, medium: int, spend: float) -> dict:
    return {
        "headline": f"Supply chain portfolio has {high} high-risk suppliers requiring immediate attention.",
        "portfolio_health": "At Risk" if high > 5 else "Moderate",
        "key_findings": [
            f"{high} suppliers classified as high risk — immediate mitigation required",
            f"Estimated ${spend:,.0f} annual spend exposed to high disruption probability",
            f"{medium} medium-risk suppliers should be monitored quarterly",
        ],
        "board_recommendation": (
            "The board should approve a strategic supply chain resilience program, "
            "including dual-sourcing for critical components, increased safety stock policies, "
            "and real-time supplier monitoring infrastructure."
        ),
        "30_day_priorities": [
            "Audit top 5 highest-risk suppliers for business continuity plans",
            "Identify and pre-qualify alternative suppliers for single-source dependencies",
            "Implement weekly risk score monitoring cadence",
        ],
    }
