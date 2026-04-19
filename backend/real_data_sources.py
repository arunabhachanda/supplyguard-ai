"""
Real external data source integrations for SupplyGuard AI.

Three genuinely independent signals — each measures a different dimension of risk:

  1. NewsAPI (newsapi.org)
       WHAT: Current news sentiment — "what is happening RIGHT NOW"
       HOW:  Fetches today's headlines from Reuters/BBC/Bloomberg, runs FinBERT NLP
       WHY:  Captures reactive, short-term market sentiment
       FREE: 100 req/day | Needs: NEWS_API_KEY in .env
       EXAMPLE: USA scores negative right now due to tariff/trade dispute headlines

  2. World Bank Political Stability Index (data.worldbank.org)
       WHAT: Structural political risk — "how stable is the SYSTEM ITSELF"
       HOW:  Expert-surveyed country stability scores (-2.5 to +2.5), updated annually
       WHY:  Captures chronic, long-term institutional risk that news doesn't show
       FREE: No key needed, no registration, no rate limits
       EXAMPLE: USA scores LOW despite negative news (strong institutions, no coup risk)
                Saudi Arabia scores MEDIUM-HIGH despite neutral news (authoritarian risk)
                China scores HIGH despite positive state media (structural control risk)

  3. GDACS (gdacs.org — UN Global Disaster Alert & Coordination System)
       WHAT: Physical disaster risk — "what PHYSICAL events have occurred"
       HOW:  Real-time disaster alerts (earthquakes, floods, cyclones, wildfires, droughts)
       WHY:  Completely independent of politics — measures environmental disruption
       FREE: No key needed, UN-backed
       EXAMPLE: Japan scores LOW on World Bank (stable democracy) but HIGH on GDACS
                (frequent earthquakes and typhoons) — signals can and should diverge

Key design principle: the three signals are intentionally measuring DIFFERENT things.
A supplier can score low on one and high on another — that's the point.
"""
from __future__ import annotations

import time
import hashlib
import logging
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timedelta

import requests
import numpy as np
from backend.sentiment_model import get_sentiment_scores_batch, get_model_info

from config import cfg

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# In-process TTL cache
# ─────────────────────────────────────────────────────────────────
_cache: Dict[str, Tuple[float, float]] = {}
CACHE_TTL_SECONDS = 3600   # 1 hour


def _cache_get(key: str) -> Optional[float]:
    entry = _cache.get(key)
    if entry and time.time() < entry[1]:
        return entry[0]
    return None


def _cache_set(key: str, value: float) -> None:
    _cache[key] = (value, time.time() + CACHE_TTL_SECONDS)


def _cache_key(*parts: str) -> str:
    return hashlib.md5("|".join(parts).encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────
# Synthetic fallback baselines
# ─────────────────────────────────────────────────────────────────
_SYNTHETIC_GEO_RISK: Dict[str, float] = {
    "East Asia": 0.45, "South Asia": 0.55, "Southeast Asia": 0.40,
    "Eastern Europe": 0.65, "Western Europe": 0.15, "Middle East": 0.75,
    "North Africa": 0.70, "Sub-Saharan Africa": 0.60, "Latin America": 0.50,
    "North America": 0.10,
}
_SYNTHETIC_DISASTER_RISK: Dict[str, float] = {
    "East Asia": 0.70, "South Asia": 0.65, "Southeast Asia": 0.75,
    "Eastern Europe": 0.25, "Western Europe": 0.20, "Middle East": 0.30,
    "North Africa": 0.35, "Sub-Saharan Africa": 0.50, "Latin America": 0.55,
    "North America": 0.30,
}

# ─────────────────────────────────────────────────────────────────
# Region → query / country mappings
# ─────────────────────────────────────────────────────────────────

# NewsAPI: headline queries per region
REGION_TO_HEADLINE_QUERY: Dict[str, str] = {
    "East Asia":          "China trade",
    "South Asia":         "India trade",
    "Southeast Asia":     "Vietnam manufacturing",
    "Eastern Europe":     "Ukraine war",
    "Western Europe":     "Germany economy",
    "Middle East":        "Middle East conflict",
    "North Africa":       "Egypt economy",
    "Sub-Saharan Africa": "Nigeria economy",
    "Latin America":      "Mexico trade",
    "North America":      "US economy",
}

# World Bank: ISO2 country codes per region
# Uses PV.EST — Political Stability and Absence of Violence/Terrorism
# Score range: -2.5 (most unstable) → +2.5 (most stable)
REGION_TO_WB_COUNTRIES: Dict[str, List[str]] = {
    "East Asia":          ["CN", "JP", "KR", "TW"],
    "South Asia":         ["IN", "PK", "BD", "LK"],
    "Southeast Asia":     ["VN", "TH", "ID", "MY", "PH"],
    "Eastern Europe":     ["UA", "PL", "RO", "HU"],
    "Western Europe":     ["DE", "FR", "IT", "ES", "GB"],
    "Middle East":        ["IR", "IQ", "SA", "YE", "IL", "SY"],
    "North Africa":       ["EG", "LY", "MA", "TN", "DZ"],
    "Sub-Saharan Africa": ["NG", "ET", "KE", "CD", "SD"],
    "Latin America":      ["MX", "BR", "CO", "AR", "PE"],
    "North America":      ["US", "CA"],
}

# GDACS: country name keywords for matching disaster events
REGION_TO_GDACS_KEYWORDS: Dict[str, List[str]] = {
    "East Asia":          ["China", "Japan", "Korea", "Taiwan"],
    "South Asia":         ["India", "Pakistan", "Bangladesh", "Nepal", "Sri Lanka"],
    "Southeast Asia":     ["Vietnam", "Thailand", "Indonesia", "Philippines", "Myanmar", "Malaysia"],
    "Eastern Europe":     ["Ukraine", "Poland", "Romania", "Hungary", "Moldova"],
    "Western Europe":     ["Germany", "France", "Italy", "Spain", "Greece", "United Kingdom"],
    "Middle East":        ["Iran", "Iraq", "Yemen", "Saudi", "Syria", "Turkey", "Israel"],
    "North Africa":       ["Egypt", "Libya", "Morocco", "Tunisia", "Algeria"],
    "Sub-Saharan Africa": ["Nigeria", "Ethiopia", "Congo", "Kenya", "Somalia", "Sudan"],
    "Latin America":      ["Mexico", "Brazil", "Colombia", "Argentina", "Peru", "Chile"],
    "North America":      ["United States", "USA", "Canada"],
}


# ─────────────────────────────────────────────────────────────────
# 1. NewsAPI — Current News Sentiment
# ─────────────────────────────────────────────────────────────────
def get_news_sentiment(region: str) -> Tuple[float, str]:
    """
    Fetches today's headlines from trusted sources (Reuters, BBC, Bloomberg)
    and computes sentiment polarity using fine-tuned SupplyChain FinBERT.

    Measures: SHORT-TERM reactive sentiment — what the market feels today.
    Range: -1.0 (very negative) to +1.0 (very positive)

    Requires: NEWS_API_KEY in .env (free at newsapi.org)
    """
    cache_k = _cache_key("news", region)
    cached = _cache_get(cache_k)
    if cached is not None:
        return cached, "live (cached)"

    api_key = cfg.news_api_key
    if not api_key:
        fallback = float(np.clip(
            -_SYNTHETIC_GEO_RISK.get(region, 0.5) * 0.6 + np.random.normal(0, 0.05),
            -1.0, 1.0,
        ))
        return fallback, "synthetic (no API key)"

    try:
        resp = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={
                "sources": "bbc-news,reuters,bloomberg,al-jazeera-english,financial-times",
                "pageSize": 30,
                "apiKey":   api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])

        if not articles:
            fallback = float(np.clip(-_SYNTHETIC_GEO_RISK.get(region, 0.5) * 0.6, -1, 1))
            return fallback, "synthetic (no articles)"

        # Filter articles relevant to this region
        query_terms = REGION_TO_HEADLINE_QUERY.get(region, region).lower().split()
        region_articles = [
            a for a in articles
            if any(
                term in (a.get("title") or "").lower() or
                term in (a.get("description") or "").lower()
                for term in query_terms
            )
        ]

        # Fall back to all articles if none match (global sentiment)
        target_articles = region_articles if region_articles else articles

        # Build per-article texts for batch inference
        # FinBERT scores each headline independently then we average —
        # more accurate than concatenating all text into one string
        article_texts = [
            ((a.get("title") or "") + " " + (a.get("description") or "")).strip()
            for a in target_articles
            if (a.get("title") or a.get("description"))
        ]

        if not article_texts:
            return 0.0, "synthetic (no articles)"

        # Batch inference — FinBERT fine-tuned on supply-chain headlines
        # Returns polarity scores in [-1.0, +1.0] per article
        scores    = get_sentiment_scores_batch(article_texts)
        polarity  = float(np.mean(scores))
        sentiment = round(float(np.clip(polarity, -1.0, 1.0)), 4)

        _cache_set(cache_k, sentiment)
        logger.info(
            f"NewsAPI sentiment [{region}]: {sentiment} "
            f"({len(region_articles)} region articles, {len(articles)} total)"
        )
        return sentiment, "live"

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        logger.debug(f"NewsAPI HTTP {status} for {region}: {e}")
        fallback = float(np.clip(-_SYNTHETIC_GEO_RISK.get(region, 0.5) * 0.6, -1, 1))
        return fallback, f"synthetic (HTTP {status})"
    except Exception as e:
        logger.debug(f"NewsAPI failed for {region}: {e}")
        fallback = float(np.clip(-_SYNTHETIC_GEO_RISK.get(region, 0.5) * 0.6, -1, 1))
        return fallback, "synthetic (error)"


# ─────────────────────────────────────────────────────────────────
# 2. World Bank Political Stability Index — Structural Geo Risk
# ─────────────────────────────────────────────────────────────────
def get_geopolitical_risk(region: str) -> Tuple[float, str]:
    """
    Fetches the World Bank Political Stability and Absence of Violence
    indicator (GOV_WGI_PV.EST) for representative countries in the region.

    Measures: STRUCTURAL, LONG-TERM political risk — how stable the system
    itself is, independent of today's news cycle.

    Key insight: This deliberately diverges from NewsAPI:
      - USA: NewsAPI negative (tariffs/politics), World Bank LOW (strong institutions)
      - Saudi Arabia: NewsAPI neutral (filtered media), World Bank MEDIUM-HIGH (authoritarian risk)
      - China: NewsAPI positive (state media), World Bank HIGH (structural control risk)
      - France: NewsAPI negative (strikes/protests), World Bank LOW (stable democracy)

    Score mapping: WB range [-2.5, +2.5] → risk [0.95, 0.05]
      risk = (-wb_score + 2.5) / 5.0

    No API key required. Updated annually. No rate limits.
    """
    cache_k = _cache_key("worldbank", region)
    cached = _cache_get(cache_k)
    if cached is not None:
        return cached, "live (cached)"

    country_codes = REGION_TO_WB_COUNTRIES.get(region, [])
    if not country_codes:
        return _SYNTHETIC_GEO_RISK.get(region, 0.5), "synthetic (no countries)"

    scores = []
    for iso2 in country_codes:
        try:
            url = (
                f"https://api.worldbank.org/v2/country/{iso2}"
                f"/indicator/GOV_WGI_PV.EST?format=json&mrv=1&per_page=1"
            )
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            # World Bank returns [metadata, [datapoints]]
            if (
                isinstance(data, list)
                and len(data) >= 2
                and isinstance(data[1], list)
                and data[1]
            ):
                value = data[1][0].get("value")
                if value is not None:
                    scores.append(float(value))
                    logger.info(f"World Bank [{iso2}]: PV.EST = {value:.3f}")

        except Exception as e:
            logger.debug(f"World Bank failed for {iso2}: {e}")
            continue

    if not scores:
        fallback = _SYNTHETIC_GEO_RISK.get(region, 0.5)
        return fallback, "synthetic (no WB data)"

    # Average stability score for the region
    avg_wb_score = float(np.mean(scores))

    # Map [-2.5, +2.5] → [0.95, 0.05] risk
    # -2.5 (most unstable) → 0.95 risk
    # 0.0  (neutral)       → 0.50 risk
    # +2.5 (most stable)   → 0.05 risk
    risk = float(np.clip((-avg_wb_score + 2.5) / 5.0, 0.05, 0.95))
    risk = round(risk, 4)

    _cache_set(cache_k, risk)
    logger.info(
        f"World Bank geo risk [{region}]: {risk} "
        f"(avg PV.EST={avg_wb_score:.3f}, {len(scores)}/{len(country_codes)} countries)"
    )
    return risk, "live"


# ─────────────────────────────────────────────────────────────────
# 3. GDACS — Physical Disaster Risk
# ─────────────────────────────────────────────────────────────────
def get_disaster_risk(region: str) -> Tuple[float, str]:
    """
    Fetches disaster alerts from GDACS (UN Global Disaster Alert &
    Coordination System) for the last 365 days.

    Measures: PHYSICAL environmental risk — completely independent of
    politics or economics. A stable democracy can have high disaster risk
    (Japan: earthquakes/typhoons). A war zone can have low disaster risk
    (Libya: dry climate, few natural disasters).

    Covers: Earthquakes (EQ), Tropical Cyclones (TC), Floods (FL),
            Volcanoes (VO), Droughts (DR), Wildfires (WF)

    Scoring: Red alert = 3pts, Orange = 2pts, Green = 0.5pts
    Normalised against max expected score of 30pts/year.

    No API key required. UN-backed real-time data.
    """
    cache_k = _cache_key("gdacs", region)
    cached = _cache_get(cache_k)
    if cached is not None:
        return cached, "live (cached)"

    keywords = REGION_TO_GDACS_KEYWORDS.get(region, [])
    if not keywords:
        return _SYNTHETIC_DISASTER_RISK.get(region, 0.4), "synthetic"

    since = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH",
            params={
                "eventlist":  "EQ,TC,FL,VO,DR,WF",
                "alertlevel": "Green,Orange,Red",
                "datestart":  since,
                "dateend":    today,
                "limit":      "200",
            },
            timeout=15,
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])

        if not features:
            fallback = _SYNTHETIC_DISASTER_RISK.get(region, 0.4)
            return fallback, "synthetic (no events)"

        score          = 0.0
        matched_events = 0
        alert_weights  = {"Red": 3.0, "Orange": 2.0, "Green": 0.5}

        for feature in features:
            props   = feature.get("properties", {})
            country = str(props.get("country", "") or props.get("countryname", "") or "")
            alert   = str(props.get("alertlevel", "Green"))

            for keyword in keywords:
                if keyword.lower() in country.lower():
                    score += alert_weights.get(alert, 0.5)
                    matched_events += 1
                    break

        if matched_events == 0:
            # No disasters — genuine low risk, but don't go to zero
            synthetic_base = _SYNTHETIC_DISASTER_RISK.get(region, 0.4)
            risk = round(float(np.clip(synthetic_base * 0.5, 0.05, 0.40)), 4)
            _cache_set(cache_k, risk)
            logger.info(f"GDACS [{region}]: {risk} (0 events — genuine low disaster risk)")
            return risk, "live"

        # Normalise: max ~30pts/year for a high-risk region
        risk = round(float(np.clip(score / 30.0, 0.05, 0.95)), 4)

        # Blend 70% live / 30% synthetic for stability
        synthetic_base = _SYNTHETIC_DISASTER_RISK.get(region, 0.4)
        blended = round(0.70 * risk + 0.30 * synthetic_base, 4)

        _cache_set(cache_k, blended)
        logger.info(
            f"GDACS [{region}]: {blended} "
            f"(score={score:.1f}, {matched_events} events)"
        )
        return blended, "live"

    except requests.exceptions.Timeout:
        logger.debug(f"GDACS timeout for {region}")
        return _SYNTHETIC_DISASTER_RISK.get(region, 0.4), "synthetic (timeout)"
    except Exception as e:
        logger.debug(f"GDACS failed for {region}: {e}")
        return _SYNTHETIC_DISASTER_RISK.get(region, 0.4), "synthetic (error)"


# ─────────────────────────────────────────────────────────────────
# 4. Connectivity probes (for sidebar badges)
# ─────────────────────────────────────────────────────────────────
def _probe_worldbank() -> bool:
    """Quick check — fetch USA stability score."""
    try:
        resp = requests.get(
            "https://api.worldbank.org/v2/country/US/indicator/PV.EST"
            "?format=json&mrv=1&per_page=1",
            timeout=8,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _probe_gdacs() -> bool:
    """Quick connectivity check for GDACS."""
    try:
        resp = requests.get(
            "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH",
            params={"eventlist": "EQ", "limit": "1"},
            timeout=8,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────
# 5. Enrichment orchestrator
# ─────────────────────────────────────────────────────────────────
def enrich_supplier_row(
    region: str,
    fallback_geo: float,
    fallback_sentiment: float,
    fallback_disaster: float,
    live_mode: bool = False,
) -> Dict:
    if not live_mode:
        return {
            "geo_risk_score":        fallback_geo,
            "region_news_sentiment": fallback_sentiment,
            "natural_disaster_risk": fallback_disaster,
            "geo_source":            "synthetic",
            "sentiment_source":      "synthetic",
            "disaster_source":       "synthetic",
        }

    geo,       geo_src  = get_geopolitical_risk(region)
    sentiment, sent_src = get_news_sentiment(region)
    disaster,  dis_src  = get_disaster_risk(region)

    return {
        "geo_risk_score":        geo,
        "region_news_sentiment": sentiment,
        "natural_disaster_risk": disaster,
        "geo_source":            geo_src,
        "sentiment_source":      sent_src,
        "disaster_source":       dis_src,
    }


def enrich_dataframe(df, live_mode: bool = False):
    """
    Enrich entire supplier DataFrame with live or synthetic signals.
    Processes unique regions only to minimise API calls.
    Returns (enriched_df, source_report).
    """
    import pandas as pd

    if not live_mode:
        return df, {
            "geo_risk_score":        "synthetic",
            "region_news_sentiment": "synthetic",
            "natural_disaster_risk": "synthetic",
        }

    enriched   = df.copy()
    source_log: Dict[str, List[str]] = {"geo": [], "sentiment": [], "disaster": []}
    region_results: Dict[str, Dict]  = {}

    unique_regions = (
        enriched["region"].unique().tolist()
        if "region" in enriched.columns else []
    )

    for region in unique_regions:
        row    = enriched[enriched["region"] == region].iloc[0]
        result = enrich_supplier_row(
            region=region,
            fallback_geo=float(row.get("geo_risk_score", 0.5)),
            fallback_sentiment=float(row.get("region_news_sentiment", 0.0)),
            fallback_disaster=float(row.get("natural_disaster_risk", 0.4)),
            live_mode=True,
        )
        region_results[region] = result
        source_log["geo"].append(result["geo_source"])
        source_log["sentiment"].append(result["sentiment_source"])
        source_log["disaster"].append(result["disaster_source"])

    for idx, row in enriched.iterrows():
        region = row.get("region")
        if region and region in region_results:
            r = region_results[region]
            enriched.at[idx, "geo_risk_score"]        = r["geo_risk_score"]
            enriched.at[idx, "region_news_sentiment"] = r["region_news_sentiment"]
            enriched.at[idx, "natural_disaster_risk"] = r["natural_disaster_risk"]

    def majority_source(sources: List[str]) -> str:
        live_count = sum(1 for s in sources if "live" in s)
        return "live" if live_count >= 1 else "synthetic"

    source_report = {
        "geo_risk_score":        majority_source(source_log["geo"]),
        "region_news_sentiment": majority_source(source_log["sentiment"]),
        "natural_disaster_risk": majority_source(source_log["disaster"]),
    }

    print("Source log:", source_log)
    print("source_report values:", source_report)
    for k, v in source_report.items():
        print(f"  {k}: '{v}'")

    return enriched, source_report


# ─────────────────────────────────────────────────────────────────
# 6. API status checker (for sidebar badges)
# ─────────────────────────────────────────────────────────────────
def get_api_status() -> Dict:
    """
    Returns which APIs are configured and reachable.
    Probes are cached for 1 hour to avoid slowing rerenders.
    """
    wb_cache_k   = _cache_key("probe", "worldbank")
    gdacs_cache_k = _cache_key("probe", "gdacs")

    wb_val = _cache_get(wb_cache_k)
    if wb_val is None:
        wb_ok = _probe_worldbank()
        _cache_set(wb_cache_k, 1.0 if wb_ok else 0.0)
    else:
        wb_ok = wb_val > 0.5

    gdacs_val = _cache_get(gdacs_cache_k)
    if gdacs_val is None:
        gdacs_ok = _probe_gdacs()
        _cache_set(gdacs_cache_k, 1.0 if gdacs_ok else 0.0)
    else:
        gdacs_ok = gdacs_val > 0.5

    return {
        "newsapi": {
            "configured": bool(cfg.news_api_key),
            "label":      "NewsAPI",
            "url":        "newsapi.org",
            "covers":     "News Sentiment",
            "key_needed": True,
            "register":   "https://newsapi.org/register",
        },
        "worldbank": {
            "configured": wb_ok,
            "label":      "World Bank",
            "url":        "data.worldbank.org",
            "covers":     "Geopolitical Risk",
            "key_needed": False,
            "register":   None,
        },
        "gdacs": {
            "configured": gdacs_ok,
            "label":      "GDACS (UN)",
            "url":        "gdacs.org",
            "covers":     "Disaster Risk",
            "key_needed": False,
            "register":   None,
        },
    }
