"""
Generates realistic synthetic supply chain supplier data for demo/training.
"""
import numpy as np
import pandas as pd
from typing import Optional

# ── Constants ────────────────────────────────────────────────────
SUPPLIER_NAMES = [
    "Apex Components GmbH", "SinoTech Manufacturing", "Delta Logistics SA",
    "Vikram Auto Parts", "Nordic Steel AB", "Gulf Industrial LLC",
    "Andean Minerals Co.", "East Euro Plastics", "Pacific Rim Electronics",
    "Sahara Textiles Ltd", "Alpine Precision AG", "Shenzhen Circuit Co.",
    "Mumbai Auto Forge", "Seoul Semiconductor", "Cairo Composites",
    "Istanbul Machinery", "Monterrey Metal Works", "Hanoi Hydraulics",
    "Warsaw Wire & Cable", "Lagos Lubricants Plc",
]

REGIONS = ["East Asia", "South Asia", "Southeast Asia", "Eastern Europe",
           "Western Europe", "Middle East", "North Africa", "Sub-Saharan Africa",
           "Latin America", "North America"]

CATEGORIES = ["Electronics", "Raw Materials", "Automotive Parts",
               "Chemicals", "Textiles", "Machinery", "Logistics Services",
               "Packaging", "Semiconductors", "Energy Components"]

TRANSPORT_MODES = ["Sea Freight", "Air Freight", "Road", "Rail", "Multimodal"]


def _geo_risk_by_region(region: str) -> float:
    """Assign baseline geopolitical risk per region (0-1)."""
    risks = {
        "East Asia": 0.45, "South Asia": 0.55, "Southeast Asia": 0.40,
        "Eastern Europe": 0.65, "Western Europe": 0.15, "Middle East": 0.75,
        "North Africa": 0.70, "Sub-Saharan Africa": 0.60, "Latin America": 0.50,
        "North America": 0.10,
    }
    base = risks.get(region, 0.5)
    return float(np.clip(base + np.random.normal(0, 0.08), 0.05, 0.95))


def _disaster_risk_by_region(region: str) -> float:
    """Natural disaster risk index per region."""
    risks = {
        "East Asia": 0.70, "South Asia": 0.65, "Southeast Asia": 0.75,
        "Eastern Europe": 0.25, "Western Europe": 0.20, "Middle East": 0.30,
        "North Africa": 0.35, "Sub-Saharan Africa": 0.50, "Latin America": 0.55,
        "North America": 0.30,
    }
    base = risks.get(region, 0.4)
    return float(np.clip(base + np.random.normal(0, 0.07), 0.05, 0.95))


def _transport_risk(mode: str) -> float:
    risks = {
        "Sea Freight": 0.55, "Air Freight": 0.25, "Road": 0.40,
        "Rail": 0.35, "Multimodal": 0.50,
    }
    return risks.get(mode, 0.4) + np.random.normal(0, 0.05)


def generate_supplier_data(
    n_suppliers: int = 40,
    seed: Optional[int] = 42,
    include_labels: bool = True,
) -> pd.DataFrame:
    """
    Generate a synthetic supply chain supplier dataset.

    Parameters
    ----------
    n_suppliers : int
        Number of supplier rows to generate.
    seed : int
        Random seed for reproducibility.
    include_labels : bool
        If True, include the target 'disruption_risk_label' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered supply chain features.
    """
    rng = np.random.default_rng(seed)

    supplier_pool = (SUPPLIER_NAMES * ((n_suppliers // len(SUPPLIER_NAMES)) + 1))[:n_suppliers]
    suffixes = rng.integers(100, 999, size=n_suppliers)
    supplier_ids = [f"SUP-{s:04d}" for s in rng.integers(1000, 9999, size=n_suppliers)]

    regions = rng.choice(REGIONS, size=n_suppliers)
    categories = rng.choice(CATEGORIES, size=n_suppliers)
    transport_modes = rng.choice(TRANSPORT_MODES, size=n_suppliers)

    geo_risk = np.array([_geo_risk_by_region(r) for r in regions])
    disaster_risk = np.array([_disaster_risk_by_region(r) for r in regions])
    transport_risk = np.array([_transport_risk(t) for t in transport_modes])

    lead_time_days = rng.integers(7, 120, size=n_suppliers).astype(float)
    lead_time_variance = rng.uniform(0.05, 0.45, size=n_suppliers) * lead_time_days

    inventory_buffer_days = rng.integers(5, 90, size=n_suppliers).astype(float)
    supplier_reliability = np.clip(rng.normal(0.75, 0.15, size=n_suppliers), 0.1, 0.99)
    financial_health = np.clip(rng.normal(0.70, 0.18, size=n_suppliers), 0.1, 0.99)
    single_source = rng.choice([0, 1], size=n_suppliers, p=[0.65, 0.35])

    # Sentiment: regions under tension get lower sentiment
    base_sentiment = -geo_risk * 0.6 + rng.normal(0, 0.2, size=n_suppliers)
    region_sentiment = np.clip(base_sentiment, -1, 1)

    past_disruptions = rng.integers(0, 8, size=n_suppliers)
    regulatory_risk = np.clip(geo_risk * 0.7 + rng.normal(0, 0.1, size=n_suppliers), 0, 1)

    annual_spend_usd = rng.integers(50_000, 5_000_000, size=n_suppliers)
    num_alternative_suppliers = rng.integers(0, 5, size=n_suppliers)

    # ── Composite Risk Score (ground truth for labelling) ─────────
    raw_score = (
        0.22 * geo_risk
        + 0.18 * (lead_time_variance / (lead_time_days + 1))
        + 0.15 * (1 - supplier_reliability)
        + 0.12 * (1 - financial_health)
        + 0.10 * single_source
        + 0.08 * disaster_risk
        + 0.07 * (past_disruptions / 8.0)
        + 0.05 * transport_risk
        + 0.03 * (-region_sentiment * 0.5 + 0.5)
        + rng.normal(0, 0.04, size=n_suppliers)        # noise
    )
    raw_score = np.clip(raw_score, 0, 1)

    df = pd.DataFrame({
        "supplier_id": supplier_ids,
        "supplier_name": [f"{n} ({sfx})" for n, sfx in zip(supplier_pool, suffixes)],
        "region": regions,
        "category": categories,
        "transport_mode": transport_modes,
        "geo_risk_score": geo_risk.round(3),
        "lead_time_days": lead_time_days,
        "lead_time_variance": lead_time_variance.round(2),
        "inventory_buffer_days": inventory_buffer_days,
        "supplier_reliability_score": supplier_reliability.round(3),
        "financial_health_score": financial_health.round(3),
        "single_source_dependency": single_source,
        "region_news_sentiment": region_sentiment.round(3),
        "natural_disaster_risk": disaster_risk.round(3),
        "past_disruptions_12mo": past_disruptions,
        "regulatory_risk_score": regulatory_risk.round(3),
        "transport_mode_risk": transport_risk.round(3),
        "annual_spend_usd": annual_spend_usd,
        "num_alternative_suppliers": num_alternative_suppliers,
        "disruption_risk_score": raw_score.round(4),
    })

    if include_labels:
        df["disruption_risk_label"] = pd.cut(
            df["disruption_risk_score"],
            bins=[0, 0.35, 0.65, 1.0],
            labels=["Low", "Medium", "High"],
        )

    # ── Hardcoded high-risk suppliers ────────────────────────────
    high_risk = pd.DataFrame({
        "supplier_id":                ["SUP-9001", "SUP-9002", "SUP-9003", "SUP-9004", "SUP-9005",
                                       "SUP-9006", "SUP-9007", "SUP-9008", "SUP-9009", "SUP-9010"],
        "supplier_name":              ["Tripoli Steel Works (Libya)", "Sana'a Textiles Ltd (Yemen)",
                                       "Benghazi Components Co (Libya)", "Aleppo Industrial LLC (Syria)",
                                       "Kabul Minerals Corp (Afghanistan)",
                                       "Tehran Energy Systems Co (Iran)", "Kharkiv Power Equipment LLC (Ukraine)",
                                       "Isfahan Oil Components Ltd (Iran)", "Zaporizhzhia Industrial Energy (Ukraine)",
                                       "Ahvaz Petroleum Parts Co (Iran)"],
        "region":                     ["North Africa", "Middle East", "North Africa", "Middle East", "South Asia",
                                       "Middle East", "Eastern Europe", "Middle East", "Eastern Europe", "Middle East"],
        "category":                   ["Raw Materials", "Textiles", "Raw Materials", "Chemicals", "Raw Materials",
                                       "Energy Components", "Energy Components", "Energy Components", "Energy Components", "Energy Components"],
        "transport_mode":             ["Sea Freight", "Air Freight", "Sea Freight", "Air Freight", "Road",
                                       "Air Freight", "Rail", "Sea Freight", "Rail", "Sea Freight"],
        "geo_risk_score":             [0.92, 0.95, 0.88, 0.91, 0.85,
                                       0.91, 0.82, 0.88, 0.79, 0.93],
        "lead_time_days":             [75.0, 90.0, 80.0, 95.0, 110.0,
                                       85.0, 70.0, 95.0, 65.0, 100.0],
        "lead_time_variance":         [28.0, 35.0, 30.0, 40.0, 45.0,
                                       32.0, 25.0, 38.0, 28.0, 42.0],
        "inventory_buffer_days":      [8.0, 5.0, 10.0, 7.0, 6.0,
                                       8.0, 10.0, 6.0, 12.0, 7.0],
        "supplier_reliability_score": [0.28, 0.22, 0.31, 0.25, 0.30,
                                       0.26, 0.35, 0.30, 0.40, 0.24],
        "financial_health_score":     [0.25, 0.20, 0.28, 0.22, 0.26,
                                       0.24, 0.38, 0.28, 0.42, 0.22],
        "single_source_dependency":   [1, 1, 1, 1, 1,
                                       1, 1, 1, 0, 1],
        "region_news_sentiment":      [-0.75, -0.85, -0.70, -0.80, -0.65,
                                       -0.78, -0.65, -0.72, -0.58, -0.82],
        "natural_disaster_risk":      [0.55, 0.60, 0.50, 0.45, 0.58,
                                       0.38, 0.32, 0.40, 0.30, 0.42],
        "past_disruptions_12mo":      [7, 8, 6, 7, 5,
                                       7, 5, 6, 4, 8],
        "regulatory_risk_score":      [0.88, 0.92, 0.85, 0.90, 0.82,
                                       0.92, 0.75, 0.88, 0.70, 0.94],
        "transport_mode_risk":        [0.65, 0.70, 0.62, 0.68, 0.72,
                                       0.70, 0.58, 0.65, 0.52, 0.68],
        "annual_spend_usd":           [850000, 620000, 740000, 580000, 920000,
                                       780000, 920000, 650000, 1100000, 580000],
        "num_alternative_suppliers":  [0, 0, 1, 0, 0,
                                       0, 1, 0, 2, 0],
        "disruption_risk_score":      [0.88, 0.92, 0.85, 0.90, 0.83,
                                       0.89, 0.76, 0.84, 0.72, 0.91],
    })

    # Match dtypes to main df
    for col in ["lead_time_days", "inventory_buffer_days"]:
        high_risk[col] = high_risk[col].astype(df[col].dtype)
    for col in ["single_source_dependency", "past_disruptions_12mo",
                "annual_spend_usd", "num_alternative_suppliers"]:
        high_risk[col] = high_risk[col].astype(df[col].dtype)

    if include_labels:
        high_risk["disruption_risk_label"] = pd.Categorical(
            ["High", "High", "High", "High", "High",
             "High", "High", "High", "High", "High"],
            categories=["Low", "Medium", "High"],
        )

    df = pd.concat([df, high_risk], ignore_index=True)

    return df.reset_index(drop=True)


def enrich_with_live_data(df: pd.DataFrame, live_mode: bool = False) -> tuple:
    """
    Wraps real_data_sources.enrich_dataframe().
    Call this after generate_supplier_data() to optionally replace
    the three synthetic signals with real API data.

    Returns (enriched_df, source_report)
    """
    from backend.real_data_sources import enrich_dataframe
    return enrich_dataframe(df, live_mode=live_mode)


FEATURE_COLUMNS = [
    "geo_risk_score", "lead_time_days", "lead_time_variance",
    "inventory_buffer_days", "supplier_reliability_score",
    "financial_health_score", "single_source_dependency",
    "region_news_sentiment", "natural_disaster_risk",
    "past_disruptions_12mo", "regulatory_risk_score",
    "transport_mode_risk",
]

FEATURE_LABELS = {
    "geo_risk_score": "Geopolitical Risk",
    "lead_time_days": "Lead Time (days)",
    "lead_time_variance": "Lead Time Variance",
    "inventory_buffer_days": "Inventory Buffer (days)",
    "supplier_reliability_score": "Supplier Reliability",
    "financial_health_score": "Financial Health",
    "single_source_dependency": "Single-Source Dependency",
    "region_news_sentiment": "Region News Sentiment",
    "natural_disaster_risk": "Natural Disaster Risk",
    "past_disruptions_12mo": "Past Disruptions (12mo)",
    "regulatory_risk_score": "Regulatory Risk",
    "transport_mode_risk": "Transport Mode Risk",
}
