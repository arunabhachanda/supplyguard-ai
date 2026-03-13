"""
SupplyGuard AI — Supply Chain Disruption Risk Intelligence Platform
Main Streamlit application entry point.
"""
import sys
import os
from typing import Optional, Tuple, List

# Ensure backend modules are importable
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import cfg
from backend.auth import (
    init_auth_state, is_authenticated,
    render_login_page, render_user_badge,
    has_permission, require_permission,
    get_current_user, get_user_role,
)
from backend.data_generator import generate_supplier_data, FEATURE_COLUMNS, FEATURE_LABELS
from backend.risk_model import predict_risk, explain_supplier, get_model_metrics, validate_upload
from backend.llm_advisor import get_mitigation_recommendations, get_portfolio_summary
from page_rebalancing import render_rebalancing_page

# ─────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SupplyGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# Global CSS — dark industrial theme, orange accents
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ────────────────────────────────── */
.stApp {
    background: #1C1C1C !important;
    color: #E2E8F0;
    font-family: 'Space Grotesk', sans-serif;
}
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: #141414 !important;
    border-right: 1px solid #1E293B;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }

/* ── Headings ────────────────────────────── */
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }
h1 { color: #F97316 !important; font-weight: 700; }
h2 { color: #F1F5F9 !important; font-weight: 600; }
h3 { color: #CBD5E1 !important; font-weight: 500; }

/* ── Metric cards ────────────────────────── */
[data-testid="metric-container"] {
    background: #2A2A2A !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: #94A3B8 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #F97316 !important; font-size: 1.8rem !important; font-weight: 700 !important; }

/* ── Buttons ─────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #F97316, #EA580C) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(249,115,22,0.4) !important;
}

/* ── Tabs ─────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { background: #2A2A2A; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #94A3B8 !important; border-radius: 6px; }
.stTabs [aria-selected="true"] { background: #F97316 !important; color: white !important; }

/* ── Dataframes ────────────────────────────── */
[data-testid="stDataFrame"] { border: 1px solid #334155; border-radius: 8px; }

/* ── Divider ─────────────────────────────── */
hr { border-color: #1E293B !important; }

/* ── Alerts ──────────────────────────────── */
.stAlert { border-radius: 8px !important; }

/* ── Upload ──────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #1E293B !important;
    border: 2px dashed #334155 !important;
    border-radius: 12px !important;
}

/* ── Select / Input ──────────────────────── */
.stSelectbox > div, .stTextInput > div {
    background: #1E293B !important;
    border-color: #334155 !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
}

/* ── Risk label badges ───────────────────── */
.badge-high   { background:#FEE2E2; color:#991B1B; border-radius:4px; padding:2px 8px; font-weight:600; font-size:0.78rem; }
.badge-medium { background:#FEF3C7; color:#92400E; border-radius:4px; padding:2px 8px; font-weight:600; font-size:0.78rem; }
.badge-low    { background:#DCFCE7; color:#166534; border-radius:4px; padding:2px 8px; font-weight:600; font-size:0.78rem; }

/* ── Card ────────────────────────────────── */
.sg-card {
    background: #2A2A2A;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.sg-card-accent { border-left: 4px solid #F97316; }

/* ── KPI strip ───────────────────────────── */
.kpi-strip {
    display: flex; gap: 0.5rem; flex-wrap: wrap;
    margin: 0.5rem 0;
}
.kpi-chip {
    background: #1A1A1A;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: #94A3B8;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Sidebar nav ─────────────────────────── */
[data-testid="stRadio"] label { font-size: 0.9rem !important; }

/* ── Progress bars ───────────────────────── */
.stProgress > div > div { background: #F97316 !important; }

/* ── Scrollbar ───────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1C1C1C; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

/* ── Disabled toggle label ───────────────── */
[data-testid="stToggle"][aria-disabled="true"] label {
    opacity: 0.4 !important;
}

/* ── Brighten dim text ───────────────────── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #CBD5E1 !important;
}
.stSlider label, .stSelectbox label,
.stMultiSelect label, .stToggle label,
.stRadio label, .stCheckbox label {
    color: #CBD5E1 !important;
}
[data-testid="stWidgetLabel"] {
    color: #CBD5E1 !important;
}

/* ── Force white text globally ───────────── */
.js-plotly-plot .plotly .legend text,
.js-plotly-plot .plotly text,
.legendtext {
    fill: #FFFFFF !important;
}
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th {
    color: #E2E8F0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
RISK_COLORS = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#22C55E"}


def load_demo_data(live_mode: bool = False, force_refresh: bool = False) -> pd.DataFrame:
    """Load demo data, optionally enriched with real API signals."""
    df = generate_supplier_data(n_suppliers=40, seed=42, include_labels=True)
    if live_mode:
        from backend.data_generator import enrich_with_live_data
        df, source_report = enrich_with_live_data(df, live_mode=True)
        st.session_state["_source_report"] = source_report
    else:
        st.session_state["_source_report"] = {
            "geo_risk_score":        "synthetic",
            "region_news_sentiment": "synthetic",
            "natural_disaster_risk": "synthetic",
        }
    return predict_risk(df)


def validate_and_score_upload(uploaded_file) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Read, validate, and score an uploaded CSV."""
    if st.session_state.get("_upload_count", 0) >= cfg.max_uploads_per_session:
        return None, ["Upload limit reached for this session."]
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, [f"Could not parse CSV: {e}"]

    if len(df) > cfg.max_rows_per_upload:
        return None, [f"Max {cfg.max_rows_per_upload} rows allowed per upload. Got {len(df)}."]

    is_valid, errors = validate_upload(df)
    if not is_valid:
        return None, errors

    st.session_state["_upload_count"] = st.session_state.get("_upload_count", 0) + 1
    scored = predict_risk(df)
    return scored, []


# ─────────────────────────────────────────────────────────────────
# Page: Dashboard
# ─────────────────────────────────────────────────────────────────
def page_dashboard(df: pd.DataFrame):
    require_permission("view_dashboard")

    st.markdown("## 📊 Portfolio Risk Dashboard")
    st.markdown("<hr>", unsafe_allow_html=True)

    high = df[df["risk_label"] == "High"]
    med  = df[df["risk_label"] == "Medium"]
    low  = df[df["risk_label"] == "Low"]
    high_spend = high["annual_spend_usd"].sum() if "annual_spend_usd" in df.columns else 0

    # ── KPI Row ──────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Suppliers", len(df))
    c2.metric("🔴 High Risk", len(high), delta=f"{len(high)/len(df)*100:.0f}%", delta_color="inverse")
    c3.metric("🟡 Medium Risk", len(med))
    c4.metric("🟢 Low Risk", len(low))
    c5.metric("💰 Spend at Risk", f"${high_spend/1e6:.1f}M" if high_spend else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row 1 ────────────────────────────────────────────
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("**Risk Distribution**")
        fig_pie = px.pie(
            names=["High", "Medium", "Low"],
            values=[len(high), len(med), len(low)],
            color=["High", "Medium", "Low"],
            color_discrete_map=RISK_COLORS,
            hole=0.55,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1", legend_font_size=12,
            margin=dict(l=0, r=0, t=20, b=0), height=260,
            showlegend=True,
        )
        fig_pie.update_traces(textfont_color="#E2E8F0")
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        if "region" in df.columns:
            st.markdown("**Risk Score by Region**")
            region_risk = df.groupby("region")["risk_score"].mean().sort_values(ascending=True)
            fig_bar = px.bar(
                x=region_risk.values,
                y=region_risk.index,
                orientation="h",
                color=region_risk.values,
                color_continuous_scale=["#22C55E", "#F59E0B", "#EF4444"],
                range_color=[0, 1],
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#CBD5E1", coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=20, b=0), height=260,
                xaxis=dict(gridcolor="#1E293B", range=[0, 1]),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row 2 ────────────────────────────────────────────
    col_c, col_d = st.columns([2, 1])

    with col_c:
        st.markdown("**Risk Score Distribution**")
        fig_hist = px.histogram(
            df, x="risk_score", nbins=25,
            color_discrete_sequence=["#F97316"],
        )
        fig_hist.add_vline(x=0.35, line_dash="dash", line_color="#F59E0B",
                           annotation_text="Medium threshold", annotation_font_color="#F59E0B")
        fig_hist.add_vline(x=0.65, line_dash="dash", line_color="#EF4444",
                           annotation_text="High threshold", annotation_font_color="#EF4444")
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1", height=240, margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(gridcolor="#1E293B"), yaxis=dict(gridcolor="#1E293B"),
            bargap=0.05, showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    with col_d:
        if "category" in df.columns:
            st.markdown("**Risk by Category**")
            cat_risk = df.groupby("category")["risk_score"].mean().sort_values(ascending=False).head(6)
            fig_cat = px.bar(
                x=cat_risk.values, y=cat_risk.index,
                orientation="h",
                color=cat_risk.values,
                color_continuous_scale=["#22C55E", "#F59E0B", "#EF4444"],
                range_color=[0, 1],
            )
            fig_cat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#CBD5E1", coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=0), height=240,
                xaxis=dict(gridcolor="#1E293B"), yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scatter: Geo Risk vs Reliability ────────────────────────
    st.markdown("**Supplier Risk Matrix — Geo Risk vs. Reliability**")
    # Drop NaN risk_label rows to avoid 'nan' in legend
    scatter_df = df[df["risk_label"].isin(["High", "Medium", "Low"])].copy()
    fig_scatter = px.scatter(
        scatter_df,
        x="geo_risk_score",
        y="supplier_reliability_score",
        color="risk_label",
        color_discrete_map=RISK_COLORS,
        size="risk_score",
        size_max=22,
        hover_name="supplier_name" if "supplier_name" in scatter_df.columns else None,
        hover_data={"risk_score": ":.3f", "risk_label": True},
        labels={"geo_risk_score": "Geopolitical Risk Score",
                "supplier_reliability_score": "Supplier Reliability Score"},
        category_orders={"risk_label": ["High", "Medium", "Low"]},
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1", height=350, margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(gridcolor="#1E293B"), yaxis=dict(gridcolor="#1E293B"),
        legend_title_font_color="#CBD5E1",
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

    # ── Data Source Provenance Panel ────────────────────────────
    st.markdown("---")
    st.markdown("### 🔬 Data Source Provenance")
    source_report = st.session_state.get("_source_report", {})
    signal_meta = {
        "geo_risk_score": {
            "label": "Geopolitical Risk",
            "api": "World Bank (PV.EST)",
            "url": "data.worldbank.org",
            "desc": "Political Stability & Absence of Violence index",
        },
        "region_news_sentiment": {
            "label": "News Sentiment",
            "api": "NewsAPI + TextBlob",
            "url": "newsapi.org",
            "desc": "Supply chain news NLP sentiment (30-day window)",
        },
        "natural_disaster_risk": {
            "label": "Disaster Risk",
            "api": "GDACS (UN)",
            "url": "gdacs.org",
            "desc": "Natural disaster alerts (EQ, TC, FL, VO, DR, WF — 365-day window)",
        },
    }
    prov_cols = st.columns(3)
    for i, (col_name, meta) in enumerate(signal_meta.items()):
        src = source_report.get(col_name, "synthetic")
        is_live = "live" in src
        badge_color = "#22C55E" if is_live else "#F59E0B"
        badge_text  = "● LIVE"  if is_live else "● SYNTHETIC"
        with prov_cols[i]:
            st.markdown(
                f'<div class="sg-card" style="text-align:center; padding:1rem;">'
                f'<div style="font-size:0.75rem; font-weight:700; color:{badge_color}; '
                f'margin-bottom:6px;">{badge_text}</div>'
                f'<div style="font-size:0.9rem; font-weight:600; color:#E2E8F0; '
                f'margin-bottom:4px;">{meta["label"]}</div>'
                f'<div style="font-size:0.75rem; color:#F97316; '
                f'font-family:monospace;">{meta["api"]}</div>'
                f'<div style="font-size:0.72rem; color:#64748B; '
                f'margin-top:4px;">{meta["desc"]}</div>'
                f'<div style="font-size:0.68rem; margin-top:6px;">'
                f'<a href="https://{meta["url"]}" target="_blank" style="color:#F97316; text-decoration:none; font-family:monospace;">🔗 {meta["url"]}</a></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if not any("live" in v for v in source_report.values()):
        st.info(
            "💡 **Enable Live Data Mode** in the sidebar to replace synthetic signals "
            "with real data from World Bank (geo risk), NewsAPI (sentiment), and "
            "GDACS (disaster risk). Add NEWS_API_KEY to your `.env` file first."
        )

    # ── AI Portfolio Summary ─────────────────────────────────────
    if has_permission("get_recommendations"):
        st.markdown("---")
        st.markdown("**🤖 AI Executive Portfolio Brief**")
        if st.button("Generate AI Portfolio Summary", key="portfolio_ai"):
            top_regions = (
                df[df["risk_label"] == "High"]["region"].value_counts().head(3).index.tolist()
                if "region" in df.columns else []
            )
            with st.spinner("Generating executive brief..."):
                summary = get_portfolio_summary(
                    high_risk_count=len(high),
                    medium_risk_count=len(med),
                    low_risk_count=len(low),
                    total_spend_at_risk=float(high_spend),
                    top_risky_regions=top_regions,
                )
            if summary:
                st.markdown(f"### {summary.get('headline', '')}")
                health = summary.get("portfolio_health", "")
                health_color = {"Critical": "#EF4444", "At Risk": "#F59E0B",
                                "Moderate": "#3B82F6", "Healthy": "#22C55E"}.get(health, "#94A3B8")
                st.markdown(
                    f'<span style="background:{health_color}20; color:{health_color}; '
                    f'border:1px solid {health_color}; border-radius:6px; padding:4px 12px; '
                    f'font-weight:600;">Portfolio Health: {health}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Key Findings**")
                    for f in summary.get("key_findings", []):
                        st.markdown(f"• {f}")
                with col2:
                    st.markdown("**30-Day Priorities**")
                    for p in summary.get("30_day_priorities", []):
                        st.markdown(f"→ {p}")

                st.markdown("**Board Recommendation**")
                st.info(summary.get("board_recommendation", ""))


# ─────────────────────────────────────────────────────────────────
# Page: Supplier Risk Analysis
# ─────────────────────────────────────────────────────────────────
def page_supplier_analysis(df: pd.DataFrame):
    require_permission("view_dashboard")

    st.markdown("## 🔍 Supplier Risk Analysis")
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Filters — always computed from original unfiltered df ────
    all_risk_labels = ["High", "Medium", "Low"]
    all_regions     = sorted(df["region"].unique().tolist()) if "region" in df.columns else []
    all_categories  = sorted(df["category"].unique().tolist()) if "category" in df.columns else []

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        risk_filter = st.multiselect(
            "Filter by Risk Level", all_risk_labels, default=all_risk_labels,
        )
    with col_f2:
        region_filter = st.multiselect(
            "Filter by Region", all_regions, default=all_regions,
        )
    with col_f3:
        cat_filter = st.multiselect(
            "Filter by Category", all_categories, default=all_categories,
        )

    filtered = df[df["risk_label"].isin(risk_filter)] if risk_filter else df.copy()
    if region_filter and "region" in df.columns:
        filtered = filtered[filtered["region"].isin(region_filter)]
    if cat_filter and "category" in df.columns:
        filtered = filtered[filtered["category"].isin(cat_filter)]

    st.markdown(f"**{len(filtered)} suppliers** matching filters")
    st.markdown("<br>", unsafe_allow_html=True)

    if filtered.empty:
        st.warning("No suppliers match the selected filters. Please adjust your filters.")
        return

    # ── Supplier Table ────────────────────────────────────────────
    display_cols = ["supplier_name", "region", "category", "risk_score",
                    "risk_label", "risk_percentile", "geo_risk_score",
                    "supplier_reliability_score", "lead_time_days",
                    "past_disruptions_12mo"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[display_cols].copy(),
        use_container_width=True,
        height=380,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=1, format="%.3f"
            ),
            "geo_risk_score": st.column_config.ProgressColumn(
                "Geo Risk", min_value=0, max_value=1, format="%.3f"
            ),
            "supplier_reliability_score": st.column_config.ProgressColumn(
                "Reliability", min_value=0, max_value=1, format="%.3f"
            ),
            "risk_percentile": st.column_config.NumberColumn("Percentile", format="%.1f%%"),
            "risk_label": st.column_config.TextColumn("Risk Level"),
        },
    )

    # ── Per-supplier deep-dive ────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧬 Supplier Deep Dive")

    if "supplier_name" in filtered.columns:
        selected_name = st.selectbox(
            "Select a supplier to analyse",
            options=filtered["supplier_name"].tolist(),
        )
        row = filtered[filtered["supplier_name"] == selected_name].iloc[0]
    else:
        idx = st.selectbox("Select supplier index", options=filtered.index.tolist())
        row = filtered.loc[idx]
        selected_name = f"Supplier #{idx}"

    col_left, col_right = st.columns([1, 2])

    with col_left:
        risk_color = RISK_COLORS.get(str(row["risk_label"]), "#94A3B8")
        detail_rows = ""
        for key, label in [
            ("region", "Region"), ("category", "Category"),
            ("transport_mode", "Transport"), ("lead_time_days", "Lead Time"),
            ("past_disruptions_12mo", "Disruptions (12mo)"),
            ("supplier_reliability_score", "Reliability"),
            ("financial_health_score", "Fin. Health"),
        ]:
            if key in row.index:
                val = row[key]
                if isinstance(val, float):
                    val = f"{val:.3f}"
                detail_rows += (
                    f'<div style="display:flex; justify-content:space-between; '
                    f'padding:4px 0; border-bottom:1px solid #1E293B;">'
                    f'<span style="color:#64748B; font-size:0.82rem;">{label}</span>'
                    f'<span style="color:#E2E8F0; font-size:0.82rem; font-family:monospace;">{val}</span>'
                    f'</div>'
                )
        st.markdown(
            f'<div class="sg-card sg-card-accent">'
            f'<div style="margin-bottom:1rem;">'
            f'<div style="font-size:0.8rem; color:#94A3B8;">Risk Score</div>'
            f'<div style="font-size:2.8rem; font-weight:700; color:{risk_color}; '
            f'font-family:\'JetBrains Mono\',monospace; line-height:1.1;">'
            f'{row["risk_score"]:.3f}</div>'
            f'<div style="background:{risk_color}20; color:{risk_color}; border:1px solid {risk_color}; '
            f'border-radius:6px; display:inline-block; padding:2px 10px; '
            f'font-weight:600; font-size:0.85rem; margin-top:4px;">'
            f'{row["risk_label"]} Risk</div>'
            f'</div>'
            f'{detail_rows}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("**Risk Factor Importance (AI Explanation)**")
        with st.spinner("Computing feature importance..."):
            importance_df = explain_supplier(row)

        fig_imp = px.bar(
            importance_df,
            x="importance",
            y="label",
            orientation="h",
            color="importance",
            color_continuous_scale=["#22C55E", "#F59E0B", "#EF4444"],
            text="direction",
        )
        fig_imp.update_traces(textposition="outside", textfont_color="#CBD5E1")
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1", coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0), height=320,
            xaxis=dict(gridcolor="#1E293B", title="Contribution to Risk Score"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", title=""),
        )
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

    # ── AI Mitigation ─────────────────────────────────────────────
    if has_permission("get_recommendations"):
        st.markdown("---")
        st.markdown("### 🤖 AI Mitigation Recommendations")
        if st.button(f"Generate Recommendations for {selected_name}", type="primary"):
            top_drivers = importance_df.to_dict("records")
            with st.spinner("AI is analysing risk profile..."):
                recs = get_mitigation_recommendations(
                    supplier_name=selected_name,
                    region=row.get("region", "Unknown"),
                    category=row.get("category", "Unknown"),
                    risk_score=float(row["risk_score"]),
                    risk_label=str(row["risk_label"]),
                    transport_mode=row.get("transport_mode", "Unknown"),
                    top_risk_drivers=top_drivers,
                )

            if recs:
                st.markdown(
                    f'<div class="sg-card sg-card-accent">'
                    f'<strong>Executive Summary</strong><br>{recs.get("executive_summary", "")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                col_i, col_s = st.columns(2)
                with col_i:
                    st.markdown("#### ⚡ Immediate Actions")
                    for action in recs.get("immediate_actions", []):
                        priority = action.get("priority", "Medium")
                        p_color = {"Critical": "#EF4444", "High": "#F59E0B",
                                   "Medium": "#3B82F6"}.get(priority, "#94A3B8")
                        st.markdown(
                            f'<div class="sg-card" style="margin-bottom:0.5rem; padding:0.85rem;">'
                            f'<div style="display:flex; justify-content:space-between; '
                            f'align-items:center; margin-bottom:4px;">'
                            f'<span style="background:{p_color}20; color:{p_color}; '
                            f'border-radius:4px; padding:2px 8px; font-size:0.75rem; '
                            f'font-weight:600;">{priority}</span>'
                            f'<span style="color:#64748B; font-size:0.75rem;">'
                            f'⏱ {action.get("timeline", "")}</span>'
                            f'</div>'
                            f'<div style="color:#E2E8F0; font-size:0.85rem;">'
                            f'{action.get("action", "")}</div>'
                            f'<div style="color:#64748B; font-size:0.75rem; margin-top:4px;">'
                            f'Cost: {action.get("cost_impact", "")}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                with col_s:
                    st.markdown("#### 🗺️ Strategic Recommendations")
                    for rec in recs.get("strategic_recommendations", []):
                        effort = rec.get("effort", "Medium")
                        e_color = {"Low": "#22C55E", "Medium": "#F59E0B",
                                   "High": "#EF4444"}.get(effort, "#94A3B8")
                        st.markdown(
                            f'<div class="sg-card" style="margin-bottom:0.5rem; padding:0.85rem;">'
                            f'<div style="color:#E2E8F0; font-size:0.85rem; margin-bottom:4px;">'
                            f'{rec.get("recommendation", "")}</div>'
                            f'<div style="color:#22C55E; font-size:0.75rem;">'
                            f'✓ {rec.get("benefit", "")}</div>'
                            f'<div style="color:{e_color}; font-size:0.75rem; margin-top:2px;">'
                            f'Effort: {effort}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown("**📈 KPIs to Monitor**")
                kpi_html = "".join(
                    f'<span class="kpi-chip">📊 {k}</span>'
                    for k in recs.get("kpi_to_monitor", [])
                )
                trend = recs.get("risk_trend", "Stable")
                trend_icon = {"Improving": "📉", "Stable": "➡️", "Deteriorating": "📈"}.get(trend, "➡️")
                st.markdown(
                    f'<div class="kpi-strip">{kpi_html}'
                    f'<span class="kpi-chip">{trend_icon} Trend: {trend}</span>'
                    f'<span class="kpi-chip">🎯 Confidence: {recs.get("confidence", "")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────
# Page: Data Upload
# ─────────────────────────────────────────────────────────────────
def page_upload():
    require_permission("upload_data")

    st.markdown("## 📁 Upload Supplier Data")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        '<div class="sg-card">Upload a CSV with your supplier data. Required columns are listed below. '
        'A <strong>sample CSV template</strong> is available to download.</div>',
        unsafe_allow_html=True,
    )

    template = generate_supplier_data(n_suppliers=5, seed=1, include_labels=False)
    template_csv = template[[c for c in ["supplier_name", "region", "category",
                                          "transport_mode"] + FEATURE_COLUMNS
                               if c in template.columns]].to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV Template",
        data=template_csv,
        file_name="supplier_template.csv",
        mime="text/csv",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📋 Required Column Reference", expanded=False):
        col_info = []
        for feat, label in FEATURE_LABELS.items():
            ranges = {
                "geo_risk_score": "0.0 – 1.0", "supplier_reliability_score": "0.0 – 1.0",
                "financial_health_score": "0.0 – 1.0", "natural_disaster_risk": "0.0 – 1.0",
                "regulatory_risk_score": "0.0 – 1.0", "transport_mode_risk": "0.0 – 1.0",
                "region_news_sentiment": "-1.0 – 1.0", "single_source_dependency": "0 or 1",
                "lead_time_days": "1 – 365", "inventory_buffer_days": "0 – 365",
                "lead_time_variance": "> 0", "past_disruptions_12mo": "0 – 50",
            }
            col_info.append({"Column": feat, "Description": label, "Range": ranges.get(feat, "—")})
        st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

    uploaded = st.file_uploader(
        "Upload your supplier CSV",
        type=["csv"],
        help=f"Max {cfg.max_rows_per_upload} rows",
    )

    if uploaded:
        with st.spinner("Validating and scoring suppliers..."):
            scored, errors = validate_and_score_upload(uploaded)
        if errors:
            for err in errors:
                st.error(f"❌ {err}")
        else:
            st.success(f"✅ Successfully scored {len(scored)} suppliers!")
            st.session_state["uploaded_df"] = scored
            st.rerun()


# ─────────────────────────────────────────────────────────────────
# Page: Export
# ─────────────────────────────────────────────────────────────────
def page_export(df: pd.DataFrame):
    require_permission("export_report")

    st.markdown("## 📤 Export Risk Report")
    st.markdown("<hr>", unsafe_allow_html=True)

    csv_data = df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Full Risk Report (CSV)",
        data=csv_data,
        file_name="supplyguard_risk_report.csv",
        mime="text/csv",
        type="primary",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    summary_stats = {
        "Total Suppliers": len(df),
        "High Risk": len(df[df["risk_label"] == "High"]),
        "Medium Risk": len(df[df["risk_label"] == "Medium"]),
        "Low Risk": len(df[df["risk_label"] == "Low"]),
        "Average Risk Score": f"{df['risk_score'].mean():.3f}",
        "Max Risk Score": f"{df['risk_score'].max():.3f}",
        "Min Risk Score": f"{df['risk_score'].min():.3f}",
    }
    rows_html = "".join(
        f'<div style="display:flex; justify-content:space-between; padding:6px 0; '
        f'border-bottom:1px solid #1E293B;">'
        f'<span style="color:#94A3B8;">{k}</span>'
        f'<span style="color:#E2E8F0; font-family:monospace; font-weight:600;">{v}</span>'
        f'</div>'
        for k, v in summary_stats.items()
    )
    st.markdown(
        f'<div class="sg-card"><strong>Summary Statistics</strong><br><br>{rows_html}</div>',
        unsafe_allow_html=True,
    )

    metrics = get_model_metrics()
    st.markdown("<br>**Model Performance (trained on 1,600 synthetic samples)**")
    m_cols = st.columns(4)
    m_cols[0].metric("ROC-AUC", metrics.get("roc_auc", "—"))
    m_cols[1].metric("F1 Score", metrics.get("f1", "—"))
    m_cols[2].metric("Train Set", metrics.get("train_size", "—"))
    m_cols[3].metric("Test Set", metrics.get("test_size", "—"))


# ─────────────────────────────────────────────────────────────────
# Page: About
# ─────────────────────────────────────────────────────────────────
def page_about():
    st.markdown("## ℹ️ About SupplyGuard AI")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        '<div class="sg-card sg-card-accent">'
        '<strong>SupplyGuard AI</strong> is a supply chain disruption risk intelligence platform '
        'combining machine learning, NLP-driven news sentiment, and LLM-generated mitigation '
        'strategies to give procurement teams and executives actionable risk intelligence.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🧠 ML Model**
        - Algorithm: Gradient Boosting + Isotonic Calibration
        - 12 engineered features (geo, financial, operational, sentiment)
        - SHAP-style feature importance per supplier
        - Trained on 2,000 synthetic supply chain scenarios

        **🤖 LLM Advisor**
        - Powered by Anthropic Claude
        - Structured JSON output (validated schema)
        - Session-scoped rate limiting (20 calls/hour)
        - Graceful fallback if API unavailable
        """)

    with col2:
        st.markdown("""
        **🔐 Security**
        - Session-based authentication + RBAC (admin/analyst/viewer)
        - Brute-force protection (5 attempts → 5-min lockout)
        - Input validation on all CSV uploads
        - Rate limiting on uploads + LLM calls
        - HTTPS enforced via AWS ALB

        **☁️ AWS Deployment**
        - Containerised via Docker
        - AWS ECR + ECS Fargate (serverless containers)
        - Application Load Balancer with ACM SSL certificate
        - Auto-scaling policies included
        """)

    st.markdown("---")
    st.markdown("**API Integration**")
    st.code("""
import requests

response = requests.post(
  "https://your-alb-domain.com/api/score",
  json={"suppliers": [{
      "supplier_name": "Apex GmbH",
      "geo_risk_score": 0.45,
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
  }]},
  headers={"Authorization": "Bearer YOUR_TOKEN"}
)
print(response.json())
    """, language="python")


# ─────────────────────────────────────────────────────────────────
# Main App Shell
# ─────────────────────────────────────────────────────────────────
def main():
    init_auth_state()

    if not is_authenticated():
        render_login_page()
        return

    # ── Sidebar ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div style="padding: 1rem 0 0.5rem 0; text-align: center;">'
            '<div style="font-size:1.8rem;">🛡️</div>'
            '<div style="font-family:\'Space Grotesk\',sans-serif; font-size:1.1rem; '
            'font-weight:700; color:#F97316;">SupplyGuard AI</div>'
            '<div style="font-size:0.72rem; color:#475569; margin-top:2px;">'
            'Risk Intelligence Platform v1.0</div>'
            '</div>'
            '<hr style="border-color:#1E293B; margin: 0.5rem 0 1rem 0;">',
            unsafe_allow_html=True,
        )

        render_user_badge()

        nav_options = ["📊 Dashboard", "🔍 Supplier Analysis", "⚖️ Rebalancing", "📤 Export"]
        if has_permission("upload_data"):
            nav_options.insert(2, "📁 Upload Data")
        nav_options.append("ℹ️ About")

        page = st.radio("Navigation", nav_options, label_visibility="collapsed")

        # ── Data source controls ──────────────────────────────────
        st.markdown("---")
        st.markdown('<span style="color:#64748B; font-size:0.8rem;">DATA SOURCE</span>', unsafe_allow_html=True)

        # Initialise mutual-exclusive mode: "live", "demo", or "none"
        if "_data_mode" not in st.session_state:
            st.session_state["_data_mode"] = "live"

        current_mode = st.session_state["_data_mode"]

        demo_on = st.toggle("Use Demo Data", value=(current_mode == "demo"))
        live_on = st.toggle("🌐 Live Data Mode", value=(current_mode == "live"),
                            help="Fetches real signals from World Bank, NewsAPI and GDACS.")

        # Mutual exclusivity logic
        if live_on and demo_on:
            if current_mode == "demo":
                st.session_state["_data_mode"] = "live"
            else:
                st.session_state["_data_mode"] = "demo"
            st.rerun()
        elif live_on and not demo_on:
            if current_mode != "live":
                st.session_state["_data_mode"] = "live"
                st.rerun()
        elif demo_on and not live_on:
            if current_mode != "demo":
                st.session_state["_data_mode"] = "demo"
                st.rerun()
        elif not live_on and not demo_on:
            if current_mode != "none":
                st.session_state["_data_mode"] = "none"
                st.rerun()

        current_mode = st.session_state["_data_mode"]
        use_demo = current_mode in ("demo", "live")
        live_mode = current_mode == "live"

        # Clear other mode's cache when switching
        if current_mode == "live" and "_demo_df" in st.session_state:
            del st.session_state["_demo_df"]
        if current_mode == "demo" and "_live_df" in st.session_state:
            del st.session_state["_live_df"]

        if current_mode == "none":
            st.markdown('<div style="font-size:0.72rem; color:#475569; padding-left:4px;">Enable a mode to load data</div>', unsafe_allow_html=True)

        # ── Load correct dataset ──────────────────────────────────
        st.markdown("---")

        if current_mode == "none":
            df = None

        elif "uploaded_df" in st.session_state and current_mode == "demo":
            df = st.session_state["uploaded_df"]
            st.success(f"✓ Custom data: {len(df)} suppliers")

        else:
            cache_key = "_live_df" if live_mode else "_demo_df"

            # Refresh button + API STATUS header (live mode only)
            if live_mode:
                ref_col1, ref_col2 = st.columns([3, 1])
                with ref_col1:
                    st.markdown('<span style="color:#64748B; font-size:0.75rem;">API STATUS</span>', unsafe_allow_html=True)
                with ref_col2:
                    force_refresh = st.button("🔄", help="Refresh live data from all APIs", key="_refresh_btn")
                if force_refresh:
                    for k in ["_live_df", "_source_report"]:
                        if k in st.session_state:
                            del st.session_state[k]
            else:
                force_refresh = False
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<span style="color:#64748B; font-size:0.75rem;">API STATUS</span>', unsafe_allow_html=True)

            # API Status badges
            from backend.real_data_sources import get_api_status
            api_status = get_api_status()
            for _, info in api_status.items():
                configured = info["configured"]
                dot   = "🟢" if configured else "🔴"
                src   = "LIVE" if configured else "SYNTHETIC"
                color = "#22C55E" if configured else "#64748B"
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; '
                    f'align-items:center; padding:3px 0;">'
                    f'<span style="font-size:0.75rem; color:#94A3B8;">{dot} {info["label"]}</span>'
                    f'<span style="font-size:0.7rem; color:{color}; '
                    f'font-family:monospace; font-weight:600;">{src}</span>'
                    f'</div>'
                    f'<div style="font-size:0.68rem; color:#475569; '
                    f'padding-left:1.2rem; margin-bottom:2px;">{info["covers"]}</div>',
                    unsafe_allow_html=True,
                )
            if live_mode and not any(v["configured"] for v in api_status.values()):
                st.warning("⚠️ No API keys found. Add NEWS_API_KEY to .env")

            st.markdown("---")

            # Load from cache or fetch fresh
            if cache_key in st.session_state and not force_refresh:
                df = st.session_state[cache_key]
                source_report = st.session_state.get("_source_report", {})
                live_cols = [k for k, v in source_report.items() if "live" in v]
                if live_cols:
                    st.success(f"✓ {len(live_cols)}/3 signals live")
                else:
                    st.info(f"ℹ️ Demo data: {len(df)} suppliers")
            else:
                with st.spinner("Loading data..." if not live_mode else "Fetching live data..."):
                    df = load_demo_data(live_mode=live_mode)
                st.session_state[cache_key] = df
                source_report = st.session_state.get("_source_report", {})
                live_cols = [k for k, v in source_report.items() if "live" in v]
                if live_cols:
                    st.success(f"✓ {len(live_cols)}/3 signals live")
                else:
                    st.info(f"ℹ️ Demo data: {len(df)} suppliers")

        st.markdown(
            '<div style="color:#334155; font-size:0.72rem; text-align:center; margin-top:1rem;">'
            'SupplyGuard AI · Built for AWS · Secured</div>',
            unsafe_allow_html=True,
        )

    # ── Page routing ──────────────────────────────────────────────
    if df is None or (hasattr(df, '__len__') and len(df) == 0):
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("👈 Enable **Use Demo Data** in the sidebar or upload your own CSV to get started.")
        st.stop()

    if "Dashboard" in page:
        page_dashboard(df)
    elif "Supplier Analysis" in page:
        page_supplier_analysis(df)
    elif "Rebalancing" in page:
        render_rebalancing_page(df)
    elif "Upload Data" in page:
        page_upload()
    elif "Export" in page:
        page_export(df)
    elif "About" in page:
        page_about()


if __name__ == "__main__":
    main()