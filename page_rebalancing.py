"""
page_rebalancing.py — Supply Chain Rebalancing UI page for SupplyGuard AI.

Renders the ⚖️ Supply Rebalancing page inside the Streamlit app.
Import and call render_rebalancing_page(df) from app.py.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.optimizer import SupplyChainOptimizer, CategoryOptimizationResult, ReallocationResult

RISK_COLORS = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#22C55E"}


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def _fmt_usd(value: float) -> str:
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value/1_000:.0f}K"
    return f"${value:.0f}"


def _risk_badge(score: float, label: str) -> str:
    color = "#EF4444" if label == "High" else ("#F59E0B" if label == "Medium" else "#22C55E")
    return (
        f'<span style="background:{color}20; color:{color}; border:1px solid {color}; '
        f'border-radius:6px; padding:2px 10px; font-weight:600; font-size:0.8rem;">'
        f'{label} ({score:.3f})</span>'
    )


def _delta_chip(value: float, suffix: str = "%", invert: bool = False) -> str:
    """Green if improvement, red if worse. invert=True for cost (increase = bad)."""
    is_good = (value < 0) if invert else (value > 0)
    color   = "#22C55E" if is_good else "#EF4444"
    arrow   = "▼" if value < 0 else "▲"
    return (
        f'<span style="color:{color}; font-weight:600; font-family:monospace;">'
        f'{arrow} {abs(value):.1f}{suffix}</span>'
    )


# ─────────────────────────────────────────────────────────────────
# Sub-renders
# ─────────────────────────────────────────────────────────────────
def _render_single_reallocation(
    result: ReallocationResult,
    all_nodes_by_id: dict,
):
    """Render one source supplier's reallocation card."""
    s = result.source
    risk_color = RISK_COLORS.get(s.risk_label, "#94A3B8")

    # Header card
    st.markdown(
        f'<div class="sg-card sg-card-accent" style="margin-bottom:0.5rem;">'
        f'<div style="display:flex; justify-content:space-between; align-items:flex-start;">'
        f'<div>'
        f'<div style="font-size:1rem; font-weight:700; color:#E2E8F0;">{s.supplier_name}</div>'
        f'<div style="font-size:0.8rem; color:#64748B; margin-top:2px;">'
        f'{s.region} · {s.transport_mode}</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'{_risk_badge(s.risk_score, s.risk_label)}<br>'
        f'<span style="font-size:0.8rem; color:#94A3B8;">Demand: '
        f'<strong style="color:#F97316;">{_fmt_usd(result.total_demand)}</strong>/yr</span>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not result.feasible and not result.allocations:
        st.warning(f"⚠️ {result.message}")
        return

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Original Risk",
        f"{result.original_risk:.3f}",
    )
    m2.metric(
        "New Avg Risk",
        f"{result.new_weighted_risk:.3f}",
        delta=f"{result.risk_reduction_pct:.1f}% reduction",
        delta_color="inverse",
    )
    m3.metric(
        "Demand Covered",
        f"{_fmt_usd(result.total_demand - result.unmet_demand)}",
        delta=f"{(1 - result.unmet_demand/result.total_demand)*100:.0f}%" if result.total_demand > 0 else "—",
    )
    m4.metric(
        "Cost Impact",
        f"{result.cost_delta_pct:+.1f}%",
        delta="vs original spend",
        delta_color="inverse" if result.cost_delta_pct > 0 else "normal",
    )

    if not result.allocations:
        st.info("No allocation possible within cost constraints.")
        return

    # Allocation breakdown table
    st.markdown("**Suggested Allocation**")
    rows = []
    for tid, amount in result.allocations.items():
        node = all_nodes_by_id.get(tid)
        if node:
            pct = amount / result.total_demand * 100 if result.total_demand > 0 else 0
            rows.append({
                "Supplier": node.supplier_name,
                "Region": node.region,
                "Risk Score": round(node.risk_score, 3),
                "Risk Label": node.risk_label if node.risk_label and node.risk_label != "nan" else ("High" if node.risk_score >= 0.65 else ("Medium" if node.risk_score >= 0.35 else "Low")),
                "Allocation (USD)": round(amount, 0),
                "Share (%)": round(pct, 1),
                "Safety Premium": f"+{(1 - node.risk_score) * 25:.1f}%",
                "Capacity Used": f"{amount / (node.annual_spend * 1.5) * 100:.0f}%",
            })

    if rows:
        alloc_df = pd.DataFrame(rows).sort_values("Risk Score")
        st.dataframe(
            alloc_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Risk Score": st.column_config.ProgressColumn(
                    "Risk Score", min_value=0, max_value=1, format="%.3f"
                ),
                "Allocation (USD)": st.column_config.NumberColumn(
                    "Allocation (USD)", format="$%,.0f"
                ),
                "Share (%)": st.column_config.ProgressColumn(
                    "Share (%)", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )

    # Allocation pie chart
    if len(rows) > 1:
        pie_df = pd.DataFrame(rows)
        fig = px.pie(
            pie_df,
            names="Supplier",
            values="Allocation (USD)",
            color="Risk Label",
            color_discrete_map=RISK_COLORS,
            hole=0.5,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF",
            font=dict(color="#FFFFFF", size=12),
            height=220,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True,
            legend=dict(font=dict(color="#FFFFFF", size=12)),
        )
        fig.update_traces(textfont_color="#FFFFFF", textfont_size=13)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if result.unmet_demand > 0:
        st.warning(
            f"⚠️ {_fmt_usd(result.unmet_demand)} of demand cannot be covered within "
            f"the cost tolerance. Consider raising the cost tolerance slider or adding "
            f"more low-risk suppliers to this category."
        )


def _render_category_summary(result: CategoryOptimizationResult):
    """Render the top summary bar for one category."""
    covered_icon = "✅" if result.fully_covered else "⚠️"
    risk_delta   = result.avg_original_risk - result.avg_new_risk
    cost_delta   = result.total_cost_delta_pct

    st.markdown(
        f'<div class="sg-card" style="margin-bottom:1rem;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; '
        f'flex-wrap:wrap; gap:1rem;">'
        # Left: category name + coverage
        f'<div>'
        f'<div style="font-size:1.1rem; font-weight:700; color:#F97316;">{result.category}</div>'
        f'<div style="font-size:0.8rem; color:#64748B; margin-top:2px;">'
        f'{len(result.source_suppliers)} supplier(s) to rebalance · '
        f'Total demand: {_fmt_usd(result.total_demand_usd)}/yr · '
        f'{covered_icon} {"Fully covered" if result.fully_covered else "Partially covered"}'
        f'</div>'
        f'</div>'
        # Right: KPI chips
        f'<div style="display:flex; gap:1rem; flex-wrap:wrap;">'
        f'<div style="text-align:center;">'
        f'<div style="font-size:0.72rem; color:#64748B;">Avg Risk Before</div>'
        f'<div style="font-size:1.2rem; font-weight:700; color:#EF4444;">'
        f'{result.avg_original_risk:.3f}</div>'
        f'</div>'
        f'<div style="font-size:1.5rem; color:#334155; align-self:center;">→</div>'
        f'<div style="text-align:center;">'
        f'<div style="font-size:0.72rem; color:#64748B;">Avg Risk After</div>'
        f'<div style="font-size:1.2rem; font-weight:700; color:#22C55E;">'
        f'{result.avg_new_risk:.3f}</div>'
        f'</div>'
        f'<div style="text-align:center;">'
        f'<div style="font-size:0.72rem; color:#64748B;">Risk Reduction</div>'
        f'<div style="font-size:1.2rem; font-weight:700; color:#22C55E;">'
        f'▼ {result.total_risk_reduction_pct:.1f}%</div>'
        f'</div>'
        f'<div style="text-align:center;">'
        f'<div style="font-size:0.72rem; color:#64748B;">Cost Impact</div>'
        f'<div style="font-size:1.2rem; font-weight:700; '
        f'color:{"#EF4444" if cost_delta > 0 else "#22C55E"};">'
        f'{"▲" if cost_delta > 0 else "▼"} {abs(cost_delta):.1f}%</div>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# Main page renderer
# ─────────────────────────────────────────────────────────────────
def render_rebalancing_page(df: pd.DataFrame):
    """Main entry point — call this from app.py."""
    st.markdown("## ⚖️ Supply Chain Rebalancing")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        '<div class="sg-card sg-card-accent">'
        'Identifies high-risk suppliers and computes the optimal redistribution of their '
        'purchase volume across low-risk alternatives — minimising portfolio risk while '
        'keeping cost increase within your tolerance. Safer suppliers carry a small cost '
        'premium (up to +25%) reflecting real-world pricing dynamics.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Controls ─────────────────────────────────────────────────
    st.markdown("### ⚙️ Optimization Settings")
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)

    with ctrl1:
        cost_tolerance = st.slider(
            "Max Cost Increase (%)",
            min_value=0, max_value=50, value=20, step=5,
            help="Maximum allowed cost increase when reallocating to safer suppliers.",
        ) / 100.0

    with ctrl2:
        risk_threshold_source = st.slider(
            "Rebalance suppliers with risk ≥",
            min_value=0.30, max_value=0.80, value=0.50, step=0.05,
            help="Suppliers above this risk score will be rebalanced.",
        )

    with ctrl3:
        risk_threshold_target = st.slider(
            "Use alternatives with risk ≤",
            min_value=0.20, max_value=0.60, value=0.45, step=0.05,
            help="Only suppliers below this risk score are considered as alternatives.",
        )

    with ctrl4:
        categories = sorted(df["category"].unique().tolist()) if "category" in df.columns else []
        view_mode = st.selectbox(
            "View Mode",
            ["All Categories", "Single Category"],
        )

    selected_category = None
    if view_mode == "Single Category" and categories:
        selected_category = st.selectbox("Select Category", categories)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Run optimization ─────────────────────────────────────────
    if st.button("🚀 Run Optimization", type="primary"):
        optimizer = SupplyChainOptimizer(
            risk_threshold_source=risk_threshold_source,
            risk_threshold_target=risk_threshold_target,
            cost_tolerance=cost_tolerance,
        )

        with st.spinner("Running optimization..."):
            if view_mode == "Single Category" and selected_category:
                cat_results = {selected_category: optimizer.optimise_category(df, selected_category)}
                cat_results = {k: v for k, v in cat_results.items() if v is not None}
            else:
                cat_results = optimizer.optimise_all_categories(df)

        st.session_state["_rebal_results"]  = cat_results
        st.session_state["_rebal_df"]       = df.copy()

    # ── Display results ───────────────────────────────────────────
    if "_rebal_results" not in st.session_state:
        # Prompt to run
        st.markdown(
            '<div style="text-align:center; padding:3rem; color:#475569;">'
            '<div style="font-size:2rem;">⚖️</div>'
            '<div style="font-size:1rem; margin-top:0.5rem;">Adjust settings above and click '
            '<strong style="color:#F97316;">Run Optimization</strong> to see rebalancing recommendations.'
            '</div></div>',
            unsafe_allow_html=True,
        )
        return

    cat_results: Dict[str, CategoryOptimizationResult] = st.session_state["_rebal_results"]
    ref_df = st.session_state["_rebal_df"]

    if not cat_results:
        st.info(
            "✅ No suppliers exceed the risk threshold for rebalancing. "
            "Try lowering the 'Rebalance suppliers with risk ≥' slider."
        )
        return

    # Build lookup dict: supplier_id → SupplierNode
    all_nodes_by_id = {}
    for res in cat_results.values():
        for r in res.reallocation_results:
            all_nodes_by_id[r.source.supplier_id] = r.source
            for t in r.target_nodes:
                all_nodes_by_id[t.supplier_id] = t

    # ── Portfolio-level summary ───────────────────────────────────
    st.markdown("### 📊 Portfolio Rebalancing Summary")

    total_demand    = sum(r.total_demand_usd for r in cat_results.values())
    total_orig_cost = sum(r.total_original_cost for r in cat_results.values())
    total_new_cost  = sum(r.total_new_cost for r in cat_results.values())
    avg_orig_risk   = float(np.mean([r.avg_original_risk for r in cat_results.values()]))
    avg_new_risk    = float(np.mean([r.avg_new_risk for r in cat_results.values()]))
    n_sources       = sum(len(r.source_suppliers) for r in cat_results.values())
    n_covered       = sum(1 for r in cat_results.values() if r.fully_covered)

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Categories Affected", len(cat_results))
    p2.metric("Suppliers to Rebalance", n_sources)
    p3.metric("Total Demand", _fmt_usd(total_demand))
    p4.metric(
        "Avg Risk Reduction",
        f"{((avg_orig_risk - avg_new_risk) / avg_orig_risk * 100):.1f}%",
        delta="improvement",
        delta_color="inverse",
    )
    p5.metric(
        "Cost Impact",
        f"{((total_new_cost - total_orig_cost) / total_orig_cost * 100) if total_orig_cost > 0 else 0:+.1f}%",
        delta="vs original spend",
        delta_color="inverse",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Before vs After risk comparison chart
    before_after_data = []
    for cat, res in cat_results.items():
        before_after_data.append({"Category": cat, "Stage": "Before", "Avg Risk": res.avg_original_risk})
        before_after_data.append({"Category": cat, "Stage": "After",  "Avg Risk": res.avg_new_risk})

    ba_df = pd.DataFrame(before_after_data)
    fig_ba = px.bar(
        ba_df, x="Category", y="Avg Risk", color="Stage", barmode="group",
        color_discrete_map={"Before": "#EF4444", "After": "#22C55E"},
    )
    fig_ba.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1", height=280, margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(gridcolor="#1E293B"), yaxis=dict(gridcolor="#1E293B", range=[0, 1]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_ba, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # ── Per-category breakdown ────────────────────────────────────
    st.markdown("### 📦 Category-by-Category Breakdown")

    for cat, cat_result in cat_results.items():
        _render_category_summary(cat_result)

        with st.expander(f"📋 View detailed allocations for {cat}", expanded=False):
            for realloc_result in cat_result.reallocation_results:
                _render_single_reallocation(realloc_result, all_nodes_by_id)
                st.markdown("<br>", unsafe_allow_html=True)

    # ── Export rebalancing plan ───────────────────────────────────
    st.markdown("---")
    st.markdown("### ⬇️ Export Rebalancing Plan")

    export_rows = []
    for cat, cat_result in cat_results.items():
        for r in cat_result.reallocation_results:
            for tid, amount in r.allocations.items():
                tnode = all_nodes_by_id.get(tid)
                if tnode:
                    export_rows.append({
                        "Category":             cat,
                        "From Supplier":        r.source.supplier_name,
                        "From Risk Score":      r.source.risk_score,
                        "From Region":          r.source.region,
                        "To Supplier":          tnode.supplier_name,
                        "To Risk Score":        tnode.risk_score,
                        "To Region":            tnode.region,
                        "Allocation USD":       round(amount, 2),
                        "Share of Demand (%)":  round(amount / r.total_demand * 100, 1) if r.total_demand > 0 else 0,
                        "Safety Premium (%)":   round((1 - tnode.risk_score) * 25, 1),
                        "Risk Reduction (%)":   round(r.risk_reduction_pct, 1),
                        "Cost Delta (%)":       round(r.cost_delta_pct, 1),
                    })

    if export_rows:
        export_df  = pd.DataFrame(export_rows)
        export_csv = export_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Rebalancing Plan (CSV)",
            data=export_csv,
            file_name="supplyguard_rebalancing_plan.csv",
            mime="text/csv",
            type="primary",
        )
        st.dataframe(export_df, use_container_width=True, hide_index=True, height=300)
