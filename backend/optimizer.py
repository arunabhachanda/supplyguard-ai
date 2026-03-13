"""
Supply Chain Rebalancing Optimizer for SupplyGuard AI.

Given a set of High/Medium risk suppliers in a category, this engine
finds the optimal redistribution of their purchase volume across
Low/Medium risk alternatives in the same category.

Optimization approach: Risk-first with cost as a constraint
  - PRIMARY objective: minimise portfolio-weighted risk score
  - CONSTRAINT: total reallocation cost ≤ original cost × (1 + cost_tolerance)
  - CAPACITY: each alternative capped at 1.5× their current annual_spend_usd
  - SAFETY PREMIUM: safer suppliers cost slightly more
      cost_rate = (spend/lead_time) × (1 + (1 - risk_score) × 0.25)

Solver: scipy.optimize.linprog (HiGHS backend — fast, no extra install)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────
@dataclass
class SupplierNode:
    """Represents one supplier in the optimization."""
    supplier_id:   str
    supplier_name: str
    region:        str
    category:      str
    risk_score:    float
    risk_label:    str
    annual_spend:  float          # USD — used as demand (source) or capacity (target)
    lead_time:     float          # days — used in cost rate calculation
    reliability:   float          # 0–1
    transport_mode: str = ""

    @property
    def cost_rate(self) -> float:
        """
        Effective cost rate proxy (USD / day of lead time).
        Safer suppliers carry a premium of up to 25%.
        safety_premium = 1 + (1 - risk_score) * 0.25
        """
        base = self.annual_spend / max(self.lead_time, 1.0)
        safety_premium = 1.0 + (1.0 - self.risk_score) * 0.25
        return base * safety_premium

    @property
    def max_capacity(self) -> float:
        """Maximum we can allocate to this supplier = 1.5× current spend."""
        return self.annual_spend * 1.5


@dataclass
class ReallocationResult:
    """Result for one source (high-risk) supplier."""
    source:             SupplierNode
    allocations:        Dict[str, float]   # target supplier_id → USD allocated
    target_nodes:       List[SupplierNode]
    total_demand:       float
    original_cost:      float
    new_cost:           float
    original_risk:      float              # source supplier risk score
    new_weighted_risk:  float              # weighted avg risk of new allocation
    unmet_demand:       float              # if no feasible solution covers all demand
    feasible:           bool
    message:            str = ""

    @property
    def cost_delta_pct(self) -> float:
        if self.original_cost == 0:
            return 0.0
        return (self.new_cost - self.original_cost) / self.original_cost * 100

    @property
    def risk_reduction_pct(self) -> float:
        if self.original_risk == 0:
            return 0.0
        return (self.original_risk - self.new_weighted_risk) / self.original_risk * 100


@dataclass
class CategoryOptimizationResult:
    """Aggregated result for an entire category."""
    category:               str
    source_suppliers:       List[SupplierNode]
    reallocation_results:   List[ReallocationResult]
    total_demand_usd:       float
    total_original_cost:    float
    total_new_cost:         float
    avg_original_risk:      float
    avg_new_risk:           float
    fully_covered:          bool           # all demand met by low-risk alternatives

    @property
    def total_cost_delta_pct(self) -> float:
        if self.total_original_cost == 0:
            return 0.0
        return (self.total_new_cost - self.total_original_cost) / self.total_original_cost * 100

    @property
    def total_risk_reduction_pct(self) -> float:
        if self.avg_original_risk == 0:
            return 0.0
        return (self.avg_original_risk - self.avg_new_risk) / self.avg_original_risk * 100


# ─────────────────────────────────────────────────────────────────
# Core optimizer
# ─────────────────────────────────────────────────────────────────
class SupplyChainOptimizer:
    """
    Optimizes supply chain rebalancing using linear programming.

    Parameters
    ----------
    risk_threshold_source : float
        Suppliers with risk_score >= this are considered for reallocation.
        Default 0.50 (catches both High and upper-Medium risk suppliers).
    risk_threshold_target : float
        Only suppliers with risk_score <= this are considered as alternatives.
        Default 0.45 (Low + lower-Medium risk).
    cost_tolerance : float
        Maximum allowed cost increase as a fraction (e.g. 0.20 = 20% increase).
    capacity_multiplier : float
        How much we can scale up an alternative supplier. Default 1.5.
    safety_premium_rate : float
        Premium per unit of safety (1 - risk_score). Default 0.25.
    """

    def __init__(
        self,
        risk_threshold_source:  float = 0.50,
        risk_threshold_target:  float = 0.45,
        cost_tolerance:         float = 0.20,
        capacity_multiplier:    float = 1.50,
        safety_premium_rate:    float = 0.25,
    ):
        self.risk_threshold_source  = risk_threshold_source
        self.risk_threshold_target  = risk_threshold_target
        self.cost_tolerance         = cost_tolerance
        self.capacity_multiplier    = capacity_multiplier
        self.safety_premium_rate    = safety_premium_rate

    def _df_to_nodes(self, df: pd.DataFrame) -> List[SupplierNode]:
        """Convert DataFrame rows to SupplierNode objects."""
        nodes = []
        for _, row in df.iterrows():
            nodes.append(SupplierNode(
                supplier_id    = str(row.get("supplier_id",   f"SUP-{_}")),
                supplier_name  = str(row.get("supplier_name", f"Supplier {_}")),
                region         = str(row.get("region",        "Unknown")),
                category       = str(row.get("category",      "Unknown")),
                risk_score     = float(row.get("risk_score",  0.5)),
                risk_label     = str(row.get("risk_label",    "Medium")),
                annual_spend   = float(row.get("annual_spend_usd", 500_000)),
                lead_time      = float(row.get("lead_time_days",   30)),
                reliability    = float(row.get("supplier_reliability_score", 0.7)),
                transport_mode = str(row.get("transport_mode", "")),
            ))
        return nodes

    def _effective_cost_rate(self, node: SupplierNode) -> float:
        """Cost rate with safety premium applied."""
        base = node.annual_spend / max(node.lead_time, 1.0)
        premium = 1.0 + (1.0 - node.risk_score) * self.safety_premium_rate
        return base * premium

    def _optimise_single(
        self,
        source: SupplierNode,
        targets: List[SupplierNode],
    ) -> ReallocationResult:
        """
        Solve the LP for one source supplier.

        Variables: x[i] = USD allocated to target supplier i

        Minimise:   Σ risk_score[i] * x[i]          (risk objective)
        Subject to:
          Σ x[i]                  = demand           (meet all demand)
          x[i]                   ≤ capacity[i]       (capacity cap)
          Σ cost_rate[i] * x[i]  ≤ budget_cap        (cost constraint)
          x[i]                   ≥ 0
        """
        demand      = source.annual_spend
        budget_cap  = source.annual_spend * (1.0 + self.cost_tolerance)
        n           = len(targets)

        if n == 0:
            return ReallocationResult(
                source=source, allocations={}, target_nodes=[],
                total_demand=demand, original_cost=source.annual_spend,
                new_cost=0.0, original_risk=source.risk_score,
                new_weighted_risk=source.risk_score, unmet_demand=demand,
                feasible=False,
                message="No low-risk alternatives available in this category.",
            )

        # Coefficients
        risk_scores  = np.array([t.risk_score for t in targets])
        cost_rates   = np.array([self._effective_cost_rate(t) for t in targets])
        capacities   = np.array([t.annual_spend * self.capacity_multiplier for t in targets])

        # Normalise cost_rates so cost constraint is in USD terms
        # cost_rate * x = effective_cost for that allocation
        # We want: Σ (cost_rate[i] / base_cost_rate[i]) * x[i] * base_spend ≤ budget_cap
        # Simplified: treat cost_rate[i] * x[i] as relative cost contribution
        # Scale so Σ cost_rate[i] * x[i] = demand means same total cost as original

        # Objective: minimise risk-weighted allocation
        c = risk_scores  # shape (n,)

        # Inequality constraints (≤):
        # 1. cost constraint: Σ (cost_rate[i] * x[i]) ≤ budget_cap
        #    cost_rate normalised: effective_cost = cost_rate[i] * x[i] / sum(cost_rates) * demand
        #    Simplified: use relative cost ratio vs average
        avg_cost_rate = float(np.mean(cost_rates)) if np.mean(cost_rates) > 0 else 1.0
        norm_cost     = cost_rates / avg_cost_rate   # relative cost multipliers

        A_ub = np.zeros((n + 1, n))
        b_ub = np.zeros(n + 1)

        # Capacity constraints: x[i] ≤ capacity[i]
        for i in range(n):
            A_ub[i, i] = 1.0
            b_ub[i]    = capacities[i]

        # Cost constraint: Σ norm_cost[i] * x[i] ≤ budget_cap
        A_ub[n, :] = norm_cost
        b_ub[n]    = budget_cap

        # Equality constraint: Σ x[i] = demand
        A_eq = np.ones((1, n))
        b_eq = np.array([demand])

        bounds = [(0, None)] * n

        if not SCIPY_AVAILABLE:
            # Greedy fallback if scipy not installed
            return self._greedy_fallback(source, targets, demand, budget_cap)

        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if result.success:
            x = result.x
            allocations = {
                targets[i].supplier_id: round(float(x[i]), 2)
                for i in range(n) if x[i] > 1.0   # ignore negligible allocations
            }
            new_cost = float(np.sum(norm_cost * x))
            new_weighted_risk = (
                float(np.dot(risk_scores, x) / demand) if demand > 0 else 0.0
            )
            return ReallocationResult(
                source=source,
                allocations=allocations,
                target_nodes=targets,
                total_demand=demand,
                original_cost=float(source.annual_spend),
                new_cost=new_cost,
                original_risk=source.risk_score,
                new_weighted_risk=round(new_weighted_risk, 4),
                unmet_demand=0.0,
                feasible=True,
                message="Optimal allocation found.",
            )
        else:
            # LP infeasible — try relaxing cost constraint with greedy
            return self._greedy_fallback(source, targets, demand, budget_cap * 1.5)

    def _greedy_fallback(
        self,
        source: SupplierNode,
        targets: List[SupplierNode],
        demand: float,
        budget_cap: float,
    ) -> ReallocationResult:
        """
        Simple greedy fallback: sort by risk_score ascending, fill capacity.
        Used when scipy is unavailable or LP is infeasible.
        """
        sorted_targets = sorted(targets, key=lambda t: t.risk_score)
        remaining      = demand
        allocations    = {}
        total_cost     = 0.0
        risk_weighted  = 0.0

        for t in sorted_targets:
            if remaining <= 0:
                break
            cap       = t.annual_spend * self.capacity_multiplier
            allocated = min(remaining, cap)
            cost      = self._effective_cost_rate(t) * allocated / max(t.lead_time, 1)

            if total_cost + cost > budget_cap and allocations:
                # Would breach budget — allocate only what fits
                affordable = (budget_cap - total_cost) / (
                    self._effective_cost_rate(t) / max(t.lead_time, 1)
                )
                allocated = min(allocated, max(affordable, 0))

            if allocated > 1.0:
                allocations[t.supplier_id] = round(allocated, 2)
                total_cost   += cost
                risk_weighted += t.risk_score * allocated
                remaining    -= allocated

        unmet = max(remaining, 0.0)
        nwr   = risk_weighted / demand if demand > 0 else source.risk_score

        return ReallocationResult(
            source=source,
            allocations=allocations,
            target_nodes=sorted_targets,
            total_demand=demand,
            original_cost=float(source.annual_spend),
            new_cost=round(total_cost, 2),
            original_risk=source.risk_score,
            new_weighted_risk=round(nwr, 4),
            unmet_demand=round(unmet, 2),
            feasible=unmet < demand * 0.05,   # feasible if <5% unmet
            message="Greedy allocation (LP fallback)." if unmet == 0
                    else f"⚠️ ${unmet:,.0f} demand unmet — insufficient low-risk capacity.",
        )

    def optimise_category(
        self,
        df: pd.DataFrame,
        category: str,
    ) -> Optional[CategoryOptimizationResult]:
        """
        Run full optimization for all high-risk suppliers in a category.

        Parameters
        ----------
        df       : scored supplier DataFrame (output of predict_risk)
        category : category name to optimize

        Returns
        -------
        CategoryOptimizationResult or None if no high-risk suppliers
        """
        cat_df = df[df["category"] == category].copy()
        if cat_df.empty:
            return None

        all_nodes    = self._df_to_nodes(cat_df)
        source_nodes = [n for n in all_nodes if n.risk_score >= self.risk_threshold_source]
        target_nodes = [n for n in all_nodes if n.risk_score <= self.risk_threshold_target]

        if not source_nodes:
            return None

        results = []
        for source in source_nodes:
            # Exclude the source itself from targets
            eligible_targets = [t for t in target_nodes if t.supplier_id != source.supplier_id]
            result = self._optimise_single(source, eligible_targets)
            results.append(result)

        total_demand       = sum(r.total_demand for r in results)
        total_orig_cost    = sum(r.original_cost for r in results)
        total_new_cost     = sum(r.new_cost for r in results)
        avg_orig_risk      = float(np.mean([r.original_risk for r in results]))
        avg_new_risk       = float(np.mean([r.new_weighted_risk for r in results]))
        fully_covered      = all(r.unmet_demand < r.total_demand * 0.05 for r in results)

        return CategoryOptimizationResult(
            category=category,
            source_suppliers=source_nodes,
            reallocation_results=results,
            total_demand_usd=total_demand,
            total_original_cost=total_orig_cost,
            total_new_cost=total_new_cost,
            avg_original_risk=avg_orig_risk,
            avg_new_risk=avg_new_risk,
            fully_covered=fully_covered,
        )

    def optimise_all_categories(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, CategoryOptimizationResult]:
        """
        Run optimization across all categories that have high-risk suppliers.

        Returns dict: category_name → CategoryOptimizationResult
        """
        results = {}
        if "category" not in df.columns:
            return results

        for category in df["category"].unique():
            result = self.optimise_category(df, category)
            if result is not None:
                results[category] = result

        return results
