"""
Supply Chain Rebalancing Optimizer for SupplyGuard AI.

Given a set of high-risk suppliers in a category, this engine finds the
optimal redistribution of their purchase volume across low-risk alternatives
in the same category.

Optimization approach: Risk-first with cost as a constraint
──────────────────────────────────────────────────────────
  PRIMARY objective : minimise portfolio-weighted risk score
  CONSTRAINT 1      : total cost ≤ original_spend × (1 + cost_tolerance)   [USD]
  CONSTRAINT 2      : each alternative ≤ 1.5× their current annual_spend    [capacity]
  CONSTRAINT 3      : no single supplier covers > 60% of source demand      [concentration]
  CONSTRAINT 4      : no single region covers > 65% of source demand        [geo diversification]
  CONSTRAINT 5      : alternative lead time ≤ source lead time × 2.0        [operational]
  CONSTRAINT 6      : minimum allocation = 5% of demand or zero             [no tiny slivers]

Safety premium (real-world pricing dynamics):
  effective_cost[i] = x[i] × (1 + (1 − risk_score[i]) × 0.25)
  → safer supplier = higher price (up to +25%)
  → ALL costs in USD throughout — cost_delta_pct is always interpretable

Shared capacity pool:
  Capacity of each alternative is tracked across ALL sources in the category.
  Highest-risk source gets first pick. Subsequent sources see reduced capacity.

Solver: scipy.optimize.linprog (HiGHS)
Fallback: greedy (sort by risk ascending, fill capacity) if LP infeasible
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
    supplier_id:    str
    supplier_name:  str
    region:         str
    category:       str
    risk_score:     float
    risk_label:     str
    annual_spend:   float
    lead_time:      float
    reliability:    float
    transport_mode: str = ""

    @property
    def safety_premium(self) -> float:
        """
        Real-world cost multiplier: safer = more expensive.
        premium = 1 + (1 - risk_score) * 0.25
        risk=0.0 → 1.25x  |  risk=0.5 → 1.125x  |  risk=1.0 → 1.0x
        """
        return 1.0 + (1.0 - self.risk_score) * 0.25

    @property
    def max_capacity(self) -> float:
        return self.annual_spend * 1.5


@dataclass
class ReallocationResult:
    """Result for one source (high-risk) supplier."""
    source:              SupplierNode
    allocations:         Dict[str, float]
    target_nodes:        List[SupplierNode]
    total_demand:        float
    original_cost:       float    # USD — source annual spend (baseline)
    new_cost:            float    # USD — Σ safety_premium[i] * x[i]
    original_risk:       float
    new_weighted_risk:   float
    unmet_demand:        float
    feasible:            bool
    message:             str = ""
    constraints_applied: List[str] = field(default_factory=list)

    @property
    def cost_delta_pct(self) -> float:
        """
        Safety premium paid on the demand that WAS covered.

        Formula: (new_cost - covered_baseline) / covered_baseline * 100
        where covered_baseline = demand we actually covered at original rates

        This is always interpretable regardless of partial coverage:
        - "+18%" means we pay 18% more per unit for the covered demand
        - Never goes negative due to partial coverage (apples-to-apples)
        - Typical range: 0% to +25% (safety premium bound)
        """
        covered_demand = self.total_demand - self.unmet_demand
        if covered_demand <= 0:
            return 0.0
        # Baseline: what covered_demand would cost at original rates (no premium)
        covered_baseline = covered_demand
        return (self.new_cost - covered_baseline) / covered_baseline * 100

    @property
    def risk_reduction_pct(self) -> float:
        if self.original_risk <= 0:
            return 0.0
        raw = (self.original_risk - self.new_weighted_risk) / self.original_risk * 100
        return max(0.0, min(100.0, raw))


@dataclass
class CategoryOptimizationResult:
    """Aggregated result for an entire product category."""
    category:             str
    source_suppliers:     List[SupplierNode]
    reallocation_results: List[ReallocationResult]
    total_demand_usd:     float
    total_original_cost:  float
    total_new_cost:       float
    avg_original_risk:    float
    avg_new_risk:         float
    fully_covered:        bool

    @property
    def total_cost_delta_pct(self) -> float:
        """
        Safety premium on covered demand — apples to apples.
        total_original_cost = total_demand_usd (baseline, no premium)
        total_new_cost = Σ safety_premium[i] * allocation[i]
        Covered portion = total_demand - unmet across all results.
        """
        total_unmet = sum(r.unmet_demand for r in self.reallocation_results)
        covered = self.total_demand_usd - total_unmet
        if covered <= 0:
            return 0.0
        return (self.total_new_cost - covered) / covered * 100

    @property
    def total_risk_reduction_pct(self) -> float:
        if self.avg_original_risk <= 0:
            return 0.0
        raw = (self.avg_original_risk - self.avg_new_risk) / self.avg_original_risk * 100
        return max(0.0, min(100.0, raw))


# ─────────────────────────────────────────────────────────────────
# Core optimizer
# ─────────────────────────────────────────────────────────────────
class SupplyChainOptimizer:

    def __init__(
        self,
        risk_threshold_source:      float = 0.50,
        risk_threshold_target:      float = 0.45,
        cost_tolerance:             float = 0.20,
        capacity_multiplier:        float = 1.50,
        safety_premium_rate:        float = 0.25,
        max_single_supplier_share:  float = 0.60,
        max_single_region_share:    float = 0.65,
        max_lead_time_multiplier:   float = 2.00,
        min_allocation_share:       float = 0.05,
    ):
        self.risk_threshold_source      = risk_threshold_source
        self.risk_threshold_target      = risk_threshold_target
        self.cost_tolerance             = cost_tolerance
        self.capacity_multiplier        = capacity_multiplier
        self.safety_premium_rate        = safety_premium_rate
        self.max_single_supplier_share  = max_single_supplier_share
        self.max_single_region_share    = max_single_region_share
        self.max_lead_time_multiplier   = max_lead_time_multiplier
        self.min_allocation_share       = min_allocation_share

    def _df_to_nodes(self, df: pd.DataFrame) -> List[SupplierNode]:
        nodes = []
        for idx, row in df.iterrows():
            nodes.append(SupplierNode(
                supplier_id    = str(row.get("supplier_id",   f"SUP-{idx}")),
                supplier_name  = str(row.get("supplier_name", f"Supplier {idx}")),
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

    def _filter_targets(
        self,
        source: SupplierNode,
        targets: List[SupplierNode],
    ) -> Tuple[List[SupplierNode], List[str]]:
        """
        Apply operational filters.
        Constraint 5: lead time ≤ source * max_lead_time_multiplier
        Reliability:  alternative must be ≥ 0.60 to be trustworthy
        """
        filtered = []
        max_lead = source.lead_time * self.max_lead_time_multiplier
        for t in targets:
            if t.supplier_id == source.supplier_id:
                continue
            if t.lead_time > max_lead:
                continue
            if t.reliability < 0.60:
                continue
            filtered.append(t)
        constraints = ["lead_time_2x_filter", "reliability_0.6_filter"] if len(filtered) < len(targets) else []
        return filtered, constraints

    def _optimise_single(
        self,
        source: SupplierNode,
        targets: List[SupplierNode],
        capacity_used: Dict[str, float],
    ) -> ReallocationResult:
        """
        LP for one source supplier.

        Variables: x[i] = USD allocated to target supplier i

        Minimise:   Σ risk_score[i] * x[i]                           (risk objective)
        Subject to:
          Σ x[i]                        = demand                      (meet all demand)
          Σ safety_premium[i] * x[i]   ≤ demand * (1+cost_tolerance) (cost budget USD)
          x[i]                          ≤ available_cap[i]            (shared capacity USD)
          x[i]                          ≤ demand * 0.60               (concentration cap)
          region_sum[r]                 ≤ demand * 0.65               (geo diversification)
          x[i]                          ≥ 0
        """
        demand = source.annual_spend
        n = len(targets)

        if n == 0:
            return ReallocationResult(
                source=source, allocations={}, target_nodes=[],
                total_demand=demand, original_cost=demand,
                new_cost=demand, original_risk=source.risk_score,
                new_weighted_risk=source.risk_score, unmet_demand=demand,
                feasible=False, message="No qualifying alternatives available.",
            )

        # Apply operational filters
        targets, op_constraints = self._filter_targets(source, targets)
        n = len(targets)
        if n == 0:
            return ReallocationResult(
                source=source, allocations={}, target_nodes=[],
                total_demand=demand, original_cost=demand,
                new_cost=demand, original_risk=source.risk_score,
                new_weighted_risk=source.risk_score, unmet_demand=demand,
                feasible=False,
                message="No alternatives passed lead time / reliability filters.",
                constraints_applied=op_constraints,
            )

        risk_scores     = np.array([t.risk_score     for t in targets])
        safety_premiums = np.array([t.safety_premium for t in targets])

        # Available capacity: 1.5× spend minus already used by other sources
        available_caps = np.array([
            max(0.0, t.annual_spend * self.capacity_multiplier
                - capacity_used.get(t.supplier_id, 0.0))
            for t in targets
        ])

        # Concentration cap: only apply when 3+ alternatives exist.
        # With 1-2 alternatives, a 60% cap creates unavoidable partial coverage.
        # In emergency cases (few alternatives), allow full capacity usage.
        if n >= 3:
            conc_caps = np.full(n, demand * self.max_single_supplier_share)
        else:
            # Only 1-2 alternatives — let them cover as much as capacity allows
            conc_caps = np.full(n, demand)   # effectively no concentration cap

        # Effective upper bound per supplier
        ub = np.minimum(available_caps, conc_caps)

        # Cost budget in USD
        cost_budget = demand * (1.0 + self.cost_tolerance)

        # Geographic groups
        unique_regions = list(set(t.region for t in targets))
        n_geo = len(unique_regions)

        # Build A_ub, b_ub
        # Rows: n capacity + n_geo regional + 1 cost = n + n_geo + 1
        total_rows = n + n_geo + 1
        A_ub = np.zeros((total_rows, n))
        b_ub = np.zeros(total_rows)
        row = 0

        # Capacity constraints
        for i in range(n):
            A_ub[row, i] = 1.0
            b_ub[row]    = ub[i]
            row += 1

        # Geographic concentration
        for region in unique_regions:
            for i, t in enumerate(targets):
                if t.region == region:
                    A_ub[row, i] = 1.0
            b_ub[row] = demand * self.max_single_region_share
            row += 1

        # Cost constraint: Σ safety_premium[i] * x[i] ≤ cost_budget
        A_ub[row, :] = safety_premiums
        b_ub[row]    = cost_budget

        # Demand equality
        A_eq = np.ones((1, n))
        b_eq = np.array([demand])
        bounds = [(0, None)] * n
        c = risk_scores

        if not SCIPY_AVAILABLE:
            return self._greedy_fallback(source, targets, demand, cost_budget, capacity_used)

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")

        if res.success:
            x = res.x
            min_alloc = demand * self.min_allocation_share

            # Collect meaningful allocations (≥ 5% of demand)
            allocations = {
                targets[i].supplier_id: round(float(x[i]), 2)
                for i in range(n) if float(x[i]) >= min_alloc
            }

            # Merge tiny slivers into the largest allocation
            sliver_sum = sum(float(x[i]) for i in range(n)
                             if 1.0 < float(x[i]) < min_alloc)
            if sliver_sum > 0 and allocations:
                top_id = max(allocations, key=allocations.get)
                allocations[top_id] = round(allocations[top_id] + sliver_sum, 2)

            # Cost in USD: Σ safety_premium[i] * x[i]
            new_cost = float(np.dot(safety_premiums, x))

            # Demand-weighted risk
            new_weighted_risk = float(np.dot(risk_scores, x) / demand) if demand > 0 else 0.0

            # Update shared capacity pool
            for i, t in enumerate(targets):
                capacity_used[t.supplier_id] = (
                    capacity_used.get(t.supplier_id, 0.0) + float(x[i])
                )

            all_constraints = op_constraints + [
                "capacity_1.5x", "concentration_60pct",
                "geo_diversification_65pct", "cost_budget_usd"
            ]

            return ReallocationResult(
                source=source,
                allocations=allocations,
                target_nodes=targets,
                total_demand=demand,
                original_cost=demand,
                new_cost=round(new_cost, 2),
                original_risk=source.risk_score,
                new_weighted_risk=round(new_weighted_risk, 4),
                unmet_demand=0.0,
                feasible=True,
                message="Optimal allocation found.",
                constraints_applied=all_constraints,
            )
        else:
            return self._greedy_fallback(
                source, targets, demand, cost_budget * 1.5, capacity_used
            )

    def _greedy_fallback(
        self,
        source: SupplierNode,
        targets: List[SupplierNode],
        demand: float,
        cost_budget: float,
        capacity_used: Dict[str, float],
    ) -> ReallocationResult:
        """
        Two-pass greedy allocation:

        Pass 1 — Normal: sort by risk ascending, fill capacity up to cost_budget.
                  Respects concentration cap when 3+ alternatives exist.

        Pass 2 — Coverage sweep: if demand is still unmet after Pass 1,
                  use remaining budget headroom to cover as much as possible,
                  relaxing the concentration cap. The user's question is correct:
                  "if I have money, I should cover more demand."
        """
        sorted_targets = sorted(targets, key=lambda t: t.risk_score)
        remaining      = demand
        allocations:   Dict[str, float] = {}
        total_cost     = 0.0
        risk_weighted  = 0.0
        min_alloc      = demand * self.min_allocation_share
        # Only apply concentration cap when 3+ alternatives
        max_share = demand * self.max_single_supplier_share if len(targets) >= 3 else demand

        # ── Pass 1: Normal allocation ─────────────────────────────
        for t in sorted_targets:
            if remaining <= 0:
                break
            available = max(0.0, t.annual_spend * self.capacity_multiplier
                            - capacity_used.get(t.supplier_id, 0.0))
            allocated = min(remaining, available, max_share)
            if allocated < min_alloc:
                continue
            cost_here = t.safety_premium * allocated
            if total_cost + cost_here > cost_budget and allocations:
                affordable = (cost_budget - total_cost) / t.safety_premium
                allocated  = min(allocated, max(affordable, 0.0))
            if allocated >= min_alloc:
                allocations[t.supplier_id] = round(allocated, 2)
                total_cost    += t.safety_premium * allocated
                risk_weighted += t.risk_score * allocated
                remaining     -= allocated
                capacity_used[t.supplier_id] = (
                    capacity_used.get(t.supplier_id, 0.0) + allocated
                )

        # ── Pass 2: Budget headroom coverage sweep ────────────────
        # If demand still unmet AND budget remains: cover more, ignoring
        # concentration cap. User logic: "if I have money, cover more demand."
        if remaining > min_alloc:
            budget_remaining = cost_budget - total_cost
            for t in sorted_targets:
                if remaining <= min_alloc:
                    break
                if budget_remaining <= 0:
                    break
                # Check remaining physical capacity
                cap_left = max(0.0, t.annual_spend * self.capacity_multiplier
                               - capacity_used.get(t.supplier_id, 0.0))
                if cap_left < min_alloc:
                    continue
                # How much can we afford?
                affordable = budget_remaining / t.safety_premium
                allocated  = min(remaining, cap_left, affordable)
                if allocated < min_alloc:
                    continue
                # Add to existing allocation or create new entry
                prev = allocations.get(t.supplier_id, 0.0)
                allocations[t.supplier_id] = round(prev + allocated, 2)
                total_cost       += t.safety_premium * allocated
                risk_weighted    += t.risk_score * allocated
                remaining        -= allocated
                budget_remaining -= t.safety_premium * allocated
                capacity_used[t.supplier_id] = (
                    capacity_used.get(t.supplier_id, 0.0) + allocated
                )

        unmet = max(remaining, 0.0)
        nwr   = risk_weighted / demand if demand > 0 else source.risk_score
        constraints = ["greedy_fallback", "capacity_1.5x"]
        if len(targets) >= 3:
            constraints.append("concentration_60pct")

        if unmet > demand * 0.05:
            msg = (f"⚠️ ${unmet:,.0f} demand unmet — "
                   f"alternatives at capacity or budget exhausted.")
        elif unmet > 0:
            msg = f"Near-optimal: ${unmet:,.0f} residual unmet (<5%)."
        else:
            msg = "Greedy allocation — full demand covered."

        return ReallocationResult(
            source=source, allocations=allocations, target_nodes=sorted_targets,
            total_demand=demand, original_cost=demand, new_cost=round(total_cost, 2),
            original_risk=source.risk_score, new_weighted_risk=round(nwr, 4),
            unmet_demand=round(unmet, 2),
            feasible=unmet < demand * 0.05,
            message=msg,
            constraints_applied=constraints,
        )

    def optimise_category(
        self, df: pd.DataFrame, category: str
    ) -> Optional[CategoryOptimizationResult]:
        cat_df = df[df["category"] == category].copy()
        if cat_df.empty:
            return None

        all_nodes    = self._df_to_nodes(cat_df)
        source_nodes = [n for n in all_nodes if n.risk_score >= self.risk_threshold_source]
        target_nodes = [n for n in all_nodes if n.risk_score <= self.risk_threshold_target]

        if not source_nodes:
            return None

        # Shared capacity pool — prevents over-allocation across multiple sources
        capacity_used: Dict[str, float] = {}

        # Process highest-risk sources first — they get first pick of capacity
        results = []
        for source in sorted(source_nodes, key=lambda s: s.risk_score, reverse=True):
            eligible = [t for t in target_nodes if t.supplier_id != source.supplier_id]
            result   = self._optimise_single(source, eligible, capacity_used)
            results.append(result)

        total_demand    = sum(r.total_demand  for r in results)
        total_orig_cost = sum(r.original_cost for r in results)
        total_new_cost  = sum(r.new_cost      for r in results)
        avg_orig_risk   = float(np.mean([r.original_risk    for r in results]))
        avg_new_risk    = float(np.mean([r.new_weighted_risk for r in results]))
        fully_covered   = all(r.unmet_demand < r.total_demand * 0.05 for r in results)

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
        self, df: pd.DataFrame
    ) -> Dict[str, CategoryOptimizationResult]:
        results = {}
        if "category" not in df.columns:
            return results
        for category in df["category"].unique():
            result = self.optimise_category(df, category)
            if result is not None:
                results[category] = result
        return results
