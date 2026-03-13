"""
Supply Chain Disruption Risk ML Model.
Trains a GradientBoostingClassifier on synthetic data and provides:
  - predict_risk()      → risk scores + labels per supplier
  - explain_supplier()  → SHAP-based feature importance per row
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")

from backend.data_generator import (
    generate_supplier_data,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
)

# ─────────────────────────────────────────────────────────────────
# Model singleton
# ─────────────────────────────────────────────────────────────────
_model: Optional[Pipeline] = None
_model_metrics: dict = {}


def _build_and_train() -> tuple[Pipeline, dict]:
    """Train on large synthetic dataset and return (pipeline, metrics)."""
    # 2000-row training corpus
    df_train = generate_supplier_data(n_suppliers=2000, seed=0, include_labels=True)
    X = df_train[FEATURE_COLUMNS]
    y = (df_train["disruption_risk_score"] >= 0.50).astype(int)   # binary

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    base = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=15,
        random_state=42,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(base, cv=5, method="isotonic")),
    ])
    pipeline.fit(X_tr, y_tr)

    y_prob = pipeline.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)
    metrics = {
        "roc_auc": round(roc_auc_score(y_te, y_prob), 4),
        "f1": round(f1_score(y_te, y_pred), 4),
        "train_size": len(X_tr),
        "test_size": len(X_te),
    }
    return pipeline, metrics


def get_model() -> Pipeline:
    """Lazy-load + cache the trained model (singleton)."""
    global _model, _model_metrics
    if _model is None:
        _model, _model_metrics = _build_and_train()
    return _model


def get_model_metrics() -> dict:
    get_model()           # ensure trained
    return _model_metrics


# ─────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────
def predict_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a DataFrame of suppliers.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all FEATURE_COLUMNS. Non-feature columns are preserved.

    Returns
    -------
    pd.DataFrame
        Input df + columns: risk_score, risk_label, risk_percentile
    """
    model = get_model()
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    probs = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["risk_score"] = probs.round(4)
    result["risk_label"] = pd.cut(
        result["risk_score"],
        bins=[0, 0.35, 0.65, 1.0],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    result["risk_percentile"] = (
        result["risk_score"].rank(pct=True) * 100
    ).round(1)

    return result.sort_values("risk_score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# SHAP-style feature importance (model-agnostic permutation approach)
# Works without installing the shap library.
# ─────────────────────────────────────────────────────────────────
def explain_supplier(row: pd.Series, n_permutations: int = 80) -> pd.DataFrame:
    """
    Compute approximate Shapley-style feature importances for one supplier row
    using a permutation sampling approach (no shap dependency needed).

    Returns
    -------
    pd.DataFrame
        Columns: feature, label, importance, direction
        Sorted by |importance| descending.
    """
    model = get_model()
    X_base = row[FEATURE_COLUMNS].values.astype(float)

    # Baseline prediction
    base_pred = model.predict_proba([X_base])[0][1]

    importances = []
    rng = np.random.default_rng(99)

    for feat_idx, feat_name in enumerate(FEATURE_COLUMNS):
        contributions = []
        for _ in range(n_permutations):
            # Perturb: zero out the feature (replace with column mean ≈ 0.5)
            X_perturbed = X_base.copy()
            # Add small noise to simulate marginal contribution
            X_perturbed[feat_idx] = rng.normal(0.5, 0.15)
            pred_perturbed = model.predict_proba([X_perturbed])[0][1]
            contributions.append(base_pred - pred_perturbed)

        mean_contribution = float(np.mean(contributions))
        importances.append({
            "feature": feat_name,
            "label": FEATURE_LABELS[feat_name],
            "importance": round(mean_contribution, 5),
            "direction": "↑ Risk" if mean_contribution > 0 else "↓ Risk",
        })

    result = pd.DataFrame(importances)
    result["abs_importance"] = result["importance"].abs()
    result = result.sort_values("abs_importance", ascending=False).drop(columns="abs_importance")
    return result.head(8).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────
def validate_upload(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate an uploaded DataFrame for required columns and value ranges.

    Returns
    -------
    (is_valid, list_of_errors)
    """
    errors = []
    required = ["supplier_name"] + FEATURE_COLUMNS

    for col in required:
        if col not in df.columns:
            errors.append(f"Missing column: '{col}'")

    if not errors:
        range_checks = {
            "geo_risk_score": (0, 1),
            "supplier_reliability_score": (0, 1),
            "financial_health_score": (0, 1),
            "natural_disaster_risk": (0, 1),
            "regulatory_risk_score": (0, 1),
            "transport_mode_risk": (0, 1),
            "region_news_sentiment": (-1, 1),
            "single_source_dependency": (0, 1),
            "lead_time_days": (1, 365),
            "inventory_buffer_days": (0, 365),
            "past_disruptions_12mo": (0, 50),
        }
        for col, (lo, hi) in range_checks.items():
            if col in df.columns:
                if df[col].isna().any():
                    errors.append(f"Column '{col}' contains NaN values.")
                elif (df[col] < lo).any() or (df[col] > hi).any():
                    errors.append(
                        f"Column '{col}' has values outside expected range [{lo}, {hi}]."
                    )

    return len(errors) == 0, errors
