"""Model monitoring: drift, decay and alert thresholds.

A model that passed validation will silently rot in production: the population
shifts, feature distributions drift, and yesterday's AUC says nothing about
today's. This module quantifies the three canonical decay modes and turns them
into alert decisions with stated thresholds:

* **feature drift** — per-feature PSI against the training reference
  (<0.1 stable, 0.1–0.25 moderate shift, >0.25 major shift);
* **data drift** — two-sample KS test per numeric feature (distribution-level,
  catches shape changes PSI's binned view can miss);
* **performance decay** — AUC by cohort slice with a trend test, so the decay
  is dated and directional rather than a vague "worse lately".

Every check returns a severity so callers can hook alerting directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats

    HAS_SCIPY = True
except Exception:  # noqa: BLE001
    HAS_SCIPY = False

# Alert thresholds (industry-standard PSI bands, stated so callers can tune).
PSI_WARN = 0.10
PSI_CRITICAL = 0.25
AUC_DROP_WARN = 0.03
AUC_DROP_CRITICAL = 0.05


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two samples.

    Bins are quantile edges of the *reference* sample (so empty actual bins
    mean genuine absence, not binning artefacts). Zero counts are floored at
    0.5 (standard continuity correction) to keep the log defined.
    """
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    if len(expected) < 20 or len(actual) < 20:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:  # constant reference
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    # np.histogram silently drops values beyond the outer edges — exactly the
    # escaped mass a drift check must catch. Clip both samples into range so
    # overflow lands in the edge bins and shows up as a huge PSI term.
    e_counts = np.histogram(np.clip(expected, edges[0], edges[-1]), bins=edges)[0].astype(float)
    a_counts = np.histogram(np.clip(actual, edges[0], edges[-1]), bins=edges)[0].astype(float)
    e = np.maximum(e_counts / max(e_counts.sum(), 1), 0.005)
    a = np.maximum(a_counts / max(a_counts.sum(), 1), 0.005)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_drift_table(
    reference: pd.DataFrame, current: pd.DataFrame, max_features: int = 12
) -> Dict[str, Any]:
    """Per-feature PSI with a severity verdict for each.

    Returns ``{"columns": [...], "n_critical": int, "n_warn": int,
    "max_psi": float, "verdict": str}`` — severity bands are the module
    constants, so alert wiring has one source of truth.
    """
    rows: List[Dict[str, Any]] = []
    shared = [c for c in reference.columns if c in current.columns]
    for col in shared:
        if not pd.api.types.is_numeric_dtype(reference[col]):
            # Categorical drift: PSI over the level distribution.
            levels = set(reference[col].dropna().unique()) | set(current[col].dropna().unique())
            if not levels or len(levels) > 30:
                continue
            e = reference[col].value_counts(normalize=True, dropna=True)
            a = current[col].value_counts(normalize=True, dropna=True)
            e = np.maximum(e.reindex(list(levels)).fillna(0).to_numpy(), 0.005)
            a = np.maximum(a.reindex(list(levels)).fillna(0).to_numpy(), 0.005)
            val = float(np.sum((a - e) * np.log(a / e)))
        else:
            val = psi(reference[col].to_numpy(), current[col].to_numpy())
        if np.isnan(val):
            continue
        rows.append({"feature": col, "psi": round(val, 4)})
    rows.sort(key=lambda r: -r["psi"])
    for r in rows:
        r["severity"] = "critical" if r["psi"] > PSI_CRITICAL else ("warn" if r["psi"] > PSI_WARN else "ok")
    n_crit = sum(1 for r in rows if r["severity"] == "critical")
    n_warn = sum(1 for r in rows if r["severity"] == "warn")
    return {
        "columns": rows[:max_features],
        "n_critical": n_crit,
        "n_warn": n_warn,
        "max_psi": rows[0]["psi"] if rows else 0.0,
        "verdict": "critical" if n_crit else ("warn" if n_warn else "ok"),
    }


def ks_data_drift(
    reference: pd.DataFrame, current: pd.DataFrame, max_features: int = 12, alpha: float = 0.05
) -> Dict[str, Any]:
    """Two-sample Kolmogorov–Smirnov test per numeric feature."""
    out: List[Dict[str, Any]] = []
    if not HAS_SCIPY:
        return {"tests": out, "n_drifted": 0, "verdict": "unknown"}
    shared = [c for c in reference.columns if c in current.columns]
    for col in shared:
        if not pd.api.types.is_numeric_dtype(reference[col]):
            continue
        a = reference[col].dropna()
        b = current[col].dropna()
        if len(a) < 30 or len(b) < 30:
            continue
        stat, p = _stats.ks_2samp(a, b)
        out.append({"feature": col, "ks": round(float(stat), 4), "p_value": float(p), "drifted": bool(p < alpha)})
    out.sort(key=lambda r: r["p_value"])
    out = out[:max_features]
    n_drifted = sum(1 for r in out if r["drifted"])
    return {"tests": out, "n_drifted": n_drifted, "verdict": "drifted" if n_drifted else "stable"}


def performance_decay(
    auc_history: List[Dict[str, Any]],
    baseline_window: int = 4,
    drop_warn: float = AUC_DROP_WARN,
    drop_critical: float = AUC_DROP_CRITICAL,
) -> Dict[str, Any]:
    """AUC decay over ordered cohort slices, with a directional verdict.

    ``auc_history``: ``[{"slice": name, "auc": float}, ...]`` in production
    order. The verdict compares the mean of the last slices against the
    baseline (first) window; a formal significance test needs score
    distributions, which this summary intentionally does not carry — the
    threshold bands do the alerting instead.
    """
    if len(auc_history) < 2:
        return {"verdict": "insufficient_history", "aucs": []}
    aucs = [float(h["auc"]) for h in auc_history]
    labels = [str(h.get("slice", i)) for i, h in enumerate(auc_history)]
    baseline = float(np.mean(aucs[:baseline_window])) if len(aucs) >= baseline_window * 2 else aucs[0]
    recent = float(np.mean(aucs[-min(3, len(aucs)):]))
    drop = baseline - recent
    slope = float(np.polyfit(np.arange(len(aucs)), np.array(aucs), 1)[0])
    verdict = "ok" if drop < drop_warn else ("warn" if drop < drop_critical else "critical")
    return {
        "verdict": verdict,
        "baseline_auc": round(baseline, 4),
        "recent_auc": round(recent, 4),
        "drop": round(drop, 4),
        "slope_per_slice": round(slope, 5),
        "aucs": [{"slice": l, "auc": round(a, 4)} for l, a in zip(labels, aucs)],
        "thresholds": {"warn": drop_warn, "critical": drop_critical},
    }


def monitor_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    auc_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Full monitoring pass: feature drift + KS drift + performance decay."""
    drift = feature_drift_table(reference, current)
    ks = ks_data_drift(reference, current)
    perf = performance_decay(auc_history or [])
    alerts: List[Dict[str, Any]] = []
    if drift["verdict"] == "critical":
        alerts.append(
            {"level": "critical", "check": "feature_drift",
             "message": f"{drift['n_critical']} features with PSI>{PSI_CRITICAL}: "
                        + ", ".join(r["feature"] for r in drift["columns"] if r["severity"] == "critical")}
        )
    elif drift["verdict"] == "warn":
        alerts.append(
            {"level": "warn", "check": "feature_drift",
             "message": f"{drift['n_warn']} features with PSI>{PSI_WARN} — watch, no retrain yet"}
        )
    if ks["verdict"] == "drifted":
        alerts.append(
            {"level": "warn", "check": "data_drift",
             "message": "KS test flags distribution shift in: "
                        + ", ".join(r["feature"] for r in ks["tests"] if r["drifted"])}
        )
    if perf.get("verdict") in ("warn", "critical"):
        alerts.append(
            {"level": perf["verdict"], "check": "performance_decay",
             "message": f"AUC {perf['baseline_auc']} → {perf['recent_auc']} (drop {perf['drop']})"}
        )
    overall = "critical" if any(a["level"] == "critical" for a in alerts) else (
        "warn" if alerts else "ok"
    )
    return {
        "feature_drift": drift,
        "data_drift": ks,
        "performance": perf,
        "alerts": alerts,
        "overall": overall,
        "thresholds": {
            "psi_warn": PSI_WARN, "psi_critical": PSI_CRITICAL,
            "auc_drop_warn": AUC_DROP_WARN, "auc_drop_critical": AUC_DROP_CRITICAL,
        },
    }
