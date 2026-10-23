"""Pre-modelling data preparation: missing values, constants, collinearity.

This module is deliberately independent of any model. Feature engineering and
preprocessing are shared pre-steps, so the same functions run before a
classification/regression model, before a propensity model in causal inference,
and inside the uplift learners. Everything returns a structured log of what was
changed, in the same shape the agent prints, so a reader can reproduce the
exact table the model saw.

Design rules
------------
* Never drop information silently: every column removed is recorded with the
  reason ("78% missing", "constant", "collinear with x2, r=0.97").
* Imputation is a *decision*, not a default: the caller chooses the numeric and
  categorical strategy, and the log says which one ran and on how many cells.
* Collinearity is handled before fitting: |r| pruning first (cheap, stable),
  then VIF pruning (catches multi-way dependence a pairwise screen misses).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

NUMERIC_STRATEGIES = ("median", "mean", "zero", "drop")
CATEGORICAL_STRATEGIES = ("mode", "missing", "drop")


def missing_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Per-column missingness, plus the row-level summary a report needs."""
    rows = []
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        rows.append(
            {
                "column": str(col),
                "dtype": str(series.dtype),
                "n_missing": n_missing,
                "pct_missing": round(100.0 * n_missing / max(len(df), 1), 2),
                "n_unique": int(series.nunique(dropna=True)),
            }
        )
    rows.sort(key=lambda r: r["pct_missing"], reverse=True)
    total_cells = int(df.shape[0] * df.shape[1])
    missing_cells = int(df.isna().sum().sum())
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "missing_cells": missing_cells,
        "missing_pct": round(100.0 * missing_cells / max(total_cells, 1), 2),
        "columns": rows,
        "columns_with_missing": [r["column"] for r in rows if r["n_missing"] > 0],
    }


def _numeric_impute(series: pd.Series, strategy: str) -> Tuple[pd.Series, str]:
    if strategy == "mean":
        value = series.mean()
        return series.fillna(value), f"mean={value:.4g}"
    if strategy == "zero":
        return series.fillna(0.0), "zero"
    value = series.median()
    return series.fillna(value), f"median={value:.4g}"


def preprocess_table(
    df: pd.DataFrame,
    *,
    numeric_strategy: str = "median",
    categorical_strategy: str = "missing",
    max_missing_ratio: float = 0.6,
    drop_constant: bool = True,
    drop_duplicates: bool = True,
    winsorise: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean a raw table: unreadable columns, NaNs, constants, duplicates.

    ``numeric_strategy``: median | mean | zero | drop (drop = drop the column).
    ``categorical_strategy``: mode | missing (own category) | drop.
    ``max_missing_ratio``: columns above this share of NaNs are dropped.
    ``winsorise``: optional two-sided quantile clip, e.g. 0.01 for 1%/99%.

    Returns ``(clean_df, log)`` where ``log`` carries ``steps`` (bilingual-ready
    English sentences), the dropped columns with reasons, and the imputed cells.
    """
    if numeric_strategy not in NUMERIC_STRATEGIES:
        raise ValueError(f"numeric_strategy must be one of {NUMERIC_STRATEGIES}")
    if categorical_strategy not in CATEGORICAL_STRATEGIES:
        raise ValueError(f"categorical_strategy must be one of {CATEGORICAL_STRATEGIES}")

    log: Dict[str, Any] = {"steps": [], "dropped": [], "imputed": {}, "duplicates_removed": 0}
    frame = df.copy()

    if drop_duplicates:
        before = len(frame)
        frame = frame.drop_duplicates()
        removed = before - len(frame)
        log["duplicates_removed"] = int(removed)
        if removed:
            log["steps"].append(f"removed {removed} duplicate rows")

    # 1) Columns too sparse to be useful.
    for col in list(frame.columns):
        ratio = float(frame[col].isna().mean()) if len(frame) else 0.0
        if ratio > max_missing_ratio:
            frame = frame.drop(columns=[col])
            log["dropped"].append({"column": str(col), "reason": f"{ratio:.0%} missing"})
    if log["dropped"]:
        log["steps"].append(
            "dropped sparse columns: " + ", ".join(f"{d['column']} ({d['reason']})" for d in log["dropped"])
        )

    # 2) NaNs are handled by type, per column, and every fill is recorded.
    imputed = 0
    for col in list(frame.columns):
        series = frame[col]
        n_missing = int(series.isna().sum())
        if not n_missing:
            continue
        if pd.api.types.is_numeric_dtype(series):
            if numeric_strategy == "drop":
                frame = frame.drop(columns=[col])
                log["dropped"].append({"column": str(col), "reason": f"{n_missing} NaN, strategy=drop"})
                continue
            filled, detail = _numeric_impute(series, numeric_strategy)
            frame[col] = filled
            log["imputed"][str(col)] = {"n": n_missing, "strategy": numeric_strategy, "detail": detail}
        else:
            if categorical_strategy == "drop":
                frame = frame.drop(columns=[col])
                log["dropped"].append({"column": str(col), "reason": f"{n_missing} NaN, strategy=drop"})
                continue
            if categorical_strategy == "mode":
                modes = series.mode(dropna=True)
                value = modes.iloc[0] if len(modes) else "unknown"
            else:
                value = "missing"
            frame[col] = series.fillna(value)
            log["imputed"][str(col)] = {"n": n_missing, "strategy": categorical_strategy, "detail": str(value)}
        imputed += n_missing
    if imputed:
        log["steps"].append(f"imputed {imputed} NaN cells across {len(log['imputed'])} columns")

    # 3) Constants carry no signal and break standardisation.
    if drop_constant:
        const_cols = [str(c) for c in frame.columns if frame[c].nunique(dropna=True) <= 1]
        if const_cols:
            frame = frame.drop(columns=const_cols)
            log["dropped"].extend({"column": c, "reason": "constant"} for c in const_cols)
            log["steps"].append("dropped constant columns: " + ", ".join(const_cols))

    # 4) Optional outlier clip on numeric columns.
    if winsorise is not None and 0 < winsorise < 0.5:
        clipped = []
        for col in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[col]):
                lo, hi = frame[col].quantile([winsorise, 1 - winsorise])
                if hi > lo:
                    frame[col] = frame[col].clip(lo, hi)
                    clipped.append(str(col))
        if clipped:
            log["steps"].append(f"winsorised {len(clipped)} numeric columns at {winsorise:.0%}/{1 - winsorise:.0%}")

    log["n_rows"] = int(len(frame))
    log["n_cols"] = int(frame.shape[1])
    return frame, log


def correlation_pairs(
    X: pd.DataFrame, threshold: float = 0.9
) -> List[Dict[str, Any]]:
    """Numeric column pairs whose |Pearson r| exceeds ``threshold``."""
    numeric = X.select_dtypes(include=[np.number])
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if numeric.shape[1] < 2:
        return []
    corr = numeric.corr().abs()
    pairs: List[Dict[str, Any]] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            value = float(corr.loc[a, b])
            if np.isfinite(value) and value >= threshold:
                pairs.append({"a": str(a), "b": str(b), "r": round(value, 4)})
    pairs.sort(key=lambda p: p["r"], reverse=True)
    return pairs


def _vif_table(X: pd.DataFrame) -> Dict[str, float]:
    """VIF per numeric column via the standard R² regression."""
    numeric = X.select_dtypes(include=[np.number]).astype(float)
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if numeric.shape[1] < 2 or len(numeric) < 3:
        return {}
    values = numeric.to_numpy()
    out: Dict[str, float] = {}
    for i, col in enumerate(numeric.columns):
        y = values[:, i]
        others = np.delete(values, i, axis=1)
        design = np.column_stack([np.ones(len(others)), others])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coef
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 0.0 if ss_tot <= 1e-12 else max(0.0, 1.0 - ss_res / ss_tot)
        out[str(col)] = round(float(1.0 / max(1.0 - r2, 1e-6)), 3)
    return out


def drop_collinear(
    X: pd.DataFrame,
    *,
    corr_threshold: float = 0.95,
    vif_threshold: float = 10.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Prune redundant numeric features, keeping the first of each cluster.

    Two passes, in this order:
    1. pairwise |r| — drop the later column of any pair above the threshold
       (deterministic, so the same table always yields the same design matrix);
    2. VIF — drop the highest-VIF column while any VIF exceeds the threshold,
       which is what catches three-way dependence that pairwise r misses.

    Non-numeric (already one-hot) columns are left alone: they are binary and
    their VIF is not meaningful in the same way.
    """
    log: Dict[str, Any] = {"steps": [], "dropped": [], "vif_before": {}, "vif_after": {}}
    frame = X.copy()
    numeric = list(frame.select_dtypes(include=[np.number]).columns)
    log["vif_before"] = _vif_table(frame[numeric]) if numeric else {}

    dropped: List[Dict[str, Any]] = []
    pairs = correlation_pairs(frame[numeric] if numeric else frame, threshold=corr_threshold)
    for pair in pairs:
        a, b = pair["a"], pair["b"]
        if a in frame.columns and b in frame.columns:
            frame = frame.drop(columns=[b])
            dropped.append({"column": b, "reason": f"r={pair['r']:.2f} with {a}"})
    removed_by_corr = len(dropped)

    if numeric:
        guard = 0
        while guard < 20:
            guard += 1
            numeric = list(frame.select_dtypes(include=[np.number]).columns)
            vif = _vif_table(frame[numeric]) if len(numeric) > 1 else {}
            if not vif:
                break
            worst_col, worst = max(vif.items(), key=lambda kv: kv[1])
            if worst <= vif_threshold or len(numeric) <= 1:
                break
            frame = frame.drop(columns=[worst_col])
            dropped.append({"column": worst_col, "reason": f"VIF={worst:.1f}"})

    if dropped:
        log["steps"].append(
            f"collinearity: dropped {len(dropped)} features "
            f"({removed_by_corr} by pairwise r, {len(dropped) - removed_by_corr} by VIF)"
        )
        log["steps"].append(
            "; ".join(f"{d['column']} ({d['reason']})" for d in dropped)
        )
    log["dropped"] = dropped
    numeric = list(frame.select_dtypes(include=[np.number]).columns)
    log["vif_after"] = _vif_table(frame[numeric]) if len(numeric) > 1 else {}
    log["n_features"] = int(frame.shape[1])
    return frame, log


def prepare_features(
    df: pd.DataFrame,
    target: Optional[str] = None,
    *,
    drop: Optional[Sequence[str]] = None,
    numeric_strategy: str = "median",
    categorical_strategy: str = "missing",
    max_missing_ratio: float = 0.6,
    corr_threshold: float = 0.95,
    vif_threshold: float = 10.0,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, Any]]:
    """The shared pre-model pipeline: clean → encode-free numeric frame.

    Used before modelling, before a propensity model in causal inference, and
    by the uplift learners. Returns ``(X, y, log)`` where the log holds one
    ``preprocess`` block and one ``feature_engineering`` block.
    """
    drop_cols = [c for c in (drop or []) if c in df.columns and c != target]
    working = df.drop(columns=drop_cols) if drop_cols else df.copy()
    y = working[target].copy() if target and target in working.columns else None
    if target and target in working.columns:
        working = working.drop(columns=[target])

    clean, pre_log = preprocess_table(
        working,
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy,
        max_missing_ratio=max_missing_ratio,
    )

    # Categorical → dummy; numeric left as-is so VIF/correlation are meaningful.
    cat_cols = [str(c) for c in clean.columns if not pd.api.types.is_numeric_dtype(clean[c])]
    if cat_cols:
        clean = pd.get_dummies(clean, columns=cat_cols, drop_first=True, dummy_na=False)
        pre_log["steps"].append(f"one-hot encoded {cat_cols}")

    clean = clean.replace([np.inf, -np.inf], np.nan)
    if clean.isna().any().any():
        clean = clean.fillna(clean.median(numeric_only=True))

    X, col_log = drop_collinear(
        clean, corr_threshold=corr_threshold, vif_threshold=vif_threshold
    )
    log = {"preprocess": pre_log, "feature_engineering": col_log, "n_features": int(X.shape[1])}
    if y is not None:
        log["n_rows"] = int(len(y))
    return X, y, log
