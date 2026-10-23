"""Modeling toolkit: supervised models, feature engineering, explanations.

The DS workhorse that other modules reuse: classification and regression with
honest held-out metrics, feature engineering for tabular data, and feature
analysis (permutation importance; true SHAP values when the ``shap`` package
is installed, otherwise a coefficient/permutation fallback labelled as such).

Every result carries a ``confidence`` block so the caller can judge how much
to trust the model before using it in a decision or a causal pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # optional: true SHAP values
    import shap

    HAS_SHAP = True
except Exception:  # noqa: BLE001 - shap pulls in numba, often painful to install
    HAS_SHAP = False

try:  # optional: scikit-learn
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.inspection import permutation_importance as sk_permutation_importance
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer

    HAS_SKLEARN = True
except Exception:  # noqa: BLE001
    HAS_SKLEARN = False


# ------------------------------------------------------------ feature engineering
def engineer_features(
    df: pd.DataFrame,
    target: str,
    drop: Optional[List[str]] = None,
    date_column: Optional[str] = None,
    *,
    preprocess: bool = True,
    prune_collinear: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, Any]]:
    """Shared pre-model pipeline: clean -> expand -> encode -> prune.

    Steps, in order (this is the pipeline causal inference reuses too —
    a propensity model needs exactly the same hygiene as a supervised one):

    1. data preprocessing: missing/NaN profiling and imputation, sparse and
       constant columns dropped, duplicates removed (``tools.preprocess``);
    2. feature construction: datetime expansion (dow/month/is_weekend) and
       one-hot encoding of categoricals;
    3. collinearity handling: pairwise |r| pruning then VIF pruning, so the
       design matrix the model sees is not rank-deficient.

    Returns ``(X, y, feature_names, log)``. ``log`` records every transform so
    the agent can show what the model actually saw: ``log["preprocess"]`` for
    step 1, ``log["steps"]`` for step 2, ``log["collinearity"]`` for step 3.
    """
    from .preprocess import drop_collinear, preprocess_table

    log: Dict[str, Any] = {"steps": [], "dropped": []}
    frame = df.copy()
    for col in [target] + (drop or []):
        if col in frame.columns:
            frame = frame.drop(columns=[col])
            log["dropped"].append(col)

    if date_column and date_column in frame.columns:
        # Expand BEFORE preprocessing: a constant snapshot date then yields
        # constant derived features, which the constant-column filter judges
        # on their own merits (and the log records the expansion either way).
        dt = pd.to_datetime(frame[date_column], errors="coerce")
        frame = frame.drop(columns=[date_column])
        frame["dow"] = dt.dt.dayofweek
        frame["month"] = dt.dt.month
        frame["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        log["steps"].append(f"datetime expansion of {date_column} -> dow/month/is_weekend")

    if preprocess:
        frame, pre_log = preprocess_table(frame)
        log["preprocess"] = pre_log
        log["steps"].extend(f"[preprocess] {s}" for s in pre_log["steps"])

    # y must follow the surviving rows: preprocessing (drop_duplicates) can
    # shrink the frame, so index-align instead of copying the original column.
    y = None
    if target in df.columns:
        y = df.loc[frame.index, target]
    cat_cols = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    num_cols = [c for c in frame.columns if c not in cat_cols]
    if cat_cols:
        frame = pd.get_dummies(frame, columns=cat_cols, drop_first=True, dummy_na=True)
        log["steps"].append(f"one-hot encoded {cat_cols}")
    for col in frame.columns:
        if frame[col].isna().any():
            frame[col] = frame[col].fillna(frame[col].median())
            log["steps"].append(f"imputed NaNs in {col} with median")
    if prune_collinear:
        frame, col_log = drop_collinear(frame)
        log["collinearity"] = col_log
        log["steps"].extend(f"[collinearity] {s}" for s in col_log["steps"])
    feature_names = list(frame.columns)
    log["n_features"] = len(feature_names)
    return frame, y, feature_names, log


# --------------------------------------------------------------------- training
def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    task: str = "auto",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train a classification or regression model with held-out metrics + CV.

    ``task``: "auto" (infer from y), "classification" or "regression".
    Linear model first, gradient boosting as the nonlinear comparison — the
    gap between them is itself a diagnostic (nonlinear signal present?).
    """
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required for modeling (pip install scikit-learn)")
    if task == "auto":
        task = "classification" if y.dtype == object or y.nunique() <= 10 else "regression"
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    result: Dict[str, Any] = {"task": task, "n_train": len(X_tr), "n_test": len(X_te)}

    if task == "classification":
        models = {
            "logistic_regression": LogisticRegression(max_iter=2000),
            "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        }
    else:
        models = {
            "linear_regression": LinearRegression(),
            "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
        }

    fitted = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        fitted[name] = model
    result["models"] = {n: (hasattr(m, "feature_importances_") or hasattr(m, "coef_")) for n, m in fitted.items()}

    # Metrics on the held-out split; pick the better model by the primary metric.
    primary = {}
    for name, model in fitted.items():
        pred = model.predict(X_te)
        if task == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_te, pred)),
                "precision": float(precision_score(y_te, pred, zero_division=0)),
                "recall": float(recall_score(y_te, pred, zero_division=0)),
                "f1": float(f1_score(y_te, pred, zero_division=0)),
            }
            if hasattr(model, "predict_proba") and y.nunique() == 2:
                proba = model.predict_proba(X_te)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_te, proba))
            primary[name] = metrics.get("roc_auc", metrics["f1"])
        else:
            metrics = {
                "r2": float(r2_score(y_te, pred)),
                "rmse": float(np.sqrt(np.mean((y_te - pred) ** 2))),
                "mae": float(mean_absolute_error(y_te, pred)),
            }
            primary[name] = metrics["r2"]
        result.setdefault("metrics", {})[name] = {k: round(v, 4) for k, v in metrics.items()}
        result.setdefault("predictions", {})[name] = pred
    result["best_model"] = max(primary, key=primary.get)
    result["metrics"]["primary_metric"] = "roc_auc or f1" if task == "classification" else "r2"

    # 5-fold CV stability of the best model on the training split. For binary
    # classification the fold metric is the AUC (accuracy hides class balance).
    best = fitted[result["best_model"]]
    if task == "classification" and y.nunique() == 2:
        from sklearn.model_selection import StratifiedKFold

        cv = cross_val_score(
            best, X_tr, y_tr,
            cv=StratifiedKFold(5, shuffle=True, random_state=random_state),
            scoring="roc_auc",
        )
    else:
        cv = cross_val_score(best, X_tr, y_tr, cv=5)
    result["cv_mean"] = round(float(cv.mean()), 4)
    result["cv_std"] = round(float(cv.std()), 4)

    # ---- closed loop: baseline, leakage screen, calibration -----------------
    result["baseline"] = strong_baseline(y_tr, y_te, task)
    result["leakage"] = check_leakage(X, y)
    if task == "classification" and y.nunique() == 2 and hasattr(fitted[result["best_model"]], "predict_proba"):
        proba = fitted[result["best_model"]].predict_proba(X_te)[:, 1]
        result["calibration"] = calibration_report(np.asarray(y_te), proba)
    result["_fitted"] = fitted
    result["_split"] = (X_tr, X_te, y_tr, y_te)
    return result


# ------------------------------------------------------------ validation suite
def check_leakage(X: pd.DataFrame, y: pd.Series) -> List[Dict[str, Any]]:
    """Screen features for target leakage before anything is trusted.

    Three signals: ID-like columns (unique keys memorise rows), proxy
    features (near-perfect correlation with the label), and post-outcome
    naming (columns that can only exist after the label was written).
    """
    import re

    flags: List[Dict[str, Any]] = []
    n = len(X)
    y_num = pd.to_numeric(y, errors="coerce")
    id_re = re.compile(r"(^|_)(id|uuid|guid|sn|serial|order_no|user_id)(_|$)", re.I)
    after_re = re.compile(r"(后|事后|之后|_post|post_|_after|after_|cum_|累计)", re.I)
    for col in X.columns:
        s = X[col]
        reasons: List[str] = []
        if id_re.search(str(col)) and n and s.nunique(dropna=True) / n > 0.9:
            reasons.append("ID-like key: unique per row, memorises the label")
        if pd.api.types.is_numeric_dtype(s) and y_num.notna().sum() > 10:
            corr = s.corr(y_num)
            if corr is not None and abs(corr) > 0.95:
                reasons.append(f"proxy feature: |corr| with target = {abs(corr):.3f}")
        if after_re.search(str(col)):
            reasons.append("post-outcome naming: may only exist after the label")
        if reasons:
            flags.append({"feature": str(col), "reasons": reasons, "action": "exclude or justify"})
    return flags


def strong_baseline(y_tr: pd.Series, y_te: pd.Series, task: str) -> Dict[str, Any]:
    """The baseline any real model must beat, stated as numbers not words."""
    y_tr, y_te = pd.Series(y_tr), pd.Series(y_te)
    if task == "classification":
        majority = y_tr.mode().iloc[0]
        acc = float((y_te == majority).mean())
        return {
            "model": "majority class",
            "accuracy": round(acc, 4),
            "prevalence": round(float(y_tr.mean()) if y_tr.dtype != object else 0.0, 4),
            "note": "a candidate model must clear this accuracy to be worth deploying",
        }
    mean_pred = float(y_tr.mean())
    rmse = float(np.sqrt(np.mean((y_te - mean_pred) ** 2)))
    ss_res = float(np.sum((y_te - mean_pred) ** 2))
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2)) or 1.0
    return {
        "model": "mean predictor",
        "rmse": round(rmse, 4),
        "r2_vs_test_mean": round(1 - ss_res / ss_tot, 4),
        "note": "a candidate model must beat this RMSE to be worth deploying",
    }


def calibration_report(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Probability calibration: are predicted 70%s right 70% of the time?

    Brier score (lower is better) and expected calibration error over equal-
    width bins; per-bin (mean predicted vs observed rate) feeds the
    reliability chart. miscalibrated scores mislead every downstream
    threshold decision, even when the AUC looks fine.
    """
    from sklearn.metrics import brier_score_loss

    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(proba, edges) - 1, 0, n_bins - 1)
    bins = []
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        mean_pred = float(proba[mask].mean())
        frac_pos = float(y_true[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(frac_pos - mean_pred)
        bins.append({"bin": f"[{edges[b]:.1f}-{edges[b+1]:.1f})", "n": int(mask.sum()),
                     "mean_predicted": round(mean_pred, 4), "observed_rate": round(frac_pos, 4)})
    return {
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "ece": round(ece, 4),
        "well_calibrated": bool(ece < 0.05),
        "bins": bins,
    }


# ---------------------------------------------------------------- explanations
def feature_analysis(result: Dict[str, Any], max_features: int = 12) -> Dict[str, Any]:
    """Feature analysis for the best model: SHAP when available, else
    permutation importance + linear coefficients (labelled honestly)."""
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required for feature analysis")
    X_tr, X_te, y_tr, y_te = result["_split"]
    best_name = result["best_model"]
    model = result["_fitted"][best_name]
    out: Dict[str, Any] = {
        "method": "shap" if HAS_SHAP else "permutation+coefficients",
        "model": best_name,
        "engine": "shap" if HAS_SHAP else "sklearn-permutation",
    }
    if HAS_SHAP:
        explainer = shap.Explainer(model, X_te)
        sv = explainer(X_te)
        mean_abs = np.abs(sv.values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:max_features]
        out["importance"] = [
            {"feature": str(X_te.columns[i]), "value": float(mean_abs[i])} for i in order
        ]
        return out
    perm = sk_permutation_importance(model, X_te, y_te, n_repeats=8, random_state=42)
    order = np.argsort(perm.importances_mean)[::-1][:max_features]
    out["importance"] = [
        {"feature": str(X_te.columns[i]), "value": float(perm.importances_mean[i])}
        for i in order
        if perm.importances_mean[i] > 0
    ]
    coef_map: Dict[str, float] = {}
    if hasattr(model, "coef_"):
        coefs = np.ravel(model.coef_)
        coef_map = {str(c): float(v) for c, v in zip(X_tr.columns, coefs)}
        out["coefficients"] = sorted(coef_map.items(), key=lambda kv: abs(kv[1]), reverse=True)[:max_features]
    out["direction"] = {f: ("+" if v > 0 else "-") for f, v in out.get("coefficients", [])}
    return out


# ------------------------------------------------------------------ confidence
def modeling_confidence(result: Dict[str, Any]) -> Dict[str, Any]:
    """PASS/WARN/FAIL checks a decision-maker should see before trusting a model."""
    checks: List[Dict[str, Any]] = []
    gap = None
    task = result["task"]
    metrics = result["metrics"][result["best_model"]]
    train_primary = result["cv_mean"]
    test_primary = metrics.get("roc_auc", metrics.get("f1", metrics.get("r2")))
    gap = train_primary - test_primary
    checks.append(
        {
            "name": "train-test gap",
            "status": "PASS" if gap is not None and abs(gap) < 0.15 else ("WARN" if abs(gap) < 0.3 else "FAIL"),
            "detail": f"CV={train_primary:.3f} vs test={test_primary:.3f} (gap {gap:+.3f})",
        }
    )
    stable = result["cv_std"] < 0.1
    checks.append(
        {
            "name": "5-fold CV stability",
            "status": "PASS" if stable else "WARN",
            "detail": f"std={result['cv_std']:.3f} across folds",
        }
    )
    n = result["n_train"] + result["n_test"]
    checks.append(
        {
            "name": "sample size",
            "status": "PASS" if n >= 500 else ("WARN" if n >= 100 else "FAIL"),
            "detail": f"n={n} rows",
        }
    )
    overall = "low" if any(c["status"] == "FAIL" for c in checks) else ("medium" if any(c["status"] == "WARN" for c in checks) else "high")
    return {"checks": checks, "overall": overall, "overfit_gap": round(gap, 4)}
