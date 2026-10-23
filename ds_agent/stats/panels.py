"""Per-analysis test batteries.

``stats.tests`` knows how to run a test; this module knows *which* tests a given
conclusion should carry. The distinction matters: an anomaly readout needs a
paired comparison against its seasonal baseline and an autocorrelation check on
the residual, while an experiment readout needs a variance check, a location
test, a rank test and a randomisation cross-check. Running all 30 tests
everywhere would be noise; running one everywhere would be a claim without
support.

Every panel returns the same shape::

    {"tests": [...], "headline_zh": ..., "headline_en": ..., "agreement": True|False|None}

``PANEL_SPEC`` names the tests each readout reports. The documentation generator
reads it, so "which tests does the anomaly conclusion include" has exactly one
answer, in code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import tests as T

# Readout -> (test keys, bilingual description). Used by docs/statistical_tests.md.
PANEL_SPEC: Dict[str, Dict[str, Any]] = {
    "anomaly": {
        "name_zh": "Metric anomaly diagnosis",
        "name_en": "Metric anomaly diagnosis",
        "tests": ["shapiro_wilk|dagostino_k2", "jarque_bera", "paired_t", "wilcoxon", "ljung_box", "durbin_watson", "adf", "kpss"],
        "why_zh": "Normality gates the normal-theory test, the paired test judges the window against its seasonal baseline, and the rank/autocorrelation checks guard the conclusion.",
        "why_en": "Normality gates the normal-theory test, the paired test judges the window against its seasonal baseline, and the rank/autocorrelation checks guard the conclusion.",
    },
    "experiment": {
        "name_zh": "Experiment readout",
        "name_en": "Experiment readout",
        "tests": ["dagostino_k2|shapiro_wilk", "levene", "bartlett", "welch_t", "student_t", "mann_whitney", "ks_two_sample", "bootstrap", "permutation", "two_proportion_z", "chi_square_independence", "fisher_exact", "chi_square_gof", "bh_fdr"],
        "why_zh": "SRM gates the split, variance/normality tests choose Welch vs Student, rank/KS/bootstrap/permutation cross-check the analytic result, and BH controls multiplicity.",
        "why_en": "SRM gates the split, variance/normality tests choose Welch vs Student, rank/KS/bootstrap/permutation cross-check the analytic result, and BH controls multiplicity.",
    },
    "did": {
        "name_zh": "Difference-in-differences",
        "name_en": "Difference-in-differences",
        "tests": ["wald_joint_f", "breusch_pagan", "durbin_watson", "jarque_bera"],
        "why_zh": "The joint F test on leads is the parallel-trends evidence; residual heteroskedasticity/autocorrelation/normality decide whether cluster or HAC errors are required.",
        "why_en": "The joint F test on leads is the parallel-trends evidence; residual heteroskedasticity/autocorrelation/normality decide whether cluster or HAC errors are required.",
    },
    "its": {
        "name_zh": "Time-series causal (ITS / structural counterfactual)",
        "name_en": "Time-series causal (ITS / structural counterfactual)",
        "tests": ["adf", "kpss", "ljung_box", "durbin_watson", "jarque_bera"],
        "why_zh": "Stationarity first (ADF and KPSS are complements), then residual whiteness and first-order autocorrelation; without them the p-values and CIs are overstated.",
        "why_en": "Stationarity first (ADF and KPSS are complements), then residual whiteness and first-order autocorrelation; without them the p-values and CIs are overstated.",
    },
    "modeling": {
        "name_zh": "Modeling (classification/regression)",
        "name_en": "Modeling (classification/regression)",
        "tests": ["jarque_bera", "durbin_watson", "breusch_pagan", "variance_inflation", "hosmer_lemeshow", "mcnemar"],
        "why_zh": "Regression: residual normality/autocorrelation/heteroskedasticity and collinearity. Classification: calibration (Hosmer-Lemeshow) and a McNemar comparison between models. Together they decide whether the model can drive a decision.",
        "why_en": "Regression: residual normality/autocorrelation/heteroskedasticity and collinearity. Classification: calibration (Hosmer-Lemeshow) and a McNemar comparison between models. Together they decide whether the model can drive a decision.",
    },
    "psm": {
        "name_zh": "Propensity score matching",
        "name_en": "Propensity score matching",
        "tests": ["paired_t", "welch_t"],
        "why_zh": "A paired t-test on matched pairs gives the analytic ATT inference (complementing the bootstrap), a Welch test on the worst covariate confirms balance alongside an SMD below 0.1, and overlap is judged by the common-support share (a KS test rejects on immaterial differences at scale).",
        "why_en": "A paired t-test on matched pairs gives the analytic ATT inference (complementing the bootstrap), a Welch test on the worst covariate confirms balance alongside an SMD below 0.1, and overlap is judged by the common-support share (a KS test rejects on immaterial differences at scale).",
    },
}


def _panel(tests: Sequence[Dict[str, Any]], primary_keys: Sequence[str], lang_note: str = "") -> Dict[str, Any]:
    """Attach a headline that says whether the tests agree."""
    ran = [t for t in tests if t["status"] == "ok"]
    primary = [t for t in ran if t["key"] in primary_keys]
    decisions = {t["decision"] for t in primary}
    agreement: Optional[bool] = len(decisions) == 1 if len(primary) >= 2 else None

    if not ran:
        headline_zh = "Not enough data to run the tests."
        headline_en = "Not enough data to run the tests."
    elif agreement is True:
        if decisions == {"reject"}:
            headline_zh = f"All {len(primary)} primary tests reject H0: the difference is significant, not sampling noise."
            headline_en = f"All {len(primary)} primary tests reject H0: the difference is significant, not sampling noise."
        else:
            headline_zh = f"All {len(primary)} primary tests retain H0: the data do not support a difference."
            headline_en = f"All {len(primary)} primary tests retain H0: the data do not support a difference."
    elif agreement is False:
        headline_zh = "Primary tests disagree: defer to the robust/non-parametric result and inspect assumptions and outliers."
        headline_en = "Primary tests disagree: defer to the robust/non-parametric result and inspect assumptions and outliers."
    else:
        headline_zh = f"{len(ran)} tests ran, checking the prerequisites and robustness of the conclusion."
        headline_en = f"{len(ran)} tests ran, checking the prerequisites and robustness of the conclusion."

    return {
        "tests": list(tests),
        "agreement": agreement,
        "headline_zh": headline_zh + (lang_note or ""),
        "headline_en": headline_en,
        "summary_zh": T.summary_line(tests, "zh"),
        "summary_en": T.summary_line(tests, "en"),
    }


def _safe(fn, *args, **kwargs) -> Dict[str, Any]:
    """Run one test; a failure becomes a skipped row instead of an exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break the answer
        key = getattr(fn, "__name__", "test")
        return T.skipped(key, f"failed: {exc}", f"failed: {exc}")


# --------------------------------------------------------------------------- #
# Anomaly
# --------------------------------------------------------------------------- #
def anomaly_panel(
    values: Sequence[float],
    expected: Sequence[float],
    series: Optional[Sequence[float]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Tests behind an anomaly verdict: is the window really below its baseline?"""
    v = np.asarray(values, dtype=float).ravel()
    e = np.asarray(expected, dtype=float).ravel()
    n = min(len(v), len(e))
    v, e = v[:n], e[:n]
    residual = v - e

    tests: List[Dict[str, Any]] = []
    tests.extend(T.normality_tests(residual, alpha)[:2])
    # The paired tests are the right location test here: actual and expected are
    # two measurements of the same day, so an equal-variance test across them
    # would be testing the wrong thing.
    tests.append(_safe(T.paired_t, e, v, alpha))
    tests.append(_safe(T.wilcoxon, e, v, alpha))
    if series is not None:
        arr = np.asarray(series, dtype=float).ravel()
        tests.append(_safe(T.ljung_box, residual, 7 if n >= 14 else None, alpha))
        tests.append(_safe(T.durbin_watson, residual, alpha))
        tests.append(_safe(T.adf, arr, alpha))
        tests.append(_safe(T.kpss, arr, alpha))
    return _panel(tests, ["paired_t", "wilcoxon"], " (residuals vs seasonal baseline)")


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def experiment_panel(
    control: Sequence[float],
    treatment: Sequence[float],
    kind: str = "mean",
    alpha: float = 0.05,
    permutation_iter: int = 500,
) -> Dict[str, Any]:
    """Tests behind an experiment readout.

    ``kind="proportion"`` expects 0/1 arrays (or counts via ``counts``); the
    proportion branch runs a z test, a chi-square and, where the table is sparse,
    an exact test.
    """
    c = np.asarray(control, dtype=float).ravel()
    t = np.asarray(treatment, dtype=float).ravel()
    c = c[np.isfinite(c)]
    t = t[np.isfinite(t)]
    tests: List[Dict[str, Any]] = []

    if kind == "proportion":
        x1, n1 = int(np.nansum(c)), len(c)
        x2, n2 = int(np.nansum(t)), len(t)
        tests.append(_safe(T.two_proportion_z, x1, n1, x2, n2, alpha))
        table = [[x1, n1 - x1], [x2, n2 - x2]]
        tests.append(_safe(T.chi_square_independence, table, alpha))
        sparse = min(x1, n1 - x1, x2, n2 - x2) < 5
        if sparse:
            tests.append(_safe(T.fisher_exact, table, alpha))
        else:
            tests.append(T._skipped("fisher_exact", "counts are large enough for the z test", "counts are large enough for the z test"))
        tests.append(_safe(T.chi_square_gof, [n1, n2], None, alpha))
        return _panel(tests, ["two_proportion_z", "chi_square_independence"], " (proportion metric)")

    tests.extend([_arm_normality(c, "control"), _arm_normality(t, "treatment")])
    tests.append(_safe(T.levene, c, t, alpha))
    tests.append(_safe(T.bartlett, c, t, alpha))
    tests.append(_safe(T.welch_t, c, t, alpha))
    tests.append(_safe(T.student_t, c, t, alpha))
    tests.append(_safe(T.mann_whitney, c, t, alpha))
    tests.append(_safe(T.ks_two_sample, c, t, alpha))
    tests.append(_safe(T.bootstrap_ci, c, t, 1000, alpha))
    values = np.concatenate([c, t])
    groups = np.concatenate([np.zeros(len(c), dtype=int), np.ones(len(t), dtype=int)])
    tests.append(_safe(T.permutation_test, values, groups, permutation_iter, alpha))
    return _panel(tests, ["welch_t", "mann_whitney", "permutation"])


def _arm_normality(values: np.ndarray, arm: str) -> Dict[str, Any]:
    """Normality per arm, labelled so the table says which arm was tested."""
    res = T.normality_tests(values)[0]
    zh = "control arm" if arm == "control" else "treatment arm"
    en = "control arm" if arm == "control" else "treatment arm"
    res["name_zh"] = f"{res['name_zh']} ({zh})"
    res["name_en"] = f"{res['name_en']} ({en})"
    return res


# --------------------------------------------------------------------------- #
# Regression-based readouts
# --------------------------------------------------------------------------- #
def regression_panel(
    diagnostics: Optional[Sequence[Dict[str, Any]]],
    joint: Optional[Dict[str, Any]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """DiD panel: a pre-supplied joint F test plus residual diagnostics."""
    tests: List[Dict[str, Any]] = [t for t in (diagnostics or []) if t]
    if joint is not None:
        tests.insert(0, joint)
    return _panel(tests, ["wald_joint_f"], "")


def its_panel(
    residuals: Sequence[float],
    series: Sequence[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """ITS panel: stationarity of the series, whiteness of the residuals."""
    tests: List[Dict[str, Any]] = [
        _safe(T.adf, series, alpha),
        _safe(T.kpss, series, alpha),
        _safe(T.ljung_box, residuals, None, alpha),
        _safe(T.durbin_watson, residuals, alpha),
        _safe(T.jarque_bera, residuals, alpha),
    ]
    return _panel(tests, ["ljung_box"], "")


def model_panel(model_result: Dict[str, Any], alpha: float = 0.05) -> Dict[str, Any]:
    """Model diagnostics from a ``train_model`` result.

    Regression: residual normality, autocorrelation, heteroskedasticity and VIF.
    Classification: calibration (Hosmer-Lemeshow) on the held-out probabilities.
    """
    task = model_result.get("task", "regression")
    best = model_result.get("best_model")
    preds = (model_result.get("predictions") or {}).get(best)
    split = model_result.get("_split")
    tests: List[Dict[str, Any]] = []

    if split is None or preds is None:
        return _panel(tests, [], "")

    X_tr, X_te, y_tr, y_te = split
    y_true = np.asarray(y_te, dtype=float).ravel()
    pred = np.asarray(preds, dtype=float).ravel()[: len(y_true)]

    if task == "classification" and len(np.unique(y_true)) == 2:
        tests.append(_safe(T.hosmer_lemeshow, y_true, pred, 10, alpha))
        # McNemar between the two fitted models on the same test rows: "is the
        # winner actually better, or did it just get lucky on this split?"
        preds_all = model_result.get("predictions") or {}
        others = [n for n in preds_all if n != best]
        if others and best:
            other = np.asarray(preds_all[others[0]], dtype=float).ravel()[: len(y_true)]
            best_ok = pred == y_true
            other_ok = other == y_true
            b = int(np.sum(best_ok & ~other_ok))
            c = int(np.sum(~best_ok & other_ok))
            res = _safe(T.mcnemar, b, c, alpha)
            res["name_zh"] = f"{res['name_zh']} ({best} vs {others[0]})"
            res["name_en"] = f"{res['name_en']} ({best} vs {others[0]})"
            sig = res.get("decision") == "reject"
            res["conclusion_zh"] = (
                "the two models differ significantly on the test set; keep the better one"
                if sig
                else "no significant difference between the models; prefer the simpler one"
            )
            res["conclusion_en"] = (
                "the two models differ significantly on the test set; keep the better one"
                if sig
                else "no significant difference between the models; prefer the simpler one"
            )
            tests.append(res)
        return _panel(tests, ["hosmer_lemeshow"], " (classification: calibration + model comparison)")

    resid = y_true - pred
    tests.append(_safe(T.jarque_bera, resid, alpha))
    tests.append(_safe(T.durbin_watson, resid, alpha))
    try:
        exog = np.asarray(X_te, dtype=float)
        exog = np.column_stack([np.ones(len(exog)), exog])
    except Exception:  # noqa: BLE001 - non-numeric encodings cannot enter an OLS diagnostic
        exog = None
    if exog is not None:
        tests.append(_safe(T.breusch_pagan, resid, exog, alpha))
        tests.append(_safe(T.variance_inflation, np.asarray(X_te, dtype=float), list(X_te.columns)))
    return _panel(tests, ["jarque_bera"], " (regression: residual diagnostics)")


def psm_panel(psm: Dict[str, Any], alpha: float = 0.05) -> Dict[str, Any]:
    """PSM panel: the matched-pairs t-test on the ATT, plus post-match balance.

    Both p-values come from the estimator. Overlap is deliberately judged by the
    common-support share rather than a KS test: at 10^4 rows a KS test rejects on
    separations far too small to affect the estimate.
    """
    tests: List[Dict[str, Any]] = []
    att_test = psm.get("att_test") or {}
    if att_test:
        tests.append(
            T.result(
                "paired_t",
                att_test.get("t"),
                att_test.get("p_value"),
                n=psm.get("n_matched_pairs"),
                alpha=alpha,
                note_zh="paired t-test over matched pairs (complements the bootstrap interval)",
                note_en="paired t-test over matched pairs (complements the bootstrap interval)",
            )
        )
        sig = tests[-1]["decision"] == "reject"
        tests[-1]["conclusion_zh"] = (
            "the matched-pair effect is non-zero: the ATT is distinguishable from 0" if sig else "the matched-pair effect is not significant: the ATT cannot be distinguished from 0"
        )
        tests[-1]["conclusion_en"] = (
            "the matched-pair effect is non-zero: the ATT is distinguishable from 0"
            if sig
            else "the matched-pair effect is not significant: the ATT cannot be distinguished from 0"
        )
    balance = psm.get("balance") or []
    if balance:
        worst = max(balance, key=lambda b: abs(float(b.get("smd_after") or 0)))
        tests.append(
            T.result(
                "welch_t",
                worst.get("t_after"),
                worst.get("p_after"),
                n=psm.get("n_matched_pairs"),
                alpha=alpha,
                note_zh=f"worst covariate {worst.get('covariate')}: |SMD| after matching="
                f"{abs(float(worst.get('smd_after') or 0)):.3f} (threshold 0.1)",
                note_en=f"worst covariate {worst.get('covariate')}: |SMD| after matching="
                f"{abs(float(worst.get('smd_after') or 0)):.3f} (threshold 0.1)",
            )
        )
        tests[-1]["name_zh"] = f"{tests[-1]['name_zh']} (post-match balance)"
        tests[-1]["name_en"] = f"{tests[-1]['name_en']} (post-match balance)"
    return _panel(tests, [], "")
