"""Statistical inference primitives used by every tool.

All functions return plain dictionaries so results can be serialised into
reports and fed back to the language model without extra conversion.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Confidence intervals
# --------------------------------------------------------------------------- #
def normal_ci(estimate: float, se: float, alpha: float = 0.05) -> Tuple[float, float]:
    """Wald confidence interval for an approximately normal estimator."""
    z = stats.norm.ppf(1 - alpha / 2)
    return estimate - z * se, estimate + z * se


def proportion_ci(x: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval, which behaves well for small counts."""
    if n <= 0:
        return 0.0, 0.0
    p = x / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def relative_lift_ci(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Delta-method CI for the relative lift (p2 - p1) / p1 on the log scale."""
    p1, p2 = x1 / n1, x2 / n2
    if p1 <= 0 or p2 <= 0:
        return float("nan"), float("nan")
    se = np.sqrt((1 - p1) / (n1 * p1) + (1 - p2) / (n2 * p2))
    z = stats.norm.ppf(1 - alpha / 2)
    return float(np.exp(np.log(p2 / p1) - z * se) - 1), float(np.exp(np.log(p2 / p1) + z * se) - 1)


# --------------------------------------------------------------------------- #
# Hypothesis tests
# --------------------------------------------------------------------------- #
def two_proportion_test(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05) -> Dict[str, float]:
    """Two-sided z-test for the difference between two proportions.

    ``x1/n1`` is treated as the control arm.
    """
    p1, p2 = (x1 / n1 if n1 else 0.0), (x2 / n2 if n2 else 0.0)
    pooled = (x1 + x2) / (n1 + n2) if (n1 + n2) else 0.0
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2)) if n1 and n2 else float("nan")
    diff = p2 - p1
    z = diff / se_pooled if se_pooled and se_pooled > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se_unpooled = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) if n1 and n2 else float("nan")
    ci_low, ci_high = normal_ci(diff, se_unpooled, alpha)
    return {
        "control": p1,
        "treatment": p2,
        "abs_diff": diff,
        "rel_lift": (diff / p1) if p1 else float("nan"),
        "z": float(z),
        "p_value": float(p_value),
        "ci_abs": (float(ci_low), float(ci_high)),
        "ci_rel": relative_lift_ci(x1, n1, x2, n2, alpha),
        "se": float(se_unpooled),
        "n_control": int(n1),
        "n_treatment": int(n2),
    }


def welch_ttest(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> Dict[str, float]:
    """Welch's unequal-variance t-test for two independent samples (control=a)."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a_arr, b_arr = a_arr[~np.isnan(a_arr)], b_arr[~np.isnan(b_arr)]
    if len(a_arr) < 2 or len(b_arr) < 2:
        return {"error": "each arm needs at least 2 observations"}
    res = stats.ttest_ind(b_arr, a_arr, equal_var=False)
    m1, m2 = float(a_arr.mean()), float(b_arr.mean())
    se = float(np.sqrt(a_arr.var(ddof=1) / len(a_arr) + b_arr.var(ddof=1) / len(b_arr)))
    df = (a_arr.var(ddof=1) / len(a_arr) + b_arr.var(ddof=1) / len(b_arr)) ** 2 / (
        (a_arr.var(ddof=1) / len(a_arr)) ** 2 / (len(a_arr) - 1)
        + (b_arr.var(ddof=1) / len(b_arr)) ** 2 / (len(b_arr) - 1)
    )
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return {
        "control_mean": m1,
        "treatment_mean": m2,
        "abs_diff": m2 - m1,
        "rel_lift": (m2 - m1) / m1 if m1 else float("nan"),
        "t": float(res.statistic),
        "p_value": float(res.pvalue),
        "ci_abs": (m2 - m1 - t_crit * se, m2 - m1 + t_crit * se),
        "se": se,
        "df": float(df),
        "n_control": int(len(a_arr)),
        "n_treatment": int(len(b_arr)),
    }


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Rank-based test, robust to heavy tails and outliers."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    res = stats.mannwhitneyu(b_arr, a_arr, alternative="two-sided")
    return {"u_stat": float(res.statistic), "p_value": float(res.pvalue)}


def bootstrap_diff_in_means(
    control: Sequence[float],
    treatment: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap for the difference in means (handles skew well)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(control, dtype=float), np.asarray(treatment, dtype=float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(b, len(b), replace=True).mean() - rng.choice(a, len(a), replace=True).mean()
    point = float(b.mean() - a.mean())
    return {
        "abs_diff": point,
        "p_value": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
        "ci_abs": (float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))),
        "se": float(diffs.std(ddof=1)),
        "n_boot": n_boot,
    }


def ratio_metric_test(
    num_control: float, den_control: float, num_treatment: float, den_treatment: float,
    var_num_c: Optional[float] = None, var_den_c: Optional[float] = None,
    var_num_t: Optional[float] = None, var_den_t: Optional[float] = None,
    cov_c: Optional[float] = None, cov_t: Optional[float] = None,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Delta-method test for a ratio of sums (e.g. GMV / sessions).

    ``var_*`` are variances of the *sums* (not per-unit), and ``cov_*`` the
    covariance of numerator and denominator. The full delta-method variance of
    a ratio is

        Var(N/D) ~ Var(N)/D^2 - 2 (N/D) Cov(N,D)/D^2 + (N/D)^2 Var(D)/D^2

    Passing only the numerator variance (the historical behaviour) silently
    drops the denominator and covariance terms and can produce absurd standard
    errors, so a partial set of moments now yields a NaN test instead of a
    confident wrong one. Use :func:`ratio_metric_test_from_samples` when you
    have per-unit rows - it is the easier and safer entry point.
    """
    r_c = num_control / den_control if den_control else float("nan")
    r_t = num_treatment / den_treatment if den_treatment else float("nan")

    def _var_ratio(num: float, den: float, ratio: float, vn, vd, cv) -> float:
        if not den or None in (vn, vd):
            return float("nan")
        cov = 0.0 if cv is None else float(cv)
        return float(
            max(float(vn) / den**2 - 2.0 * ratio * cov / den**2 + ratio**2 * float(vd) / den**2, 0.0)
        )

    var_c = _var_ratio(num_control, den_control, r_c, var_num_c, var_den_c, cov_c)
    var_t = _var_ratio(num_treatment, den_treatment, r_t, var_num_t, var_den_t, cov_t)
    diff = r_t - r_c
    if np.isfinite(var_c) and np.isfinite(var_t):
        se = float(np.sqrt(var_c + var_t))
        z = diff / se if se > 0 else 0.0
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
        ci = normal_ci(diff, se, alpha) if se > 0 else (float("nan"), float("nan"))
        note = "delta_method"
    else:
        se, z, p_value, ci = float("nan"), float("nan"), float("nan"), (float("nan"), float("nan"))
        note = "insufficient moments: pass var_den_* and cov_* or use ratio_metric_test_from_samples"
    return {
        "control": r_c,
        "treatment": r_t,
        "abs_diff": diff,
        "rel_lift": diff / r_c if r_c else float("nan"),
        "se": se,
        "z": float(z),
        "p_value": p_value,
        "ci_abs": ci,
        "method": note,
    }


def _ratio_variance(numerator: np.ndarray, denominator: np.ndarray) -> float:
    """Delta-method variance of ``sum(num) / sum(den)`` from per-unit rows."""
    n = np.asarray(numerator, dtype=float)
    d = np.asarray(denominator, dtype=float)
    if len(n) < 2 or len(d) < 2:
        return float("nan")
    mean_d = float(d.mean())
    if not np.isfinite(mean_d) or mean_d == 0.0:
        return float("nan")
    ratio = float(n.mean()) / mean_d
    cov = np.cov(n, d, ddof=1)
    var_unit = cov[0, 0] - 2.0 * ratio * cov[0, 1] + ratio**2 * cov[1, 1]
    return float(max(var_unit, 0.0) / (len(n) * mean_d**2))


def ratio_metric_test_from_samples(
    num_control: np.ndarray, den_control: np.ndarray,
    num_treatment: np.ndarray, den_treatment: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Delta-method test for a ratio metric using per-unit rows.

    This is the correct test for "revenue per visit"-style metrics: the unit of
    randomisation is the user, the metric is a ratio of two sums, and the
    variance has to account for the correlation between numerator and
    denominator (heavy spenders also visit more often).
    """
    n_c, d_c = np.asarray(num_control, dtype=float), np.asarray(den_control, dtype=float)
    n_t, d_t = np.asarray(num_treatment, dtype=float), np.asarray(den_treatment, dtype=float)
    r_c = float(n_c.sum() / d_c.sum()) if d_c.sum() else float("nan")
    r_t = float(n_t.sum() / d_t.sum()) if d_t.sum() else float("nan")
    var_c = _ratio_variance(n_c, d_c)
    var_t = _ratio_variance(n_t, d_t)
    diff = r_t - r_c
    se = float(np.sqrt(var_c + var_t)) if np.isfinite(var_c) and np.isfinite(var_t) else float("nan")
    if np.isfinite(se) and se > 0:
        z = float(diff / se)
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
        ci = normal_ci(diff, se, alpha)
    else:
        z, p_value, ci = float("nan"), float("nan"), (float("nan"), float("nan"))
    return {
        "control": r_c,
        "treatment": r_t,
        "abs_diff": diff,
        "rel_lift": diff / r_c if r_c else float("nan"),
        "se": se,
        "z": z,
        "p_value": p_value,
        "ci_abs": ci,
        "method": "delta_method",
    }


# --------------------------------------------------------------------------- #
# Experiment quality checks
# --------------------------------------------------------------------------- #
def srm_check(observed: Sequence[int], expected_ratio: Optional[Sequence[float]] = None) -> Dict[str, object]:
    """Sample-ratio-mismatch check via chi-square goodness of fit."""
    observed = np.asarray(observed, dtype=float)
    total = observed.sum()
    if expected_ratio is None:
        expected_ratio = np.ones(len(observed)) / len(observed)
    expected = np.asarray(expected_ratio, dtype=float) * total
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    p_value = float(1 - stats.chi2.cdf(chi2, df=len(observed) - 1))
    return {
        "observed": observed.astype(int).tolist(),
        "expected": expected.round(1).tolist(),
        "chi2": chi2,
        "p_value": p_value,
        "passed": p_value >= 0.001,  # industry practice: flag at 0.001, not 0.05
        "message": "SRM detected: randomisation or logging likely broken" if p_value < 0.001 else "No SRM",
    }


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> Dict[str, object]:
    """CUPED variance reduction using a pre-period covariate ``x``.

    Returns the adjusted outcome, theta, and the achieved variance reduction.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(x))
    y, x = y[mask], x[mask]
    theta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)) if np.var(x, ddof=1) > 0 else 0.0
    y_adj = y - theta * (x - x.mean())
    reduction = 1 - np.var(y_adj, ddof=1) / np.var(y, ddof=1) if np.var(y, ddof=1) > 0 else 0.0
    corr = float(np.corrcoef(y, x)[0, 1])
    return {
        "theta": theta,
        "variance_reduction": float(reduction),
        "corr": corr,
        "y_adjusted": y_adj,
        "n_used": int(len(y)),
    }


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled_sd = np.sqrt(((len(a_arr) - 1) * a_arr.var(ddof=1) + (len(b_arr) - 1) * b_arr.var(ddof=1)) / (len(a_arr) + len(b_arr) - 2))
    return float((b_arr.mean() - a_arr.mean()) / pooled_sd) if pooled_sd else 0.0


def bh_correction(p_values: Sequence[float], alpha: float = 0.05) -> List[Dict[str, float]]:
    """Benjamini-Hochberg FDR control, returns adjusted p-values in input order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return [
        {"p_value": float(pi), "p_adjusted": float(ai), "significant": bool(ai < alpha)}
        for pi, ai in zip(p, out)
    ]


def obrien_fleming_spend(n_looks: int, alpha: float = 0.05) -> List[float]:
    """Cumulative alpha spent at each interim look (O'Brien-Fleming-like)."""
    if n_looks < 1:
        return []
    fractions = np.arange(1, n_looks + 1) / n_looks
    z_target = stats.norm.ppf(1 - alpha / 2)
    spend = 2 * (1 - stats.norm.cdf(z_target / np.sqrt(fractions)))
    return [float(min(s, alpha)) for s in spend]


def sequential_p_value(estimate: float, se: float, tau: Optional[float] = None) -> Dict[str, float]:
    """Always-valid p-value based on a normal-mixture SPRT (mSPRT style).

    ``tau`` encodes the prior spread of plausible effects; it defaults to 1 se.
    """
    if se <= 0:
        return {"p_value": 1.0, "method": "mSPRT"}
    tau = tau or se
    z = estimate / se
    # Mixture likelihood ratio integral under a normal prior on the true effect.
    exponent = min((z**2) / 2 * (tau**2 / (se**2 + tau**2)), 700.0)  # clip to avoid float overflow
    ratio = np.sqrt(se**2 / (se**2 + tau**2)) * np.exp(exponent)
    p = float(min(1.0, 1 / max(ratio, 1e-12)))
    return {"p_value": p, "method": "mSPRT", "tau": tau}
