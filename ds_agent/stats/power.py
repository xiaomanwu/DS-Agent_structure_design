"""Sample size, power and MDE calculators.

A single generic solver powers every metric family, so the numbers stay
internally consistent: power() is the exact inverse of sample_size().
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import optimize, stats


def _power_two_sided(delta: float, se: float, alpha: float) -> float:
    """Power of a two-sided normal test given an effect and its standard error."""
    if se <= 0:
        return 1.0 if abs(delta) > 0 else alpha
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z = abs(delta) / se
    return float(stats.norm.cdf(z - z_alpha) + stats.norm.cdf(-z - z_alpha))


def _solve(fn: Callable[[float], float], target: float, lo: float, hi: float, increasing: bool = True) -> float:
    """Robust 1-D root find on a monotone function."""
    f = (lambda x: fn(x) - target) if increasing else (lambda x: target - fn(x))
    if f(lo) > 0:
        return lo
    if f(hi) < 0:
        return hi
    return float(optimize.brentq(f, lo, hi, maxiter=200))


def _se_proportion(p1: float, p2: float, n1: float, n2: float) -> float:
    return float(np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))


def sample_size_proportion(
    baseline: float,
    mde_rel: Optional[float] = None,
    mde_abs: Optional[float] = None,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 1.0,
) -> Dict[str, float]:
    """Sample size per arm for a conversion-rate style metric.

    ``allocation`` is treatment:control (1.0 == 50/50). Returned ``n_control``
    is the control arm size; ``n_treatment = allocation * n_control``.
    """
    if mde_abs is None and mde_rel is None:
        raise ValueError("provide mde_rel or mde_abs")
    p1 = float(baseline)
    p2 = float(baseline) + (mde_abs if mde_abs is not None else baseline * mde_rel)
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("baseline and mde must keep both proportions in (0, 1)")

    def power_at(n: float) -> float:
        return _power_two_sided(p2 - p1, _se_proportion(p1, p2, n, allocation * n), alpha)

    n_control = _solve(power_at, power, 10, 5e8)
    n_control = int(np.ceil(n_control))
    return {
        "metric_type": "proportion",
        "control_rate": p1,
        "treatment_rate": p2,
        "abs_effect": p2 - p1,
        "rel_effect": (p2 - p1) / p1,
        "n_control": n_control,
        "n_treatment": int(np.ceil(n_control * allocation)),
        "n_total": int(np.ceil(n_control * (1 + allocation))),
        "alpha": alpha,
        "power": power,
        "achieved_power": power_at(n_control),
    }


def sample_size_mean(
    baseline_mean: float,
    sd: float,
    mde_rel: Optional[float] = None,
    mde_abs: Optional[float] = None,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 1.0,
    cv_treatment: Optional[float] = None,
) -> Dict[str, float]:
    """Sample size per arm for a continuous metric (revenue per user, time spent)."""
    if mde_abs is None and mde_rel is None:
        raise ValueError("provide mde_rel or mde_abs")
    delta = mde_abs if mde_abs is not None else baseline_mean * mde_rel
    sd_t = sd if cv_treatment is None else cv_treatment * (baseline_mean + delta)

    def power_at(n: float) -> float:
        se = np.sqrt(sd**2 / n + sd_t**2 / (allocation * n))
        return _power_two_sided(delta, float(se), alpha)

    n_control = int(np.ceil(_solve(power_at, power, 10, 5e8)))
    return {
        "metric_type": "mean",
        "baseline_mean": baseline_mean,
        "abs_effect": delta,
        "rel_effect": delta / baseline_mean,
        "sd": sd,
        "n_control": n_control,
        "n_treatment": int(np.ceil(n_control * allocation)),
        "n_total": int(np.ceil(n_control * (1 + allocation))),
        "alpha": alpha,
        "power": power,
        "achieved_power": power_at(n_control),
    }


def sample_size_count(
    baseline_mean: float,
    mde_rel: float,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 1.0,
    dispersion: float = 1.0,
) -> Dict[str, float]:
    """Sample size for a count metric (orders per user) using a quasi-Poisson variance.

    ``dispersion`` > 1 accounts for over-dispersion (variance = dispersion * mean).
    """
    delta = baseline_mean * mde_rel
    var = dispersion * baseline_mean

    def power_at(n: float) -> float:
        se = np.sqrt(var / n + var / (allocation * n))
        return _power_two_sided(delta, float(se), alpha)

    n_control = int(np.ceil(_solve(power_at, power, 10, 5e8)))
    return {
        "metric_type": "count",
        "baseline_mean": baseline_mean,
        "abs_effect": delta,
        "rel_effect": mde_rel,
        "dispersion": dispersion,
        "n_control": n_control,
        "n_treatment": int(np.ceil(n_control * allocation)),
        "n_total": int(np.ceil(n_control * (1 + allocation))),
        "alpha": alpha,
        "power": power,
        "achieved_power": power_at(n_control),
    }


def sample_size_ratio_metric(
    mean_numerator: float,
    mean_denominator: float,
    cv_numerator: float = 1.0,
    cv_denominator: float = 0.5,
    corr: float = 0.3,
    mde_rel: float = 0.02,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 1.0,
) -> Dict[str, float]:
    """Sample size for a ratio-of-sums metric (GMV/session, CTR, ARPU) via the delta method."""
    r = mean_numerator / mean_denominator
    var_num = (cv_numerator * mean_numerator) ** 2
    var_den = (cv_denominator * mean_denominator) ** 2
    cov = corr * np.sqrt(var_num * var_den)
    # Var(X/Y) ~= (1/(n * E[Y]^2)) * (Var(X) - 2R*Cov(X,Y) + R^2*Var(Y))
    per_unit_var = (var_num - 2 * r * cov + r**2 * var_den) / mean_denominator**2
    delta = r * mde_rel

    def power_at(n: float) -> float:
        se = np.sqrt(per_unit_var / n + per_unit_var / (allocation * n))
        return _power_two_sided(delta, float(se), alpha)

    n_control = int(np.ceil(_solve(power_at, power, 10, 5e8)))
    return {
        "metric_type": "ratio",
        "baseline_ratio": r,
        "abs_effect": delta,
        "rel_effect": mde_rel,
        "per_unit_variance": float(per_unit_var),
        "n_control": n_control,
        "n_treatment": int(np.ceil(n_control * allocation)),
        "n_total": int(np.ceil(n_control * (1 + allocation))),
        "alpha": alpha,
        "power": power,
        "achieved_power": power_at(n_control),
        "note": "Delta-method approximation; simulate_power() is advisable for heavy tails.",
    }


def mde_proportion(
    n_per_arm: int, baseline: float, alpha: float = 0.05, power: float = 0.8, allocation: float = 1.0
) -> Dict[str, float]:
    """Minimum detectable absolute/relative effect for a given sample size."""
    def achievable(delta: float) -> float:
        p2 = baseline + delta
        return _power_two_sided(delta, _se_proportion(baseline, p2, n_per_arm, allocation * n_per_arm), alpha)

    lo, hi = 1e-6, min(1 - baseline - 1e-6, 0.5)
    delta = _solve(achievable, power, lo, hi)
    return {
        "mde_abs": float(delta),
        "mde_rel": float(delta / baseline),
        "n_control": n_per_arm,
        "n_treatment": int(n_per_arm * allocation),
    }


def duration_days(n_total: int, daily_traffic: float, allocation: float = 1.0, ramp_up: float = 1.0) -> Dict[str, float]:
    """Calendar days needed to collect ``n_total`` users, plus a full-week rounding."""
    eligible = max(daily_traffic * (allocation / (1 + allocation)) * 2, 1e-9) * ramp_up
    raw_days = n_total / eligible
    return {
        "eligible_users_per_day": round(eligible, 1),
        "raw_days": round(float(raw_days), 1),
        "recommended_days": int(np.ceil(max(raw_days, 7) / 7) * 7),
        "note": "Always cover at least one full week and one business cycle (2 weeks for B2B).",
    }


def cluster_design_effect(cluster_size: float, icc: float) -> Dict[str, float]:
    """Design effect for cluster/geo randomisation: 1 + (m - 1) * ICC."""
    deff = 1 + (cluster_size - 1) * icc
    return {
        "cluster_size": cluster_size,
        "icc": icc,
        "design_effect": float(deff),
        "inflation_factor": float(np.sqrt(deff)),
        "note": "Multiply the individual-level sample size by design_effect.",
    }


def simulate_power(
    metric: str = "proportion",
    baseline: float = 0.03,
    rel_effect: float = 0.05,
    n_per_arm: int = 20_000,
    alpha: float = 0.05,
    reps: int = 400,
    seed: int = 0,
    heavy_tail_sigma: float = 1.2,
) -> Dict[str, float]:
    """Monte-Carlo power check; recommended for ratio or heavy-tailed metrics."""
    rng = np.random.default_rng(seed)
    significant = 0
    for _ in range(reps):
        if metric == "proportion":
            p1 = baseline
            p2 = baseline * (1 + rel_effect)
            x1 = rng.binomial(n_per_arm, p1)
            x2 = rng.binomial(n_per_arm, p2)
            p_hat = (x1 + x2) / (2 * n_per_arm)
            se = np.sqrt(p_hat * (1 - p_hat) * 2 / n_per_arm)
            z = ((x2 - x1) / n_per_arm) / se if se else 0
        else:  # "mean" with a log-normal revenue distribution
            mu = np.log(max(baseline, 1e-9)) - heavy_tail_sigma**2 / 2
            a = rng.lognormal(mu, heavy_tail_sigma, n_per_arm)
            b = rng.lognormal(mu, heavy_tail_sigma, n_per_arm) * (1 + rel_effect)
            se = np.sqrt(a.var(ddof=1) / n_per_arm + b.var(ddof=1) / n_per_arm)
            z = (b.mean() - a.mean()) / se if se else 0
        if abs(z) > stats.norm.ppf(1 - alpha / 2):
            significant += 1
    return {
        "empirical_power": significant / reps,
        "reps": reps,
        "n_per_arm": n_per_arm,
        "metric": metric,
        "rel_effect": rel_effect,
    }


def power_curve(
    metric: str = "proportion",
    baseline: float = 0.03,
    mde_grid: Sequence[float] = (0.01, 0.02, 0.03, 0.05, 0.08, 0.10),
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 1.0,
) -> List[Dict[str, float]]:
    """Sample size for a grid of relative MDEs, ready for plotting."""
    out: List[Dict[str, float]] = []
    for mde in mde_grid:
        if metric == "proportion":
            res = sample_size_proportion(baseline, mde_rel=mde, alpha=alpha, power=power, allocation=allocation)
        else:
            res = sample_size_mean(baseline, sd=baseline, mde_rel=mde, alpha=alpha, power=power, allocation=allocation)
        out.append({"mde_rel": mde, "n_total": res["n_total"], "n_per_arm": res["n_control"]})
    return out
