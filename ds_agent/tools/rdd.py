"""Regression discontinuity: sharp, fuzzy, and the checks that decide credibility.

An RDD estimate is only as good as three things, and all three are reported here
rather than assumed:

1. **A real discontinuity in treatment** - the first-stage jump. Fuzzy designs
   divide the outcome jump by the compliance jump (Wald ratio).
2. **No manipulation of the running variable** - if units can sort across the
   cutoff, the comparison is between two self-selected groups. Tested with a
   Poisson density-jump test around the cutoff.
3. **Local linearity and bandwidth** - the estimate is a limit from both sides,
   so it is re-estimated across bandwidths. A sign that flips with the bandwidth
   is a sign that the design is too weak to answer the question.

Covariates that cannot respond to treatment must be continuous at the cutoff; a
significant jump in one of them is the cheapest available falsification test.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def _kernel(u: np.ndarray, kind: str = "triangular") -> np.ndarray:
    u = np.abs(np.asarray(u, dtype=float))
    if kind == "uniform":
        return (u <= 1).astype(float)
    if kind == "epanechnikov":
        return np.where(u <= 1, 0.75 * (1 - u**2), 0.0)
    return np.where(u <= 1, 1 - u, 0.0)  # triangular (default)


def _local_linear(
    x: np.ndarray, y: np.ndarray, cutoff: float, bandwidth: float, side: str, kernel: str = "triangular"
) -> Dict[str, float]:
    """Weighted local linear fit on one side of the cutoff, evaluated at it.

    A local *constant* (kernel mean) is biased whenever the outcome has a slope;
    the linear term removes that first-order bias, which is why every credible
    RDD uses it.
    """
    offset = x - cutoff
    mask = (offset >= 0) & (offset <= bandwidth) if side == "right" else (offset <= 0) & (offset >= -bandwidth)
    if mask.sum() < 10:
        return {"intercept": float("nan"), "slope": float("nan"), "se": float("nan"), "n": int(mask.sum())}
    d = offset[mask]
    yy = y[mask]
    w = _kernel(d / bandwidth, kernel)
    if w.sum() < 5:
        return {"intercept": float("nan"), "slope": float("nan"), "se": float("nan"), "n": int(mask.sum())}

    X = np.column_stack([np.ones(len(d)), d])
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], yy * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    dof = max(1, len(d) - 2)
    # HC1 sandwich: residual heteroskedasticity is the norm around a cutoff.
    xtx_inv = np.linalg.pinv(Xw.T @ Xw)
    meat = (Xw * resid[:, None] ** 2).T @ Xw
    cov = xtx_inv @ meat @ xtx_inv * (len(d) / dof)
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "se": float(np.sqrt(max(cov[0, 0], 0.0))),
        "n": int(mask.sum()),
    }


def _rule_of_thumb_bandwidth(x: np.ndarray) -> float:
    """CCT-style starting bandwidth: h = 5.4 * sd(x) * n^(-1/5)."""
    n = len(x)
    return float(5.4 * np.std(x, ddof=1) * n ** (-0.2))


def _cv_mse(x: np.ndarray, y: np.ndarray, cutoff: float, bandwidth: float, seed: int = 0, max_points: int = 120) -> float:
    """Leave-one-out MSE of the local linear fit near the cutoff.

    Evaluated on points within half a bandwidth of the cutoff, because that is
    the region the estimate actually uses; doing CV over the whole support would
    happily pick a bandwidth so wide that the "local" fit is a global regression.
    """
    near = np.flatnonzero(np.abs(x - cutoff) <= bandwidth / 2)
    if len(near) < 20:
        return float("inf")
    rng = np.random.default_rng(seed)
    if len(near) > max_points:
        near = rng.choice(near, max_points, replace=False)
    errors = []
    for i in near:
        side = "right" if x[i] >= cutoff else "left"
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        fit = _local_linear(x[mask], y[mask], cutoff, bandwidth, side)
        if np.isfinite(fit["intercept"]):
            offset = x[i] - cutoff
            errors.append((y[i] - (fit["intercept"] + fit["slope"] * offset)) ** 2)
    return float(np.mean(errors)) if errors else float("inf")


def select_bandwidth(
    x: np.ndarray, y: np.ndarray, cutoff: float, candidates: Optional[Sequence[float]] = None, seed: int = 0
) -> Dict[str, object]:
    """Pick the bandwidth with the lowest leave-one-out MSE near the cutoff."""
    h0 = _rule_of_thumb_bandwidth(x)
    candidates = list(candidates) if candidates is not None else [h0 * m for m in (0.5, 0.75, 1.0, 1.25, 1.5)]
    rows = []
    for h in candidates:
        rows.append({"bandwidth": float(h), "cv_mse": _cv_mse(x, y, cutoff, h, seed=seed)})
    table = pd.DataFrame(rows).sort_values("cv_mse")
    best = float(table.iloc[0]["bandwidth"]) if len(table) and np.isfinite(table.iloc[0]["cv_mse"]) else h0
    return {"bandwidth": best, "table": table, "rule_of_thumb": h0}


def density_test(x: np.ndarray, cutoff: float, bandwidth: Optional[float] = None, n_bins: int = 20) -> Dict[str, float]:
    """Poisson density-jump test (the McCrary idea, in histogram form).

    Counts in equal-width bins either side of the cutoff are modelled with a
    Poisson GLM on ``|bin centre - cutoff|`` plus a right-side indicator. A
    significant indicator means the density jumps at the cutoff, i.e. units can
    sort across it and the design is compromised.
    """
    span = bandwidth or _rule_of_thumb_bandwidth(x)
    window = (x >= cutoff - 2 * span) & (x <= cutoff + 2 * span)
    sample = x[window]
    if len(sample) < 100:
        return {"coef": float("nan"), "p_value": float("nan"), "ratio": float("nan"), "n": int(len(sample))}
    width = 4 * span / n_bins
    edges = np.arange(cutoff - 2 * span, cutoff + 2 * span + width * 0.5, width)
    counts, edges = np.histogram(sample, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    frame = pd.DataFrame(
        {
            "count": counts,
            "right": (centers > cutoff).astype(float),
            "distance": np.abs(centers - cutoff) / span,
        }
    )
    try:
        fit = sm.GLM(frame["count"], sm.add_constant(frame[["right", "distance"]]), family=sm.families.Poisson()).fit()
        coef = float(fit.params["right"])
        return {
            "coef": coef,
            "p_value": float(fit.pvalues["right"]),
            "ratio": float(np.exp(coef)),
            "n": int(len(sample)),
            "manipulation": bool(fit.pvalues["right"] < 0.05),
        }
    except Exception:  # noqa: BLE001 - degenerate histograms are reported, not fatal
        return {"coef": float("nan"), "p_value": float("nan"), "ratio": float("nan"), "n": int(len(sample))}


def regression_discontinuity(
    df: pd.DataFrame,
    outcome: str,
    running: str,
    cutoff: float = 0.0,
    bandwidth: Optional[float] = None,
    kernel: str = "triangular",
    fuzzy: Optional[str] = None,
    covariates: Sequence[str] = (),
    alpha: float = 0.05,
    bandwidth_grid: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5),
    lang: str = "zh",
) -> Dict[str, object]:
    """Estimate a sharp or fuzzy regression-discontinuity effect.

    ``running`` is the assignment variable, ``cutoff`` the threshold. Pass
    ``fuzzy`` with the name of a 0/1 treatment column when compliance is
    imperfect: the estimate becomes the Wald ratio (jump in outcome / jump in
    treatment), which is a LATE for compliers at the cutoff, not an ATE.
    """
    work = df[[outcome, running] + ([fuzzy] if fuzzy else []) + list(covariates)].dropna().copy()
    x = work[running].to_numpy(dtype=float)
    y = work[outcome].to_numpy(dtype=float)
    if len(x) < 100:
        raise ValueError("RDD needs at least 100 observations")

    selection = select_bandwidth(x, y, cutoff)
    h = float(bandwidth) if bandwidth is not None else float(selection["bandwidth"])

    right = _local_linear(x, y, cutoff, h, "right", kernel)
    left = _local_linear(x, y, cutoff, h, "left", kernel)
    if not (np.isfinite(right["intercept"]) and np.isfinite(left["intercept"])):
        raise ValueError("not enough observations within the bandwidth on both sides")

    jump = right["intercept"] - left["intercept"]
    se = float(np.sqrt(right["se"] ** 2 + left["se"] ** 2))
    p_value = float(2 * (1 - stats.norm.cdf(abs(jump / se)))) if se > 0 else float("nan")
    crit = float(stats.norm.ppf(1 - alpha / 2))

    first_stage = None
    estimate, estimate_se = jump, se
    if fuzzy:
        d = (work[fuzzy].to_numpy(dtype=float) > 0).astype(float)
        d_right = _local_linear(x, d, cutoff, h, "right", kernel)
        d_left = _local_linear(x, d, cutoff, h, "left", kernel)
        jump_d = d_right["intercept"] - d_left["intercept"]
        se_d = float(np.sqrt(d_right["se"] ** 2 + d_left["se"] ** 2))
        first_stage = {
            "jump": float(jump_d),
            "se": se_d,
            "p_value": float(2 * (1 - stats.norm.cdf(abs(jump_d / se_d)))) if se_d > 0 else float("nan"),
            "compliance": bool(jump_d > 0.1),
        }
        if abs(jump_d) < 1e-6:
            raise ValueError("first stage is zero: nothing changes at the cutoff")
        estimate = jump / jump_d
        # Delta method, covariances ignored (conservative).
        estimate_se = float(
            np.sqrt((se / jump_d) ** 2 + (jump * se_d / jump_d**2) ** 2)
        )
        p_value = float(2 * (1 - stats.norm.cdf(abs(estimate / estimate_se)))) if estimate_se > 0 else float("nan")

    # Bandwidth sensitivity: the single most informative robustness table.
    sens = []
    for mult in bandwidth_grid:
        hh = h * mult
        r = _local_linear(x, y, cutoff, hh, "right", kernel)
        l = _local_linear(x, y, cutoff, hh, "left", kernel)
        if np.isfinite(r["intercept"]) and np.isfinite(l["intercept"]):
            tau = r["intercept"] - l["intercept"]
            sens.append({"bandwidth": float(hh), "effect": float(tau), "n": int(r["n"] + l["n"])})
    sens_table = pd.DataFrame(sens)
    stable = bool(len(sens_table) > 1 and np.all(np.sign(sens_table["effect"]) == np.sign(estimate)))

    # Plot data: binned means make the discontinuity visible without overplotting
    # ten thousand points, and the two local-linear fits are what the estimate
    # actually rests on.
    inside = np.abs(x - cutoff) <= h
    bins = pd.DataFrame({"x": x[inside], "y": y[inside]})
    if len(bins) >= 20:
        bins["bin"] = pd.cut(bins["x"], bins=24)
        plot_bins = (
            bins.groupby("bin", observed=True)
            .agg(x=("x", "mean"), y=("y", "mean"))
            .reset_index(drop=True)
            .dropna()
        )
    else:
        plot_bins = bins[["x", "y"]]
    grid = np.linspace(-h, h, 60)
    plot_left = pd.DataFrame(
        {"x": cutoff + grid[grid < 0], "fit": left["intercept"] + left["slope"] * grid[grid < 0]}
    )
    plot_right = pd.DataFrame(
        {"x": cutoff + grid[grid >= 0], "fit": right["intercept"] + right["slope"] * grid[grid >= 0]}
    )

    # Covariate balance: a jump in a pre-determined covariate falsifies the design.
    balance = []
    for cov in covariates:
        cy = work[cov].to_numpy(dtype=float)
        r = _local_linear(x, cy, cutoff, h, "right", kernel)
        l = _local_linear(x, cy, cutoff, h, "left", kernel)
        if not (np.isfinite(r["intercept"]) and np.isfinite(l["intercept"])):
            continue
        cj = r["intercept"] - l["intercept"]
        cse = float(np.sqrt(r["se"] ** 2 + l["se"] ** 2))
        balance.append(
            {
                "covariate": cov,
                "jump": float(cj),
                "p_value": float(2 * (1 - stats.norm.cdf(abs(cj / cse)))) if cse > 0 else np.nan,
            }
        )
    balance_ok = all(row["p_value"] > 0.05 for row in balance if np.isfinite(row["p_value"])) if balance else True

    # Placebo cutoffs inside each side: a "jump" there would mean the model, not
    # the treatment, is producing the discontinuity.
    placebos = []
    for offset in (-0.5, 0.5):
        fake = cutoff + offset * h
        r = _local_linear(x, y, fake, h / 2, "right", kernel)
        l = _local_linear(x, y, fake, h / 2, "left", kernel)
        if np.isfinite(r["intercept"]) and np.isfinite(l["intercept"]):
            pj = r["intercept"] - l["intercept"]
            pse = float(np.sqrt(r["se"] ** 2 + l["se"] ** 2))
            placebos.append(
                {
                    "cutoff": float(fake),
                    "effect": float(pj),
                    "p_value": float(2 * (1 - stats.norm.cdf(abs(pj / pse)))) if pse > 0 else np.nan,
                }
            )

    density = density_test(x, cutoff, h)

    relative = estimate / left["intercept"] if left["intercept"] else float("nan")
    narrative = _tr(
        lang,
        f"RDD ({'fuzzy' if fuzzy else 'sharp'}, bandwidth {h:.4g}): jump {jump:+.4g} "
        f"(left {left['intercept']:.4g} -> right {right['intercept']:.4g}), "
        + (f"first stage {first_stage['jump']:.3f}, " if first_stage else "")
        + f"{'LATE' if fuzzy else 'effect'} {estimate:+.4g} ({relative:+.2%} relative, 95% CI "
        f"{estimate - crit * estimate_se:+.4g}..{estimate + crit * estimate_se:+.4g}, p={p_value:.4f}); "
        f"density test p={density['p_value']:.3f} ({'no manipulation detected' if density['p_value'] > 0.05 else 'manipulation risk'}), "
        f"bandwidth sensitivity {'stable' if stable else 'unstable'}, covariate balance {'passes' if balance_ok else 'fails'}.",
        f"RDD ({'fuzzy' if fuzzy else 'sharp'}, bandwidth {h:.4g}): jump {jump:+.4g} "
        f"(left {left['intercept']:.4g} -> right {right['intercept']:.4g}), "
        + (f"first stage {first_stage['jump']:.3f}, " if first_stage else "")
        + f"{'LATE' if fuzzy else 'effect'} {estimate:+.4g} ({relative:+.2%} relative, 95% CI "
        f"{estimate - crit * estimate_se:+.4g}..{estimate + crit * estimate_se:+.4g}, p={p_value:.4f}); "
        f"density test p={density['p_value']:.3f} ({'no manipulation detected' if density['p_value'] > 0.05 else 'manipulation risk'}), "
        f"bandwidth sensitivity {'stable' if stable else 'unstable'}, covariate balance {'passes' if balance_ok else 'fails'}.",
    )

    return {
        "method": "fuzzy_rdd" if fuzzy else "sharp_rdd",
        "effect": float(estimate),
        "se": float(estimate_se),
        "p_value": p_value,
        "ci": (float(estimate - crit * estimate_se), float(estimate + crit * estimate_se)),
        "relative": float(relative),
        "jump_outcome": float(jump),
        "left_intercept": float(left["intercept"]),
        "right_intercept": float(right["intercept"]),
        "left_slope": float(left["slope"]),
        "right_slope": float(right["slope"]),
        "bandwidth": float(h),
        "bandwidth_selection": selection,
        "n_left": int(left["n"]),
        "n_right": int(right["n"]),
        "first_stage": first_stage,
        "density": density,
        "bandwidth_sensitivity": sens_table,
        "bandwidth_stable": stable,
        "covariate_balance": balance,
        "covariate_balance_ok": bool(balance_ok),
        "placebo_cutoffs": placebos,
        "cutoff": float(cutoff),
        "plot": {"bins": plot_bins, "left": plot_left, "right": plot_right},
        "assumption": "Potential outcomes are continuous in the running variable at the cutoff.",
        "narrative": narrative,
    }


def _tr(lang: str, zh: str, en: str) -> str:
    return zh if lang == "zh" else en
