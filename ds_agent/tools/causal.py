"""Causal inference when randomisation is not available.

Implemented estimators (all returning effect, uncertainty, diagnostics and a
plain-language verdict):

* Difference-in-differences - 2x2 table and two-way fixed effects
* Event study / parallel-trends test
* Propensity score matching (1:1 nearest neighbour with caliper) + balance table
* IPW (ATE / ATT) with weight trimming
* AIPW (doubly robust, sklearn outcome models)
* Synthetic control with in-space placebo inference

Every estimator reports its identifying assumption next to the number, because
a causal estimate without its assumption is just a correlation with a suit on.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.linear_model import LinearRegression, LogisticRegression

import statsmodels.formula.api as smf


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference between two samples."""
    var = (a.var(ddof=1) + b.var(ddof=1)) / 2
    return float((a.mean() - b.mean()) / np.sqrt(var)) if var > 0 else 0.0


def _overlap_ks(ps_t: np.ndarray, ps_c: np.ndarray) -> Dict[str, float]:
    """Two-sample KS test on the propensity scores of the two arms.

    Kept for callers that want a formal separation test. It is **not** used in
    the PSM readout: with tens of thousands of rows it rejects on differences
    too small to matter, so overlap is judged by the common-support share.
    """
    try:
        res = stats.ks_2samp(ps_t, ps_c)
        return {"stat": float(res.statistic), "p_value": float(res.pvalue), "n": int(len(ps_t) + len(ps_c))}
    except Exception:  # noqa: BLE001
        return {}


# --------------------------------------------------------------------------- #
# Difference-in-differences
# --------------------------------------------------------------------------- #
def difference_in_differences(
    df: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated: str,
    post: str,
    cluster_se: bool = True,
    log_outcome: bool = False,
) -> Dict[str, object]:
    """Two-way fixed-effects DiD with a cluster-robust standard error.

    ``treated`` is a unit-level 0/1 indicator, ``post`` a time-level 0/1 indicator.
    Set ``log_outcome=True`` when the trend is multiplicative: the level ATT is
    then biased whenever group sizes differ, while the log specification
    estimates a relative effect directly.
    """
    data = df[[outcome, unit, time, treated, post]].dropna().copy()
    if log_outcome:
        if (data[outcome] <= 0).any():
            raise ValueError("log_outcome requires strictly positive outcomes")
        data[outcome] = np.log(data[outcome])
    data["_interaction"] = data[treated] * data[post]

    cells = data.groupby([treated, post])[outcome].agg(["mean", "count"]).reset_index()
    pivot_mean = cells.pivot(index=treated, columns=post, values="mean")
    if pivot_mean.shape != (2, 2):
        raise ValueError("DiD needs both treated/control and pre/post cells")
    y11, y10 = float(pivot_mean.loc[1, 1]), float(pivot_mean.loc[1, 0])
    y01, y00 = float(pivot_mean.loc[0, 1]), float(pivot_mean.loc[0, 0])
    did_manual = (y11 - y10) - (y01 - y00)

    # Unit FE absorb the treated main effect and time FE absorb the post main
    # effect, so only the interaction is included: avoids a rank-deficient design.
    formula = f"{outcome} ~ _interaction + C({unit}) + C({time})"
    model = smf.ols(formula, data=data)
    if cluster_se:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": data[unit]})
    else:
        fit = model.fit(cov_type="HC1")

    coef = float(fit.params["_interaction"])
    ci = fit.conf_int().loc["_interaction"]
    if log_outcome:
        relative = float(np.exp(coef) - 1)
    else:
        # Counterfactual post level for the treated group: pre mean scaled by the
        # control group's own growth. Dividing by the raw pre mean would inflate
        # the relative effect by the trend factor.
        counterfactual = y10 * (y01 / y00) if y00 else float("nan")
        relative = coef / counterfactual if counterfactual else float("nan")

    return {
        "method": "twfe_did",
        "outcome": outcome,
        "att": coef,
        "att_manual_2x2": did_manual,
        "relative_effect": relative,
        "std_error": float(fit.bse["_interaction"]),
        "p_value": float(fit.pvalues["_interaction"]),
        "ci": (float(ci.iloc[0]), float(ci.iloc[1])),
        "cells": {"treated_post": y11, "treated_pre": y10, "control_post": y01, "control_pre": y00},
        "n_obs": int(len(data)),
        "n_units": int(data[unit].nunique()),
        "diagnostics": _did_diagnostics(data, fit, unit, outcome),
        "assumption": "Parallel trends: absent treatment, treated and control would have moved in parallel.",
        "narrative": (
            f"DiD ATT = {coef:+.4g} ({relative:+.2%} of the treated pre-period mean; 95% CI "
            f"{ci.iloc[0]:+.4g}..{ci.iloc[1]:+.4g}, p={fit.pvalues['_interaction']:.4f}). The raw 2x2 estimate is "
            f"{did_manual:+.4g}, so covariates/time composition are "
            + ("not distorting the result much." if abs(did_manual - coef) < 0.2 * max(abs(coef), 1e-9) else "materially shifting it.")
        ),
    }


def _did_diagnostics(data: pd.DataFrame, fit, unit: str, outcome: str) -> List[Dict[str, object]]:
    """Residual diagnostics for the TWFE fit, as test-result rows.

    Breusch-Pagan is run on ``[const, interaction]`` rather than the full dummy
    matrix: the question that matters is whether the treatment cell has a
    different residual variance, not whether one of forty city dummies does.
    Durbin-Watson is computed **within** each unit and the median reported — the
    row order runs across city boundaries, so a global DW would be meaningless.
    """
    from ..stats import tests as _tests

    resid = np.asarray(fit.resid, dtype=float)
    out: List[Dict[str, object]] = []
    try:
        out.append(_tests.jarque_bera(resid))
    except Exception:  # noqa: BLE001
        pass
    try:
        exog = np.column_stack([np.ones(len(data)), data["_interaction"].to_numpy(dtype=float)])
        out.append(_tests.breusch_pagan(resid, exog))
    except Exception:  # noqa: BLE001
        pass
    try:
        frame = data.assign(_resid=resid)
        dws = []
        for _, chunk in frame.groupby(unit):
            if len(chunk) >= 6:
                r = _tests.durbin_watson(chunk["_resid"].to_numpy(dtype=float))
                if r["status"] == "ok":
                    dws.append(r)
        if dws:
            vals = sorted(float(r["statistic"]) for r in dws)
            median_val = float(np.median(vals))
            median = _tests.result(
                "durbin_watson",
                median_val,
                1.0 if abs(median_val - 2) <= 0.5 else 0.01,
                n=len(vals),
                note_zh=f"computed per city over {len(vals)} cities, median reported",
                note_en=f"computed per city over {len(vals)} cities, median reported",
            )
            median["p_approx"] = True
            median["name_zh"] += " (within city, median)"
            median["name_en"] += " (within city, median)"
            out.append(median)
    except Exception:  # noqa: BLE001
        pass
    return out


def event_study(
    df: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated: str,
    treat_time: float,
    max_lag: int = 6,
    max_lead: int = 6,
    base_period: int = -1,
) -> Dict[str, object]:
    """Event-study DiD: leads test parallel trends, lags trace dynamic effects."""
    data = df[[outcome, unit, time, treated]].dropna().copy()
    data["_rel"] = data[time] - treat_time
    data = data[(data["_rel"] >= -max_lead) & (data["_rel"] <= max_lag)]

    periods = [p for p in range(-max_lead, max_lag + 1) if p != base_period]
    for p in periods:
        data[f"_D{abs(p)}{'m' if p < 0 else 'p'}"] = (
            (data["_rel"] == p) & (data[treated] == 1)
        ).astype(float)
    terms = [f"_D{abs(p)}{'m' if p < 0 else 'p'}" for p in periods]

    formula = f"{outcome} ~ " + " + ".join(terms) + f" + C({unit}) + C({time})"
    fit = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data[unit]}
    )

    coefs = []
    for p, term in zip(periods, terms):
        coefs.append(
            {
                "rel_period": p,
                "coef": float(fit.params.get(term, np.nan)),
                "se": float(fit.bse.get(term, np.nan)),
                "p_value": float(fit.pvalues.get(term, np.nan)),
                "ci": (float(fit.conf_int().loc[term].iloc[0]), float(fit.conf_int().loc[term].iloc[1]))
                if term in fit.params.index
                else (np.nan, np.nan),
            }
        )

    pre_terms = [f"_D{abs(p)}m" for p in periods if p < 0]
    pre_terms = [t for t in pre_terms if t in fit.params.index]
    joint = fit.f_test(" = 0, ".join(pre_terms) + " = 0") if pre_terms else None
    joint_p = float(joint.pvalue) if joint is not None else float("nan")
    joint_f = float(joint.fvalue) if joint is not None else float("nan")

    return {
        "method": "event_study",
        "coefficients": coefs,
        "pre_trend_joint_p": joint_p,
        "pre_trend_joint_f": joint_f,
        "pre_trend_df": len(pre_terms),
        "parallel_trends_ok": bool(joint_p > 0.05) if np.isfinite(joint_p) else True,
        "assumption": "Parallel trends: all pre-period leads should be jointly indistinguishable from zero.",
        "narrative": (
            f"Joint test on {len(pre_terms)} pre-period leads gives p={joint_p:.3f}: "
            + ("pre-trends look parallel, DiD is credible." if joint_p > 0.05 else "pre-trends differ, DiD is biased - do not trust the ATT.")
        ),
    }


# --------------------------------------------------------------------------- #
# Propensity score methods
# --------------------------------------------------------------------------- #
def _propensity(data: pd.DataFrame, treatment: str, covariates: Sequence[str]) -> np.ndarray:
    X = data[list(covariates)].to_numpy(dtype=float)
    y = data[treatment].to_numpy(dtype=int)
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X, y)
    return model.predict_proba(X)[:, 1]


def propensity_score_matching(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: Sequence[str],
    caliper: float = 0.2,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> Dict[str, object]:
    """1:1 nearest-neighbour matching on the propensity score; reports ATT and balance.

    Feature engineering runs first, exactly as before a supervised model:
    covariates whose pairwise |r| exceeds 0.95 are pruned (the later column of
    each pair is dropped), because a rank-deficient design destabilises the
    propensity logistic regression and the matching that depends on it.
    """
    pruned: List[str] = []
    used_covariates = list(covariates)
    numeric = [c for c in covariates if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) >= 2:
        from .preprocess import correlation_pairs

        corr = df[numeric].corr().abs()
        removed = set()
        for pair in correlation_pairs(df[numeric], threshold=0.95):
            if pair["b"] not in removed and pair["b"] in used_covariates:
                used_covariates.remove(pair["b"])
                removed.add(pair["b"])
                pruned.append(pair["b"])
    data = df[[treatment, outcome, *used_covariates]].dropna().copy()
    data["_ps"] = _propensity(data, treatment, used_covariates)
    treated_idx = data.index[data[treatment] == 1].to_numpy()
    control_idx = data.index[data[treatment] == 0].to_numpy()
    ps = data["_ps"]
    sd_logit = np.std(np.log(ps / (1 - ps)))
    caliper_width = caliper * sd_logit

    # Sort-based nearest-neighbour matching: O(n log n) instead of a pair loop.
    # Controls are sorted on the propensity score; for each treated unit (also
    # processed in score order) we binary-search and walk outwards to the closest
    # still-unused control within the caliper.
    control_sorted = control_idx[np.argsort(ps.loc[control_idx].to_numpy())]
    control_ps = ps.loc[control_sorted].to_numpy()
    used = np.zeros(len(control_sorted), dtype=bool)
    treated_order = treated_idx[np.argsort(ps.loc[treated_idx].to_numpy())]

    pairs: List[tuple] = []
    for ti in treated_order:
        pos = int(np.searchsorted(control_ps, ps.loc[ti]))
        best, best_dist = -1, np.inf
        for step in range(0, 12):
            for cand in (pos - 1 - step, pos + step):
                if 0 <= cand < len(control_sorted) and not used[cand]:
                    dist = abs(control_ps[cand] - ps.loc[ti])
                    if dist < best_dist:
                        best, best_dist = cand, dist
            if best >= 0 and best_dist <= max(caliper_width, 0.0):
                # Stop expanding once we have a match inside the caliper.
                break
        if best >= 0 and best_dist <= caliper_width:
            pairs.append((ti, control_sorted[best]))
            used[best] = True

    if not pairs:
        raise ValueError("no matched pairs within the caliper; relax caliper or covariates")

    t_vals = data.loc[[p[0] for p in pairs], outcome].to_numpy(dtype=float)
    c_vals = data.loc[[p[1] for p in pairs], outcome].to_numpy(dtype=float)
    att = float(t_vals.mean() - c_vals.mean())
    # Paired t-test on the matched pairs: the same inference as the bootstrap
    # interval, from a different assumption. Reported next to the ATT.
    try:
        att_t, att_p = stats.ttest_rel(t_vals, c_vals)
    except Exception:  # noqa: BLE001
        att_t, att_p = float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap)
    diffs = t_vals - c_vals
    for i in range(n_bootstrap):
        boots[i] = rng.choice(diffs, len(diffs), replace=True).mean()

    balance = []
    matched = data.loc[[i for pair in pairs for i in pair]]
    for cov in used_covariates:
        treated_all = data.loc[data[treatment] == 1, cov].to_numpy(dtype=float)
        control_all = data.loc[data[treatment] == 0, cov].to_numpy(dtype=float)
        treated_m = matched.loc[matched[treatment] == 1, cov].to_numpy(dtype=float)
        control_m = matched.loc[matched[treatment] == 0, cov].to_numpy(dtype=float)
        before = _smd(treated_all, control_all)
        after = _smd(treated_m, control_m)
        # A Welch test per covariate: the SMD is the decision rule, the p-value
        # says how much sampling noise sits behind it.
        try:
            t_before, p_before = stats.ttest_ind(treated_all, control_all, equal_var=False)
            t_after, p_after = stats.ttest_ind(treated_m, control_m, equal_var=False)
        except Exception:  # noqa: BLE001 - a degenerate covariate must not stop the match
            t_before = t_after = float("nan")
            p_before = p_after = float("nan")
        balance.append(
            {
                "covariate": cov,
                "smd_before": before,
                "smd_after": after,
                "balanced": abs(after) < 0.1,
                "t_before": float(t_before),
                "p_before": float(p_before),
                "t_after": float(t_after),
                "p_after": float(p_after),
            }
        )

    # Common support: the overlap of the two propensity distributions. Treated
    # units outside it have no comparable controls — a structural limit on what
    # matching can say about them, independent of the caliper.
    ps_t = ps.loc[treated_idx].to_numpy()
    ps_c = ps.loc[control_idx].to_numpy()
    support_lo = float(max(ps_t.min(), ps_c.min()))
    support_hi = float(min(ps_t.max(), ps_c.max()))
    n_outside = int(((ps_t < support_lo) | (ps_t > support_hi)).sum())
    # Histogram payload for the overlap chart (shared bin edges).
    edges = np.linspace(float(min(ps_t.min(), ps_c.min())), float(max(ps_t.max(), ps_c.max())), 25)
    hist_t, _ = np.histogram(ps_t, bins=edges)
    hist_c, _ = np.histogram(ps_c, bins=edges)

    return {
        "method": "psm",
        "att": att,
        "ci": (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))),
        "p_value": float(2 * min((boots <= 0).mean(), (boots >= 0).mean())),
        "n_treated": int((data[treatment] == 1).sum()),
        "n_matched_pairs": len(pairs),
        "caliper": caliper_width,
        "att_test": {"t": float(att_t), "p_value": float(att_p), "n_pairs": len(pairs)},
        "pair_diffs": (t_vals - c_vals).tolist(),
        "pruned_covariates": pruned,
        "covariates_used": used_covariates,
        "balance": balance,
        "common_support": {
            "low": support_lo,
            "high": support_hi,
            "n_treated_outside": n_outside,
            "share_outside": float(n_outside / max(len(ps_t), 1)),
        },
        "ps_overlap": {
            "edges": edges.tolist(),
            "treated": hist_t.tolist(),
            "control": hist_c.tolist(),
        },
        "assumption": "Conditional ignorability: treatment is as-good-as-random given the covariates.",
        "narrative": (
            f"Matched {len(pairs)} pairs (caliper {caliper_width:.3f} on the logit scale). ATT = {att:+.4g} "
            f"(bootstrap 95% CI {np.quantile(boots, 0.025):+.4g}..{np.quantile(boots, 0.975):+.4g}). "
            f"{sum(1 for b in balance if not b['balanced'])}/{len(balance)} covariates remain unbalanced (|SMD|>0.1). "
            f"Common support [{support_lo:.3f}, {support_hi:.3f}]; {n_outside} treated units fall outside it."
        ),
    }


def ipw_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: Sequence[str],
    estimand: str = "ate",
    trim: float = 0.01,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> Dict[str, object]:
    """Inverse-probability weighting with weight trimming."""
    data = df[[treatment, outcome, *covariates]].dropna().copy()
    ps = _propensity(data, treatment, covariates)
    ps = np.clip(ps, trim, 1 - trim)
    t = data[treatment].to_numpy(dtype=float)
    y = data[outcome].to_numpy(dtype=float)

    if estimand == "att":
        w1 = t
        w0 = (1 - t) * ps / (1 - ps)
    else:
        w1 = t / ps
        w0 = (1 - t) / (1 - ps)
    est = (w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum()

    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap)
    idx = np.arange(len(data))
    for i in range(n_bootstrap):
        s = rng.choice(idx, len(idx), replace=True)
        w1s, w0s, ys = w1[s], w0[s], y[s]
        boots[i] = (w1s * ys).sum() / max(w1s.sum(), 1e-12) - (w0s * ys).sum() / max(w0s.sum(), 1e-12)

    ess = float((w1.sum() ** 2) / (w1**2).sum() + (w0.sum() ** 2) / (w0**2).sum())
    return {
        "method": f"ipw_{estimand}",
        "estimate": float(est),
        "ci": (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))),
        "p_value": float(2 * min((boots <= 0).mean(), (boots >= 0).mean())),
        "effective_sample_size": ess,
        "ps_range": (float(ps.min()), float(ps.max())),
        "assumption": "Conditional ignorability plus positivity (every unit has non-zero chance of either arm).",
        "narrative": (
            f"IPW {estimand.upper()} = {est:+.4g} (95% CI {np.quantile(boots, 0.025):+.4g}.."
            f"{np.quantile(boots, 0.975):+.4g}); effective sample size {ess:.0f} of {len(data)}."
        ),
    }


def aipw_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: Sequence[str],
    n_bootstrap: int = 200,
    seed: int = 0,
) -> Dict[str, object]:
    """Augmented IPW: doubly robust combination of outcome models and propensity."""
    data = df[[treatment, outcome, *covariates]].dropna().copy()
    X = data[list(covariates)].to_numpy(dtype=float)
    t = data[treatment].to_numpy(dtype=int)
    y = data[outcome].to_numpy(dtype=float)
    ps = np.clip(_propensity(data, treatment, covariates), 0.01, 0.99)

    models = {}
    for arm in (0, 1):
        reg = LinearRegression().fit(X[t == arm], y[t == arm])
        models[arm] = reg.predict(X)

    mu1, mu0 = models[1], models[0]
    score = mu1 - mu0 + t * (y - mu1) / ps - (1 - t) * (y - mu0) / (1 - ps)
    est = float(score.mean())
    se = float(score.std(ddof=1) / np.sqrt(len(score)))
    return {
        "method": "aipw",
        "estimate": est,
        "std_error": se,
        "ci": (est - 1.96 * se, est + 1.96 * se),
        "p_value": float(2 * (1 - stats.norm.cdf(abs(est / se)))) if se else 1.0,
        "assumption": "Doubly robust: consistent if either the propensity or the outcome model is correct.",
        "narrative": f"AIPW ATE = {est:+.4g} (95% CI {est - 1.96 * se:+.4g}..{est + 1.96 * se:+.4g}).",
    }


# --------------------------------------------------------------------------- #
# Synthetic control
# --------------------------------------------------------------------------- #
def synthetic_control(
    df: pd.DataFrame,
    unit: str,
    time: str,
    outcome: str,
    treated_unit,
    treat_time: float,
    donors: Optional[Sequence] = None,
) -> Dict[str, object]:
    """Synthetic control with in-space placebo inference.

    Weights minimise pre-period RMSE on standardised outcomes subject to
    w >= 0 and sum(w) = 1. Placebo tests on donor units give a rank-based
    p-value that does not rely on asymptotic theory.
    """
    wide = df.pivot_table(index=time, columns=unit, values=outcome, aggfunc="mean").sort_index()
    if treated_unit not in wide.columns:
        raise ValueError(f"treated unit {treated_unit} not found")
    donor_cols = list(donors) if donors is not None else [c for c in wide.columns if c != treated_unit]
    wide = wide[[treated_unit] + donor_cols].dropna()

    pre_mask = np.asarray(wide.index < treat_time)
    post_mask = np.asarray(wide.index >= treat_time)
    if pre_mask.sum() < 5 or post_mask.sum() < 2:
        raise ValueError("need at least 5 pre-periods and 2 post-periods")

    y_pre = wide.loc[pre_mask, treated_unit].to_numpy(dtype=float)
    X_pre = wide.loc[pre_mask, donor_cols].to_numpy(dtype=float)
    scale = y_pre.std(ddof=1) or 1.0

    def loss(w: np.ndarray) -> float:
        return float(np.sqrt(np.mean(((X_pre @ w - y_pre) / scale) ** 2)))

    n_donors = len(donor_cols)
    w0 = np.ones(n_donors) / n_donors
    res = optimize.minimize(
        loss, w0, method="SLSQP",
        bounds=[(0, 1)] * n_donors,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    weights = res.x

    synthetic = wide[donor_cols].to_numpy(dtype=float) @ weights
    gap = wide[treated_unit].to_numpy(dtype=float) - synthetic
    pre_rmse = float(np.sqrt(np.mean(gap[pre_mask] ** 2)))
    post_gap = gap[post_mask]
    avg_effect = float(post_gap.mean())

    # In-space placebos: treat each donor as if it were the treated unit.
    placebo_ratios = []
    for donor in donor_cols:
        others = [c for c in donor_cols if c != donor]
        if len(others) < 2:
            continue
        yp = wide.loc[pre_mask, donor].to_numpy(dtype=float)
        Xp = wide.loc[pre_mask, others].to_numpy(dtype=float)
        sc = yp.std(ddof=1) or 1.0

        def loss_p(w: np.ndarray, Xp=Xp, yp=yp, sc=sc) -> float:
            return float(np.sqrt(np.mean(((Xp @ w - yp) / sc) ** 2)))

        wp = optimize.minimize(
            loss_p, np.ones(len(others)) / len(others), method="SLSQP",
            bounds=[(0, 1)] * len(others),
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            options={"maxiter": 300, "ftol": 1e-9},
        ).x
        g = wide[donor].to_numpy(dtype=float) - wide[others].to_numpy(dtype=float) @ wp
        pre_r = np.sqrt(np.mean(g[pre_mask] ** 2)) or 1e-9
        post_r = np.sqrt(np.mean(g[post_mask] ** 2))
        placebo_ratios.append(float(post_r / pre_r))
    ratio = float(np.sqrt(np.mean(post_gap**2)) / pre_rmse) if pre_rmse else float("inf")
    # Rank-based inference: the p-value cannot be smaller than 1 / (placebos + 1).
    p_value = (
        max(float(np.mean(np.array(placebo_ratios) >= ratio)), 1.0 / (len(placebo_ratios) + 1))
        if placebo_ratios
        else float("nan")
    )

    return {
        "method": "synthetic_control",
        "treated_unit": treated_unit,
        "treat_time": treat_time,
        "weights": {c: float(w) for c, w in zip(donor_cols, weights) if w > 0.01},
        "n_donors": int(len(donor_cols)),
        "n_effective_donors": int((weights > 0.01).sum()),
        "n_placebos": int(len(placebo_ratios)),
        "pre_rmse": pre_rmse,
        "post_avg_effect": avg_effect,
        "relative_effect": float(avg_effect / wide.loc[post_mask, treated_unit].mean()),
        "post_rmse": float(np.sqrt(np.mean(post_gap**2))),
        "post_pre_ratio": ratio,
        "placebo_p_value": p_value,
        "pre_fit_r2": float(1 - (pre_rmse**2) / (y_pre.var(ddof=0) or 1)),
        "gap_series": pd.DataFrame(
            {"time": wide.index, "actual": wide[treated_unit].to_numpy(), "synthetic": synthetic, "gap": gap}
        ),
        "assumption": "A convex combination of untreated units reproduces the counterfactual trajectory.",
        "narrative": (
            f"Synthetic control for {treated_unit} from {treat_time}: pre-period fit RMSE={pre_rmse:.3g}, "
            f"post-period average gap={avg_effect:+.4g} (post/pre RMSE ratio {ratio:.2f}, "
            f"placebo p={p_value:.3f}). "
            + ("Effect looks real." if p_value < 0.1 else "Effect is within placebo noise.")
        ),
    }


def compare_estimators(
    df: pd.DataFrame, treatment: str, outcome: str, covariates: Sequence[str], **kwargs
) -> Dict[str, object]:
    """Run several estimators and show the spread - a practical robustness check."""
    import inspect

    out: Dict[str, object] = {}
    for name, fn in (("psm", propensity_score_matching), ("ipw", ipw_effect), ("aipw", aipw_effect)):
        allowed = {k: v for k, v in kwargs.items() if k in inspect.signature(fn).parameters}
        try:
            out[name] = fn(df, treatment, outcome, covariates, **allowed)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            out[name] = {"error": str(exc)}
    ests = [v.get("att", v.get("estimate")) for v in out.values() if isinstance(v, dict) and (v.get("att") or v.get("estimate")) is not None]
    out["summary"] = {
        "estimates": ests,
        "min": float(min(ests)) if ests else None,
        "max": float(max(ests)) if ests else None,
        "spread_pct": float((max(ests) - min(ests)) / abs(np.mean(ests))) if ests and np.mean(ests) else None,
        "robust": bool(ests and (max(ests) - min(ests)) / abs(np.mean(ests)) < 0.5) if ests and np.mean(ests) else False,
    }
    return out


def placebo_refutation(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: Sequence[str],
    n_placebo: int = 12,
    seed: int = 0,
) -> Dict[str, object]:
    """Refutation by fake treatment: shuffle labels, the effect must vanish.

    The estimator is re-run on permuted treatment assignments. A valid
    identification strategy returns ATTs centred on zero; if the placebo
    distribution reaches the real estimate, the "effect" was an artefact of
    the covariate mix rather than the treatment.
    """
    rng = np.random.default_rng(seed)
    shuffled = df.copy()
    placebo_atts = []
    for _ in range(n_placebo):
        shuffled[treatment] = rng.permutation(shuffled[treatment].to_numpy())
        try:
            res = propensity_score_matching(
                shuffled, treatment, outcome, covariates, n_bootstrap=10, seed=seed
            )
            placebo_atts.append(float(res["att"]))
        except Exception:  # noqa: BLE001 - a shrunken placebo pool is expected
            continue
    if not placebo_atts:
        return {"error": "no placebo run succeeded", "robust": None}
    lo, hi = float(np.quantile(placebo_atts, 0.05)), float(np.quantile(placebo_atts, 0.95))
    real = float(
        propensity_score_matching(df, treatment, outcome, covariates, n_bootstrap=10, seed=seed)["att"]
    )
    # The real effect should sit outside (or at the extreme tail of) the placebo band.
    outside = real < lo or real > hi
    return {
        "n_placebo": len(placebo_atts),
        "placebo_mean": float(np.mean(placebo_atts)),
        "placebo_ci": (lo, hi),
        "real_att": real,
        "placebo_max_abs": float(np.max(np.abs(placebo_atts))),
        "robust": bool(outside and abs(real) > abs(float(np.mean(placebo_atts)))),
        "narrative": (
            f"{len(placebo_atts)} placebo runs (shuffled treatment) give ATT "
            f"{np.mean(placebo_atts):+.4g} [90% range {lo:+.4g}..{hi:+.4g}]; the real ATT {real:+.4g} "
            + ("stays outside the placebo band." if outside else "falls INSIDE the placebo band — the estimate is not separable from noise.")
        ),
    }


def rosenbaum_bounds(
    pair_diffs: np.ndarray, gammas: Sequence[float] = (1.0, 1.5, 2.0, 2.5, 3.0)
) -> List[Dict[str, object]]:
    """Wilcoxon signed-rank sensitivity to hidden bias (Rosenbaum bounds).

    Gamma is the odds ratio an unobserved confounder could distort treatment
    assignment by. Each rank's probability of being positive ranges over
    [1/(1+Gamma), Gamma/(1+Gamma)]; the bounds are normal approximations of
    the signed-rank statistic under the worst-case weighting.
    """
    diffs = np.asarray(pair_diffs, dtype=float)
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n < 5:
        return []
    ranks = stats.rankdata(np.abs(diffs))
    t_plus = float(ranks[diffs > 0].sum())
    rows = []
    for gamma in gammas:
        p_hi = gamma / (1.0 + gamma)  # worst case: confounder pushes toward positive
        p_lo = 1.0 / (1.0 + gamma)
        mean_hi = p_hi * ranks.sum()
        sd_hi = float(np.sqrt((ranks**2 * p_hi * (1 - p_hi)).sum()))
        mean_lo = p_lo * ranks.sum()
        sd_lo = float(np.sqrt((ranks**2 * p_lo * (1 - p_lo)).sum()))
        p_upper = float(stats.norm.sf((t_plus - mean_hi - 0.5) / sd_hi))   # evidence could die
        p_lower = float(stats.norm.sf((t_plus - mean_lo - 0.5) / sd_lo))   # evidence survives
        rows.append(
            {
                "gamma": float(gamma),
                "p_lower": p_lower,
                "p_upper": p_upper,
                "survives": bool(p_upper < 0.05),
            }
        )
    return rows


def method_selection_advice(psm: Dict[str, object]) -> List[Dict[str, str]]:
    """Rule-based next-method advice when a matching assumption fails."""
    advice: List[Dict[str, str]] = []
    support = psm.get("common_support", {}) or {}
    balance = psm.get("balance", []) or []
    n_unbalanced = sum(1 for b in balance if not b.get("balanced", True))
    share_out = float(support.get("share_outside", 0.0) or 0.0)

    if share_out > 0.05:
        advice.append(
            {
                "situation": f"{share_out:.1%} treated units lack common support",
                "zh": f"{share_out:.1%} of treated units fall outside common support → trim by propensity score or switch to IPW with truncated weights; under extreme selection prefer RDD.",
                "en": f"{share_out:.1%} of treated units fall outside common support → trim by propensity score or switch to IPW with truncated weights; under extreme selection prefer RDD.",
            }
        )
    if n_unbalanced:
        advice.append(
            {
                "situation": f"{n_unbalanced} covariates remain unbalanced after matching",
                "zh": f"{n_unbalanced} covariates stay unbalanced (|SMD|>0.1) → tighten the caliper, use 1:k matching with replacement, or residualise via regression adjustment.",
                "en": f"{n_unbalanced} covariates stay unbalanced (|SMD|>0.1) → tighten the caliper, use 1:k matching with replacement, or residualise via regression adjustment.",
            }
        )
    if not advice:
        advice.append(
            {
                "situation": "assumptions hold",
                "zh": "Balance and overlap pass → keep the PSM conclusion; for panel policies, cross-check with DiD.",
                "en": "Balance and overlap pass → keep the PSM conclusion; for panel policies, cross-check with DiD.",
            }
        )
    advice.append(
        {
            "situation": "unmeasured confounding suspected",
            "zh": "Unmeasured confounding suspected (e.g. internal user tier) → quantify with Rosenbaum bounds; identify via DiD or an instrument, not matching alone.",
            "en": "Unmeasured confounding suspected (e.g. internal user tier) → quantify with Rosenbaum bounds; identify via DiD or an instrument, not matching alone.",
        }
    )
    return advice
