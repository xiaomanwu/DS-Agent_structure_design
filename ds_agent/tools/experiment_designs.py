"""Design-specific analyses for the non-standard experiment archetypes.

Standard A/B readouts assume independent units, one randomisation layer and a
stable treatment over time. None of those hold for the designs below, and using
the standard test on them is how teams ship noise:

* ``analyze_switchback``      - 时间片轮转实验: treatment alternates by time
  block, so observations are serially correlated and the unit of inference is
  the block, not the user. Uses HAC standard errors plus a paired block
  comparison, and reports carryover explicitly.
* ``analyze_reverse``         - 反转实验: arms are swapped halfway through. The
  reversal separates a real treatment effect from instrumentation drift and
  measures novelty decay; a symmetric sign flip is the strongest cheap evidence
  that the effect is causal.
* ``analyze_nested_experiment`` - 父子实验 / 正交分层: a child experiment runs
  inside a parent experiment's arm. Checks layer orthogonality, then estimates
  the child effect with parent fixed effects and tests the interaction.
* ``analyze_two_sided``       - 双边实验: marketplace experiments where treating
  one side spills onto the other. Estimates direct, cross-side and interaction
  effects with a market-level cluster bootstrap.
* ``sensitivity_analysis``    - 敏感性分析: re-estimates the same effect under
  winsorising, trimming, transformations, CUPED, bootstrap and an A/A placebo,
  so a conclusion that only survives one specification is visibly fragile.

All functions return numbers *and* the diagnostic that decides whether the
number may be trusted.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from ..stats import inference as inf


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _inference(est: float, se: float, alpha: float = 0.05, df: Optional[float] = None) -> Dict[str, float]:
    """Normal (or t) reference distribution for an estimate."""
    if not np.isfinite(se) or se <= 0:
        return {"p_value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    if df is not None and df > 0:
        crit = float(stats.t.ppf(1 - alpha / 2, df))
        p = float(2 * (1 - stats.t.cdf(abs(est / se), df)))
    else:
        crit = float(stats.norm.ppf(1 - alpha / 2))
        p = float(2 * (1 - stats.norm.cdf(abs(est / se))))
    return {"p_value": p, "ci_low": float(est - crit * se), "ci_high": float(est + crit * se)}


def _cluster_bootstrap(
    df: pd.DataFrame,
    cluster_col: str,
    statistic: Callable[[pd.DataFrame], float],
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 7,
) -> Dict[str, float]:
    """Percentile bootstrap that resamples whole clusters (markets, blocks).

    Resampling users inside a market would understate the uncertainty created by
    market-level assignment, which is the entire point of this design.
    """
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].dropna().unique()
    if len(clusters) < 3:
        return {"se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    point = float(statistic(df))
    draws = np.empty(n_boot)
    grouped = {c: df[df[cluster_col] == c] for c in clusters}
    for i in range(n_boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([grouped[c] for c in pick], ignore_index=True)
        try:
            draws[i] = statistic(sample)
        except Exception:  # noqa: BLE001 - a degenerate resample is skipped
            draws[i] = np.nan
    draws = draws[np.isfinite(draws)]
    if len(draws) < 10:
        return {"se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": int(len(draws))}
    return {
        "se": float(np.std(draws, ddof=1)),
        "ci_low": float(np.quantile(draws, alpha / 2)),
        "ci_high": float(np.quantile(draws, 1 - alpha / 2)),
        "n_boot": int(len(draws)),
        "boot_mean": float(np.mean(draws)),
        "point": point,
    }


# Four cells times two arms: below this the interaction test is pure noise.
MIN_NESTED_USERS = 1000


def _tr(lang: str, zh: str, en: str) -> str:
    return zh if lang == "zh" else en


# --------------------------------------------------------------------------- #
# 1. Switchback (time-sliced rotation)
# --------------------------------------------------------------------------- #
def analyze_switchback(
    df: pd.DataFrame,
    metric: str = "value",
    date_col: str = "date",
    arm_col: str = "arm",
    treatment: str = "treatment",
    control: str = "control",
    washout: int = 1,
    maxlags: int = 7,
    alpha: float = 0.05,
    market_col: Optional[str] = None,
) -> Dict[str, object]:
    """Analyse a time-blocked (switchback) experiment.

    The inference unit is the *time block*, not the user-day. Treating 448
    market-days as independent observations (the naive readout) understates the
    standard error by an order of magnitude, because every market shares the same
    arm and the same daily shock. Two estimators are reported:

    * HAC regression ``y ~ treat + day-of-week`` on the collapsed daily series,
      robust to serial correlation up to ``maxlags``. Day-of-week is included
      because the block schedule is rarely balanced across weekdays.
    * paired block differences - consecutive blocks of opposite arms within the
      same market. Immune to slow drift, less efficient, and a useful disagreement
      detector.

    ``washout`` drops the first day(s) of every block: the effect of the previous
    arm does not vanish the moment the switch flips. ``metric`` should be an
    extensive quantity (GMV, orders, sessions) when several markets are present,
    because the series is summed across markets.
    """
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    if metric not in work.columns:
        raise KeyError(f"metric column '{metric}' not found")
    if arm_col not in work.columns:
        raise KeyError(f"arm column '{arm_col}' not found")

    market_col = market_col or ("market" if "market" in work.columns else None)
    work = work[work[arm_col].isin([control, treatment])].copy()
    if work.empty:
        raise ValueError("no rows with the requested arms")

    work["_treat"] = (work[arm_col] == treatment).astype(float)
    work["_y"] = work[metric].astype(float)
    work["_dow"] = work[date_col].dt.dayofweek
    # A slow trend must be modelled explicitly: with alternating blocks the first
    # arm of the series tends to sit on lower-trend days, which reads as a
    # treatment effect of the wrong sign (or a shrunken one).
    work["_t"] = (work[date_col] - work[date_col].min()).dt.days.astype(float)

    # Staggered assignment (markets switch at different times) needs market fixed
    # effects; a common schedule can be collapsed into one daily series.
    staggered = bool(market_col and work.groupby(date_col)[arm_col].nunique().max() > 1)

    if staggered:
        work = work.sort_values([market_col, date_col]).reset_index(drop=True)
        run_id = work.groupby(market_col)[arm_col].transform(lambda s: s.ne(s.shift()).cumsum())
        work["_block"] = work[market_col].astype(str) + "#" + run_id.astype(str)
        work["_pos"] = work.groupby("_block").cumcount()
        used = work[work["_pos"] >= washout].copy()
        if used.empty:
            raise ValueError("washout removed every observation")
        fit = smf.ols("_y ~ _treat + C(_dow) + _t + C(%s)" % market_col, data=used).fit(
            cov_type="cluster", cov_kwds={"groups": used[date_col]}
        )
        tau = float(fit.params["_treat"])
        se = float(fit.bse["_treat"])
        p_value = float(fit.pvalues["_treat"])
        ci_values = (float(fit.conf_int().loc["_treat"].iloc[0]), float(fit.conf_int().loc["_treat"].iloc[1]))
        ctrl_mean = float(used.loc[used["_treat"] == 0, "_y"].mean())
        trt_mean = float(used.loc[used["_treat"] == 1, "_y"].mean())
        resid = np.asarray(fit.resid, dtype=float)
        n_obs = int(len(used))
    else:
        daily = (
            work.groupby(date_col, as_index=False)
            .agg(_y=("_y", "sum"), _treat=("_treat", "max"), _dow=("_dow", "first"),
                 _t=("_t", "first"), n_units=(arm_col, "size"))
            .sort_values(date_col)
            .reset_index(drop=True)
        )
        daily["_block"] = daily["_treat"].ne(daily["_treat"].shift()).cumsum()
        daily["_pos"] = daily.groupby("_block").cumcount()
        used = daily[daily["_pos"] >= washout].copy()
        if used.empty:
            raise ValueError("washout removed every observation")
        # An arm that alternates every day is collinear with day-of-week; the
        # rank check drops the dow term instead of returning a singular fit.
        formula = "_y ~ _treat + C(_dow) + _t"
        rank = np.linalg.matrix_rank(
            np.column_stack(
                [
                    np.ones(len(used)),
                    used["_treat"].to_numpy(),
                    used["_t"].to_numpy(),
                    pd.get_dummies(used["_dow"], drop_first=True).to_numpy(dtype=float),
                ]
            )
        )
        if rank < 3 + used["_dow"].nunique() - 1:
            formula = "_y ~ _treat + _t"
        fit = smf.ols(formula, data=used).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, maxlags)})
        tau = float(fit.params["_treat"])
        se = float(fit.bse["_treat"])
        p_value = float(fit.pvalues["_treat"])
        ci_values = (float(fit.conf_int().loc["_treat"].iloc[0]), float(fit.conf_int().loc["_treat"].iloc[1]))
        ctrl_mean = float(used.loc[used["_treat"] == 0, "_y"].mean())
        trt_mean = float(used.loc[used["_treat"] == 1, "_y"].mean())
        resid = np.asarray(fit.resid, dtype=float)
        n_obs = int(len(used))

    # Paired block comparison: adjacent blocks with opposite arms.
    pairs: List[float] = []
    if staggered and market_col:
        grouped = used.groupby(market_col)
    else:
        grouped = [("all", used)]
    for _, sub in grouped:
        blocks = (
            sub.groupby("_block")
            .agg(treat=("_treat", "max"), mean=("_y", "mean"))
            .reset_index()
            .sort_values("_block")
        )
        for i in range(len(blocks) - 1):
            # 1 = treatment block, 0 = control block; differencing is always
            # treatment minus control regardless of the order.
            if blocks.iloc[i]["treat"] != blocks.iloc[i + 1]["treat"]:
                a, b = float(blocks.iloc[i]["mean"]), float(blocks.iloc[i + 1]["mean"])
                pairs.append(a - b if blocks.iloc[i]["treat"] == 1 else b - a)
    paired = {
        "n_pairs": len(pairs),
        "mean_diff": float(np.mean(pairs)) if pairs else float("nan"),
        "se": float(np.std(pairs, ddof=1) / np.sqrt(len(pairs))) if len(pairs) >= 2 else float("nan"),
        "p_value": float(stats.ttest_1samp(pairs, 0).pvalue) if len(pairs) >= 3 else float("nan"),
    }

    # Carryover: a lagged treatment indicator should be irrelevant once the block
    # is long enough; a significant lag means the washout is too short.
    carryover = {"lag_coef": float("nan"), "lag_p": float("nan"), "detected": False}
    if used["_block"].nunique() > 2:
        lag_frame = used.sort_values(date_col).copy()
        lag_frame["_treat_lag"] = lag_frame["_treat"].shift(1)
        lag_frame = lag_frame.dropna(subset=["_treat_lag"])
        try:
            lag_fit = smf.ols("_y ~ _treat + _treat_lag", data=lag_frame).fit(
                cov_type="HAC", cov_kwds={"maxlags": max(1, maxlags)}
            )
            carryover = {
                "lag_coef": float(lag_fit.params["_treat_lag"]),
                "lag_p": float(lag_fit.pvalues["_treat_lag"]),
                # How much of the treatment effect survives into the day after a
                # switch: 1.0 means yesterday's arm still explains today.
                "ratio": float(lag_fit.params["_treat_lag"] / tau) if abs(tau) > 1e-12 else float("nan"),
                "detected": bool(lag_fit.pvalues["_treat_lag"] < 0.05),
                "contaminated": bool(lag_fit.pvalues["_treat_lag"] < 0.05),
            }
        except Exception:  # noqa: BLE001 - diagnostics are best effort
            pass

    rho = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if len(resid) > 3 else float("nan")

    blocks = int(used["_block"].nunique())
    block_var = float(used.groupby("_block")["_y"].mean().var(ddof=1)) if blocks > 2 else float("nan")
    blocks_per_arm_needed = float("nan")
    if np.isfinite(block_var) and np.isfinite(tau) and abs(tau) > 1e-12:
        # Paired-block sizing: every block yields one difference, so the required
        # number of blocks scales with block-level variance, not with user count.
        blocks_per_arm_needed = float(2 * (1.96 + 0.84) ** 2 * block_var / tau**2)

    agreement = bool(
        np.isfinite(paired["mean_diff"])
        and np.sign(paired["mean_diff"]) == np.sign(tau)
        and abs(paired["mean_diff"] - tau) < 0.5 * abs(tau)
    )
    narrative = _tr(
        "zh",
        f"时间片轮转：HAC 回归效应 {tau:+.4g}（相对 {tau / ctrl_mean:+.2%}，95% CI {ci_values[0]:+.4g}..{ci_values[1]:+.4g}，"
        f"p={p_value:.4f}），{blocks} 个时间块 / {n_obs} 个时间点。残差一阶自相关 {rho:+.2f}，"
        f"{'必须用 HAC 或块级标准误' if abs(rho) > 0.3 else '自相关较弱'}；"
        f"配对块估计 {paired['mean_diff']:+.4g}（{paired['n_pairs']} 对），"
        f"{'与回归一致' if agreement else '与回归不一致，需检查 washout 与块长'}。"
        + ("检测到延续效应，建议加长 washout。" if carryover["detected"] else ""),
        f"Switchback: HAC effect {tau:+.4g} ({tau / ctrl_mean:+.2%} relative, 95% CI {ci_values[0]:+.4g}..{ci_values[1]:+.4g}, "
        f"p={p_value:.4f}) over {blocks} blocks / {n_obs} time points. Residual lag-1 autocorrelation {rho:+.2f}; "
        f"paired-block estimate {paired['mean_diff']:+.4g} across {paired['n_pairs']} pairs, "
        f"{'consistent with the regression' if agreement else 'inconsistent - check washout and block length'}. "
        + ("Carryover detected, lengthen the washout." if carryover["detected"] else ""),
    )

    return {
        "method": "switchback",
        "effect": tau,
        "se": se,
        "p_value": p_value,
        "ci": ci_values,
        "relative": tau / ctrl_mean if ctrl_mean else float("nan"),
        "control_mean": ctrl_mean,
        "treatment_mean": trt_mean,
        "n_blocks": blocks,
        "n_obs": n_obs,
        "staggered": staggered,
        "washout_days": washout,
        "hac_lags": maxlags,
        "residual_autocorr": rho,
        "paired_blocks": paired,
        "carryover": carryover,
        "blocks_per_arm_needed": blocks_per_arm_needed,
        "significant": bool(p_value < alpha),
        "assumption": "No time-varying confounding between blocks; washout removes carryover.",
        "narrative": narrative,
    }


# --------------------------------------------------------------------------- #
# 2. Reverse experiment
# --------------------------------------------------------------------------- #
def analyze_reverse(
    df: pd.DataFrame,
    metric: str = "value",
    date_col: str = "date",
    arm_col: str = "arm",
    switch_time=None,
    treatment: str = "treatment",
    control: str = "control",
    alpha: float = 0.05,
    phase_col: Optional[str] = None,
) -> Dict[str, object]:
    """Analyse a reversal (swap) design.

    Phase 1 gives ``tau1 = E[Y|T] - E[Y|C]``. Phase 2 swaps the arms, so
    ``tau2 = E[Y|C_after] - E[Y|T_after]`` is the same contrast measured with
    the roles exchanged.

    * ``tau1 + tau2 ~ 0`` (sum test): the effect is reversible, which rules out
      instrumentation drift -- a broken tracker moves the same way in both
      phases and would not flip sign.
    * ``tau2 / -tau1``: persistence. Below 1 means the effect decays (novelty),
      above 1 means it strengthens (learning).
    * ``(tau1 - tau2) / 2``: pooled estimate, twice as efficient as either phase
      alone when the effect is symmetric.

    ``phase_col`` is used when the input already carries the phase (user-level
    data with re-randomisation per phase); otherwise the phase is derived from
    ``switch_time`` against ``date_col``.
    """
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    if phase_col and phase_col in work.columns:
        work["_phase"] = work[phase_col].astype(int)
        switch = work.loc[work["_phase"] == 2, date_col].min()
    else:
        switch = pd.Timestamp(switch_time) if switch_time is not None else work[date_col].quantile(0.5)
        work["_phase"] = np.where(work[date_col] < switch, 1, 2)
    work["_y"] = work[metric].astype(float)
    work["_treat"] = (work[arm_col] == treatment).astype(float)

    per_phase = {}
    for phase in (1, 2):
        sub = work[work["_phase"] == phase]
        c = sub.loc[sub["_treat"] == 0, "_y"]
        t = sub.loc[sub["_treat"] == 1, "_y"]
        if len(c) < 3 or len(t) < 3:
            raise ValueError(f"phase {phase} lacks observations in one arm")
        test = inf.welch_ttest(c.to_numpy(), t.to_numpy(), alpha)
        per_phase[phase] = {
            "effect": float(test["abs_diff"]),
            "se": float(test["se"]),
            "p_value": float(test["p_value"]),
            "control_mean": float(c.mean()),
            "treatment_mean": float(t.mean()),
            "n": int(len(sub)),
            "start": str(sub[date_col].min().date()),
            "end": str(sub[date_col].max().date()),
        }

    tau1, tau2 = per_phase[1]["effect"], per_phase[2]["effect"]
    se_sum = float(np.sqrt(per_phase[1]["se"] ** 2 + per_phase[2]["se"] ** 2))
    sum_test = _inference(tau1 + tau2, se_sum, alpha)

    pooled = (tau1 - tau2) / 2
    pooled_se = se_sum / 2
    pooled_test = _inference(pooled, pooled_se, alpha)

    reversal = bool(np.sign(tau1) != np.sign(tau2))
    persistence = -tau2 / tau1 if abs(tau1) > 1e-12 else float("nan")
    infra_valid = bool(reversal and per_phase[1]["p_value"] < 0.1 and per_phase[2]["p_value"] < 0.1)

    base = per_phase[1]["control_mean"]
    narrative = _tr(
        "zh",
        f"反转实验：第一期效应 {tau1:+.4g}（p={per_phase[1]['p_value']:.4f}），互换后第二期 {tau2:+.4g}"
        f"（p={per_phase[2]['p_value']:.4f}），"
        + ("方向反转，" if reversal else "方向未反转，")
        + f"对称性检验 p={sum_test['p_value']:.3f}（{'效应可逆，非埋点漂移' if sum_test['p_value'] > 0.05 else '两期不对称，可能存在新奇效应或污染'}）。"
        f"持续性 {persistence:.2f}，合并估计 {pooled:+.4g}（相对 {pooled / base:+.2%}，p={pooled_test['p_value']:.4f}）。",
        f"Reversal: phase 1 effect {tau1:+.4g} (p={per_phase[1]['p_value']:.4f}), phase 2 {tau2:+.4g} "
        f"(p={per_phase[2]['p_value']:.4f}); "
        + ("the sign flips, " if reversal else "the sign does not flip, ")
        + f"symmetry test p={sum_test['p_value']:.3f} "
        + ("(reversible effect, not instrumentation drift)" if sum_test["p_value"] > 0.05 else "(asymmetric - novelty or contamination)")
        + f". Persistence {persistence:.2f}, pooled estimate {pooled:+.4g} ({pooled / base:+.2%} relative, "
        f"p={pooled_test['p_value']:.4f}).",
    )

    return {
        "method": "reverse",
        "phases": per_phase,
        "tau1": tau1,
        "tau2": tau2,
        "switch_time": str(pd.Timestamp(switch).date()),
        "reversal_detected": reversal,
        "symmetry": {"sum": tau1 + tau2, "se": se_sum, **sum_test},
        "symmetric": bool(sum_test["p_value"] > 0.05),
        "persistence": float(persistence),
        "pooled": {"effect": pooled, "se": pooled_se, "relative": pooled / base if base else float("nan"), **pooled_test},
        "infrastructure_valid": infra_valid,
        "assumption": "Swapping the arms reverses the treatment contrast; no period-specific confounder.",
        "narrative": narrative,
    }


# --------------------------------------------------------------------------- #
# 3. Parent-child / orthogonal layers
# --------------------------------------------------------------------------- #
def analyze_nested_experiment(
    df: pd.DataFrame,
    metric: str,
    parent_col: str = "parent_arm",
    child_col: str = "child_arm",
    treatment: str = "treatment",
    control: str = "control",
    alpha: float = 0.05,
    lang: str = "zh",
    parent_treatment: Optional[str] = None,
) -> Dict[str, object]:
    """Analyse a child experiment running inside a parent experiment.

    Three things can go wrong and all three are tested here:

    1. **Layer collision** - the child bucketing is not independent of the
       parent, so the two effects are entangled (chi-square test).
    2. **Simpson's paradox** - the child effect differs across parent arms, so
       the pooled number is a weighted average of heterogeneous effects
       (parent fixed effects + interaction test).
    3. **Under-powered cells** - orthogonal layers split traffic multiplicatively
       (parent x child), so the child cells are much smaller than they look.
    """
    work = df[[metric, parent_col, child_col]].dropna().copy()
    if len(work) < MIN_NESTED_USERS:
        raise ValueError(
            f"parent-child readout needs at least {MIN_NESTED_USERS} users to keep the "
            f"parent x child cells above 30 per arm (got {len(work)})"
        )
    work["_y"] = work[metric].astype(float)
    # Arm labels are usually "treatment"/"control", but a real experiment often
    # ships "ranking_v2" as the variant name. Fall back to "the non-control
    # level" rather than silently producing an all-zero treatment indicator.
    child_levels = list(pd.unique(work[child_col]))
    if treatment not in child_levels:
        treatment = next((lvl for lvl in child_levels if lvl != control), treatment)
    if treatment not in child_levels:
        raise ValueError(f"cannot identify the child treatment level among {child_levels}")
    work["_child_treat"] = (work[child_col] == treatment).astype(float)

    table = pd.crosstab(work[parent_col], work[child_col])
    chi2, chi_p, dof, _ = stats.chi2_contingency(table.to_numpy())
    orthogonal = bool(chi_p > 0.01)

    c = work.loc[work["_child_treat"] == 0, "_y"]
    t = work.loc[work["_child_treat"] == 1, "_y"]
    naive = inf.welch_ttest(c.to_numpy(), t.to_numpy(), alpha)

    # Child effect within each parent arm, then pooled with parent fixed effects.
    within = []
    for parent_arm, sub in work.groupby(parent_col):
        cc = sub.loc[sub["_child_treat"] == 0, "_y"]
        tt = sub.loc[sub["_child_treat"] == 1, "_y"]
        if len(cc) < 30 or len(tt) < 30:
            continue
        test = inf.welch_ttest(cc.to_numpy(), tt.to_numpy(), alpha)
        within.append(
            {
                "parent_arm": str(parent_arm),
                "effect": float(test["abs_diff"]),
                "relative": float(test["rel_lift"]),
                "p_value": float(test["p_value"]),
                "n": int(len(sub)),
            }
        )

    adjusted = None
    base_y = float(work.loc[work["_child_treat"] == 0, "_y"].mean())
    try:
        fit = smf.ols("_y ~ _child_treat + C(%s)" % parent_col, data=work).fit(cov_type="HC1")
        coef = float(fit.params["_child_treat"])
        adjusted = {
            "effect": coef,
            "relative": float(coef / base_y) if base_y else float("nan"),
            "se": float(fit.bse["_child_treat"]),
            "p_value": float(fit.pvalues["_child_treat"]),
            "ci": (float(fit.conf_int().loc["_child_treat"].iloc[0]), float(fit.conf_int().loc["_child_treat"].iloc[1])),
        }
    except Exception:  # noqa: BLE001 - a single-parent frame is still valid
        pass

    # Interaction: does the child effect depend on the parent arm?
    interaction = {"coefficient": float("nan"), "p_value": float("nan"), "detected": False}
    arms = list(pd.unique(work[parent_col]))
    if len(arms) == 2:
        p_treat = parent_treatment if parent_treatment in arms else next((a for a in arms if a != control), arms[-1])
        work["_parent_treat"] = (work[parent_col] == p_treat).astype(float)
        try:
            fit_int = smf.ols("_y ~ _child_treat * _parent_treat", data=work).fit(cov_type="HC1")
            term = "_child_treat:_parent_treat"
            interaction = {
                "coefficient": float(fit_int.params[term]),
                "relative": float(fit_int.params[term] / base_y) if base_y else float("nan"),
                "p_value": float(fit_int.pvalues[term]),
                "detected": bool(fit_int.pvalues[term] < 0.05),
            }
        except Exception:  # noqa: BLE001
            pass

    cell_sizes = table.stack().rename("n").reset_index()
    min_cell = int(cell_sizes["n"].min()) if len(cell_sizes) else 0

    narrative = _tr(
        lang,
        f"父子实验：父子层{'正交' if orthogonal else '不正交'}（卡方 p={chi_p:.4f}），"
        f"子实验朴素效应 {naive['rel_lift']:+.2%}（p={naive['p_value']:.4f}），"
        + (f"控制父实验后 {adjusted['effect'] / naive['control_mean']:+.2%}（p={adjusted['p_value']:.4f}），" if adjusted else "")
        + f"父x子交互 {'显著' if interaction['detected'] else '不显著'}（p={interaction['p_value']:.3f}）；最小单元格 {min_cell:,}。",
        f"Parent-child: layers are {'orthogonal' if orthogonal else 'NOT orthogonal'} (chi-square p={chi_p:.4f}); "
        f"naive child effect {naive['rel_lift']:+.2%} (p={naive['p_value']:.4f}), "
        + (f"parent-adjusted {adjusted['effect'] / naive['control_mean']:+.2%} (p={adjusted['p_value']:.4f}), " if adjusted else "")
        + f"parent x child interaction {'significant' if interaction['detected'] else 'not significant'} "
        f"(p={interaction['p_value']:.3f}); smallest cell {min_cell:,}.",
    )

    return {
        "method": "nested_parent_child",
        "orthogonal": orthogonal,
        "orthogonality": {"chi2": float(chi2), "p_value": float(chi_p), "dof": int(dof)},
        "naive": {
            "control": float(naive["control_mean"]),
            "treatment": float(naive["treatment_mean"]),
            "effect": float(naive["abs_diff"]),
            "relative": float(naive["rel_lift"]),
            "p_value": float(naive["p_value"]),
        },
        "adjusted": adjusted,
        "within_parent": within,
        "interaction": interaction,
        "cells": table,
        "min_cell": min_cell,
        "n": int(len(work)),
        "assumption": "Child bucketing is independent of the parent layer; no cross-layer interaction.",
        "narrative": narrative,
    }


# --------------------------------------------------------------------------- #
# 4. Two-sided marketplace experiment
# --------------------------------------------------------------------------- #
def analyze_two_sided(
    df: pd.DataFrame,
    metrics: Dict[str, str],
    market_col: str = "market",
    buyer_treat_col: str = "buyer_treat",
    seller_treat_col: str = "seller_treat",
    n_boot: int = 200,
    alpha: float = 0.05,
    lang: str = "zh",
) -> Dict[str, object]:
    """Analyse a 2x2 market-level design that treats both sides.

    ``metrics`` maps a role to a column, e.g. ``{"buyer": "buyer_conversion",
    "seller": "seller_listings", "market": "gmv"}``. Each market belongs to one
    of four cells (neither / buyer only / seller only / both), so:

    * direct effect = average over the other side's assignment,
    * cross-side spillover = effect of treating side A on side B's metric,
    * interaction = the extra (or missing) effect when both sides are treated,
      estimated as a difference-in-differences across cells.

    Inference resamples markets, because markets -- not users -- are the
    independent unit once both sides are treated.
    """
    work = df.copy()
    for col in (buyer_treat_col, seller_treat_col):
        if col not in work.columns:
            raise KeyError(f"missing column '{col}'")
    work["_cell"] = (
        work[buyer_treat_col].astype(int).astype(str) + work[seller_treat_col].astype(int).astype(str)
    )

    def cell_mean(frame: pd.DataFrame, metric: str, cell: str) -> float:
        sub = frame.loc[frame["_cell"] == cell, metric]
        return float(sub.mean()) if len(sub) else np.nan

    results: Dict[str, object] = {}
    for role, col in metrics.items():
        if col not in work.columns:
            continue
        direct_buyer = (cell_mean(work, col, "10") + cell_mean(work, col, "11")) / 2 - (
            cell_mean(work, col, "00") + cell_mean(work, col, "01")
        ) / 2
        direct_seller = (cell_mean(work, col, "01") + cell_mean(work, col, "11")) / 2 - (
            cell_mean(work, col, "00") + cell_mean(work, col, "10")
        ) / 2
        interaction = (
            cell_mean(work, col, "11") - cell_mean(work, col, "10") - cell_mean(work, col, "01") + cell_mean(work, col, "00")
        )
        stat = lambda frame, col=col: _cell_contrast(frame, col)  # noqa: E731
        boot = _cluster_bootstrap(work, market_col, stat, n_boot=n_boot, alpha=alpha)
        inference = _inference(boot.get("point", float("nan")), boot.get("se", float("nan")), alpha)
        # Relative to the untreated cell: the only scale on which a conversion
        # rate and a listing rate can sit in the same sentence.
        base = cell_mean(work, col, "00")
        results[role] = {
            "metric": col,
            "buyer_direct": float(direct_buyer),
            "seller_direct": float(direct_seller),
            "interaction": float(interaction),
            "buyer_direct_relative": float(direct_buyer / base) if base else float("nan"),
            "seller_direct_relative": float(direct_seller / base) if base else float("nan"),
            "interaction_relative": float(interaction / base) if base else float("nan"),
            "baseline": float(base),
            "interaction_p": inference["p_value"],
            "interaction_ci": (inference["ci_low"], inference["ci_high"]),
            "se": boot.get("se"),
            "cells": {cell: cell_mean(work, col, cell) for cell in ("00", "10", "01", "11")},
        }

    # Cross-side spillover relative to the direct effect on the same side.
    spillovers = {}
    for role, res in results.items():
        opposite = "seller" if role == "buyer" else "buyer"
        direct = res.get(f"{role}_direct", float("nan"))
        cross = res.get(f"{opposite}_direct", float("nan"))
        spillovers[role] = {
            "direct": float(direct),
            "cross_side": float(cross),
            "ratio": float(cross / direct) if direct and np.isfinite(direct) and abs(direct) > 1e-12 else float("nan"),
        }

    market_cells = work.groupby("_cell")[market_col].nunique().to_dict()
    def _metric_summary(res: Dict[str, object], zh: bool) -> str:
        """One line per metric: both direct effects and the interaction.

        Every metric lives in all four cells, so reporting only the "own side"
        direct effect would drop the market-level interaction, which is usually
        the number the decision turns on.
        """
        if zh:
            return (
                f"{res['metric']}：买家侧干预 {res['buyer_direct']:+.4g}、"
                f"卖家侧干预 {res['seller_direct']:+.4g}、"
                f"交互 {res['interaction']:+.4g}（p={res['interaction_p']:.3f}）"
            )
        return (
            f"{res['metric']}: buyer-side {res['buyer_direct']:+.4g}, "
            f"seller-side {res['seller_direct']:+.4g}, "
            f"interaction {res['interaction']:+.4g} (p={res['interaction_p']:.3f})"
        )

    cell_text_zh = "、".join(f"{k}={v}" for k, v in market_cells.items())
    narrative = _tr(
        lang,
        "双边实验（按市场聚类自助法）："
        + "；".join(_metric_summary(res, True) for res in results.values())
        + f"。四个单元格的市场数：{cell_text_zh}。",
        "Two-sided design (market-cluster bootstrap): "
        + "; ".join(_metric_summary(res, False) for res in results.values())
        + f". Markets per cell: {market_cells}.",
    )

    return {
        "method": "two_sided",
        "cells": market_cells,
        "results": results,
        "spillover": spillovers,
        "n_markets": int(work[market_col].nunique()),
        "assumption": "Markets are independent; within-market interference is absorbed by the cell design.",
        "narrative": narrative,
    }


def _cell_contrast(frame: pd.DataFrame, metric: str) -> float:
    """Interaction contrast, used as the bootstrap statistic."""
    means = {cell: frame.loc[frame["_cell"] == cell, metric].mean() for cell in ("00", "10", "01", "11")}
    return float(means["11"] - means["10"] - means["01"] + means["00"])


# --------------------------------------------------------------------------- #
# 5. Sensitivity analysis
# --------------------------------------------------------------------------- #
def sensitivity_analysis(
    df: pd.DataFrame,
    metric: str,
    group_col: str = "group",
    control: str = "control",
    treatment: str = "treatment",
    covariate: Optional[str] = None,
    segment_col: Optional[str] = None,
    n_boot: int = 400,
    alpha: float = 0.05,
    lang: str = "zh",
    kind: str = "auto",
) -> Dict[str, object]:
    """Re-estimate one effect under every defensible specification.

    A result that survives winsorising, trimming, a log transform, CUPED and a
    placebo A/A split is a result. A result that only appears in one of them is
    a specification artefact, and the table makes that visible instead of
    hiding it behind a single p-value.

    ``kind="auto"`` detects a 0/1 metric and switches to the proportion test;
    trimming and log transforms are then meaningless and are skipped rather than
    reported as if they had been run.
    """
    extra = ([covariate] if covariate else []) + ([segment_col] if segment_col and segment_col in df.columns else [])
    work = df[[metric, group_col] + extra].dropna().copy()
    work["_y"] = work[metric].astype(float)
    work["_treat"] = (work[group_col] == treatment).astype(float)
    ctrl = work.loc[work[group_col] == control, "_y"].to_numpy()
    trt = work.loc[work[group_col] == treatment, "_y"].to_numpy()
    if len(ctrl) < 30 or len(trt) < 30:
        raise ValueError("need at least 30 observations per arm")

    specs: List[Dict[str, object]] = []
    unique_values = set(np.unique(np.concatenate([ctrl, trt])).tolist())
    if kind == "auto":
        kind = "proportion" if unique_values <= {0.0, 1.0} else "mean"

    def add(name: str, y_c: np.ndarray, y_t: np.ndarray, note: str = "", metric_kind: Optional[str] = None) -> None:
        if len(y_c) < 20 or len(y_t) < 20:
            return
        effective = metric_kind or kind
        if effective == "proportion":
            res = inf.two_proportion_test(int(np.nansum(y_c)), len(y_c), int(np.nansum(y_t)), len(y_t), alpha)
            lift = float(res.get("rel_lift", np.nan))
            p = float(res.get("p_value", np.nan))
            se = float(res.get("se", np.nan))
        else:
            res = inf.welch_ttest(y_c, y_t, alpha)
            lift = float(res["rel_lift"])
            p = float(res["p_value"])
            se = float(res["se"])
        specs.append(
            {
                "spec": name,
                "note": note,
                "control_mean": float(np.nanmean(y_c)),
                "treatment_mean": float(np.nanmean(y_t)),
                "abs_diff": float(res.get("abs_diff", np.nan)),
                "rel_lift": lift,
                "se": se,
                "p_value": p,
                "significant": bool(p < alpha),
            }
        )

    add(_tr(lang, "基准（Welch）" if kind == "mean" else "基准（两比例 z）",
            "baseline (Welch)" if kind == "mean" else "baseline (two-proportion z)"), ctrl, trt)

    # Segment-adjusted estimate: a legitimate specification for either metric
    # kind, and the one that exposes a segment mix difference between arms.
    if segment_col and segment_col in work.columns:
        try:
            fit = smf.ols("_y ~ _treat + C(%s)" % segment_col, data=work).fit(cov_type="HC1")
            coef = float(fit.params["_treat"])
            base_mean = float(work.loc[work["_treat"] == 0, "_y"].mean())
            specs.append(
                {
                    "spec": _tr(lang, "分层调整", "segment-adjusted"),
                    "note": _tr(lang, f"按 {segment_col} 固定效应调整", f"fixed effects on {segment_col}"),
                    "control_mean": base_mean,
                    "treatment_mean": base_mean + coef,
                    "abs_diff": coef,
                    "rel_lift": coef / base_mean if base_mean else np.nan,
                    "se": float(fit.bse["_treat"]),
                    "p_value": float(fit.pvalues["_treat"]),
                    "significant": bool(fit.pvalues["_treat"] < alpha),
                }
            )
        except Exception:  # noqa: BLE001 - a degenerate segment column is skipped
            pass

    if kind == "mean":
        # Winsorising the top 0.5%: the standard defence against a single whale.
        lo, hi = np.nanquantile(np.concatenate([ctrl, trt]), [0.005, 0.995])
        add(_tr(lang, "缩尾 0.5%", "winsorise 0.5%"), np.clip(ctrl, lo, hi), np.clip(trt, lo, hi))

        # Trimming both tails changes the estimand, so it is a stress test only.
        lo, hi = np.nanquantile(np.concatenate([ctrl, trt]), [0.01, 0.99])
        add(
            _tr(lang, "截尾 1%", "trim 1%"),
            ctrl[(ctrl >= lo) & (ctrl <= hi)],
            trt[(trt >= lo) & (trt <= hi)],
            note=_tr(lang, "估计量变为截尾均值", "estimand changes to the trimmed mean"),
        )

        # Drop the top 1% of users (the "is it one big customer?" test).
        cutoff = np.nanquantile(np.concatenate([ctrl, trt]), 0.99)
        add(
            _tr(lang, "剔除头部 1%", "drop top 1%"),
            ctrl[ctrl < cutoff],
            trt[trt < cutoff],
            note=_tr(lang, "用户级截断", "user-level truncation"),
        )

        if (ctrl >= 0).all() and (trt >= 0).all():
            add(_tr(lang, "log1p 变换", "log1p transform"), np.log1p(ctrl), np.log1p(trt))

    if covariate:
        y_all = work["_y"].to_numpy(dtype=float)
        x_all = work[covariate].to_numpy(dtype=float)
        adj = inf.cuped_adjust(y_all, x_all)
        groups = work[group_col].to_numpy()
        adj_c = adj["y_adjusted"][groups == control]
        adj_t = adj["y_adjusted"][groups == treatment]
        add(
            _tr(lang, "CUPED", "CUPED"),
            adj_c,
            adj_t,
            note=_tr(lang, f"theta={adj['theta']:.3f}，方差缩减 {adj['variance_reduction']:.0%}", f"theta={adj['theta']:.3f}, VR={adj['variance_reduction']:.0%}"),
        )

    # Bootstrap CI: does the conclusion depend on the normal approximation?
    boot = inf.bootstrap_diff_in_means(ctrl, trt, n_boot=n_boot, alpha=alpha)
    specs.append(
        {
            "spec": _tr(lang, "Bootstrap 置信区间", "bootstrap CI"),
            "note": _tr(lang, f"{n_boot} 次重抽样", f"{n_boot} resamples"),
            "control_mean": float(np.mean(ctrl)),
            "treatment_mean": float(np.mean(trt)),
            "abs_diff": float(np.mean(trt) - np.mean(ctrl)),
            "rel_lift": float((np.mean(trt) - np.mean(ctrl)) / np.mean(ctrl)) if np.mean(ctrl) else np.nan,
            "se": float(boot.get("se", np.nan)),
            "p_value": float(boot.get("p_value", np.nan)),
            "significant": bool(boot["ci_abs"][0] > 0 or boot["ci_abs"][1] < 0),
        }
    )

    # A/A placebo: split the control arm at random and count false positives.
    rng = np.random.default_rng(11)
    placebo_p = []
    for _ in range(50):
        idx = rng.permutation(len(ctrl))
        half = len(idx) // 2
        test = inf.welch_ttest(ctrl[idx[:half]], ctrl[idx[half:]], alpha)
        placebo_p.append(float(test["p_value"]))
    placebo_rate = float(np.mean(np.array(placebo_p) < alpha))

    # Segment jackknife: would dropping any one segment change the sign?
    jackknife = []
    if segment_col and segment_col in work.columns:
        work_seg = work.assign(_seg=work[segment_col])
        for seg, sub in work_seg.groupby("_seg"):
            if len(sub) < 100:
                continue
            rest = work_seg[work_seg["_seg"] != seg]
            rest_c = rest.loc[rest[group_col] == control, "_y"]
            rest_t = rest.loc[rest[group_col] == treatment, "_y"]
            if len(rest_c) < 30 or len(rest_t) < 30:
                continue
            res = inf.welch_ttest(rest_c.to_numpy(), rest_t.to_numpy(), alpha)
            jackknife.append({"dropped": str(seg), "rel_lift": float(res["rel_lift"]), "p_value": float(res["p_value"])})

    frame = pd.DataFrame(specs)
    main = frame.iloc[0]
    lifts = frame["rel_lift"].dropna().to_numpy()
    signs = np.sign(lifts)
    sign_stable = bool(len(np.unique(signs)) == 1)
    sig_count = int(frame["significant"].sum())
    spread = float(np.max(lifts) - np.min(lifts)) if len(lifts) else float("nan")
    if sign_stable and sig_count >= max(2, len(frame) - 2):
        verdict = _tr(lang, "结论稳健：多数设定下同向且显著。", "Robust: consistent sign and significance across most specifications.")
    elif sign_stable:
        verdict = _tr(lang, "方向稳健但显著性不稳：不要只报告最有利的设定。", "Sign is stable but significance is not: do not cherry-pick a specification.")
    else:
        verdict = _tr(lang, "结论脆弱：不同设定下方向相反，需先查清原因再决策。", "Fragile: the sign flips across specifications; investigate before deciding.")

    narrative = _tr(
        lang,
        f"敏感性分析：基准 {main['rel_lift']:+.2%}（p={main['p_value']:.4f}），"
        f"{len(frame)} 种设定下 {sig_count} 个显著，极差 {spread:.2%}，"
        f"A/A 安慰剂假阳性率 {placebo_rate:.0%}。{verdict}",
        f"Sensitivity: baseline {main['rel_lift']:+.2%} (p={main['p_value']:.4f}); {sig_count} of {len(frame)} "
        f"specifications significant, spread {spread:.2%}, A/A placebo false-positive rate {placebo_rate:.0%}. {verdict}",
    )

    return {
        "method": "sensitivity",
        "specs": frame,
        "baseline": {
            "rel_lift": float(main["rel_lift"]),
            "abs_diff": float(main["abs_diff"]),
            "p_value": float(main["p_value"]),
        },
        "sign_stable": sign_stable,
        "significant_count": sig_count,
        "n_specs": int(len(frame)),
        "lift_range": (float(np.min(lifts)), float(np.max(lifts))) if len(lifts) else (np.nan, np.nan),
        "placebo_false_positive_rate": placebo_rate,
        "jackknife": jackknife,
        "verdict": verdict,
        "narrative": narrative,
    }
