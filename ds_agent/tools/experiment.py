"""Experiment design and analysis for internet-scale AB testing.

Covers the experiment archetypes that actually show up in consumer internet
companies (feed ranking, checkout funnels, marketplace liquidity, growth
campaigns), the design maths behind each, and the analysis path with SRM,
CUPED, sequential monitoring and multiple-comparison control.

Design and analysis share one metric taxonomy, so the sample size promised in
the design is the test actually run at readout time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..stats import inference as inf
from ..stats import power as pw

# --------------------------------------------------------------------------- #
# Experiment archetype catalogue
# --------------------------------------------------------------------------- #
EXPERIMENT_TYPES: Dict[str, Dict[str, object]] = {
    "ab": {
        "name_en": "Standard A/B test",
        "name_zh": "Standard A/B test",
        "when_zh": "One change, independent users, enough traffic. The default.",
        "when": "One change, independent users, enough traffic. The default.",
        "randomization_unit": "user_id",
        "analysis": "Two-proportion z-test or Welch t-test; CUPED when a pre-period covariate exists.",
        "pitfalls": ["Peeking without alpha spending", "Novelty effect in the first days", "Segment-level fishing"],
    },
    "abn": {
        "name_en": "A/B/n (multi-arm)",
        "name_zh": "A/B/n (multi-arm)",
        "when_zh": "Several variants compete; you want one winner among many.",
        "when": "Several variants compete; you want one winner among many.",
        "randomization_unit": "user_id",
        "analysis": "Omnibus test first (chi-square / ANOVA), then pairwise with BH or Holm correction.",
        "pitfalls": ["Alpha inflation across arms", "Unequal traffic per arm weakens power", "Choosing the winner from noisy pairs"],
    },
    "factorial": {
        "name_en": "Factorial / multivariate",
        "name_zh": "Factorial / multivariate",
        "when_zh": "Two or more independent levers (e.g. copy x price badge) and interaction matters.",
        "when": "Two or more independent levers (e.g. copy x price badge) and interaction matters.",
        "randomization_unit": "user_id",
        "analysis": "Two-way ANOVA or regression with interaction terms; report main effects and interaction.",
        "pitfalls": ["Power for the interaction term is much lower than for main effects", "Cells multiply traffic requirements"],
    },
    "stratified": {
        "name_en": "Stratified randomisation",
        "name_zh": "Stratified randomisation",
        "when_zh": "Strong covariates (device, country, new/returning) create imbalance risk.",
        "when": "Strong covariates (device, country, new/returning) create imbalance risk.",
        "randomization_unit": "user_id within strata",
        "analysis": "Stratified t-test / regression with strata fixed effects; variance reduction 5-30%.",
        "pitfalls": ["Must pre-register strata", "Empty or tiny strata destabilise the estimate"],
    },
    "cluster": {
        "name_en": "Cluster / geo randomisation",
        "name_zh": "Cluster / geo randomisation",
        "when_zh": "Interference between users (social graph, supply-demand coupling) or a channel-level change.",
        "when": "Interference between users (social graph, supply-demand coupling) or a channel-level change.",
        "randomization_unit": "city / school / seller",
        "analysis": "Cluster-robust SE or delta method on cluster means; design effect 1+(m-1)*ICC.",
        "pitfalls": ["Few clusters make the CLT unreliable", "Balance check on pre-period is mandatory"],
    },
    "switchback": {
        "name_en": "Switchback (time-based)",
        "name_zh": "Switchback (time-based)",
        "when_zh": "Marketplace / two-sided platform where treating one side contaminates the other (pricing, dispatch).",
        "when": "Marketplace / two-sided platform where treating one side contaminates the other (pricing, dispatch).",
        "randomization_unit": "time block x market",
        "analysis": "Paired comparison across blocks with autocorrelation-robust SE (HAC / block bootstrap).",
        "pitfalls": ["Carryover between adjacent blocks needs washout", "Hour-of-day confounds require balanced blocks"],
    },
    "holdout": {
        "name_en": "Long-term holdout",
        "name_zh": "Long-term holdout",
        "when_zh": "Measuring durable effects after the change is fully launched.",
        "when": "Measuring durable effects after the change is fully launched.",
        "randomization_unit": "user_id (small holdout %, e.g. 1-5%)",
        "analysis": "Repeated readouts vs holdout, cumulative effect curve, decay analysis.",
        "pitfalls": ["Holdout users drift away over time", "Non-compliance and spillover from shared devices"],
    },
    "bandit": {
        "name_en": "Multi-armed bandit",
        "name_zh": "Multi-armed bandit",
        "when_zh": "Optimising a reward during rollout and losing traffic is expensive; you do not need a clean ATE.",
        "when": "Optimising a reward during rollout and losing traffic is expensive; you do not need a clean ATE.",
        "randomization_unit": "user_id / impression",
        "analysis": "Thompson sampling posterior comparison; do NOT report p-values from adaptively collected data.",
        "pitfalls": ["Adaptive allocation breaks classic inference", "Reward hacking when the proxy is gameable"],
    },
    "sequential": {
        "name_en": "Sequential / always-valid test",
        "name_zh": "Sequential / always-valid test",
        "when_zh": "Stakeholders insist on daily readouts, or stopping early for a big win matters.",
        "when": "Stakeholders insist on daily readouts, or stopping early for a big win matters.",
        "randomization_unit": "user_id",
        "analysis": "Alpha spending (O'Brien-Fleming) or mSPRT; never use the naive p < 0.05 rule repeatedly.",
        "pitfalls": ["Wider CIs than a fixed-horizon test", "Poorly chosen prior in mSPRT inflates false positives"],
    },
    "cuped": {
        "name_en": "CUPED variance reduction",
        "name_zh": "CUPED variance reduction",
        "when_zh": "A correlated pre-period metric exists (always, for logged-in users).",
        "when": "A correlated pre-period metric exists (always, for logged-in users).",
        "randomization_unit": "user_id",
        "analysis": "Y_adj = Y - theta * (X - mean X); variance reduction equals corr(X, Y)^2.",
        "pitfalls": ["Covariate must be unaffected by treatment", "Theta estimated on pooled data can bias small samples"],
    },
    "triggered": {
        "name_en": "Triggered / activation analysis",
        "name_zh": "Triggered / activation analysis",
        "when_zh": "Only a subset of users can possibly be affected (e.g. users who saw the banner).",
        "when": "Only a subset of users can possibly be affected (e.g. users who saw the banner).",
        "randomization_unit": "user_id (randomise all, analyse triggered)",
        "analysis": "Compare triggered users only; report both ITT and triggered-only effect.",
        "pitfalls": ["Trigger is post-randomisation: check it is not affected by treatment", "Dilution hides real effects in ITT"],
    },
    "quasi": {
        "name_en": "Quasi-experiment (DID / synthetic control)",
        "name_zh": "Quasi-experiment (DID / synthetic control)",
        "when_zh": "Randomisation is impossible (policy change, pricing, competitor shock).",
        "when": "Randomisation is impossible (policy change, pricing, competitor shock).",
        "randomization_unit": "geo / entity",
        "analysis": "Difference-in-differences with parallel-trends test, or synthetic control with placebo tests.",
        "pitfalls": ["Parallel trends is an assumption, not a fact", "Concurrent shocks masquerade as effects"],
    },
    "nested": {
        "name_en": "Parent-child (nested / orthogonal layers)",
        "name_zh": "Parent-child (nested / orthogonal layers)",
        "when_zh": "Several experiments run on the same surface at once and must not contaminate each other.",
        "when": "Several experiments run on the same surface at once and must not contaminate each other.",
        "randomization_unit": "user_id within an orthogonal layer; parent arm fixed per user",
        "analysis": "Orthogonality check (chi-square on the parent x child crosstab), then child effect adjusted for parent fixed effects; report the interaction to prove there is none.",
        "pitfalls": [
            "Layers leak when one experiment's treatment changes the traffic of another",
            "Adjusting for the parent without checking orthogonality hides selection bias",
            "Tiny cells make the interaction test useless - size for the interaction, not just the main effect",
        ],
    },
    "reversal": {
        "name_en": "Reversal (swap-back) experiment",
        "name_zh": "Reversal (swap-back) experiment",
        "when_zh": "You suspect instrumentation drift, novelty effects or a time trend, so the two arms swap experiences mid-flight.",
        "when": "You suspect instrumentation drift, novelty effects or a time trend, so the two arms swap experiences mid-flight.",
        "randomization_unit": "user_id, with an arm-to-experience swap at the midpoint",
        "analysis": "Estimate the effect in both phases; tau1 and tau2 should be equal and opposite. Their sum tests for drift, their ratio measures persistence.",
        "pitfalls": [
            "Carryover: users remember the first experience, so tau2 understates the effect",
            "The swap date must be identical for both arms or the phases are not comparable",
            "A null sum test does not prove the effect is real - it only rules out drift",
        ],
    },
    "two_sided": {
        "name_en": "Two-sided / marketplace 2x2",
        "name_zh": "Two-sided / marketplace 2x2",
        "when_zh": "The platform changes both the demand and the supply side and you need direct effects plus cross-side spillovers.",
        "when": "The platform changes both the demand and the supply side and you need direct effects plus cross-side spillovers.",
        "randomization_unit": "market / city, crossed buyer-treatment x seller-treatment",
        "analysis": "2x2 cell means: direct effects from the marginals, cross-side spillover from the off-diagonal, interaction from the difference-in-differences of cells; cluster bootstrap on markets.",
        "pitfalls": [
            "Randomising at user level when the two sides trade with each other guarantees contamination",
            "Few markets make the interaction test powerless",
            "Totals confound market size - analyse rates, or include market fixed effects",
        ],
    },
}


def recommend_experiment_type(context: str) -> Dict[str, object]:
    """Pick an archetype from a natural-language description of the situation."""
    text = (context or "").lower()
    scores: Dict[str, int] = {k: 0 for k in EXPERIMENT_TYPES}

    def bump(keys: Sequence[str], weight: int = 3) -> None:
        for k in keys:
            scores[k] += weight

    # Marketplace / two-sided keywords point to switchback first, cluster second.
    if any(w in text for w in ["双边", "骑手", "司机", "供给", "marketplace", "dispatch", "抢单"]):
        bump(["switchback"], 5)
        bump(["cluster"], 2)
    if any(w in text for w in ["动态定价", "定价", "pricing", "补贴", "费率"]):
        bump(["switchback"], 4)
    if any(w in text for w in ["地域", "城市", "geo", "省份", "区域", "国家", "country"]):
        bump(["cluster", "quasi"], 3)
    if any(w in text for w in ["政策", "法规", "无法随机", "不能随机", "自然实验", "policy", "rollout"]):
        bump(["quasi"], 4)
    if any(w in text for w in ["长期", "留存", "ltv", "long term", "durable", "持续"]):
        bump(["holdout", "sequential"], 3)
    if any(w in text for w in ["每天看", "随时看", "提前停止", "peek", "sequential", "daily readout"]):
        bump(["sequential"], 4)
    if any(w in text for w in ["多变量", "两个变量", "组合", "factorial", "交互", "interaction", "价格和"]):
        bump(["factorial", "abn"], 3)
    if any(w in text for w in ["多个版本", "a/b/n", "多组", "several variants"]):
        bump(["abn"], 3)
    if any(w in text for w in ["流量少", "样本少", "low traffic", "小样本", "预算"]):
        bump(["bandit", "sequential", "cuped"], 2)
    if any(w in text for w in ["推荐", "排序", "rank", "feed", "信息流"]):
        bump(["ab", "cuped"], 2)
    if any(w in text for w in ["曝光", "看到", "trigger", "banner", "push"]):
        bump(["triggered"], 3)
    if any(w in text for w in ["分层", "strata", "stratified"]):
        bump(["stratified"], 3)
    if any(w in text for w in ["方差", "波动大", "噪声大", "cuped", "variance"]):
        bump(["cuped"], 3)
    if any(w in text for w in ["父子", "嵌套", "正交", "分层实验", "nested", "orthogonal", "layer", "多个实验同时"]):
        bump(["nested"], 5)
    if any(w in text for w in ["反转", "对调", "交换", "reversal", "swap", "埋点漂移", "新奇效应", "novelty"]):
        bump(["reversal"], 5)
    if any(w in text for w in ["双边实验", "两侧", "买卖", "买家", "卖家", "供需两端", "two-sided", "buyer", "seller"]):
        bump(["two_sided"], 5)
    if any(w in text for w in ["敏感性", "稳健性", "sensitivity", "robustness", "换口径"]):
        bump(["cuped", "ab"], 2)
    if max(scores.values()) == 0:
        scores["ab"] = 3

    best = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    primary = best[0][0]
    secondary = [k for k, v in best[1:4] if v > 0]
    return {
        "primary": primary,
        "secondary": secondary,
        "detail": EXPERIMENT_TYPES[primary],
        "alternatives": [{"type": k, "detail": EXPERIMENT_TYPES[k]} for k in secondary],
        "reasoning": f"Matched keywords against {len(EXPERIMENT_TYPES)} archetypes; '{primary}' scored highest.",
    }


# --------------------------------------------------------------------------- #
# Design
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentDesign:
    """Requested design parameters."""

    goal: str = ""
    experiment_type: str = "ab"
    metric_kind: str = "proportion"     # proportion | mean | count | ratio
    baseline: float = 0.05              # rate, mean, or ratio baseline
    mde_rel: Optional[float] = 0.05
    mde_abs: Optional[float] = None
    sd: Optional[float] = None          # required for mean metrics
    dispersion: float = 1.0             # for count metrics
    allocation: float = 1.0             # treatment:control
    n_arms: int = 2
    alpha: float = 0.05
    power: float = 0.8
    daily_traffic: float = 50_000
    icc: float = 0.02
    cluster_size: float = 1.0
    guardrail_metrics: List[str] = field(default_factory=list)
    lang: str = "en"                 # "zh" or "en": controls narrative language


def design_experiment(design: ExperimentDesign) -> Dict[str, object]:
    """Full design: sizing, duration, randomisation, analysis plan and risks."""
    kind = design.metric_kind
    if kind == "proportion":
        sizing = pw.sample_size_proportion(
            design.baseline, mde_rel=design.mde_rel, mde_abs=design.mde_abs,
            alpha=design.alpha, power=design.power, allocation=design.allocation,
        )
    elif kind == "mean":
        sd = design.sd if design.sd is not None else abs(design.baseline)
        sizing = pw.sample_size_mean(
            design.baseline, sd=sd, mde_rel=design.mde_rel, mde_abs=design.mde_abs,
            alpha=design.alpha, power=design.power, allocation=design.allocation,
        )
    elif kind == "count":
        sizing = pw.sample_size_count(
            design.baseline, mde_rel=design.mde_rel or 0.05, alpha=design.alpha,
            power=design.power, allocation=design.allocation, dispersion=design.dispersion,
        )
    elif kind == "ratio":
        sizing = pw.sample_size_ratio_metric(
            mean_numerator=design.baseline, mean_denominator=1.0, mde_rel=design.mde_rel or 0.03,
            alpha=design.alpha, power=design.power, allocation=design.allocation,
        )
    else:
        raise ValueError(f"unsupported metric_kind: {kind}")

    # Multi-arm correction: alpha per comparison, and traffic split.
    n_arms = max(2, design.n_arms)
    comparisons = n_arms - 1
    alpha_adjusted = design.alpha / comparisons if comparisons > 1 else design.alpha
    if comparisons > 1:
        per_arm = sizing["n_control"] * (1 + design.allocation)
        sizing["n_control"] = int(np.ceil(per_arm))
        sizing["n_treatment"] = int(np.ceil(per_arm * design.allocation))
        sizing["n_total"] = int(np.ceil(per_arm * (1 + design.allocation)))
        sizing["alpha_adjusted"] = alpha_adjusted
        sizing["note_multi_arm"] = (
            f"{n_arms} arms => {comparisons} comparisons; applied Bonferroni alpha={alpha_adjusted:.4f}. "
            "Prefer BH/Holm at readout for slightly better power."
        )

    # Cluster randomisation inflates the required sample.
    deff = 1.0
    if design.cluster_size > 1:
        deff_info = pw.cluster_design_effect(design.cluster_size, design.icc)
        deff = deff_info["design_effect"]
        sizing["cluster_design_effect"] = deff_info
        sizing["n_total"] = int(np.ceil(sizing["n_total"] * deff))
        sizing["n_control"] = int(np.ceil(sizing["n_control"] * deff))
        sizing["n_treatment"] = int(np.ceil(sizing["n_treatment"] * deff))

    duration = pw.duration_days(sizing["n_total"], design.daily_traffic, design.allocation)
    archetype = EXPERIMENT_TYPES.get(design.experiment_type, EXPERIMENT_TYPES["ab"])

    guardrails = design.guardrail_metrics or ["latency_p95", "crash_rate", "refund_rate", "unsubscribe_rate"]

    analysis_plan = _analysis_plan(design, sizing)
    risks = _risks(design)

    return {
        "goal": design.goal,
        "experiment_type": design.experiment_type,
        "archetype": archetype,
        "params": design.__dict__,
        "sizing": sizing,
        "duration": duration,
        "design_effect": deff,
        "guardrails": guardrails,
        "analysis_plan": analysis_plan,
        "risks": risks,
        "decision_rule": _decision_rule(design, sizing),
        "markdown": _design_markdown(design, sizing, duration, analysis_plan, risks, archetype),
    }


def _tr(lang: str, zh: str, en: str) -> str:
    """Inline translation helper for bilingual design documents."""
    return zh if lang == "zh" else en


def _analysis_plan(design: ExperimentDesign, sizing: Dict[str, float]) -> List[str]:
    lang = design.lang
    plan = []
    if design.metric_kind == "proportion":
        plan.append(_tr(lang,
            "Primary test: two-proportion z-test on the success rate; report absolute and relative lift with 95% CI.",
            "Primary test: two-proportion z-test on the success rate; report absolute and relative lift with 95% CI."))
    elif design.metric_kind == "mean":
        plan.append(_tr(lang,
            "Primary test: Welch t-test on per-user values; bootstrap CI if the distribution is heavy-tailed.",
            "Primary test: Welch t-test on per-user values; bootstrap CI if the distribution is heavy-tailed."))
        plan.append(_tr(lang,
            "Winsorise the top 0.1% of revenue outliers and report results with and without winsorisation.",
            "Winsorise the top 0.1% of revenue outliers and report results with and without winsorisation."))
    elif design.metric_kind == "count":
        plan.append(_tr(lang,
            "Primary test: quasi-Poisson regression of counts per user with a group indicator.",
            "Primary test: quasi-Poisson regression of counts per user with a group indicator."))
    else:
        plan.append(_tr(lang,
            "Primary test: delta-method or bootstrap CI for the ratio of sums (numerator/denominator).",
            "Primary test: delta-method or bootstrap CI for the ratio of sums (numerator/denominator)."))

    plan.append(_tr(lang,
        "Run an SRM check before any metric readout; if p < 0.001, stop and fix instrumentation.",
        "Run an SRM check before any metric readout; if p < 0.001, stop and fix instrumentation."))
    plan.append(_tr(lang,
        "Apply CUPED on a pre-period covariate of the same metric; expect 10-50% variance reduction.",
        "Apply CUPED on a pre-period covariate of the same metric; expect 10-50% variance reduction."))
    if design.experiment_type == "sequential":
        plan.append(_tr(lang,
            "Interim looks: O'Brien-Fleming alpha spending, or mSPRT if looks are data-driven.",
            "Interim looks: O'Brien-Fleming alpha spending, or mSPRT if looks are data-driven."))
    if design.n_arms > 2:
        plan.append(_tr(lang,
            f"Multiple comparisons: BH-FDR across {design.n_arms - 1} treatment arms and all secondary metrics.",
            f"Multiple comparisons: BH-FDR across {design.n_arms - 1} treatment arms and all secondary metrics."))
    if design.experiment_type == "cluster":
        plan.append(_tr(lang,
            "Cluster-robust standard errors (or block bootstrap) using the randomisation unit.",
            "Cluster-robust standard errors (or block bootstrap) using the randomisation unit."))
    plan.append(_tr(lang,
        "Pre-register one primary metric, <=5 secondary metrics and guardrails before launch.",
        "Pre-register one primary metric, <=5 secondary metrics and guardrails before launch."))
    plan.append(_tr(lang,
        "Exclude the first 24h (logging ramp) and check novelty/primacy effects by week.",
        "Exclude the first 24h (logging ramp) and check novelty/primacy effects by week."))
    return plan


def _risks(design: ExperimentDesign) -> List[str]:
    lang = design.lang
    risks = []
    if design.daily_traffic * design.allocation / (1 + design.allocation) < 200:
        risks.append(_tr(lang,
            "Very low daily traffic per arm: consider sequential testing or a cheaper proxy metric.",
            "Very low daily traffic per arm: consider sequential testing or a cheaper proxy metric."))
    if design.mde_rel and design.mde_rel < 0.01:
        risks.append(_tr(lang,
            "Sub-1% MDE usually needs weeks of exposure and is sensitive to any instrumentation drift.",
            "Sub-1% MDE usually needs weeks of exposure and is sensitive to any instrumentation drift."))
    if design.metric_kind == "mean":
        risks.append(_tr(lang,
            "Heavy-tailed revenue metrics: one whale can dominate; plan a trimmed or capped readout.",
            "Heavy-tailed revenue metrics: one whale can dominate; plan a trimmed or capped readout."))
    if design.experiment_type in ("switchback", "cluster"):
        risks.append(_tr(lang,
            "Interference/carryover: define a washout window and verify pre-period balance.",
            "Interference/carryover: define a washout window and verify pre-period balance."))
    if design.experiment_type == "bandit":
        risks.append(_tr(lang,
            "Adaptive allocation invalidates fixed-horizon p-values; use posterior probability instead.",
            "Adaptive allocation invalidates fixed-horizon p-values; use posterior probability instead."))
    risks.append(_tr(lang,
        "Instrumentation drift is the most common cause of fake wins: log both arms with the same code path.",
        "Instrumentation drift is the most common cause of fake wins: log both arms with the same code path."))
    return risks


def _decision_rule(design: ExperimentDesign, sizing: Dict[str, float]) -> str:
    lang = design.lang
    mde = design.mde_rel if design.mde_rel is not None else design.mde_abs
    if lang == "zh":
        return (
            f"Ship if the primary metric shows a statistically significant lift (alpha={sizing['alpha']:.3f}) "
            f"with the CI excluding an effect smaller than {mde}, no guardrail regression beyond the agreed "
            f"threshold, and the effect is consistent across at least two of three pre-registered segments."
        )
    return (
        f"Ship if the primary metric shows a statistically significant lift (alpha={sizing['alpha']:.3f}) "
        f"with the CI excluding an effect smaller than {mde}, no guardrail regression beyond the agreed "
        f"threshold, and the effect is consistent across at least two of three pre-registered segments."
    )


def _design_markdown(
    design: ExperimentDesign,
    sizing: Dict[str, float],
    duration: Dict[str, float],
    analysis_plan: List[str],
    risks: List[str],
    archetype: Dict[str, object],
) -> str:
    lang = design.lang
    when = archetype.get("when_zh") if lang == "zh" else archetype.get("when")
    lines = [
        f"## {_tr(lang, 'Experiment design', 'Experiment design')}: {design.goal or 'untitled'}",
        "",
        f"**{_tr(lang, 'Archetype', 'Archetype')}**: {archetype['name_zh']} / {archetype['name_en']}",
        f"**{_tr(lang, 'Randomisation unit', 'Randomisation unit')}**: {archetype['randomization_unit']}",
        "",
        f"### {_tr(lang, 'Sizing', 'Sizing')}",
        f"- {_tr(lang, 'Baseline', 'Baseline')}: {sizing.get('control_rate', design.baseline):.4g}",
        f"- {_tr(lang, 'Detectable effect', 'Detectable effect')}: "
        f"{design.mde_rel if design.mde_rel is not None else design.mde_abs} "
        f"({'relative' if design.mde_rel is not None else 'absolute'})",
        f"- {_tr(lang, 'Per arm', 'Per arm')}: control {sizing['n_control']:,} / treatment {sizing['n_treatment']:,} "
        f"({_tr(lang, 'total', 'total')} {sizing['n_total']:,})",
        f"- {_tr(lang, 'Achieved power', 'Achieved power')}: {sizing.get('achieved_power', 0):.3f} @ alpha={sizing['alpha']}",
        f"- {_tr(lang, 'Duration', 'Duration')}: ~{duration['raw_days']} {_tr(lang, 'days', 'days')} "
        f"({duration['recommended_days']} {_tr(lang, 'days recommended', 'days recommended')})",
    ]
    if "cluster_design_effect" in sizing:
        lines.append(f"- {_tr(lang, 'Cluster design effect', 'Cluster design effect')}: {sizing['cluster_design_effect']['design_effect']:.2f}x")
    lines += ["", f"### {_tr(lang, 'Analysis plan', 'Analysis plan')}"] + [f"1. {p}" for p in analysis_plan]
    lines += ["", f"### {_tr(lang, 'Guardrails', 'Guardrails')}"] + [
        f"- {g}" for g in (design.guardrail_metrics or ["latency_p95", "crash_rate", "refund_rate"])
    ]
    lines += ["", f"### {_tr(lang, 'Risks', 'Risks')}"] + [f"- {r}" for r in risks]
    if when:
        lines += ["", f"> {when}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyze_experiment(
    df: pd.DataFrame,
    metrics: Sequence[Dict[str, str]],
    group_col: str = "group",
    control: str = "control",
    treatment: Optional[str] = None,
    covariate_col: Optional[str] = "pre_revenue",
    alpha: float = 0.05,
    n_looks: int = 1,
    segment_cols: Sequence[str] = (),
) -> Dict[str, object]:
    """Read out an AB test: SRM, per-metric tests, CUPED, segments, decisions.

    ``metrics`` items: {"name": ..., "kind": "proportion|mean|ratio", "numerator": col,
    "denominator": col (optional), "higher_is_better": bool}
    """
    if group_col not in df.columns:
        raise KeyError(f"missing group column '{group_col}'")
    groups = list(pd.unique(df[group_col]))
    if control not in groups:
        raise ValueError(f"control group '{control}' not found in {groups}")
    treatment = treatment or next((g for g in groups if g != control), None)
    if treatment is None:
        raise ValueError("no treatment group found")

    counts = df[group_col].value_counts()
    srm = inf.srm_check([int(counts.get(g, 0)) for g in groups])

    ctrl = df[df[group_col] == control]
    trt = df[df[group_col] == treatment]
    results: List[Dict[str, object]] = []

    for spec in metrics:
        name, kind = spec["name"], spec.get("kind", "mean")
        num_col = spec.get("numerator", spec.get("column", name))
        den_col = spec.get("denominator")
        higher_better = spec.get("higher_is_better", True)

        if kind == "proportion":
            x1 = float((ctrl[num_col] > 0).sum()) if ctrl[num_col].dtype == bool else float(ctrl[num_col].sum())
            x2 = float((trt[num_col] > 0).sum()) if trt[num_col].dtype == bool else float(trt[num_col].sum())
            n1, n2 = float(len(ctrl)), float(len(trt))
            stats_res = inf.two_proportion_test(int(x1), int(n1), int(x2), int(n2), alpha)
        elif kind == "ratio" and den_col:
            stats_res = inf.ratio_metric_test_from_samples(
                ctrl[num_col].to_numpy(), ctrl[den_col].to_numpy(),
                trt[num_col].to_numpy(), trt[den_col].to_numpy(),
                alpha=alpha,
            )
            num_c, num_t = float(ctrl[num_col].sum()), float(trt[num_col].sum())
            den_c, den_t = float(ctrl[den_col].sum()), float(trt[den_col].sum())
            stats_res["control"] = num_c / den_c if den_c else np.nan
            stats_res["treatment"] = num_t / den_t if den_t else np.nan
        else:
            stats_res = inf.welch_ttest(ctrl[num_col].to_numpy(), trt[num_col].to_numpy(), alpha)
            boot = inf.bootstrap_diff_in_means(
                ctrl[num_col].to_numpy(), trt[num_col].to_numpy(), n_boot=800, alpha=alpha
            )
            stats_res["ci_bootstrap"] = boot["ci_abs"]

        # CUPED: adjust both arms with the pre-period covariate.
        cuped_info = None
        if covariate_col and covariate_col in df.columns:
            y_all = df[num_col].to_numpy(dtype=float)
            x_all = df[covariate_col].to_numpy(dtype=float)
            valid = ~(np.isnan(y_all) | np.isnan(x_all))
            adj = inf.cuped_adjust(y_all[valid], x_all[valid])
            groups_valid = df[group_col].to_numpy()[valid]
            mask_c = groups_valid == control
            mask_t = groups_valid == treatment
            if mask_c.sum() > 2 and mask_t.sum() > 2:
                cuped_test = inf.welch_ttest(adj["y_adjusted"][mask_c], adj["y_adjusted"][mask_t], alpha)
                cuped_info = {
                    "theta": adj["theta"],
                    "variance_reduction": adj["variance_reduction"],
                    "corr": adj["corr"],
                    "abs_diff": cuped_test["abs_diff"],
                    "rel_lift": cuped_test["rel_lift"],
                    "p_value": cuped_test["p_value"],
                    "ci_abs": cuped_test["ci_abs"],
                }

        p_value = float(stats_res.get("p_value", 1.0))
        seq = inf.sequential_p_value(float(stats_res.get("abs_diff", 0.0)), float(stats_res.get("se", 0.0) or 0.0))
        results.append(
            {
                "metric": name,
                "kind": kind,
                "higher_is_better": higher_better,
                # The primitives name the arm means differently (control vs
                # control_mean); normalise so every reader sees the same keys.
                "control": stats_res.get("control", stats_res.get("control_mean")),
                "treatment": stats_res.get("treatment", stats_res.get("treatment_mean")),
                "abs_diff": stats_res.get("abs_diff"),
                "rel_lift": stats_res.get("rel_lift"),
                "p_value": p_value,
                "ci_abs": stats_res.get("ci_abs"),
                "ci_rel": stats_res.get("ci_rel"),
                "significant": p_value < alpha,
                "cuped": cuped_info,
                "sequential_p": seq["p_value"],
                "stats": {k: v for k, v in stats_res.items() if k not in ("ci_abs", "ci_rel")},
            }
        )

    # Multiplicity control across the metric family.
    adjusted = inf.bh_correction([r["p_value"] for r in results], alpha)
    for r, adj in zip(results, adjusted):
        r["p_adjusted"] = adj["p_adjusted"]
        r["significant_adjusted"] = adj["significant"]

    segments = (
        _segment_readout(df, metrics[0], group_col, control, treatment, segment_cols) if segment_cols else []
    )

    primary = results[0]
    direction_ok = (primary["abs_diff"] or 0) * (1 if primary["higher_is_better"] else -1) > 0
    if primary["significant_adjusted"] and direction_ok:
        recommendation = f"Ship: {primary['metric']} moves {primary['rel_lift']:+.2%} with p_adj={primary['p_adjusted']:.4f}."
    elif primary["significant_adjusted"]:
        recommendation = f"Do not ship: {primary['metric']} moves in the wrong direction ({primary['rel_lift']:+.2%})."
    else:
        recommendation = (
            f"Inconclusive: {primary['metric']} lift {primary['rel_lift']:+.2%} (p={primary['p_value']:.3f}). "
            "Check power, consider extending or shipping as a neutral change."
        )

    return {
        "groups": groups,
        "control": control,
        "treatment": treatment,
        "n": {g: int(counts.get(g, 0)) for g in groups},
        "srm": srm,
        "metrics": results,
        "segments": segments,
        "alpha": alpha,
        "n_looks": n_looks,
        "obrien_fleming": inf.obrien_fleming_spend(n_looks, alpha) if n_looks > 1 else [],
        "recommendation": recommendation,
        "narrative": _analysis_narrative(results, srm, primary),
    }


def _segment_readout(
    df: pd.DataFrame,
    metric: Dict[str, str],
    group_col: str,
    control: str,
    treatment: str,
    segment_cols: Sequence[str] = (),
) -> List[Dict[str, object]]:
    """Per-segment effects (exploratory: no multiplicity correction applied)."""
    out: List[Dict[str, object]] = []
    for seg_col in [c for c in segment_cols if c in df.columns]:
        if df[seg_col].nunique() <= 12:
            for value, chunk in df.groupby(seg_col):
                c = chunk[chunk[group_col] == control][metric.get("numerator", metric["name"])]
                t = chunk[chunk[group_col] == treatment][metric.get("numerator", metric["name"])]
                if len(c) < 30 or len(t) < 30:
                    continue
                if metric.get("kind") == "proportion":
                    test = inf.two_proportion_test(int(c.sum()), len(c), int(t.sum()), len(t))
                else:
                    test = inf.welch_ttest(c.to_numpy(), t.to_numpy())
                out.append(
                    {
                        "segment_col": seg_col,
                        "segment": str(value),
                        "control": test.get("control", test.get("control_mean")),
                        "treatment": test.get("treatment", test.get("treatment_mean")),
                        "rel_lift": test.get("rel_lift"),
                        "p_value": test.get("p_value"),
                        "n": int(len(c) + len(t)),
                    }
                )
    return out


def _analysis_narrative(results: List[Dict[str, object]], srm: Dict[str, object], primary: Dict[str, object]) -> str:
    text = []
    if not srm["passed"]:
        text.append("WARNING: SRM detected, the readout is not trustworthy until randomisation is fixed.")
    text.append(
        f"Primary metric {primary['metric']}: control {primary['control']:.4g} vs treatment "
        f"{primary['treatment']:.4g}, lift {primary['rel_lift']:+.2%} "
        f"(95% CI {primary['ci_abs'][0]:+.4g}..{primary['ci_abs'][1]:+.4g}, p={primary['p_value']:.4f})."
    )
    if primary.get("cuped"):
        cup = primary["cuped"]
        text.append(
            f"CUPED with corr={cup['corr']:.2f} cuts variance by {cup['variance_reduction']:.0%}; "
            f"adjusted lift {cup['rel_lift']:+.2%} (p={cup['p_value']:.4f})."
        )
    sig = [r["metric"] for r in results if r["significant_adjusted"]]
    text.append(
        f"{len(sig)} of {len(results)} metrics remain significant after BH correction: {', '.join(sig) or 'none'}."
    )
    return " ".join(text)
