"""Scenario-based method selection.

The agent's hardest job is not running an estimator — it is choosing one.
This module turns a free-text scenario description into parsed scenario
features, then applies a decision tree over the full method catalogue to
produce a ranked recommendation with reasons, assumptions and pitfalls.

The catalogue covers every estimator the package ships, so the advice is
always actionable: each recommendation names the tool that implements it.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Method catalogue: one entry per shipped estimator. `tool` is importable from
# ds_agent.tools, so the recommendation is executable, not just advice.
# ---------------------------------------------------------------------------
CATALOGUE: Dict[str, Dict[str, str]] = {
    "ab_test": {
        "name_zh": "Standard A/B test",
        "name_en": "Standard A/B test",
        "tool": "experiment.design_experiment",
        "assumption_zh": "Independent units, randomisable traffic",
        "assumption_en": "Independent units, randomisable traffic",
        "pitfall_zh": "Peeking; SRM; spillover across units",
        "pitfall_en": "Peeking; SRM; spillover across units",
    },
    "cuped": {
        "name_zh": "CUPED variance reduction",
        "name_en": "CUPED variance reduction",
        "tool": "stats.cuped_adjust",
        "assumption_zh": "A pre-period covariate correlated with the metric",
        "assumption_en": "A pre-period covariate correlated with the metric",
        "pitfall_zh": "Covariate must be pre-treatment, else it biases the estimate",
        "pitfall_en": "Covariate must be pre-treatment, else it biases the estimate",
    },
    "switchback": {
        "name_zh": "Switchback experiment",
        "name_en": "Switchback experiment",
        "tool": "experiment_designs.analyze_switchback",
        "assumption_zh": "Effect washes out within the block; no leakage across blocks",
        "assumption_en": "Effect washes out within the block; no leakage across blocks",
        "pitfall_zh": "Inference unit is the time block, not the user; carryover inflates the effect",
        "pitfall_en": "Inference unit is the time block, not the user; carryover inflates the effect",
    },
    "nested": {
        "name_zh": "Parent-child (nested) experiment",
        "name_en": "Parent-child (nested) experiment",
        "tool": "experiment_designs.analyze_nested_experiment",
        "assumption_zh": "The two layers are orthogonal (stratified randomisation)",
        "assumption_en": "The two layers are orthogonal (stratified randomisation)",
        "pitfall_zh": "Without the orthogonality check the parent effect leaks into the child",
        "pitfall_en": "Without the orthogonality check the parent effect leaks into the child",
    },
    "two_sided": {
        "name_zh": "Two-sided market experiment",
        "name_en": "Two-sided market experiment",
        "tool": "experiment_designs.analyze_two_sided",
        "assumption_zh": "Cluster by market; intervene on each side separately",
        "assumption_en": "Cluster by market; intervene on each side separately",
        "pitfall_zh": "User-level assignment is contaminated by cross-side spillover",
        "pitfall_en": "User-level assignment is contaminated by cross-side spillover",
    },
    "cluster": {
        "name_zh": "Cluster experiment",
        "name_en": "Cluster experiment",
        "tool": "experiment.design_experiment",
        "assumption_zh": "Enough clusters (>=20); cluster-level bootstrap for SEs",
        "assumption_en": "Enough clusters (>=20); cluster-level bootstrap for SEs",
        "pitfall_zh": "Few clusters make SEs unstable; power is far below user-level splits",
        "pitfall_en": "Few clusters make SEs unstable; power is far below user-level splits",
    },
    "did": {
        "name_zh": "Difference-in-differences",
        "name_en": "Difference-in-differences",
        "tool": "causal.difference_in_differences",
        "assumption_zh": "Parallel trends between treated and control",
        "assumption_en": "Parallel trends between treated and control",
        "pitfall_zh": "Check pre-trends with an event study; staggered timing needs TWFE care",
        "pitfall_en": "Check pre-trends with an event study; staggered timing needs TWFE care",
    },
    "event_study": {
        "name_zh": "Event study",
        "name_en": "Event study",
        "tool": "causal.event_study",
        "assumption_zh": "A clean pre-period exists",
        "assumption_en": "A clean pre-period exists",
        "pitfall_zh": "Anticipation effects look like pre-trend violations",
        "pitfall_en": "Anticipation effects look like pre-trend violations",
    },
    "psm": {
        "name_zh": "Propensity score matching",
        "name_en": "Propensity score matching",
        "tool": "causal.propensity_score_matching",
        "assumption_zh": "Conditional ignorability plus overlap; identifies the ATT",
        "assumption_en": "Conditional ignorability plus overlap; identifies the ATT",
        "pitfall_zh": "Balances only observed covariates; hidden selection survives",
        "pitfall_en": "Balances only observed covariates; hidden selection survives",
    },
    "ipw": {
        "name_zh": "IPW / AIPW",
        "name_en": "IPW / AIPW",
        "tool": "causal.aipw_effect",
        "assumption_zh": "Propensity model right (AIPW is doubly robust)",
        "assumption_en": "Propensity model right (AIPW is doubly robust)",
        "pitfall_zh": "Extreme propensities explode the weights; trim them",
        "pitfall_en": "Extreme propensities explode the weights; trim them",
    },
    "synthetic_control": {
        "name_zh": "Synthetic control",
        "name_en": "Synthetic control",
        "tool": "causal.synthetic_control",
        "assumption_zh": "One treated unit plus untouched donors; good pre-period fit",
        "assumption_en": "One treated unit plus untouched donors; good pre-period fit",
        "pitfall_zh": "Unreliable with few donors or poor pre-fit; use placebo inference",
        "pitfall_en": "Unreliable with few donors or poor pre-fit; use placebo inference",
    },
    "rdd": {
        "name_zh": "Regression discontinuity",
        "name_en": "Regression discontinuity",
        "tool": "rdd.regression_discontinuity",
        "assumption_zh": "Units cannot precisely manipulate the running variable",
        "assumption_en": "Units cannot precisely manipulate the running variable",
        "pitfall_zh": "Density test rejection signals manipulation; run bandwidth sensitivity",
        "pitfall_en": "Density test rejection signals manipulation; run bandwidth sensitivity",
    },
    "its": {
        "name_zh": "Interrupted time series (ARIMA ITS)",
        "name_en": "Interrupted time series (ARIMA ITS)",
        "tool": "ts_causal.arima_intervention",
        "assumption_zh": "Last resort without controls: trend/seasonality must be modelled",
        "assumption_en": "Last resort without controls: trend/seasonality must be modelled",
        "pitfall_zh": "Concurrent events are indistinguishable; mitigate with placebos and controls",
        "pitfall_en": "Concurrent events are indistinguishable; mitigate with placebos and controls",
    },
    "causal_impact": {
        "name_zh": "Structural counterfactual (CausalImpact)",
        "name_en": "Structural counterfactual (CausalImpact)",
        "tool": "ts_causal.causal_impact",
        "assumption_zh": "Correlated control series untouched by the intervention",
        "assumption_en": "Correlated control series untouched by the intervention",
        "pitfall_zh": "Contaminated controls bias the counterfactual",
        "pitfall_en": "Contaminated controls bias the counterfactual",
    },
    "uplift": {
        "name_zh": "Uplift (causal forest / meta-learners)",
        "name_en": "Uplift (causal forest / meta-learners)",
        "tool": "uplift.causal_forest",
        "assumption_zh": "Randomised data (or ignorability); for heterogeneity and targeting",
        "assumption_en": "Randomised data (or ignorability); for heterogeneity and targeting",
        "pitfall_zh": "On observational data it learns selection bias; validate with Qini/AIPW",
        "pitfall_en": "On observational data it learns selection bias; validate with Qini/AIPW",
    },
    "sensitivity": {
        "name_zh": "Specification-curve sensitivity",
        "name_en": "Specification-curve sensitivity",
        "tool": "experiment_designs.sensitivity_analysis",
        "assumption_zh": "The sign should hold across a reasonable spec set",
        "assumption_en": "The sign should hold across a reasonable spec set",
        "pitfall_zh": "Cherry-picking significant specs is p-hacking",
        "pitfall_en": "Cherry-picking significant specs is p-hacking",
    },
    "anomaly": {
        "name_zh": "Time-series anomaly detection (incl. Holt-Winters)",
        "name_en": "Time-series anomaly detection (incl. Holt-Winters)",
        "tool": "anomaly.AnomalyDetector",
        "assumption_zh": "Daily series with learnable seasonal structure",
        "assumption_en": "Daily series with learnable seasonal structure",
        "pitfall_zh": "A random walk has no baseline; check stability first",
        "pitfall_en": "A random walk has no baseline; check stability first",
    },
    "forecast": {
        "name_zh": "Forecasting (HW / SARIMA / STL)",
        "name_en": "Forecasting (HW / SARIMA / STL)",
        "tool": "forecast.select_method",
        "assumption_zh": "Patterns persist; intervals widen with horizon",
        "assumption_en": "Patterns persist; intervals widen with horizon",
        "pitfall_zh": "History is useless after a structural break",
        "pitfall_en": "History is useless after a structural break",
    },
    "descriptive": {
        "name_zh": "Descriptives + contribution decomposition",
        "name_en": "Descriptives + contribution decomposition",
        "tool": "contribution.attribute_change",
        "assumption_zh": "Answers where it moved, not why",
        "assumption_en": "Answers where it moved, not why",
        "pitfall_zh": "Reading decomposition as causal explanation",
        "pitfall_en": "Reading decomposition as causal explanation",
    },
}

# ---------------------------------------------------------------------------
# Scenario feature extraction
# ---------------------------------------------------------------------------
_FEATURES = {
    # design signals
    "experiment": r"ab\s*实验|ab\s*测试|实验组|随机分流|分流|randomi[sz]|controlled trial|\bab\s?test\b",
    "switchback": r"时间片|轮转|switchback",
    "nested": r"父子|嵌套|正交|nested|orthogonal layer",
    "two_sided": r"双边|两侧|买卖|买家.{0,6}卖家|供需|two-sided|marketplace",
    "cluster": r"按城市|按地区|市场级|城市级|cluster|by region|by city",
    # treatment / assignment signals
    "threshold": r"门槛|阈值|分数|评分|额度| cutoff |断点|rdd|discontinuity",
    "staggered": r"逐步|分批|陆续|灰度|staggered|rollout|phased",
    "single_event": r"上线|改版|发布|某天|当日| intervention |campaign|一次活动|某活动",
    "ongoing": r"政策|常态|持续|长期|ongoing|policy",
    "observational": r"没有.{0,2}实验|未做实验|没做实验|自然实验|观察数据|历史数据|手动|人工选|运营选|自选|observational|retrospective",
    # data-shape signals
    "panel": r"面板|多个城市|多个地区|多地区|城市间|panel|multi-?(city|region|market)|across\s+(cities|city|regions?|markets)",
    "time_series": r"时序|时间序列|每日|按天|日粒度|daily|time series",
    "cross_section": r"用户级|个体|横截面|用户画像|user-?level|cross-?section",
    # goal signals
    "effect": r"效应|影响|效果|提升多少|带来|因果|effect|impact|causal|evaluat|estimat",
    "heterogeneity": r"哪些人|哪类|异质|分层看|人群定向|定向|补贴谁|uplift|heterogen|target",
    "anomaly": r"异常|骤降|骤升|突刺|毛刺|anomaly|spike|drop",
    "forecast": r"预测|未来|预估|下个月|forecast|predict",
    "where": r"哪里|哪个渠道|哪个品类|贡献|拆解|归因|attribution|decompos",
}


def parse_scenario(question: str) -> Dict[str, bool]:
    """Regex-parse the free-text scenario into boolean features."""
    q = f" {question.lower()} "
    return {name: bool(re.search(pattern, q, re.I)) for name, pattern in _FEATURES.items()}


def recommend_methods(question: str, lang: str = "zh") -> Dict[str, Any]:
    """Rank the catalogue against a scenario description.

    Returns parsed features, ranked recommendations (each with reason,
    assumptions, pitfalls and the implementing tool), and a universal
    robustness checklist.
    """
    zh = lang == "zh"
    f = parse_scenario(question)
    ranked: List[Dict[str, Any]] = []

    def add(key: str, score: int, reason_zh: str, reason_en: str) -> None:
        m = CATALOGUE[key]
        ranked.append(
            {
                "method": key,
                "name": m["name_zh"] if zh else m["name_en"],
                "tool": m["tool"],
                "reason": reason_zh if zh else reason_en,
                "assumption": m["assumption_zh"] if zh else m["assumption_en"],
                "pitfall": m["pitfall_zh"] if zh else m["pitfall_en"],
                "score": score,
            }
        )

    # ---- Branch 1: a real experiment exists --------------------------------
    if f["experiment"] or f["switchback"] or f["nested"] or f["two_sided"]:
        if f["switchback"]:
            add("switchback", 95, "Treatment rotates over time blocks: switchback analysis with block-level inference.", "Treatment rotates over time blocks: switchback analysis with block-level inference.")
        if f["nested"]:
            add("nested", 92, "Two layers run at once: verify orthogonality, then estimate the child effect.", "Two layers run at once: verify orthogonality, then estimate the child effect.")
        if f["two_sided"]:
            add("two_sided", 92, "Both sides treated: the 2×2 grid separates direct effects from cross-side spillover.", "Both sides treated: the 2×2 grid separates direct effects from cross-side spillover.")
        if f["cluster"]:
            add("cluster", 80, "Assignment by city/region: design and infer at cluster level.", "Assignment by city/region: design and infer at cluster level.")
        if not ranked or not (f["switchback"] or f["nested"] or f["two_sided"] or f["cluster"]):
            add("ab_test", 90, "Units are independently assignable: standard A/B is the default.", "Units are independently assignable: standard A/B is the default.")
        add("cuped", 55, "With pre-period data, CUPED cuts variance and shortens the test.", "With pre-period data, CUPED cuts variance and shortens the test.")
        add("sensitivity", 40, "Run A/A and spec-curve checks before shipping to bound false positives.", "Run A/A and spec-curve checks before shipping to bound false positives.")
        add("uplift", 35, "Mine the same experiment for heterogeneity and targeting.", "Mine the same experiment for heterogeneity and targeting.")
    # ---- Branch 2: no experiment — observational identification ------------
    else:
        if f["threshold"]:
            add("rdd", 95, "Assignment follows a threshold on a score: RDD around the cutoff.", "Assignment follows a threshold on a score: RDD around the cutoff.",)
        elif f["panel"] and (f["effect"] or f["ongoing"] or f["staggered"]):
            add("did", 92, "Treated/control panel around a policy: DiD plus an event study on pre-trends.", "Treated/control panel around a policy: DiD plus an event study on pre-trends.")
            add("event_study", 70, "The event study also yields dynamic effects and the pre-trend test.", "The event study also yields dynamic effects and the pre-trend test.")
        elif f["single_event"] or (f["time_series"] and f["effect"]):
            add("its", 85, "Single series with a known intervention date: ITS models trend/seasonality, then the level shift.", "Single series with a known intervention date: ITS models trend/seasonality, then the level shift.")
            add("causal_impact", 80, "With correlated controls, the structural counterfactual is steadier than bare ITS.", "With correlated controls, the structural counterfactual is steadier than bare ITS.")
        elif f["cross_section"] or f["observational"]:
            add("psm", 90, "Selection-biased observational data: match first, re-estimate on the matched sample.", "Selection-biased observational data: match first, re-estimate on the matched sample.")
            add("ipw", 75, "AIPW is doubly robust — use it to cross-check PSM.", "AIPW is doubly robust — use it to cross-check PSM.")
        else:
            add("psm", 60, "Without a sharper strategy, start from matching and state the assumptions.", "Without a sharper strategy, start from matching and state the assumptions.")
        if f["heterogeneity"]:
            add("uplift", 88, "\"Who does it work for\": causal forest for CATEs and targeting gain.", "\"Who does it work for\": causal forest for CATEs and targeting gain.")
        if f["where"]:
            add("descriptive", 55, "Decompose the move first (descriptive, not causal).", "Decompose the move first (descriptive, not causal).")
        if f["anomaly"]:
            add("anomaly", 88, "Establish that the move is anomalous before attributing it.", "Establish that the move is anomalous before attributing it.")
        if f["forecast"]:
            add("forecast", 85, "Forecasting: pick HW/SARIMA/STL on one backtest framework.", "Forecasting: pick HW/SARIMA/STL on one backtest framework.")
        if ranked and ranked[0]["score"] >= 85:
            add("sensitivity", 45, "Observational claims always need a spec-curve pass.", "Observational claims always need a spec-curve pass.")

    ranked.sort(key=lambda r: -r["score"])
    primary = ranked[0] if ranked else None

    checklist = [
        {
            "item": self_item_zh,
            "item_en": self_item_en,
        }
        for self_item_zh, self_item_en in (
            ("Do the identifying assumptions hold — argued from design, not statistics?", "Do the identifying assumptions hold — argued from design, not statistics?"),
            ("Placebo or counterfactual checks: fake intervention date, A/A, in-space placebo?", "Placebo or counterfactual checks: fake intervention date, A/A, in-space placebo?"),
            ("Is the sign stable across specs, bandwidths, covariate sets?", "Is the sign stable across specs, bandwidths, covariate sets?"),
            ("Is the effect driven by a handful of units?", "Is the effect driven by a handful of units?"),
            ("Scope of the claim: LATE/ATT or ATE?", "Scope of the claim: LATE/ATT or ATE?"),
        )
    ]

    if zh:
        narrative = (
            f"Parsed features: {', '.join(k for k, v in f.items() if v) or 'none'}. "
            + (f"First choice **{primary['name']}**: {primary['reason']}" if primary else "Not enough information; describe the data structure and treatment.")
        )
    else:
        narrative = (
            f"Parsed features: {', '.join(k for k, v in f.items() if v) or 'none'}. "
            + (f"First choice **{primary['name']}**: {primary['reason']}" if primary else "Not enough information; describe the data structure and treatment.")
        )

    return {
        "features": f,
        "recommendations": ranked,
        "primary": primary,
        "checklist": checklist,
        "narrative": narrative,
    }
