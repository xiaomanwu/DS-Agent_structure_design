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
        "name_zh": "标准 A/B 实验",
        "name_en": "Standard A/B test",
        "tool": "experiment.design_experiment",
        "assumption_zh": "单元相互独立、流量可随机分流",
        "assumption_en": "Independent units, randomisable traffic",
        "pitfall_zh": "偷看结果；SRM；网络效应污染分流",
        "pitfall_en": "Peeking; SRM; spillover across units",
    },
    "cuped": {
        "name_zh": "CUPED 方差缩减",
        "name_en": "CUPED variance reduction",
        "tool": "stats.cuped_adjust",
        "assumption_zh": "有实验前协变量且与指标相关",
        "assumption_en": "A pre-period covariate correlated with the metric",
        "pitfall_zh": "协变量必须是干预前测量的，否则引入偏差",
        "pitfall_en": "Covariate must be pre-treatment, else it biases the estimate",
    },
    "switchback": {
        "name_zh": "时间片轮转实验",
        "name_en": "Switchback experiment",
        "tool": "experiment_designs.analyze_switchback",
        "assumption_zh": "处理效应在清洗期内消退；时间片间无泄漏",
        "assumption_en": "Effect washes out within the block; no leakage across blocks",
        "pitfall_zh": "以时间片为推断单元（不是用户）； carryover 会高估",
        "pitfall_en": "Inference unit is the time block, not the user; carryover inflates the effect",
    },
    "nested": {
        "name_zh": "父子（嵌套）实验",
        "name_en": "Parent-child (nested) experiment",
        "tool": "experiment_designs.analyze_nested_experiment",
        "assumption_zh": "两层实验正交（流量分层随机化）",
        "assumption_en": "The two layers are orthogonal (stratified randomisation)",
        "pitfall_zh": "不查正交性会把父层效应算到子层头上",
        "pitfall_en": "Without the orthogonality check the parent effect leaks into the child",
    },
    "two_sided": {
        "name_zh": "双边市场实验",
        "name_en": "Two-sided market experiment",
        "tool": "experiment_designs.analyze_two_sided",
        "assumption_zh": "按市场整体分组，买卖两侧分别干预",
        "assumption_en": "Cluster by market; intervene on each side separately",
        "pitfall_zh": "用户级分流会被跨侧溢出污染（买家补贴影响卖家行为）",
        "pitfall_en": "User-level assignment is contaminated by cross-side spillover",
    },
    "cluster": {
        "name_zh": "集群（地区级）实验",
        "name_en": "Cluster experiment",
        "tool": "experiment.design_experiment",
        "assumption_zh": "集群数足够多（≥20），单元内相关用集群 bootstrap",
        "assumption_en": "Enough clusters (>=20); cluster-level bootstrap for SEs",
        "pitfall_zh": "集群少时方差估计不稳，功效远低于用户级分流",
        "pitfall_en": "Few clusters make SEs unstable; power is far below user-level splits",
    },
    "did": {
        "name_zh": "双重差分（DiD）",
        "name_en": "Difference-in-differences",
        "tool": "causal.difference_in_differences",
        "assumption_zh": "平行趋势：无政策时两组走势平行",
        "assumption_en": "Parallel trends between treated and control",
        "pitfall_zh": "前趋势不平行时用事件研究诊断；政策时点错开要用交叠 DiD",
        "pitfall_en": "Check pre-trends with an event study; staggered timing needs TWFE care",
    },
    "event_study": {
        "name_zh": "事件研究",
        "name_en": "Event study",
        "tool": "causal.event_study",
        "assumption_zh": "有干净的预处理期",
        "assumption_en": "A clean pre-period exists",
        "pitfall_zh": "预期效应（提前囤货）会被误判为前趋势违反",
        "pitfall_en": "Anticipation effects look like pre-trend violations",
    },
    "psm": {
        "name_zh": "倾向得分匹配（PSM）",
        "name_en": "Propensity score matching",
        "tool": "causal.propensity_score_matching",
        "assumption_zh": "条件可忽略性 + 重叠性；识别 ATT",
        "assumption_en": "Conditional ignorability plus overlap; identifies the ATT",
        "pitfall_zh": "只平衡观测到的协变量；隐性选择（老板拍板）平衡不掉",
        "pitfall_en": "Balances only observed covariates; hidden selection survives",
    },
    "ipw": {
        "name_zh": "逆概率加权（IPW/AIPW）",
        "name_en": "IPW / AIPW",
        "tool": "causal.aipw_effect",
        "assumption_zh": "倾向模型正确（AIPW 双稳健）",
        "assumption_en": "Propensity model right (AIPW is doubly robust)",
        "pitfall_zh": "极端倾向得分会让权重爆炸，需截断",
        "pitfall_en": "Extreme propensities explode the weights; trim them",
    },
    "synthetic_control": {
        "name_zh": "合成控制",
        "name_en": "Synthetic control",
        "tool": "causal.synthetic_control",
        "assumption_zh": "单个处理单元 + 一组未处理捐赠者，前置期拟合好",
        "assumption_en": "One treated unit plus untouched donors; good pre-period fit",
        "pitfall_zh": "供体池太小或前置拟合差时不可信；用安慰剂检验推断",
        "pitfall_en": "Unreliable with few donors or poor pre-fit; use placebo inference",
    },
    "rdd": {
        "name_zh": "断点回归（RDD）",
        "name_en": "Regression discontinuity",
        "tool": "rdd.regression_discontinuity",
        "assumption_zh": "断点附近个体不可精确操纵分配变量",
        "assumption_en": "Units cannot precisely manipulate the running variable",
        "pitfall_zh": "密度检验拒绝连续性说明有操纵；带宽选择要做敏感性",
        "pitfall_en": "Density test rejection signals manipulation; run bandwidth sensitivity",
    },
    "its": {
        "name_zh": "中断时间序列（ARIMA ITS）",
        "name_en": "Interrupted time series (ARIMA ITS)",
        "tool": "ts_causal.arima_intervention",
        "assumption_zh": "无对照时的最后手段：趋势/季节必须建模，否则归给干预",
        "assumption_en": "Last resort without controls: trend/seasonality must be modelled",
        "pitfall_zh": "同期发生了其他事件时无法区分；用安慰剂与对照序列缓解",
        "pitfall_en": "Concurrent events are indistinguishable; mitigate with placebos and controls",
    },
    "causal_impact": {
        "name_zh": "结构时间序列反事实（CausalImpact）",
        "name_en": "Structural counterfactual (CausalImpact)",
        "tool": "ts_causal.causal_impact",
        "assumption_zh": "有相关且未被干预的对照序列",
        "assumption_en": "Correlated control series untouched by the intervention",
        "pitfall_zh": "对照序列本身受污染则反事实有偏",
        "pitfall_en": "Contaminated controls bias the counterfactual",
    },
    "uplift": {
        "name_zh": "Uplift（因果森林/元学习器）",
        "name_en": "Uplift (causal forest / meta-learners)",
        "tool": "uplift.causal_forest",
        "assumption_zh": "随机化数据或满足可忽略性；用于异质性与人群定向",
        "assumption_en": "Randomised data (or ignorability); for heterogeneity and targeting",
        "pitfall_zh": "在观察数据上跑会学到选择偏差；用 Qini/AIPW 校验",
        "pitfall_en": "On observational data it learns selection bias; validate with Qini/AIPW",
    },
    "sensitivity": {
        "name_zh": "多口径敏感性分析",
        "name_en": "Specification-curve sensitivity",
        "tool": "experiment_designs.sensitivity_analysis",
        "assumption_zh": "结论应在合理口径集合内方向一致",
        "assumption_en": "The sign should hold across a reasonable spec set",
        "pitfall_zh": "只挑显著口径就是 p-hacking",
        "pitfall_en": "Cherry-picking significant specs is p-hacking",
    },
    "anomaly": {
        "name_zh": "时序异常检测（含 Holt-Winters）",
        "name_en": "Time-series anomaly detection (incl. Holt-Winters)",
        "tool": "anomaly.AnomalyDetector",
        "assumption_zh": "日粒度序列有可学习的季节结构",
        "assumption_en": "Daily series with learnable seasonal structure",
        "pitfall_zh": "纯随机游走没有基线可言；先看平稳性",
        "pitfall_en": "A random walk has no baseline; check stability first",
    },
    "forecast": {
        "name_zh": "时序预测（HW/SARIMA/STL）",
        "name_en": "Forecasting (HW / SARIMA / STL)",
        "tool": "forecast.select_method",
        "assumption_zh": "历史模式延续；区间随期距变宽",
        "assumption_en": "Patterns persist; intervals widen with horizon",
        "pitfall_zh": "结构性断点后历史不可用",
        "pitfall_en": "History is useless after a structural break",
    },
    "descriptive": {
        "name_zh": "描述统计 + 归因拆解",
        "name_en": "Descriptives + contribution decomposition",
        "tool": "contribution.attribute_change",
        "assumption_zh": "只回答\"哪里变了\"，不回答\"为什么变了\"",
        "assumption_en": "Answers where it moved, not why",
        "pitfall_zh": "把相关拆解当因果解释",
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
            add("switchback", 95, "处理按时间片轮转：推断单元是时间片，直接轮转分析。", "Treatment rotates over time blocks: switchback analysis with block-level inference.")
        if f["nested"]:
            add("nested", 92, "两层实验同时跑：先验证正交性再估子层效应。", "Two layers run at once: verify orthogonality, then estimate the child effect.")
        if f["two_sided"]:
            add("two_sided", 92, "买卖两侧都干预：2×2 网格分别估直接效应与跨侧溢出。", "Both sides treated: the 2×2 grid separates direct effects from cross-side spillover.")
        if f["cluster"]:
            add("cluster", 80, "以城市/地区分流：按集群设计与推断。", "Assignment by city/region: design and infer at cluster level.")
        if not ranked or not (f["switchback"] or f["nested"] or f["two_sided"] or f["cluster"]):
            add("ab_test", 90, "用户可独立分流：标准 A/B 是默认选择。", "Units are independently assignable: standard A/B is the default.")
        add("cuped", 55, "有实验前数据时用 CUPED 降方差、缩短实验。", "With pre-period data, CUPED cuts variance and shortens the test.")
        add("sensitivity", 40, "上线前先做 A/A 与多口径敏感性，防假阳性。", "Run A/A and spec-curve checks before shipping to bound false positives.")
        add("uplift", 35, "实验数据顺便做异质性分析，找出高响应人群。", "Mine the same experiment for heterogeneity and targeting.")
    # ---- Branch 2: no experiment — observational identification ------------
    else:
        if f["threshold"]:
            add("rdd", 95, "分配由连续分数的阈值决定：断点附近即局部随机实验。", "Assignment follows a threshold on a score: RDD around the cutoff.",)
        elif f["panel"] and (f["effect"] or f["ongoing"] or f["staggered"]):
            add("did", 92, "有处理/对照组和政策前后面板：DiD + 事件研究检验平行趋势。", "Treated/control panel around a policy: DiD plus an event study on pre-trends.")
            add("event_study", 70, "事件研究同时给出动态效应和平行趋势诊断。", "The event study also yields dynamic effects and the pre-trend test.")
        elif f["single_event"] or (f["time_series"] and f["effect"]):
            add("its", 85, "单序列 + 明确干预日：中断时间序列建模趋势/季节后估水平位移。", "Single series with a known intervention date: ITS models trend/seasonality, then the level shift.")
            add("causal_impact", 80, "有相关对照序列时用结构反事实，比纯 ITS 更稳。", "With correlated controls, the structural counterfactual is steadier than bare ITS.")
        elif f["cross_section"] or f["observational"]:
            add("psm", 90, "有选择偏差的观察数据：先匹配，再在匹配样本上重估。", "Selection-biased observational data: match first, re-estimate on the matched sample.")
            add("ipw", 75, "AIPW 双稳健，作为 PSM 的交叉验证。", "AIPW is doubly robust — use it to cross-check PSM.")
        else:
            add("psm", 60, "没有明确识别策略时，从匹配开始并诚实标注假设。", "Without a sharper strategy, start from matching and state the assumptions.")
        if f["heterogeneity"]:
            add("uplift", 88, "要找\"对谁有效\"：因果森林给个体效应与定向收益。", "\"Who does it work for\": causal forest for CATEs and targeting gain.")
        if f["where"]:
            add("descriptive", 55, "先拆贡献定位变化来源（描述性，不是因果）。", "Decompose the move first (descriptive, not causal).")
        if f["anomaly"]:
            add("anomaly", 88, "先判定异动是否显著，再谈归因。", "Establish that the move is anomalous before attributing it.")
        if f["forecast"]:
            add("forecast", 85, "预测任务：在同一回测框架下选 HW/SARIMA/STL。", "Forecasting: pick HW/SARIMA/STL on one backtest framework.")
        if ranked and ranked[0]["score"] >= 85:
            add("sensitivity", 45, "观察性结论务必做多口径敏感性。", "Observational claims always need a spec-curve pass.")

    ranked.sort(key=lambda r: -r["score"])
    primary = ranked[0] if ranked else None

    checklist = [
        {
            "item": self_item_zh,
            "item_en": self_item_en,
        }
        for self_item_zh, self_item_en in (
            ("识别假设是否成立？能否用设计而非统计来论证？", "Do the identifying assumptions hold — argued from design, not statistics?"),
            ("有没有安慰剂/反事实检验？（假干预日期、A/A、空间安慰剂）", "Placebo or counterfactual checks: fake intervention date, A/A, in-space placebo?"),
            ("换口径/换带宽/换协变量，方向是否稳定？", "Is the sign stable across specs, bandwidths, covariate sets?"),
            ("效应是否被少数异常个体驱动？", "Is the effect driven by a handful of units?"),
            ("结论的适用范围：局部效应（LATE/ATT）还是平均效应（ATE）？", "Scope of the claim: LATE/ATT or ATE?"),
        )
    ]

    if zh:
        narrative = (
            f"场景解析：{'、'.join(k for k, v in f.items() if v) or '无明显特征'}。"
            + (f"首选 **{primary['name']}**：{primary['reason']}" if primary else "信息不足，请补充数据结构与干预方式。")
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
