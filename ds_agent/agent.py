"""The agent: natural language in, analysed answer out.

Flow
----
1. ``route``       - map the question to one capability (keyword scoring, or the
                     LLM planner when an API key is configured).
2. ``handler``     - execute tools, which return structured payloads.
3. ``compose``     - turn payloads into a narrative plus rich blocks (KPI, chart,
                     table) for the CLI and the web UI.
4. ``record``      - keep findings so a follow-up "write me a report" works.

Everything is deterministic by default, which makes the agent testable and
cheap: the LLM is an optional narrator, not a dependency.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import charts
from .config import settings
from .confidence import (
    anomaly_band,
    anomaly_confidence,
    confidence_table,
    did_confidence,
    did_placebo,
    experiment_confidence,
    matching_confidence,
    permutation_test,
    ts_causal_confidence,
    verdict_line,
)
from .llm import LLMClient
from .metric_dict import MetricDefinitionError, MetricDictionary
from .readouts import ReadoutHandlers
from .registry import (
    Extension,
    extension_catalog,
    extension_handlers,
    extension_scores,
    register_extension,
)
from .reply import AgentReply
from .reply import fmt_num as _fmt_num
from .reply import fmt_pct as _fmt_pct
from .semantic import Metric, Table, semantic
from .stats import panels as test_panels
from .stats import tests as stat_tests
from .tools.anomaly import AnomalyConfig, AnomalyDetector, compare_windows as _compare, scan_segment_anomalies
from .tools.causal import (
    compare_estimators,
    difference_in_differences,
    event_study,
    synthetic_control,
)
from .tools.causal_oss import (
    HAS_DOWHY,
    available_engines,
    dowhy_estimate,
    dowhy_refutations,
    econml_causal_forest,
    oss_confidence,
)
from .tools.contribution import attribute_change, contribution_table
from .tools.datasource import DataSourceError, DataSourceRegistry
from .tools.dl_anomaly import (
    autoencoder_scores,
    dl_anomaly_confidence,
    dl_anomaly_table,
    isolation_forest_scores,
)
from .tools.experiment import EXPERIMENT_TYPES, ExperimentDesign, analyze_experiment, design_experiment, recommend_experiment_type
from .tools.report import Report, Section, to_markdown_table
from .tools.root_cause import root_cause_tree, tree_table
from .tools.sql import SQLTool, TimeWindow, Warehouse, parse_time_window, sanitize_sql


def full_catalog() -> Dict[str, Dict[str, str]]:
    """The complete capability catalog: presets plus runtime-registered extensions."""
    return {**CAPABILITIES, **extension_catalog()}


CAPABILITIES: Dict[str, Dict[str, str]] = {
    "sql": {
        "zh": "自然语言查询：指标准确性由语义层保证，SQL 只允许只读，结果自动截断",
        "en": "Natural-language querying with a governed semantic layer and read-only SQL",
        "example_zh": "近30天各渠道 GMV 排名",
        "example_en": "GMV by channel in the last 30 days",
    },
    "diagnose": {
        "zh": "指标异动诊断：异常识别 + 贡献度拆解 + 多维度归因；基线可切 Holt-Winters / SARIMA / STL+ETS / 自动回测择优",
        "en": "Metric movement diagnosis: anomaly detection, contribution, attribution; baseline switchable to Holt-Winters / SARIMA / STL+ETS / auto-backtested",
        "example_zh": "最近7天 GMV 为什么跌了？（或：用 holt-winters 看下异常）",
        "example_en": "Why did GMV drop in the last 7 days? (or: check anomalies with holt-winters)",
    },
    "experiment_design": {
        "zh": "AB 实验设计：样本量/时效/分流/检验方案，覆盖 15 种互联网常见实验类型",
        "en": "Experiment design: sizing, duration, randomisation unit, analysis plan across 15 archetypes",
        "example_zh": "我要做一个支付转化率实验，基线3%，希望检测2%的相对提升，每天5万流量，怎么设计？",
        "example_en": "Design a checkout conversion test: baseline 3%, detect a 2% relative lift, 50k daily traffic",
    },
    "experiment_analysis": {
        "zh": "实验分析：SRM、显著性检验、CUPED、多重比较、分群异质性、上线建议",
        "en": "Experiment readout: SRM, significance, CUPED, multiplicity, segments, ship decision",
        "example_zh": "看下这个 AB 实验的结果",
        "example_en": "Read out the AB test results",
    },
    "switchback": {
        "zh": "时间片轮转实验：以时间片为推断单元，HAC 标准误、清洗期与滞后检验",
        "en": "Switchback: time block as the inference unit, HAC standard errors, washout and carryover tests",
        "example_zh": "看下时间片轮转实验的结果",
        "example_en": "Read out the switchback test",
    },
    "reversal": {
        "zh": "反转实验：期中两组对调，用 τ₁+τ₂ 检验埋点漂移、用 |τ₂|/|τ₁| 衡量持续性",
        "en": "Reversal experiment: swap the arms mid-flight, test drift with tau1+tau2 and persistence with |tau2|/|tau1|",
        "example_zh": "做一个反转实验验证这个效应",
        "example_en": "Run a reversal experiment to validate the effect",
    },
    "nested": {
        "zh": "父子实验：正交性检验、父层固定效应校正、父子交互项",
        "en": "Parent-child experiments: orthogonality check, parent fixed-effect adjustment, interaction test",
        "example_zh": "父子实验怎么验证互不干扰",
        "example_en": "Check that my parent and child experiments do not interfere",
    },
    "two_sided": {
        "zh": "双边实验：2x2 买卖两侧，拆直接效应、跨侧溢出与交互项",
        "en": "Two-sided experiment: 2x2 buyer x seller cells separating direct effects, cross-side spillover and interaction",
        "example_zh": "双边实验，买家侧和卖家侧同时干预",
        "example_en": "Two-sided test treating both buyers and sellers",
    },
    "sensitivity": {
        "zh": "敏感性分析：缩尾/截尾/对数/CUPED/分群固定效应/自助法，加 A/A 安慰剂与刀切法",
        "en": "Sensitivity analysis: winsorise, trim, log, CUPED, segment FE, bootstrap, plus A/A placebo and jackknife",
        "example_zh": "这个实验结论稳健吗，做个敏感性分析",
        "example_en": "Is the result robust? Run a sensitivity analysis",
    },
    "uplift": {
        "zh": "Uplift 建模：因果树、因果森林、T/S/X 学习器，Qini 与人群定向收益",
        "en": "Uplift modelling: causal tree, causal forest, T/S/X learners, Qini curve and targeting gain",
        "example_zh": "用因果森林看哪类用户对补贴最敏感",
        "example_en": "Which users respond most to the promo? Fit a causal forest",
    },
    "rdd": {
        "zh": "断点回归：局部线性拟合、带宽选择、密度操纵检验、模糊 RDD",
        "en": "Regression discontinuity: local linear fits, bandwidth selection, manipulation test, fuzzy RDD",
        "example_zh": "用断点回归评估会员门槛的效果",
        "example_en": "Evaluate the membership threshold with a regression discontinuity",
    },
    "ts_causal": {
        "zh": "时序因果：ARIMA 中断时间序列 + 结构时序反事实，含安慰剂与口径敏感性",
        "en": "Time-series causal: ARIMA interrupted time series plus a structural counterfactual, with placebo and specification checks",
        "example_zh": "这次改版对 GMV 的影响，用时序因果评估",
        "example_en": "Estimate the launch effect on GMV with a time-series causal model",
    },
    "method_select": {
        "zh": "方法选型：根据场景（有无实验、干预方式、数据结构、目标）推荐合适的统计/因果方法并说明理由与坑",
        "en": "Method selection: recommends the right statistical or causal method from your scenario, with reasons and pitfalls",
        "example_zh": "这个场景该用什么方法分析？",
        "example_en": "Which method fits this scenario?",
    },
    "tests": {
        "zh": "统计检验：内置 30+ 项检验（正态性/方差/位置/比例/时序/回归/重抽样/多重检验），给出原假设、公式、前提、判定与文献；每个分析结论都会附带检验表",
        "en": "Statistical tests: 30+ built-in tests (normality/variance/location/proportion/time series/regression/resampling/multiplicity) with null hypothesis, formula, assumptions, decision rule and citation; every readout ships its own test table",
        "example_zh": "异常检测用了哪些检验？/ 什么是 Welch t 检验",
        "example_en": "Which tests does anomaly detection run? / What is a Welch t-test?",
    },
    "sampling": {
        "zh": "抽样调查：样本量计算（比例/均值，含功效公式与文献）、简单随机/系统/分层抽样（可复现）、抽样分布与标准误、样本代表性检验（卡方拟合优度 / Welch t / KS）",
        "en": "Sampling surveys: sample-size determination (proportion/mean, power formula with citation), simple-random/systematic/stratified draws (reproducible), sampling distribution and standard errors, representativeness tests (chi-square GOF / Welch t / KS)",
        "example_zh": "做用户满意度调查，基线满意率 70%，要检测 3 个百分点的变化，需要多少样本？",
        "example_en": "How large a sample do I need for a satisfaction survey, baseline 70%, detect a 3pp change?",
    },
    "knowledge": {
        "zh": "业务知识问答：指标口径、维度语义、数据字典、已知埋点与业务事件（RAG 检索内部知识库）",
        "en": "Business knowledge Q&A: metric calibers, dimension semantics, data dictionary and known incidents, via RAG over the internal knowledge base",
        "example_zh": "GMV 的口径是什么？和 revenue 有什么区别？",
        "example_en": "What is the GMV caliber and how does it differ from revenue?",
    },
    "metric": {
        "zh": "指标字典管理：注册自定义指标（公式校验后进入语义层，全链路同口径）、列出/查看口径、下线指标",
        "en": "Metric dictionary: register custom metrics (validated against the schema, activated into the semantic layer), list calibers, deprecate metrics",
        "example_zh": "注册指标：利润率 = gmv - refund_amount，分母 gmv",
        "example_en": "Register metric profit_rate = (gmv - refund_amount) / gmv",
    },
    "datasource": {
        "zh": "数据源关联：接入企业数据库（DuckDB/SQLite/Postgres/MySQL）与离线表格（CSV/Excel/Parquet/JSON），注册后可直接用自然语言查询",
        "en": "Data sources: attach enterprise databases (DuckDB/SQLite/Postgres/MySQL) and offline spreadsheets (CSV/Excel/Parquet/JSON); registered sources are directly queryable in natural language",
        "example_zh": "关联数据源 /Users/me/sales.csv 叫 sales",
        "example_en": "Attach data source /Users/me/sales.csv as sales",
    },
    "causal": {
        "zh": "因果推断：DID/事件研究、PSM、IPW、AIPW、合成控制（含安慰剂检验）",
        "en": "Causal inference: DiD/event study, PSM, IPW, AIPW, synthetic control with placebo tests",
        "example_zh": "用双重差分评估补贴政策对城市 GMV 的影响",
        "example_en": "Evaluate the subsidy policy effect with difference-in-differences",
    },
    "causal_oss": {
        "zh": "开源引擎交叉验证：用 DoWhy/EconML 等现成因果推断开源包重估同一问题，跑安慰剂处理/随机共同原因反驳检验，并与自研实现对比一致性",
        "en": "Open-source cross-validation: re-estimate with DoWhy/EconML, run placebo/random-common-cause refutations, and compare against the in-house estimators",
        "example_zh": "用 DoWhy 开源包交叉验证促销的因果效应",
        "example_en": "Cross-validate the promo effect with the DoWhy package",
    },
    "dl_anomaly": {
        "zh": "深度学习异常检测：自编码器重建误差（窗口形态敏感）+ 孤立森林多变量评分，含重建质量/告警率/与季节基线交叉验证的置信卡",
        "en": "Deep-learning anomaly detection: autoencoder reconstruction error (window-shape sensitive) plus isolation-forest multivariate scoring, with a reconstruction/alert-rate/seasonal-cross-check confidence card",
        "example_zh": "用自编码器/深度学习方法看下 GMV 有没有异常",
        "example_en": "Check GMV anomalies with an autoencoder / deep-learning detector",
    },
    "modeling": {
        "zh": "建模能力（四段流水线，回归/因果共用）：① 数据预处理（缺失值/NaN 填补、稀疏列与常数列剔除）→ ② 特征工程（时间展开/独热/共线性处理：成对 |r| 与 VIF 双重剔除）→ ③ 分类与回归（线性/梯度提升对比、5折CV）→ ④ 模型验证之后的特征分析（装 shap 用真 SHAP，否则置换重要性+系数方向）+ 过拟合/CV稳定性置信卡",
        "en": "Modelling as a four-stage pipeline shared by regression and causal inference: 1. data preprocessing (missing/NaN imputation, sparse and constant columns dropped) -> 2. feature engineering (datetime expansion / one-hot / collinearity handling via pairwise |r| and VIF pruning) -> 3. classification & regression (linear vs gradient boosting, 5-fold CV) -> 4. post-validation feature analysis (true SHAP if installed, else permutation+coefficients) + overfit/CV-stability confidence card",
        "example_zh": "训练一个预测用户是否转化的分类模型，看特征重要性",
        "example_en": "Train a classifier to predict conversion and show feature importance",
    },
    "dashboard": {
        "zh": "看板能力：一键生成独立 HTML 监控看板（KPI 环比卡、趋势与均线、渠道/设备拆解、退款率、近期异动表），数字与 Agent 查询同源同口径",
        "en": "Dashboard: one-click standalone HTML monitoring view (KPI cards with WoW, trend + moving average, channel/device breakdowns, refund rate, recent anomalies) — same source and calibers as agent answers",
        "example_zh": "给我生成一个 GMV 监控看板",
        "example_en": "Generate a GMV monitoring dashboard",
    },
    "report": {
        "zh": "报告撰写：把前面的发现组织成结论先行、可复现的中英双语报告（Markdown + HTML）",
        "en": "Report writing: turns findings into an answer-first, reproducible Markdown + HTML report",
        "example_zh": "把上面的分析写成周报",
        "example_en": "Write this up as a weekly report",
    },
}


def _episode_kind(episode: Dict[str, Any], lang: str = "zh") -> str:
    """Human label for how an episode was detected."""
    kind = episode.get("kind", "point")
    if lang == "zh":
        return "持续位移" if kind == "level_shift" else "单日尖峰"
    return "level shift" if kind == "level_shift" else "single-day spike"


_SEVERITY_ZH = {"critical": "严重", "high": "高", "medium": "中", "low": "低"}

_RELATIVE_TIME = re.compile(
    r"近\s*\d|最近|前\s*\d|过去|上周|本周|上月|本月|昨天|昨日|今天|今日|"
    r"\d+\s*(天|日|周|月|个?星期)|last\s|past\s|yesterday|today|this week|last week",
    re.I,
)


def _severity(value: str, lang: str = "zh") -> str:
    """Localise the severity band produced by the detector."""
    if lang != "zh":
        return value
    return _SEVERITY_ZH.get(value, value)


_WINDOW_ZH = {
    "yesterday": "昨天",
    "day before": "前天",
    "last week": "上周",
    "week before last": "上上周",
    "this week": "本周",
    "last month": "上月",
    "month before last": "上上月",
    "custom": "自定义窗口",
}


def localize_window(window: TimeWindow, lang: str) -> TimeWindow:
    """Translate the human label of a time window (dates are already ISO)."""
    if lang != "zh":
        return window
    label = window.label
    if label in _WINDOW_ZH:
        window.label = _WINDOW_ZH[label]
        return window
    match = re.match(r"(last|previous) (\d+) (days|weeks)", label)
    if match:
        prefix = "近" if match.group(1) == "last" else "上一个"
        unit = "天" if match.group(3) == "days" else "周"
        window.label = f"{prefix}{match.group(2)}{unit}"
    return window


class DSQueryAgent(ReadoutHandlers):
    """Stateful agent: remembers the last metric/window for follow-up questions."""

    def __init__(
        self,
        warehouse: Optional[Path] = None,
        artifact_dir: Optional[Path] = None,
        llm: Optional[LLMClient] = None,
        metric_dict: Optional[MetricDictionary] = None,
        datasources: Optional[DataSourceRegistry] = None,
    ) -> None:
        self.sql = SQLTool(Warehouse(warehouse)) if warehouse else SQLTool()
        self.artifact_dir = Path(artifact_dir or settings.artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.llm = llm if llm is not None else LLMClient()
        self.metric_dict = metric_dict or MetricDictionary(semantic=semantic, warehouse=self.sql.warehouse)
        self.datasources = datasources or DataSourceRegistry(self.sql.warehouse)
        self.session: Dict[str, Any] = {}
        self.findings: List[Section] = []
        self.summary_bullets: List[str] = []
        self._knowledge = None

    def reconfigure_llm(self) -> None:
        """Pick up LLM settings changed at runtime (web settings panel)."""
        self.llm = LLMClient()

    def register_extension(
        self,
        name: str,
        handler,
        keywords: List[str],
        zh: str = "",
        en: str = "",
        example_zh: str = "",
        example_en: str = "",
        weight: int = 9,
    ) -> Extension:
        """Plug a new capability into this agent at runtime (see ``registry.py``)."""
        return register_extension(
            Extension(
                name=name,
                handler=handler,
                keywords=keywords,
                weight=weight,
                zh=zh,
                en=en,
                example_zh=example_zh,
                example_en=example_en,
            )
        )

    @property
    def knowledge(self):
        """Lazily built RAG index over docs/business_knowledge.md (None if missing)."""
        if self._knowledge is None:
            try:
                from .knowledge import KnowledgeBase

                kb = KnowledgeBase()
                self._knowledge = kb if len(kb) else False
            except Exception:  # noqa: BLE001 - the KB is an enhancement, not a dependency
                self._knowledge = False
        return self._knowledge or None

    def kb_context(self, query: str, k: int = 2) -> str:
        """Retrieved business knowledge for a question ('' when unavailable)."""
        kb = self.knowledge
        return kb.context_for(query, k=k) if kb else ""

    # ------------------------------------------------------------------ utils
    @property
    def lang(self) -> str:
        return self.session.get("lang", "zh")

    def _t(self, zh: str, en: str) -> str:
        return zh if self.lang == "zh" else en

    def close(self) -> None:
        self.sql.warehouse.close()

    @staticmethod
    def _detect_lang(question: str) -> str:
        return "zh" if re.search(r"[\u4e00-\u9fff]", question) else "en"

    def _series(
        self,
        metric: Metric,
        table: Table,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Daily series with numerator/denominator, used by the anomaly detector."""
        sql = (
            f"SELECT {table.time_column} AS d, SUM({metric.numerator}) AS numerator"
            + (f", SUM({metric.denominator}) AS denominator" if metric.is_ratio else "")
            + f" FROM {table.name} WHERE {table.time_column} BETWEEN DATE '{start}' AND DATE '{end}'"
            + f" GROUP BY 1 ORDER BY 1"
        )
        df = self.sql.run(sql).dataframe.copy()
        df["date"] = pd.to_datetime(df["d"])
        if metric.is_ratio:
            df["value"] = df["numerator"] / df["denominator"].replace(0, np.nan)
        else:
            df["value"] = df["numerator"].astype(float)
        return df

    def _window_frame(self, metric: Metric, dims: Sequence[str], window: TimeWindow, table: Table) -> pd.DataFrame:
        """Aggregate numerator/denominator over a window, grouped by dimensions."""
        if window.start is None or window.end is None:
            raise ValueError("window needs concrete start/end dates")
        select = [f"{table.dimension_expr(d)} AS {d}" for d in dims] + [
            f"SUM({metric.numerator}) AS {metric.numerator}"
        ]
        if metric.is_ratio:
            select.append(f"SUM({metric.denominator}) AS {metric.denominator}")
        sql = (
            f"SELECT {', '.join(select)} FROM {table.name} "
            f"WHERE {table.time_column} BETWEEN DATE '{window.start}' AND DATE '{window.end}' "
            f"GROUP BY {', '.join(str(i + 1) for i in range(len(dims)))}"
        )
        return self.sql.run(sql, max_rows=100000).dataframe

    def _window_total(self, metric: Metric, window: TimeWindow, table: Table) -> float:
        select = f"SUM({metric.numerator})"
        if metric.is_ratio:
            select = f"SUM({metric.numerator}) / NULLIF(SUM({metric.denominator}), 0)"
        sql = (
            f"SELECT {select} AS v FROM {table.name} "
            f"WHERE {table.time_column} BETWEEN DATE '{window.start}' AND DATE '{window.end}'"
        )
        value = self.sql.run(sql).dataframe["v"].iloc[0]
        return float(value) if value is not None else float("nan")

    def _segment_scan(
        self,
        metric: Metric,
        table: Table,
        dim: str,
        start: date,
        end: date,
        window: Optional[Tuple[date, date]] = None,
        comparison: Optional[Tuple[date, date]] = None,
        top_k: int = 6,
    ) -> Dict[str, Any]:
        """Detect anomalies segment by segment along one dimension.

        The topline can look flat while a segment is down 30%; this is the scan
        that surfaces it. Decomposition uses the full history, reporting is
        restricted to the window the user asked about.
        """
        select = [
            f"{table.dimension_expr(dim)} AS segment",
            f"{table.time_column} AS d",
            f"SUM({metric.numerator}) AS numerator",
        ]
        if metric.is_ratio:
            select.append(f"SUM({metric.denominator}) AS denominator")
        sql = (
            f"SELECT {', '.join(select)} FROM {table.name} "
            f"WHERE {table.time_column} BETWEEN DATE '{start}' AND DATE '{end}' "
            f"GROUP BY 1, 2 ORDER BY 2"
        )
        df = self.sql.run(sql, max_rows=100000).dataframe
        df["date"] = pd.to_datetime(df["d"])
        df["value"] = (
            df["numerator"] / df["denominator"].replace(0, np.nan) if metric.is_ratio else df["numerator"].astype(float)
        )
        result = scan_segment_anomalies(
            df,
            denominator_col="denominator" if metric.is_ratio else None,
            window=window,
            comparison=comparison,
            top_k=top_k,
        )
        result["dimension"] = dim
        return result

    def _metric_meta(self, metric: Metric) -> Dict[str, Any]:
        return {
            "name": metric.name,
            "kind": metric.kind,
            "unit": metric.unit,
            "higher_is_better": metric.higher_is_better,
        }

    @staticmethod
    def _plain(text: str) -> str:
        """Strip inline markdown so it can go into escaped HTML prose."""
        return re.sub(r"\*\*|`", "", text)

    def _strip_steps(self, lines: Sequence[str]) -> List[str]:
        """Extract the numbered next-step lines without their numeric prefix."""
        return [re.sub(r"^\d+\.\s*", "", line) for line in lines if re.match(r"^\d+\.", line)]

    def _record(self, bullet: str, section: Optional[Section] = None) -> None:
        """Append a finding, replacing a same-titled one instead of duplicating it.

        Asking the same question twice (common in a REPL) used to produce a
        report with the same section repeated, which reads like an error.
        """
        if section is not None:
            for idx, existing in enumerate(self.findings):
                if existing.title == section.title:
                    self.findings[idx] = section
                    break
            else:
                self.findings.append(section)
        if bullet not in self.summary_bullets:
            self.summary_bullets.append(bullet)

    # ------------------------------------------------------------------ route
    def _rule_scores(self, q: str) -> Dict[str, int]:
        """Deterministic keyword scores per capability (shared by route/rank)."""
        scores = {k: 0 for k in CAPABILITIES}
        q = q.lower()
        has_experiment = bool(re.search(r"实验|\bab\b|a/b|ab test|experiment|分流", q))
        has_design = bool(re.search(r"设计|样本量|需要多少|多大样本|多久|时长|功效|power|mde|最小|方案|够不够|sample size|duration", q))
        has_result = bool(re.search(r"结果|结论|分析|显著|看下|readout|result|lift|上线|该不该|效果", q))
        has_causal = bool(re.search(r"因果|双重差分|\bdid\b|倾向得分|倾向|propensity|\bpsm\b|\bipw\b|aipw|合成控制|政策|效应估计|causal|counterfactual", q))
        has_report = bool(re.search(r"报告|周报|汇报|总结成|形成文档|write.*report|writeup|report", q))
        has_why = bool(re.search(r"为什么|为何|原因|归因|贡献|拆解|驱动|谁导致|怎么回事|why|cause|driver|attribute|contribution", q))
        has_anomaly = bool(re.search(r"异常|异动|波动|监控|告警|突然|暴跌|大跌|暴涨|anomaly|spike|drop|shift|monitor", q))
        has_query = bool(re.search(r"查询|查一下|多少|排名|top|明细|列出|统计|看一下|看下|sql|趋势|分布|占比|对比|breakdown", q))

        if has_experiment and has_design:
            scores["experiment_design"] += 6
        if has_experiment and has_result:
            scores["experiment_analysis"] += 6
        if has_experiment and not has_design and not has_result:
            scores["experiment_design"] += 2
            scores["experiment_analysis"] += 2
        if has_causal:
            scores["causal"] += 6
        if has_report:
            scores["report"] += 5
        if has_why:
            scores["diagnose"] += 5
        if has_anomaly:
            scores["diagnose"] += 3
            scores["sql"] += 1
        if has_query:
            scores["sql"] += 3
        if re.search(r"有哪些.*实验|实验类型|experiment types", q):
            scores["experiment_design"] += 4

        # Business-knowledge questions: definitions, calibers, data dictionary.
        # These ask "what does X mean here", never "compute X".
        if re.search(r"口径|数据字典|业务知识|知识库|是什么意思|怎么定义|如何定义|定义是什么|怎么算的|如何定义|business knowledge|data dictionary|glossary", q):
            scores["knowledge"] += 10

        # Specific advanced designs come first. Every one of these phrasings also
        # contains "实验" or "因果", so without a heavier weight they would be
        # split with the generic buckets and land on the wrong handler.
        advanced = {
            "switchback": r"时间片|轮转|switchback",
            "reversal": r"反转实验|反转|对调|swap|reversal",
            "nested": r"父子|嵌套实验|正交分层|nested|orthogonal layer",
            "two_sided": r"双边实验|两侧|2x2|买卖双方|buyer.{0,8}seller|both sides",
            "sensitivity": r"敏感性|稳健性|sensitivity|robustness|规格曲线|specification curve",
            "uplift": r"uplift|因果森林|因果树|增益模型|qini|人群定向|定向人群",
            "rdd": r"断点回归|断点|regression discontinuity|\brdd\b",
            "ts_causal": r"时序因果|中断时间序列|interrupted time series|causal impact|干预日|反事实序列",
        }
        for intent, pattern in advanced.items():
            if re.search(pattern, q, re.I):
                scores[intent] += 9

        # Open-source causal engines and deep-learning detectors are explicit
        # engine requests; they must outrank the generic causal/diagnose buckets.
        if re.search(r"开源|dowhy|econml|causalml|现成.{0,4}(包|库)|open[- ]?source package|交叉验证.{0,6}(引擎|包)", q, re.I):
            scores["causal_oss"] += 9
        if re.search(r"自编码器|自动编码器|autoencoder|孤立森林|isolation forest|深度学习.{0,10}异常|神经网络.{0,10}异常|deep[- ]learning anomaly", q, re.I):
            scores["dl_anomaly"] += 9

        # Modeling and dashboard are explicit capability requests.
        if re.search(r"建模|训练.{0,6}(模型|分类|回归)|分类模型|回归模型|特征(工程|重要性)|shap|机器学习|train.{0,10}model|machine learning|feature importance|classification model|regression model", q, re.I):
            scores["modeling"] += 9
        if re.search(r"看板|大屏|监控页|仪表盘|dashboard|monitoring page", q, re.I):
            scores["dashboard"] += 9

        # Metric dictionary: registration/management verbs must outrank the
        # knowledge intent (which also matches "如何定义"). "口径是什么" stays
        # knowledge; explicit register/list/deprecate verbs go to the dictionary.
        if re.search(r"指标字典|metric dictionary", q, re.I):
            scores["metric"] += 12
        if re.search(r"(注册|新增|创建|登记|定义|下线|废弃|停用).{0,8}指标|指标.{0,4}(注册|登记)|register.{0,14}metric|define.{0,14}metric|deprecate.{0,6}metric", q, re.I):
            scores["metric"] += 12

        # Data sources: attach / import / list external databases or files.
        if re.search(
            r"(关联|接入|连接|挂载|attach|connect).{0,12}(数据源|数据库|csv|excel|parquet|json|表格|duckdb|sqlite|postgres|mysql)"
            r"|导入.{0,10}(csv|excel|离线|外部)"
            r"|数据源(列表|管理|有哪些|已关联)|(关联|接入|移除|删除|断开|注册).{0,6}数据源"
            r"|离线表格|import.{0,12}(csv|excel|spreadsheet|database)|data source",
            q,
            re.I,
        ):
            scores["datasource"] += 10

        # Statistical-test questions: "which tests do you run", "what is a Welch
        # t-test", "which tests does the experiment readout report". These are
        # meta-questions about the method, not requests to run it, so they must
        # outrank the analysis buckets their wording also touches.
        if re.search(
            r"检验方法|统计检验|假设检验|检验清单|有哪些检验|用了哪些检验|检验有哪些|"
            r"(什么|如何|怎么|解释|原理|前提|区别|比较).{0,24}检验|"
            r"检验.{0,8}(原理|前提|公式|参考|文献|怎么选|区别)|"
            r"statistical test|test catalogue|which tests|what tests|hypothesis test",
            q,
            re.I,
        ):
            scores["tests"] += 11

        # Sampling-survey questions: sample size for a survey/draw, sampling
        # methods, sample representativeness. Deliberately gated on the absence
        # of experiment wording — "实验…样本量" belongs to experiment design.
        if not has_experiment and re.search(
            r"抽样|样本量|样本大小|多少\s*样本|需要.{0,6}样本|要.{0,4}多少.{0,4}样本|代表性|代表性强|抽样分布|标准误"
            r"|sampling|sample size|sample survey|respondent|how many (samples|people)|representativ",
            q,
            re.I,
        ):
            scores["sampling"] += 11

        # Runtime-registered extensions are first-class routing targets.
        for ext_name, ext_score in extension_scores(q).items():
            scores[ext_name] = scores.get(ext_name, 0) + ext_score

        # Method selection: the user asks WHAT to use, not for a result. These
        # phrasings must outrank the concrete-method keywords below them.
        if re.search(r"该用什么方法|用什么方法|方法选型|怎么分析|如何分析|选哪种方法|分析方法|which method|how (should|to) analyz|method selection|适合什么方法", q, re.I):
            scores["method_select"] += 12

        # Asking for a specific anomaly baseline ("用 holt-winters 看异常") is a
        # diagnosis request even when the wording also contains a comparison word
        # that would otherwise hand the question to the SQL handler.
        if re.search(r"时序基线|预测基线|基线对比|哪个基线|holt|arima|sarima|stl_ets|forecast baseline", q, re.I):
            scores["diagnose"] += 6

        return scores

    def ranked_intents(self, question: str, exclude: Tuple[str, ...] = ()) -> List[str]:
        """Capability intents ranked by rule score (used for re-planning)."""
        scores = self._rule_scores(question)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [name for name, score in ranked if score > 0 and name not in exclude]

    def route(self, question: str) -> Tuple[str, Dict[str, Any], float]:
        """Keyword scoring router. Returns (intent, slots, confidence)."""
        q = question.lower()
        scores = self._rule_scores(q)

        best = max(scores.items(), key=lambda kv: kv[1])
        total = sum(scores.values()) or 1
        confidence = best[1] / total

        slots = self._rule_slots(question)
        if best[1] == 0:
            # A terse follow-up ("按品类再拆一下") carries no intent keywords of
            # its own: it only makes sense against the previous turn.
            if self._is_followup(question, slots):
                last_intent = self.session.get("last_intent")
                if last_intent in ("diagnose", "sql"):
                    slots["metric"] = slots["metric"] or self.session.get("last_metric")
                    self._inherit_window(question, slots, slots["anchor"])
                    return last_intent, slots, 0.4
            return ("sql" if slots.get("metric") else "help"), slots, 0.0
        return best[0], slots, confidence

    def _rule_slots(self, question: str) -> Dict[str, Any]:
        """Deterministic slot filling shared by the rule router and the LLM planner."""
        slots: Dict[str, Any] = {}
        table = semantic.default_table()
        metric = semantic.resolve_metric(question, table.name)
        if metric is None and re.search(r"实验|ab", question.lower()):
            metric = semantic.resolve_metric(question, "experiment_users")
        slots["metric"] = metric
        slots["dimensions"] = [d for d in table.dimensions if d in (question.lower()) or any(
            alias.lower() in question.lower() for alias in table.dimensions[d]
        )]
        anchor = self.sql.warehouse.max_date(table.name, table.time_column) or date.today()
        slots["anchor"] = anchor
        slots["window"], slots["comparison"] = parse_time_window(question, anchor)
        localize_window(slots["window"], self.session.get("lang", "zh"))
        localize_window(slots["comparison"], self.session.get("lang", "zh"))
        return slots

    @staticmethod
    def _merge_replies(primary: AgentReply, extra: AgentReply, step: str) -> AgentReply:
        """Concatenate a chained step's output onto the primary reply."""
        merged = AgentReply(
            text=primary.text + "\n\n" + extra.text,
            blocks=primary.blocks + extra.blocks,
            tool_calls=primary.tool_calls + [
                {"tool": f"chain:{step}", "args": {}, "result": {"merged": len(extra.blocks)}}
            ]
            + extra.tool_calls,
            artifacts=primary.artifacts + extra.artifacts,
            data=dict(primary.data),
        )
        merged.data.update({k: v for k, v in extra.data.items() if k not in merged.data})
        return merged

    _FOLLOWUP = re.compile(
        r"再拆|拆一下|拆开|再分|下钻|细分|换个维度|换一个维度|再按|按.{0,8}(拆|分|看)|继续|那.{0,2}呢|"
        r"drill|breakdown|split by|by (channel|device|region|category|user type)",
        re.I,
    )

    def _is_followup(self, question: str, slots: Dict[str, Any]) -> bool:
        """True when the question only makes sense relative to the last turn."""
        if not self.session.get("last_intent"):
            return False
        if self._FOLLOWUP.search(question):
            return True
        # A bare dimension name ("渠道呢") is also a follow-up.
        return bool(slots.get("dimensions")) and len(question) <= 12

    def _inherit_window(self, question: str, slots: Dict[str, Any], anchor: date) -> None:
        """Keep the previous metric scope unless the follow-up restates it."""
        last = self.session.get("last_window")
        restated = bool(_RELATIVE_TIME.search(question))
        if last and not restated:
            slots["window"], slots["comparison"] = last
            return
        slots["window"], slots["comparison"] = parse_time_window(question, anchor)
        localize_window(slots["window"], self.session.get("lang", "zh"))
        if slots["comparison"] is not None:
            localize_window(slots["comparison"], self.session.get("lang", "zh"))

    # ------------------------------------------------------------------- ask
    def ask(self, question: str) -> AgentReply:
        question = (question or "").strip()
        if not question:
            return self._help()
        self.session["lang"] = self._detect_lang(question)
        if re.search(r"^(help|帮助|\?|？|你是谁|what can you do)", question.strip(), re.I):
            return self._help()

        # Planning: when an LLM endpoint is configured, it maps the question to
        # an intent first; the keyword router stays as the deterministic
        # fallback (LLM unavailable / invalid answer / network failure).
        # Terse follow-ups bypass the LLM: they only resolve against the
        # previous turn's scope, which the planner cannot see.
        plan = None
        if self.llm.available and not self._FOLLOWUP.search(question):
            try:
                from .planner import LLMPlanner

                plan = LLMPlanner(self.llm).plan(question, full_catalog())
            except Exception:  # noqa: BLE001 - planning is an enhancement
                plan = None
        if plan is not None:
            intent = plan["intent"]
            slots: Dict[str, Any] = self._rule_slots(question)
            confidence = 0.9
            self.session["planner"] = plan.get("engine", "llm")
        else:
            intent, slots, confidence = self.route(question)
            self.session["planner"] = "rules"
        self.session.update({"last_question": question, "last_intent": intent})

        handlers = {
            "sql": self._handle_sql,
            "diagnose": self._handle_diagnose,
            "experiment_design": self._handle_experiment_design,
            "experiment_analysis": self._handle_experiment_analysis,
            "switchback": self._handle_switchback,
            "reversal": self._handle_reversal,
            "nested": self._handle_nested,
            "two_sided": self._handle_two_sided,
            "sensitivity": self._handle_sensitivity,
            "uplift": self._handle_uplift,
            "rdd": self._handle_rdd,
            "ts_causal": self._handle_ts_causal,
            "method_select": self._handle_method_select,
            "tests": self._handle_tests,
            "sampling": self._handle_sampling,
            "knowledge": self._handle_knowledge,
            "metric": self._handle_metric,
            "datasource": self._handle_datasource,
            "causal": self._handle_causal,
            "causal_oss": self._causal_oss,
            "dl_anomaly": self._handle_dl_anomaly,
            "modeling": self._handle_modeling,
            "dashboard": self._handle_dashboard,
            "report": self._handle_report,
            "help": self._handle_help,
        }
        handlers.update(extension_handlers())
        handler = handlers.get(intent, self._handle_sql)
        try:
            reply = handler(question, slots)
        except Exception as exc:  # noqa: BLE001 - trigger the re-plan loop
            # Re-planning: execution failed under the planned intent. The
            # deterministic rule ranking provides the next-best capabilities
            # (always available, no LLM needed); each candidate is tried once.
            attempts: List[Dict[str, Any]] = [{"intent": intent, "error": str(exc)}]
            reply = None
            tried = {intent, "help"}
            for alt in self.ranked_intents(question, exclude=tuple(tried))[:2]:
                alt_handler = handlers.get(alt)
                if alt_handler is None:
                    continue
                tried.add(alt)
                try:
                    reply = alt_handler(question, slots)
                except Exception as exc2:  # noqa: BLE001 - try the next candidate
                    attempts.append({"intent": alt, "error": str(exc2)})
                    continue
                reply.data["replan"] = {"from": intent, "to": alt, "attempts": attempts}
                intent = alt
                break
            if reply is None:
                tried_note = "、".join(a["intent"] for a in attempts)
                reply = AgentReply(
                    text=self._t(
                        f"这一步没能完成（已尝试 {tried_note}）：\n"
                        + "\n".join(f"- `{a['intent']}`：{a['error']}" for a in attempts)
                        + "\n\n可以试着把问题说得更具体，比如「近7天各渠道 GMV 环比变化」，或输入 `help` 看能力清单。",
                        f"That step failed (tried: {tried_note}):\n"
                        + "\n".join(f"- `{a['intent']}`: {a['error']}" for a in attempts)
                        + "\n\nTry being more specific (e.g. \"GMV by channel, last 7 days vs prior\"), or type `help`.",
                    )
                )
                reply.data["replan_failed"] = attempts
        # Task chain from the LLM plan (e.g. diagnose -> report): run the
        # remaining steps against the same resolved slots and merge replies.
        chain = (plan or {}).get("chain") if isinstance(plan, dict) else None
        if chain:
            for step in chain:
                step_handler = handlers.get(step)
                if step_handler is None:
                    continue
                try:
                    step_reply = step_handler(question, slots)
                except Exception:  # noqa: BLE001 - chain steps are best-effort
                    continue
                reply = self._merge_replies(reply, step_reply, step)
            reply.data["chain"] = chain
        reply.data.setdefault("intent", intent)
        reply.data.setdefault("confidence", round(confidence, 3))
        reply.data.setdefault("planner", self.session.get("planner", "rules"))
        reply.tool_calls.insert(0, {"tool": "route", "args": {"question": question}, "result": {"intent": intent, "confidence": round(confidence, 3)}})
        # Narration: with an LLM configured, offer a polished prose answer on
        # top of the verified tool output. The deterministic text is always
        # kept as the fallback, so a failed call changes nothing.
        if self.llm.available and intent != "help" and not reply.data.get("limitation"):
            try:
                from .planner import LLMPlanner

                digest = {
                    "question": question,
                    "intent": intent,
                    "deterministic_answer": reply.text,
                    "tool_calls": [
                        {"tool": tc.get("tool"), "result": tc.get("result")}
                        for tc in reply.tool_calls[1:6]
                    ],
                }
                narrated = LLMPlanner(self.llm).narrate(question, digest, lang=self.lang)
                if narrated:
                    # The LLM prose becomes an executive summary in front of
                    # the verified deterministic answer (kept verbatim below).
                    reply.data["narrated"] = True
                    reply.data["deterministic_text"] = reply.text
                    reply.text = narrated + "\n\n" + reply.text
            except Exception:  # noqa: BLE001 - narration must never break a reply
                pass
        # Remember the resolved scope so a terse follow-up ("按渠道再拆一下")
        # keeps the same metric and window instead of restarting at defaults.
        self.session["last_window"] = (slots["window"], slots.get("comparison"))
        self.session["last_dimensions"] = list(slots.get("dimensions") or [])
        self.session["last_metric"] = slots.get("metric")
        return reply

    # --------------------------------------------------------------- handlers
    def _llm_sql_fallback(self, question: str) -> AgentReply:
        """Semantic layer missed: LLM SQL when configured, limitation otherwise.

        The generated SQL is always re-validated by ``sanitize_sql`` and run
        through the guarded SQLTool — the LLM never touches the warehouse.
        """
        # Deterministic shortcut first: a request that names a concrete table
        # (preset or an attached data source) is answerable without any LLM.
        direct = self._direct_table_query(question)
        if direct is not None:
            return direct
        sql_error: Optional[str] = None
        if self.llm.available:
            try:
                from .planner import LLMPlanner

                plan_sql = LLMPlanner(self.llm).generate_sql(
                    question, semantic.describe_for_prompt(), lang=self.lang
                )
                if plan_sql:
                    cleaned = sanitize_sql(plan_sql["sql"])
                    result = self.sql.run(cleaned)
                    df = result.dataframe
                    if result.row_count > 0:
                        table_block = self._table_block(
                            df.head(15),
                            self._t(
                                "查询结果（LLM 生成 SQL · 只读护栏校验通过）",
                                "Query result (LLM-generated SQL, read-only guard passed)",
                            ),
                        )
                        text = self._t(
                            f"该问题超出了语义层的预设指标，改由大模型生成 SQL，经只读护栏校验后执行，返回 {result.row_count} 行。"
                            f"{plan_sql['explanation']}",
                            f"This goes beyond the semantic layer's preset metrics, so the LLM generated the SQL; "
                            f"it passed the read-only guard and returned {result.row_count} rows. {plan_sql['explanation']}",
                        )
                        return AgentReply(
                            text=text,
                            blocks=[table_block, {"type": "markdown", "markdown": f"`{cleaned}`"}],
                            tool_calls=[
                                {
                                    "tool": "llm_sql",
                                    "args": {"question": question},
                                    "result": {"sql": cleaned, "row_count": result.row_count},
                                }
                            ],
                        )
            except Exception as exc:  # noqa: BLE001 - fall through to limitation
                sql_error = f"{type(exc).__name__}: {exc}"

        base = self._t(
            f"没识别出对应的预设指标。当前可用指标：{', '.join(m.name for m in semantic.default_table().metrics.values())}。",
            f"I could not match a preset metric. Available: {', '.join(m.name for m in semantic.default_table().metrics.values())}.",
        )
        limitation = None
        if self.llm.available:
            try:
                from .planner import LLMPlanner

                limitation = LLMPlanner(self.llm).assess_limitation(
                    question,
                    CAPABILITIES,
                    reason=sql_error or "semantic layer has no matching metric and LLM SQL returned nothing usable",
                    lang=self.lang,
                )
            except Exception:  # noqa: BLE001
                limitation = None
        reply = AgentReply(text=(limitation + "\n\n" + base) if limitation else base)
        if limitation:
            reply.data["limitation"] = True
        return reply

    def _direct_table_query(self, question: str) -> Optional[AgentReply]:
        """Preview rows of a named table without the LLM.

        Handles phrasings like 「查一下 budget 前10行」/「sales_extra 明细」 —
        any question that names an existing table and asks for raw rows.
        """
        q = question.lower()
        if not re.search(r"前\s*\d+\s*行|全部行|全部数据|前几行|明细|样例|样本|preview|sample|first\s+\d+|top rows", q):
            return None
        names = self.sql.warehouse.tables()
        for name in sorted(names, key=len, reverse=True):
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name.lower())}(?![A-Za-z0-9_])", q):
                m = re.search(r"前\s*(\d+)\s*行|first\s+(\d+)", q)
                limit = min(int(m.group(1) or m.group(2)) if m else 10, 100)
                result = self.sql.run(f"SELECT * FROM {name} LIMIT {limit}")
                if result.row_count == 0:
                    return None
                return AgentReply(
                    text=self._t(
                        f"`{name}` 的前 {result.row_count} 行（共 {result.row_count} 行预览；自然语言聚合请指定指标，或配置大模型后用自由 SQL）。",
                        f"First {result.row_count} row(s) of `{name}`. For aggregates, name a metric — or configure an LLM for free-form SQL.",
                    ),
                    blocks=[self._table_block(result.dataframe, self._t(f"{name} 明细", f"{name} rows"))],
                    tool_calls=[{"tool": "sql", "args": {"sql": result.sql}, "result": {"row_count": result.row_count}}],
                    data={"intent": "sql", "direct_table": name},
                )
        return None

    def _handle_help(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """Capability list; for unmatched real questions, add a limitation note."""
        reply = self._help()
        if self.llm.available and question and not re.search(
            r"^(help|帮助|\?|？|你是谁|what can you do)$", question.strip(), re.I
        ):
            try:
                from .planner import LLMPlanner

                limitation = LLMPlanner(self.llm).assess_limitation(
                    question, full_catalog(), reason="no capability matched", lang=self.lang
                )
                if limitation:
                    reply.data["limitation"] = True
                    reply.text = limitation + "\n\n" + reply.text
            except Exception:  # noqa: BLE001
                pass
        return reply

    def _handle_sql(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        metric: Optional[Metric] = slots.get("metric")
        table = semantic.default_table()
        if metric is None:
            # Semantic layer has no match: with an LLM, generate flexible SQL
            # (still read-only-guarded); otherwise assess the limitation.
            return self._llm_sql_fallback(question)

        dims = slots.get("dimensions") or []
        window: TimeWindow = slots["window"]
        grain = self._infer_grain(question)
        df = self._series(metric, table, window.start, window.end)

        total = self._window_total(metric, window, table)
        if dims:
            frame = self._window_frame(metric, dims, window, table)
            if metric.is_ratio:
                frame["metric_value"] = frame[metric.numerator] / frame[metric.denominator].replace(0, np.nan)
            else:
                frame["metric_value"] = frame[metric.numerator]
            frame = frame.sort_values("metric_value", ascending=False).head(15)
            table_block = self._table_block(
                frame[dims + ["metric_value"]], self._t("查询结果", "Query result")
            )
            total = self._window_total(metric, window, table)
            top = frame.iloc[0]
            share1 = top["metric_value"] / total if total else float("nan")
            share3 = frame["metric_value"].head(3).sum() / total if total else float("nan")
            text = self._t(
                f"**{metric.name}**（{window.label}）：总计 {_fmt_num(total)}，按 `{'/'.join(dims)}` 拆分（Top 15）。"
                f"第一名 {dims[0]}={top[dims[0]]} 为 {_fmt_num(top['metric_value'])}（占 {share1:.1%}），"
                f"Top3 合计占 {share3:.1%}。",
                f"**{metric.name}** ({window.label}): total {_fmt_num(total)}, split by `{'/'.join(dims)}` (top 15). "
                f"Leader {dims[0]}={top[dims[0]]} at {_fmt_num(top['metric_value'])} ({share1:.1%} share); "
                f"top-3 make up {share3:.1%}.",
            )
            blocks = [table_block]
        else:
            if grain:
                frame = df.copy()
                frame["metric_value"] = frame["value"]
                chart_uri = charts.anomaly_chart(
                    pd.DataFrame({
                        "date": frame["date"],
                        "actual": frame["metric_value"],
                        "expected": frame["metric_value"].rolling(7, min_periods=1).mean(),
                        "flagged": False,
                    }),
                    title=f"{metric.name} daily trend",
                )
                blocks = [{"type": "chart", "title": f"{metric.name} trend", "image": chart_uri}]
                first = float(frame["metric_value"].iloc[0])
                last = float(frame["metric_value"].iloc[-1])
                drift = (last - first) / first if first else float("nan")
                text = self._t(
                    f"**{metric.name}** 日趋势（{window.label}）：首日 {_fmt_num(first)} → 末日 {_fmt_num(last)}"
                    f"（{_fmt_pct(drift)}），区间均值 {_fmt_num(frame['metric_value'].mean())}。",
                    f"**{metric.name}** daily trend ({window.label}): {_fmt_num(first)} on the first day → "
                    f"{_fmt_num(last)} on the last ({_fmt_pct(drift)}); average {_fmt_num(frame['metric_value'].mean())}.",
                )
            else:
                total = self._window_total(metric, window, table)
                prev = self._window_total(metric, slots["comparison"], table)
                change = (total - prev) / prev if prev else float("nan")
                arrow = "\u2191" if change > 0 else "\u2193"
                blocks = [
                    {
                        "type": "kpi",
                        "items": [
                            {"label": metric.name, "value": _fmt_num(total), "sub": window.label},
                            {
                                "label": self._t("环比", "vs prior"),
                                "value": _fmt_pct(change),
                                "sub": f"{arrow} {_fmt_num(total - prev)}",
                                "good": (change > 0) == metric.higher_is_better,
                            },
                        ],
                    }
                ]
                text = self._t(
                    f"**{metric.name}**：{window.label} 为 {_fmt_num(total)}，上一周期 {_fmt_num(prev)}，"
                    f"环比 {_fmt_pct(change)}（{_fmt_num(total - prev)}）。",
                    f"**{metric.name}**: {window.label} = {_fmt_num(total)} vs {_fmt_num(prev)} prior, "
                    f"{_fmt_pct(change)} ({_fmt_num(total - prev)}).",
                )
            table_block = None

        sql_text = self._explain_sql(metric, table, dims, window, grain)
        # RAG: surface the relevant business caliber straight in the answer, so
        # "查 GMV" also reminds the reader what GMV does and does not include.
        kb_note = self.kb_context(f"{metric.name} 口径 定义", k=1)
        if kb_note:
            text += "\n\n" + self._t("📌 业务口径提示（来自知识库）：\n", "Business caliber (from the knowledge base):\n") + kb_note
        return AgentReply(
            text=text,
            blocks=blocks,
            tool_calls=[{"tool": "sql_query", "args": {"metric": metric.name, "dimensions": dims, "window": window.label}, "result": {"sql": sql_text}}],
            data={"metric": metric.name, "total": total, "sql": sql_text},
        )

    # ------------------------------------------------------------- knowledge
    def _handle_metric(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """Metric dictionary: register / list / deprecate metric definitions.

        Registration follows ``docs/metric_dictionary.md``: a formula with a
        name, validated against the live warehouse, then activated into the
        semantic layer so every tool uses the same caliber immediately.
        """
        q = question.strip()
        md = self.metric_dict
        kb_hits = ""
        kb = self.knowledge
        if kb:
            try:
                kb_hits = kb.context_for(f"指标定义 口径 {q}", k=1)
            except Exception:  # noqa: BLE001
                kb_hits = ""

        # --- deprecate -----------------------------------------------------
        if re.search(r"(下线|废弃|停用|弃用|deprecate|retire)", q, re.I):
            match = re.search(r"(?:指标|metric)\s*[：:]?\s*[「\"]?([A-Za-z一-龥][\w一-龥]*)", q, re.I)
            name = (match.group(1) if match else "").lower()
            entry = md.entries.get(name) or md.resolve(q)
            if entry is None:
                return AgentReply(
                    text=self._t(
                        f"字典里没有找到指标 `{name or q}`。用「指标字典」可查看全部指标。",
                        f"Metric `{name or q}` is not in the dictionary. Use \"metric dictionary\" to list all.",
                    )
                )
            if entry.get("status") == "deprecated":
                return AgentReply(
                    text=self._t(f"指标 `{entry['name']}` 已于 {entry.get('updated_at')} 下线，无需重复操作。", f"Metric `{entry['name']}` was already deprecated on {entry.get('updated_at')}.")
                )
            md.deprecate(entry["name"], reason=q[:120])
            return AgentReply(
                text=self._t(
                    f"指标 **{entry['name']}** 已下线（保留在字典中可追溯，历史引用不断链）。\n\n如需新口径，请以 v2 名称重新注册，不要原地改定义。",
                    f"Metric **{entry['name']}** is deprecated (kept in the dictionary for lineage).\n\nRegister a v2 name for a new caliber instead of overwriting.",
                ),
                tool_calls=[{"tool": "metric_dict", "args": {"op": "deprecate"}, "result": {"name": entry["name"]}}],
            )

        # --- register from a formula --------------------------------------
        formula = md.parse_formula(q)
        wants_register = bool(re.search(r"(注册|新增|创建|登记|定义|register|define)", q, re.I))
        if formula and wants_register:
            # Operands like "gmv - refund_amount" keep SQL expression syntax.
            body_text = formula["numerator"] if not formula["denominator"] else f"{formula['numerator']} / {formula['denominator']}"
            name = formula["name"]
            aliases = []
            m_alias = re.search(r"[（(]?(?:别名|alias)[：:]\s*([^，,。;；)]+)", q)
            if m_alias:
                aliases = [a.strip() for a in re.split(r"[/,、]", m_alias.group(1)) if a.strip()]
            if re.search(r"[一-龥]", name):
                aliases.append(name)
            try:
                entry = md.register(
                    name=name,
                    numerator=formula["numerator"],
                    denominator=formula["denominator"],
                    aliases=aliases,
                    description=body_text,
                    formula_text=body_text,
                    table=(slots.get("metric").table if slots.get("metric") else "orders_daily"),
                )
            except MetricDefinitionError as exc:
                return AgentReply(
                    text=self._t(f"指标注册被驳回：{exc}\n\n自查清单见 docs/metric_dictionary.md（分子分母、去重、时区、过滤条件）。", f"Registration rejected: {exc}\n\nChecklist in docs/metric_dictionary.md."),
                    data={"limitation": False},
                )
            kind = self._t("比率指标", "ratio metric") if entry.get("denominator") else self._t("加总指标", "additive metric")
            lines = [
                self._t(
                    f"指标 **{entry['name']}** 已注册并通过校验，语义层已生效——现在可以直接用自然语言查询，所有工具同口径。",
                    f"Metric **{entry['name']}** registered and validated; it is live in the semantic layer — query it directly in natural language, one caliber everywhere.",
                ),
                "",
                f"- 公式 formula：`{entry['formula_text']}`（{kind}）",
                f"- 绑定表 table：`{entry['table']}`",
                f"- 别名 aliases：{', '.join(entry.get('aliases') or []) or '—'}",
                f"- 单位 unit：{entry.get('unit') or '—'}",
                f"- 负责人 owner：{entry.get('owner') or '—'}",
            ]
            if kb_hits:
                lines += ["", self._t("**口径要点提醒**（来自指标定义知识库）：", "**Caliber reminders** (from the metric-definition knowledge base):"), kb_hits[:400]]
            df = pd.DataFrame(
                [[entry["name"], entry["formula_text"], entry["table"], entry.get("unit", ""), ", ".join(entry.get("aliases") or [])]],
                columns=[self._t("指标", "metric"), self._t("公式", "formula"), self._t("绑定表", "table"), self._t("单位", "unit"), self._t("别名", "aliases")],
            )
            return AgentReply(
                text="\n".join(lines),
                blocks=[self._table_block(df, self._t("指标字典 · 新注册", "Metric dictionary · registered"))],
                tool_calls=[{"tool": "metric_dict", "args": {"op": "register", "name": entry["name"]}, "result": {"formula": entry["formula_text"], "table": entry["table"]}}],
                data={"metric_dict": {"registered": entry["name"]}},
            )

        # --- list ----------------------------------------------------------
        if re.search(r"字典|列出|列举|查看|有哪些|清单|list|show|catalog", q, re.I):
            rows = []
            for table in semantic.tables.values():
                for m in table.metrics.values():
                    formula_text = f"{m.numerator}/{m.denominator}" if m.is_ratio else f"SUM({m.numerator})"
                    rows.append([m.name, formula_text, table.name, m.unit or "—", self._t("预置", "preset"), self._t("生效", "active")])
            for entry in md.entries.values():
                rows.append([
                    entry["name"],
                    entry.get("formula_text", ""),
                    entry.get("table", ""),
                    entry.get("unit") or "—",
                    self._t("自定义", "custom"),
                    self._t("生效", "active") if entry.get("status") == "active" else self._t("已下线", "deprecated"),
                ])
            df = pd.DataFrame(rows, columns=[self._t("指标", "metric"), self._t("公式", "formula"), self._t("绑定表", "table"), self._t("单位", "unit"), self._t("来源", "source"), self._t("状态", "status")])
            return AgentReply(
                text=self._t(
                    f"指标字典共 {len(df)} 个指标（预置 {len(semantic.all_metrics())} + 自定义 {len(md.entries)}）。查询、诊断、归因、报告全部走这里的口径，保证一致性。",
                    f"The dictionary holds {len(df)} metrics (preset {len(semantic.all_metrics())} + custom {len(md.entries)}). Every tool resolves calibers here.",
                ),
                blocks=[self._table_block(df, self._t("指标字典", "Metric dictionary"), max_rows=25)],
                tool_calls=[{"tool": "metric_dict", "args": {"op": "list"}, "result": {"count": len(df)}}],
                data={"metric_dict": {"count": len(df)}},
            )

        # --- guidance ------------------------------------------------------
        return AgentReply(
            text=self._t(
                "指标字典支持三种操作：\n"
                "1. **注册**：`注册指标：客单价2 = gmv / 订单量`（分子分母会先在数仓上校验，通过即进语义层）\n"
                "2. **查看**：`指标字典`（列出全部预置 + 自定义指标口径）\n"
                "3. **下线**：`下线指标 xxx`（保留可追溯，建议新口径用 v2 名称）\n\n"
                "注册前自查：去重口径、时间窗与时区、过滤条件、单位与方向——详见 docs/metric_dictionary.md。",
                "The metric dictionary supports:\n"
                "1. **Register**: `register metric: aov2 = gmv / orders` (validated against the warehouse, then live in the semantic layer)\n"
                "2. **List**: `metric dictionary`\n"
                "3. **Deprecate**: `deprecate metric x` (kept for lineage; use a v2 name for new calibers)\n\n"
                "Before registering, check dedup rules, time zone, filters and direction — see docs/metric_dictionary.md.",
            ),
            data={"metric_dict": {"custom_count": len(md.entries)}},
        )

    def _handle_datasource(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """Attach external databases / offline spreadsheets into the session."""
        q = question.strip()
        reg = self.datasources

        # --- remove ---------------------------------------------------------
        if re.search(r"(移除|删除|断开|卸载|下线|remove|detach|drop)", q, re.I) and re.search(r"数据源|datasource|data source", q, re.I):
            match = re.search(r"(?:数据源|source)\s*[：:]?\s*[「\"]?([A-Za-z_][\w]*)", q, re.I)
            name = match.group(1) if match else ""
            try:
                removed = reg.drop(name) if name else None
            except DataSourceError as exc:
                return AgentReply(text=self._t(f"移除失败：{exc}", f"Removal failed: {exc}"))
            if removed is None:
                names = ", ".join(s["name"] for s in reg.list()) or self._t("（空）", "(none)")
                return AgentReply(
                    text=self._t(f"没有指定要移除的数据源。当前已关联：{names}", f"No source specified. Currently attached: {names}")
                )
            return AgentReply(
                text=self._t(f"数据源 **{removed['name']}** 已移除。", f"Data source **{removed['name']}** removed."),
                tool_calls=[{"tool": "datasource", "args": {"op": "drop"}, "result": {"name": removed["name"]}}],
            )

        # --- register a file / database -------------------------------------
        path_match = re.search(r"([\w./~\-]+\.(?:csv|tsv|txt|parquet|pq|json|ndjson|xlsx|xls))", q, re.I)
        db_match = re.search(r"(?:数据库|database|db)\s*[：:]?\s*[「\"]?([\w./~:\-@]+(?:\.duckdb|sqlite://[\w./~\-]+|(?:postgres|postgresql|mysql)://\S+))", q, re.I)
        sqlite_match = re.search(r"([\w./~\-]+\.sqlite(?:db)?)", q, re.I)
        duckdb_match = re.search(r"([\w./~\-]+\.duckdb)", q, re.I)
        target = path_match.group(1) if path_match else None
        if target is None and db_match:
            target = db_match.group(1)
        if target is None and sqlite_match:
            target = sqlite_match.group(1)
        if target is None and duckdb_match and duckdb_match.group(1) != str(self.sql.warehouse.path):
            target = duckdb_match.group(1)
        if target:
            m_name = re.search(r"(?:叫|命名|名为|作为|名字|as|name)\s*[「\"]?([A-Za-z_][\w]*)", q, re.I)
            default_name = Path(target).stem.lower().replace("-", "_").replace(".", "_")
            name = m_name.group(1).lower() if m_name else default_name
            try:
                if re.search(r"\.(csv|tsv|txt|parquet|pq|json|ndjson|xlsx|xls)$", target, re.I):
                    info = reg.register_file(name, target)
                elif re.search(r"\.duckdb$", target, re.I):
                    info = reg.register_database(name, target)
                elif "sqlite" in target.lower():
                    info = reg.register_database(name, target)
                else:
                    info = reg.register_database(name, target)
            except DataSourceError as exc:
                return AgentReply(
                    text=self._t(
                        f"数据源接入失败：{exc}\n\n支持：CSV/TSV/Parquet/JSON/Excel 文件；DuckDB 文件；sqlite:// 路径；postgres:// 或 mysql:// 连接串。",
                        f"Attachment failed: {exc}\n\nSupported: CSV/TSV/Parquet/JSON/Excel files; DuckDB files; sqlite:// paths; postgres:// or mysql:// URIs.",
                    )
                )
            lines = [
                self._t(
                    f"数据源 **{info['name']}** 已关联（{info.get('format') or info.get('db_type')}）。",
                    f"Data source **{info['name']}** attached ({info.get('format') or info.get('db_type')}).",
                )
            ]
            if info.get("columns"):
                lines.append(self._t(f"- 列：{', '.join(info['columns'][:12])}{' …' if len(info['columns']) > 12 else ''}", f"- columns: {', '.join(info['columns'][:12])}{' …' if len(info['columns']) > 12 else ''}"))
            if info.get("row_count") is not None:
                lines.append(self._t(f"- 行数：{info['row_count']:,}", f"- rows: {info['row_count']:,}"))
            if info.get("tables"):
                lines.append(self._t(f"- 表：{', '.join(info['tables'][:10])}", f"- tables: {', '.join(info['tables'][:10])}"))
            lines.append(
                self._t(
                    "\n现在可以直接查询了，例如：`查一下 " + info["name"] + " 前10行`，或在 SQL/分析问题里直接引用这张表。",
                    "\nIt is queryable right away, e.g.: `show top rows of " + info["name"] + "`, or reference the table in any SQL / analysis question.",
                )
            )
            sample_df = None
            if info.get("sample"):
                sample_df = pd.DataFrame(info["sample"])
            blocks = [self._table_block(sample_df, self._t(f"数据源 {info['name']} 预览", f"Source {info['name']} preview"), max_rows=5)] if sample_df is not None else []
            return AgentReply(
                text="\n".join(lines),
                blocks=blocks,
                tool_calls=[{"tool": "datasource", "args": {"op": "register", "target": target}, "result": {"name": info["name"], "mode": info.get("mode")}}],
                data={"datasource": {"registered": info["name"]}},
            )

        # --- list -----------------------------------------------------------
        sources = reg.list()
        if sources:
            rows = []
            for s in sources:
                rows.append([
                    s["name"],
                    s.get("source_kind", "file"),
                    s.get("format") or s.get("db_type", ""),
                    s.get("path") or s.get("uri", ""),
                    self._t("是", "yes") if s.get("stale") else self._t("正常", "ok"),
                ])
            df = pd.DataFrame(rows, columns=[self._t("名称", "name"), self._t("类型", "kind"), self._t("格式", "format"), self._t("位置", "location"), self._t("状态", "status")])
            return AgentReply(
                text=self._t(
                    f"当前已关联 {len(sources)} 个外部数据源，查询时直接引用表名即可（如 `查一下 {sources[0]['name']} 按天的汇总`）。",
                    f"{len(sources)} external source(s) attached; reference the table name directly (e.g. `daily summary of {sources[0]['name']}`).",
                ),
                blocks=[self._table_block(df, self._t("已关联数据源", "Attached data sources"))],
                tool_calls=[{"tool": "datasource", "args": {"op": "list"}, "result": {"count": len(sources)}}],
                data={"datasource": {"count": len(sources)}},
            )
        return AgentReply(
            text=self._t(
                "还没有关联外部数据源。可以：\n"
                "- **离线表格**：`关联数据源 /Users/me/sales.csv 叫 sales`（支持 CSV/TSV/Parquet/JSON/Excel）\n"
                "- **企业数据库**：`连接数据库 postgres://user:pass@host:5432/dw` 或 `关联数据库 /data/warehouse.duckdb`（DuckDB/SQLite/Postgres/MySQL）\n"
                "- **查看**：`数据源列表`\n\n"
                "接入后表会挂进当前会话，自然语言查询、SQL、诊断、建模、看板都能直接用。",
                "No external sources attached yet. You can:\n"
                "- **Spreadsheets**: `attach data source /Users/me/sales.csv as sales` (CSV/TSV/Parquet/JSON/Excel)\n"
                "- **Databases**: `connect database postgres://user:pass@host:5432/dw` or `attach database /data/warehouse.duckdb` (DuckDB/SQLite/Postgres/MySQL)\n"
                "- **List**: `data sources`\n\n"
                "Attached sources are usable from NL querying, SQL, diagnosis, modeling and dashboards.",
            ),
            data={"datasource": {"count": 0}},
        )

    def _handle_knowledge(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """Answer caliber / glossary questions from the RAG knowledge base."""
        kb = self.knowledge
        if not kb:
            return AgentReply(
                text=self._t(
                    "知识库不可用（缺少 docs/business_knowledge.md）。",
                    "The knowledge base is unavailable (docs/business_knowledge.md missing).",
                )
            )
        hits = kb.search(question, k=3)
        if not hits:
            return AgentReply(
                text=self._t(
                    "知识库里没找到相关条目。可以问指标口径、维度语义、数据字典或已知业务事件。",
                    "Nothing relevant found. Try metric calibers, dimension semantics, the data dictionary or known incidents.",
                )
            )
        lines = [self._t(f"从业务知识库检索到 {len(hits)} 条相关内容：", f"{len(hits)} relevant entries from the knowledge base:"), ""]
        for hit in hits:
            body = re.sub(r"\n{2,}", "\n", hit["text"]).strip()
            lines.append(f"**{hit['title']}**")
            if len(body) > 500:
                body = body[:500] + "…"
            lines.append(body)
            lines.append("")
        rows = [[h["title"], h["source"], f"{h['score']:.3f}"] for h in hits]
        blocks = [
            self._table_block(
                pd.DataFrame(rows, columns=[self._t("知识条目", "entry"), self._t("来源", "source"), self._t("相关度", "score")]),
                self._t("知识库检索结果", "Knowledge base hits"),
            )
        ]
        return AgentReply(
            text="\n".join(lines),
            blocks=blocks,
            tool_calls=[{"tool": "knowledge_search", "args": {"query": question[:120]}, "result": {"hits": [h["title"] for h in hits]}}],
            data={"knowledge": {"hits": [{"title": h["title"], "source": h["source"], "score": h["score"]} for h in hits]}},
        )

    # ------------------------------------------------------- statistical tests
    _TEST_KINDS = (
        (r"异动|异常|诊断|anomal(y|ies)|diagnos", "anomaly"),
        (r"实验|ab|readout|分流", "experiment"),
        (r"双重差分|\bdid\b|平行趋势|difference-in-differences", "did"),
        (r"时序因果|中断时间序列|\bits\b|时间序列", "its"),
        (r"建模|分类模型|回归模型|模型|model", "modeling"),
        (r"倾向|匹配|\bpsm\b|propensity", "psm"),
    )

    def _handle_tests(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """Explain the tests behind a conclusion: what, why, and how to read them.

        Three modes: one named test (full theory card), one analysis kind (which
        tests that readout reports and why), or the whole catalogue.
        """
        zh = self.lang != "en"
        hit = stat_tests.find_by_text(question)
        if hit is not None:
            used_by = [
                spec["name_zh"] if zh else spec["name_en"]
                for spec in test_panels.PANEL_SPEC.values()
                if any(hit["key"] in key for key in spec["tests"])
            ]
            rows = [
                [self._t("原假设 H0", "null hypothesis"), hit["h0_zh"] if zh else hit["h0_en"]],
                [self._t("统计量", "test statistic"), f"{hit['stat_name']} · {hit['formula']}"],
                [self._t("前提假设", "assumptions"), hit["assume_zh"] if zh else hit["assume_en"]],
                [self._t("什么时候用", "when to use"), hit["use_zh"] if zh else hit["use_en"]],
                [self._t("怎么读结论", "how to read it"), hit["reject_zh"] if zh else hit["reject_en"]],
                [self._t("本 Agent 用在哪", "where this agent uses it"), "、".join(used_by) if zh else ", ".join(used_by)],
                [self._t("参考", "reference"), hit["ref"]],
            ]
            df = pd.DataFrame(rows, columns=[self._t("项目", "item"), self._t("内容", "detail")])
            text = self._t(
                f"**{hit['name_zh']}**（{hit['name_en']}）。原假设：{hit['h0_zh']}；"
                f"统计量 {hit['stat_name']}，α=0.05 下 p<0.05 拒绝 H0 → {hit['reject_zh']}，否则 {hit['retain_zh']}。"
                f"前提是{hit['assume_zh']}；主要用于{hit['use_zh']}。参考文献：{hit['ref']}"
                + (f"\n\n本 Agent 在以下结论中会附带这一检验：{'、'.join(used_by)}。" if used_by else ""),
                f"**{hit['name_en']}**. H0: {hit['h0_en']}. Statistic {hit['stat_name']}; at alpha=0.05, "
                f"p<0.05 rejects H0 → {hit['reject_en']}, otherwise {hit['retain_en']}. "
                f"Assumptions: {hit['assume_en']}. Used for {hit['use_en']}. Reference: {hit['ref']}"
                + (f"\n\nThis agent attaches it to: {', '.join(used_by)}." if used_by else ""),
            )
            return AgentReply(
                text=text,
                blocks=[self._table_block(df, self._t("检验说明", "Test definition"), max_rows=12)],
                tool_calls=[{"tool": "describe_test", "args": {"test": hit["key"]}, "result": {"h0": hit["h0_en"]}}],
                data={"test": hit["key"]},
            )

        kind = next((k for pattern, k in self._TEST_KINDS if re.search(pattern, question, re.I)), None)
        if kind:
            spec = test_panels.PANEL_SPEC[kind]
            rows = []
            for key in spec["tests"]:
                for part in key.split("|"):
                    meta = stat_tests.describe(part)
                    if meta:
                        rows.append(
                            [
                                meta["family_zh"] if zh else meta["family_en"],
                                meta["name_zh"] if zh else meta["name_en"],
                                meta["h0_zh"] if zh else meta["h0_en"],
                                meta["stat_name"],
                            ]
                        )
            df = pd.DataFrame(rows, columns=[self._t("类别", "family"), self._t("检验", "test"), self._t("原假设", "H0"), self._t("统计量", "stat")])
            text = self._t(
                f"「{spec['name_zh']}」的结论会附带 {len(rows)} 项检验。**为什么是这些**：{spec['why_zh']}"
                f"\n\n每一项都会在结论表里给出统计量、p 值与判定，并注明参考文献；问「什么是 Welch t 检验」可以看单项的理论细节。",
                f"A '{spec['name_en']}' conclusion carries {len(rows)} tests. **Why these**: {spec['why_en']}"
                f"\n\nEach row reports the statistic, the p-value, the decision and its reference. "
                f"Ask 'what is a Welch t-test' for the theory of any single test.",
            )
            return AgentReply(
                text=text,
                blocks=[
                    {
                        "type": "kpi",
                        "items": [
                            {"label": self._t("检验数", "tests"), "value": str(len(rows))},
                            {"label": self._t("分析场景", "readout"), "value": spec["name_zh"] if zh else spec["name_en"]},
                            {"label": self._t("误报控制", "multiplicity"), "value": "BH-FDR" if kind == "experiment" else "—"},
                        ],
                    },
                    self._table_block(df, self._t("该结论附带的检验", "Tests in this readout"), max_rows=20),
                ],
                tool_calls=[{"tool": "test_panel", "args": {"kind": kind}, "result": {"tests": len(rows)}}],
                data={"panel": kind, "tests": rows},
            )

        # Full catalogue, grouped by family.
        rows = [
            [
                c["family_zh"] if zh else c["family_en"],
                c["name_zh"] if zh else c["name_en"],
                c["h0_zh"] if zh else c["h0_en"],
                c["use_zh"] if zh else c["use_en"],
            ]
            for c in stat_tests.catalogue()
        ]
        df = pd.DataFrame(rows, columns=[self._t("类别", "family"), self._t("检验", "test"), self._t("原假设", "H0"), self._t("用途", "use")])
        n_families = len({c["family_zh"] for c in stat_tests.catalogue()})
        text = self._t(
            f"目前内置 **{len(rows)} 项统计检验**，覆盖 {n_families} 个类别；每一项都写明了原假设、公式、前提、判定规则与文献出处。"
            f"\n\n读法：α=0.05 下 p<α 只说明“在原假设下不太可能观察到这个结果”，不等于效应大；所以结论里同时给效应量、置信区间和检验前提的检查结果。"
            f"\n\n每个分析结论都会自动附带对应检验（异动诊断、实验读数、DiD、ITS、建模、PSM），完整清单与公式见 `docs/statistical_tests.md`。"
            f"\n\n可以问：「异常检测用了哪些检验」「实验读数用了哪些检验」「什么是 Welch t 检验」。",
            f"**{len(rows)} statistical tests** across {n_families} families, each with its null hypothesis, formula, assumptions, decision rule and citation."
            f"\n\nReading rule: at alpha=0.05, p<alpha only means the result is unlikely under H0 — not that the effect is large. "
            f"That is why every conclusion also carries an effect size, a confidence interval and the assumption checks."
            f"\n\nEach readout attaches its own battery (anomaly, experiment, DiD, ITS, modeling, PSM); the full catalogue is in `docs/statistical_tests.md`."
            f"\n\nTry: 'which tests does anomaly detection run', 'what is a Welch t-test'.",
        )
        return AgentReply(
            text=text,
            blocks=[
                {
                    "type": "kpi",
                    "items": [
                        {"label": self._t("检验方法", "tests"), "value": str(len(rows))},
                        {"label": self._t("类别", "families"), "value": str(n_families)},
                        {"label": self._t("覆盖读数", "readouts"), "value": str(len(test_panels.PANEL_SPEC))},
                    ],
                },
                self._table_block(df, self._t("统计检验方法清单", "Statistical test catalogue"), max_rows=40),
            ],
            tool_calls=[{"tool": "test_catalogue", "args": {}, "result": {"tests": len(rows)}}],
            data={"tests": [c["key"] for c in stat_tests.catalogue()]},
        )

    @staticmethod
    def _infer_grain(question: str) -> Optional[str]:
        if re.search(r"按天|每天|每日|趋势|daily|trend|by day", question, re.I):
            return "day"
        if re.search(r"按周|每周|weekly|by week", question, re.I):
            return "week"
        if re.search(r"按月|每月|monthly|by month", question, re.I):
            return "month"
        return None

    def _explain_sql(self, metric: Metric, table: Table, dims: Sequence[str], window: TimeWindow, grain: Optional[str]) -> str:
        select = []
        if grain:
            select.append(f"DATE_TRUNC('{grain}', {table.time_column}) AS {grain}")
        select += list(dims)
        select.append(f"{metric.sql()} AS {metric.name}")
        group = ", ".join(str(i + 1) for i in range(len(select) - 1)) or "ALL"
        return (
            f"SELECT {', '.join(select)} FROM {table.name} "
            f"WHERE {window.clause(table.time_column)} GROUP BY {group} ORDER BY {len(select)} DESC"
        )

    @staticmethod
    def _format_section_table(df: pd.DataFrame, pct_cols: Sequence[str] = ()) -> pd.DataFrame:
        """Format a frame for static report rendering (percent columns become text)."""
        out = df.copy()
        for col in pct_cols:
            if col in out.columns:
                out[col] = out[col].map(lambda v: _fmt_pct(float(v)) if pd.notna(v) else "-")
        return out

    def _table_block(
        self, df: pd.DataFrame, title: str, max_rows: int = 15, pct_cols: Sequence[str] = ()
    ) -> Dict[str, Any]:
        view = df.head(max_rows).copy()
        columns = [str(c) for c in view.columns]
        rows = []
        for _, row in view.iterrows():
            cells = []
            for col in view.columns:
                val = row[col]
                if col in pct_cols and isinstance(val, (int, float, np.integer, np.floating)):
                    cells.append(_fmt_pct(float(val)))
                elif isinstance(val, (int, float, np.integer, np.floating)):
                    cells.append(_fmt_num(float(val)))
                else:
                    cells.append(str(val))
            rows.append(cells)
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join("---" for _ in columns) + " |"
        body = "\n".join("| " + " | ".join(cells) + " |" for cells in rows)
        return {
            "type": "table",
            "title": title,
            "columns": columns,
            "rows": rows,
            "markdown": "\n".join([header, divider, body]),
        }

    # -------------------------------------------------------------- diagnose
    # A question can ask for a time-series baseline instead of the default
    # decomposition. The mapping is deliberately narrow: "holt-winters" in a
    # question is a request for that model, not for whatever scores best.
    _BASELINE_HINTS = (
        (r"乘法|相乘|multiplicative|mul\b", "holt_winters_mul"),
        (r"霍尔特|holt[\s-]?winters|\bhw\b|指数平滑|三次指数", "holt_winters"),
        (r"arima|sarima|自回归", "arima"),
        (r"\bstl\b|\bets\b", "stl_ets"),
        (r"时序基线|时序方法|预测基线|哪个基线|基线对比|对比基线|forecast baseline|which baseline", "auto"),
    )

    def _baseline_from_question(self, question: str) -> str:
        """Pick the anomaly baseline the question asks for (``trailing`` by default)."""
        for pattern, baseline in self._BASELINE_HINTS:
            if re.search(pattern, question, re.I):
                return baseline
        return "trailing"

    def _handle_diagnose(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        table = semantic.default_table()
        metric: Metric = slots.get("metric") or table.metrics["gmv"]
        anchor: date = slots["anchor"]
        window: TimeWindow = slots["window"]
        comparison: TimeWindow = slots["comparison"]

        series_start = anchor - timedelta(days=89)
        daily = self._series(metric, table, series_start, anchor)
        baseline = self._baseline_from_question(question)
        cfg = AnomalyConfig(baseline=baseline)
        if baseline == "auto":
            # The library default is a cheap trio because `auto` also runs inside
            # batch segment scans. A user asking for a baseline comparison wants
            # the time-series models in it, so the full candidate set is used here.
            cfg.forecast_candidates = (
                "holt_winters",
                "holt_winters_mul",
                "arima",
                "stl_ets",
                "seasonal_naive",
            )
        detector = AnomalyDetector(cfg)
        denominators = daily["denominator"].to_numpy() if metric.is_ratio else None
        detection = detector.detect(daily["date"], daily["value"], denominators=denominators, label=metric.name)
        baseline_detail = detection.get("baseline_detail") or {}

        total_cur = self._window_total(metric, window, table)
        total_base = self._window_total(metric, comparison, table)
        delta = total_cur - total_base
        rel = delta / total_base if total_base else float("nan")

        in_window = daily[(daily["date"] >= pd.Timestamp(window.start)) & (daily["date"] <= pd.Timestamp(window.end))]
        before = daily[(daily["date"] >= pd.Timestamp(comparison.start)) & (daily["date"] <= pd.Timestamp(comparison.end))]
        # Significance is judged against the seasonal baseline, not against the
        # previous window: when an incident is already 10 days old, the prior
        # week is contaminated and a WoW test understates the problem.
        model_expected = pd.merge(in_window[["date"]], detection["frame"][["date", "expected"]], on="date", how="left")[
            "expected"
        ].to_numpy(dtype=float)
        sig = (
            _compare(in_window["value"].to_numpy(dtype=float), model_expected)
            if len(in_window) > 1 and np.isfinite(model_expected).all()
            else None
        )

        # Attribution across every dimension the table offers; an explicitly
        # requested dimension wins even if another one scores higher.
        all_dims = list(table.dimensions.keys())
        requested = list(slots.get("dimensions") or [])
        ordered_dims = list(dict.fromkeys(requested + all_dims))
        cur_frame = self._window_frame(metric, ordered_dims, window, table)
        base_frame = self._window_frame(metric, ordered_dims, comparison, table)
        attribution = attribute_change(
            cur_frame, base_frame, metric, ordered_dims, max_depth=2, focus=requested[0] if requested else None
        )

        blocks: List[Dict[str, Any]] = []
        is_good = (rel > 0) == metric.higher_is_better
        blocks.append(
            {
                "type": "kpi",
                "items": [
                    {"label": metric.name, "value": _fmt_num(total_cur), "sub": window.label},
                    {
                        "label": self._t("环比变化", "change"),
                        "value": _fmt_pct(rel),
                        "sub": _fmt_num(delta),
                        "good": is_good,
                    },
                    {
                        "label": self._t("统计显著性", "significance"),
                        "value": f"p={sig['p_value']:.3f}" if sig else "-",
                        "sub": self._t("相对季节性基线", "vs seasonal baseline"),
                    },
                ],
            }
        )
        chart_frame = daily.copy()
        chart_frame["actual"] = chart_frame["value"]
        chart_frame["expected"] = detection["frame"]["expected"].to_numpy()
        chart_frame["flagged"] = detection["frame"]["flagged"].to_numpy()
        chart_uri = charts.anomaly_chart(chart_frame, title=f"{metric.name}: actual vs expected")
        blocks.append({"type": "chart", "title": self._t("指标走势与异常点", "Actual vs expected"), "image": chart_uri})

        # When a forecasting baseline was asked for, say which model answered and
        # how well it backtested. A baseline with a bad MASE produces confident
        # nonsense, so its accuracy belongs next to the finding, not in a log.
        chosen_method = str(baseline_detail.get("method") or "")
        backtest = baseline_detail.get("backtest")
        if baseline != "trailing" and chosen_method:
            acc = baseline_detail.get("accuracy") or {}
            blocks[0]["items"].append(
                {
                    "label": self._t("基线模型", "baseline model"),
                    "value": str(baseline_detail.get("label", chosen_method)),
                    "sub": f"MASE {acc['mase']:.3f}" if acc.get("mase") else self._t("滚动回测", "rolling backtest"),
                }
            )

        contrib_uri = charts.contribution_bar(
            attribution["primary"]["segments"], attribution["primary"]["dimension"],
            title=f"{metric.name}: contribution to change",
        )
        blocks.append({"type": "chart", "title": self._t("贡献度拆解", "Contribution"), "image": contrib_uri})

        table_df = contribution_table(attribution["primary"], top_k=10)
        blocks.append(self._table_block(
            table_df, self._t("各分段贡献", "Segment contributions"), pct_cols=("share_of_change", "growth")
        ))

        # Segment-level anomaly scan: the topline is often flat while one
        # segment bleeds. Decompose on 90 days, report the user's window.
        scan = self._segment_scan(
            metric,
            table,
            attribution["primary"]["dimension"],
            series_start,
            anchor,
            window=(window.start, window.end),
            comparison=(comparison.start, comparison.end),
        )
        if scan["segments"]:
            scan_df = pd.DataFrame(
                [
                    [
                        s["segment"],
                        _fmt_pct(s["window_change"]) if s["window_change"] is not None else "-",
                        f"{s['p_value']:.3f}" if s["p_value"] is not None else "-",
                        _fmt_pct(s["worst_rel_dev"]),
                        s["worst_date"],
                        f"{s['max_abs_z']:.1f}",
                        _severity(s["severity"], self.lang),
                    ]
                    for s in scan["segments"]
                ],
                columns=[
                    self._t("分段", "segment"),
                    self._t("窗口变化", "window change"),
                    "p",
                    self._t("最大偏离", "worst dev"),
                    self._t("偏离日", "worst date"),
                    "|z|",
                    self._t("严重度", "severity"),
                ],
            )
            blocks.append(self._table_block(scan_df, self._t("分段异常扫描", "Segment anomaly scan")))

        # Root-cause decision tree: multi-level drill-down beyond the one-level
        # contribution peel. Each node re-picks the most informative remaining
        # dimension inside its slice; leaves are ranked root-cause paths.
        tree = None
        try:
            tree = root_cause_tree(
                cur_frame, base_frame, metric, ordered_dims,
                max_depth=3, value_labels=table.value_labels,
            )
            blocks.append(
                {
                    "type": "chart",
                    "title": self._t("根因下钻决策树", "Root-cause drill-down tree"),
                    "image": charts.root_cause_tree_chart(tree["nodes"], title=f"{metric.name}: drill-down tree"),
                }
            )
            blocks.append(self._table_block(tree_table(tree, self.lang), self._t("根因路径（逐层下钻）", "Root-cause paths")))
        except Exception:  # noqa: BLE001 - the tree must never fail the diagnosis
            tree = None

        # Confidence validation: standard checks + the noise-band chart, so the
        # "how much to trust this" story ships with the answer.
        mase = (baseline_detail.get("accuracy") or {}).get("mase")
        conf = anomaly_confidence(
            sig,
            in_window["value"].to_numpy(dtype=float),
            model_expected,
            None,
            chosen_method,
            mase,
            detection["episodes"],
            scan["segments"][0] if scan["segments"] else None,
            delta,
            float(detection.get("sigma") or 0.0),
        )
        try:
            band_frame = anomaly_band(detection["frame"], float(detection.get("sigma") or 0.0))
            band_chart = charts.confidence_band_chart(band_frame, title=f"{metric.name}: actual vs expected ± 2.5σ")
            blocks.append({"type": "chart", "title": self._t("置信度：噪声带诊断", "Confidence: noise band"), "image": band_chart})
        except Exception:  # noqa: BLE001
            band_chart = ""
        blocks.append(
            self._table_block(
                confidence_table(conf["checks"], self.lang, conf["verdict"]),
                self._t("置信度验证", "Confidence validation"),
            )
        )

        # Statistical-test panel: the conclusion names every test it rests on,
        # not just the headline p-value. Skipped tests stay visible with reasons.
        try:
            panel = test_panels.anomaly_panel(
                in_window["value"].to_numpy(dtype=float),
                model_expected,
                series=daily["value"].to_numpy(dtype=float),
            )
            blocks.append(
                self._table_block(
                    stat_tests.to_frame(panel["tests"], self.lang),
                    self._t("统计检验（方法 / 统计量 / p / 结论）", "Statistical tests (method / statistic / p / conclusion)"),
                    max_rows=12,
                )
            )
        except Exception:  # noqa: BLE001 - the tests must never break the diagnosis
            panel = None

        if baseline != "trailing" and chosen_method and isinstance(backtest, pd.DataFrame) and len(backtest):
            cols = [c for c in ("label", "mae", "rmse", "mase", "bias") if c in backtest.columns]
            ranked = backtest.sort_values("mase")
            base_df = ranked[cols].rename(columns={"label": self._t("方法", "method")}).reset_index(drop=True)
            base_df.insert(
                0,
                self._t("选中", "chosen"),
                ["✓" if m == chosen_method else "" for m in ranked["method"]],
            )
            blocks.append(
                self._table_block(
                    base_df,
                    self._t("基线模型滚动回测（同一 rolling origin）", "Baseline backtest on one rolling origin"),
                )
            )

        direction = self._t("下降", "down") if delta < 0 else self._t("上升", "up")
        headline = self._t(
            f"**{metric.name}** 在{window.label}为 {_fmt_num(total_cur)}，较{comparison.label} "
            f"{direction} {_fmt_num(abs(delta))}（{_fmt_pct(rel)}）"
            + (f"，相对季节性基线 p={sig['p_value']:.3f}，{'显著' if sig['significant'] else '不显著'}。" if sig else "。"),
            f"**{metric.name}** over {window.label} is {_fmt_num(total_cur)}, "
            + ("down" if delta < 0 else "up")
            + f" {_fmt_num(abs(delta))} vs {comparison.label} ({_fmt_pct(rel)})"
            + (f"; vs its seasonal baseline p={sig['p_value']:.3f} ({'significant' if sig['significant'] else 'not significant'})." if sig else "."),
        )

        driver = attribution["primary"]["segments"].iloc[0]
        share_note = ""
        if abs(driver["contribution_pct"]) > 1.1:
            share_note = self._t("（占比超过 100% 说明有其他分段反向抵消）", " (above 100% means other segments offset it)")
        driver_line = self._t(
            f"主因落在 `{attribution['primary']['dimension']}` = **{driver['segment']}**：贡献 {_fmt_num(driver['contribution'])}"
            f"（占变动 {driver['contribution_pct']:+.0%}{share_note}，该分段自身变化 {_fmt_pct(driver['growth'])}）。",
            f"Primary driver: `{attribution['primary']['dimension']}` = **{driver['segment']}**, contributing "
            f"{_fmt_num(driver['contribution'])} ({driver['contribution_pct']:+.0%} of the move{share_note}; "
            f"its own change {_fmt_pct(driver['growth'])}).",
        )
        lines = [headline, "", driver_line]

        if attribution["drilldown"]:
            sub = attribution["drilldown"][0]
            sub_top = sub["segments"].iloc[0]
            lines.append(
                self._t(
                    f"下钻一层：在 **{driver['segment']}** 内部，`{sub['dimension']}` = {sub_top['segment']} 贡献 "
                    f"{_fmt_num(sub_top['contribution'])}（{sub_top['contribution_pct']:+.0%}）。",
                    f"One level deeper: inside **{driver['segment']}**, `{sub['dimension']}` = {sub_top['segment']} "
                    f"contributes {_fmt_num(sub_top['contribution'])} ({sub_top['contribution_pct']:+.0%}).",
                )
            )

        if metric.is_ratio:
            lines.append(
                self._t(
                    f"量价拆解：变化 {_fmt_num(attribution['total_delta'], 4)} 中，率效应 {_fmt_num(attribution['primary']['rate_effect_total'], 4)}、"
                    f"结构（mix）效应 {_fmt_num(attribution['primary']['mix_effect_total'], 4)}、交互项 "
                    f"{_fmt_num(attribution['primary']['interaction_total'], 4)}。",
                    f"Rate vs mix: of the {_fmt_num(attribution['total_delta'], 4)} change, rate effect "
                    f"{_fmt_num(attribution['primary']['rate_effect_total'], 4)}, mix effect "
                    f"{_fmt_num(attribution['primary']['mix_effect_total'], 4)}, interaction "
                    f"{_fmt_num(attribution['primary']['interaction_total'], 4)}.",
                )
            )

        others = [d for d in attribution["dimension_scores"] if d["dimension"] != attribution["primary"]["dimension"]][:3]
        if others:
            lines.append(
                self._t(
                    "其他维度：" + "；".join(f"{d['dimension']}（Top1={d['top_segment']}，集中度 {d['concentration_top1']:.0%}）" for d in others),
                    "Other dimensions: " + "; ".join(f"{d['dimension']} (top1={d['top_segment']}, concentration {d['concentration_top1']:.0%})" for d in others),
                )
            )

        episodes = detection["episodes"]
        if episodes:
            top_ep = episodes[0]
            lines.append(
                self._t(
                    f"异常扫描（近90天，周季节性基线 + 稳健 z）：发现 {len(episodes)} 段异常区间，最大一段 "
                    f"{top_ep['start']}~{top_ep['end']}（{top_ep['days']}天，{_episode_kind(top_ep, self.lang)}），实际 "
                    f"{_fmt_num(top_ep['actual_sum'])} vs 预期 "
                    f"{_fmt_num(top_ep['expected_sum'])}（{_fmt_pct(top_ep['impact_pct'])}，p={top_ep['p_value']:.4f}），"
                    f"严重度 {_severity(top_ep['severity'], self.lang)}。",
                    f"Anomaly scan (90d, weekly-seasonal baseline + robust z): {len(episodes)} episode(s); largest "
                    f"{top_ep['start']}~{top_ep['end']} ({top_ep['days']}d, {_episode_kind(top_ep, self.lang)}), actual "
                    f"{_fmt_num(top_ep['actual_sum'])} vs expected "
                    f"{_fmt_num(top_ep['expected_sum'])} ({_fmt_pct(top_ep['impact_pct'])}, p={top_ep['p_value']:.4f}), "
                    f"severity {_severity(top_ep['severity'], self.lang)}.",
                )
            )
        if detection["level_shifts"]:
            shift = detection["level_shifts"][0]
            lines.append(
                self._t(
                    f"检测到可能的水平位移：{shift['date']} 之后均值 {_fmt_pct(shift['rel_change'])}，建议排查当日发布/口径变更。",
                    f"Possible level shift around {shift['date']} ({_fmt_pct(shift['rel_change'])} in level); check releases or logging changes.",
                )
            )
        if baseline != "trailing" and chosen_method:
            acc = baseline_detail.get("accuracy") or {}
            margin = baseline_detail.get("selection_margin")
            margin_note = (
                self._t(f"，比次优低 {margin:.1%}", f", {margin:.1%} better than the runner-up")
                if isinstance(margin, float) and np.isfinite(margin) and 0 < margin < 1
                else ""
            )
            lines.append(
                self._t(
                    f"基线口径：{baseline_detail.get('label', chosen_method)}"
                    f"（{'自动回测择优选中的' if baseline == 'auto' else '按你的要求指定'}）"
                    + (f"，滚动回测 MASE {acc['mase']:.3f}" if acc.get("mase") else "")
                    + f"{margin_note}。这个基线只用当天之前的数据，所以不会把事故本身当成预期。",
                    f"Baseline: {baseline_detail.get('label', chosen_method)} "
                    f"({'picked by backtest' if baseline == 'auto' else 'as requested'})"
                    + (f", rolling MASE {acc['mase']:.3f}" if acc.get("mase") else "")
                    + f"{margin_note}. Every expectation uses only data before that day, so an incident cannot leak into its own baseline.",
                )
            )
        if tree and tree["leaves"] and tree["root_delta"]:
            leaf = tree["leaves"][0]
            share = abs(float(leaf["delta"])) / abs(float(tree["root_delta"]))
            path_txt = " → ".join(f"{s['dimension']}={s['value']}" for s in leaf["path"]) or "ALL"
            more = ""
            if len(tree["leaves"]) > 1:
                second = tree["leaves"][1]
                share2 = abs(float(second["delta"])) / abs(float(tree["root_delta"]))
                more = self._t(
                    f"；次因路径 {second['path_label']}（{share2:.0%}）",
                    f"; next path {second['path_label']} ({share2:.0%})",
                )
            lines.append(
                self._t(
                    f"**根因决策树**（最多 3 层逐维下钻）：{'跌幅' if delta < 0 else '涨幅'}最集中的路径是 **{path_txt}**"
                    f"（约占总变动 {share:.0%}）{more}。树状图与路径表见上方卡片。",
                    f"**Root-cause tree** (up to 3 levels of drill-down): the {'decline' if delta < 0 else 'increase'} concentrates at "
                    f"**{path_txt}** (~{share:.0%} of the root move){more}. Tree chart and path table above.",
                )
            )
        if panel is not None:
            lines.append(
                self._t(
                    f"**统计检验**（检验方法 / 统计量 / p / 结论）：{panel['headline_zh']}{panel['summary_zh']}",
                    f"**Statistical tests** (method / statistic / p / conclusion): {panel['headline_en']} {panel['summary_en']}",
                )
            )
        lines.append(verdict_line(conf["verdict"], self.lang))
        if scan["segments"]:
            top_scan = scan["segments"][0]
            sig_part = (
                self._t(f"，窗口变化 {_fmt_pct(top_scan['window_change'])}（p={top_scan['p_value']:.4f}）",
                        f", window change {_fmt_pct(top_scan['window_change'])} (p={top_scan['p_value']:.4f})")
                if top_scan["window_change"] is not None
                else ""
            )
            lines.append(
                self._t(
                    f"分段扫描（`{scan['dimension']}`，分段季节性残差 z）：{scan['summary']}。最异常的是 "
                    f"**{top_scan['segment']}**{sig_part}，{top_scan['worst_date']} 偏离 {_fmt_pct(top_scan['worst_rel_dev'])}"
                    f"（|z|={top_scan['max_abs_z']:.1f}，{_severity(top_scan['severity'], self.lang)}）。",
                    f"Segment scan (`{scan['dimension']}`, per-segment seasonal residual z): {scan['summary']}. Worst is "
                    f"**{top_scan['segment']}**{sig_part}, deviating {_fmt_pct(top_scan['worst_rel_dev'])} on "
                    f"{top_scan['worst_date']} (|z|={top_scan['max_abs_z']:.1f}, {_severity(top_scan['severity'], self.lang)}).",
                )
            )

        lines += ["", self._t("**下一步建议**", "**Suggested next steps**")]
        steps = [
            self._t(
                f"核对该分段的埋点与口径（{driver['segment']} 的分子/分母是否同源）。",
                f"Audit instrumentation for {driver['segment']} (numerator/denominator from the same source?).",
            ),
            self._t(
                "用同口径的对照指标交叉验证（如 GMV 跌而订单不跌，多半是价格或结构问题）。",
                "Cross-check with a paired metric (GMV down without an order drop usually means price/mix).",
            ),
        ]
        if sig and not sig["significant"]:
            segment_sig = scan["segments"][0] if scan["segments"] else None
            if segment_sig and segment_sig.get("p_value") is not None and segment_sig["p_value"] < 0.05:
                steps.insert(
                    0,
                    self._t(
                        f"总量层面不显著（p={sig['p_value']:.3f}），但分段层面显著：`{scan['dimension']}`={segment_sig['segment']} "
                        f"窗口变化 {_fmt_pct(segment_sig['window_change'])}（p={segment_sig['p_value']:.4f}）。"
                        "结论应以分段为准，不要因为总量不显著就认为没问题。",
                        f"The topline change is not significant (p={sig['p_value']:.3f}) but the segment is: "
                        f"`{scan['dimension']}`={segment_sig['segment']} moved {_fmt_pct(segment_sig['window_change'])} "
                        f"(p={segment_sig['p_value']:.4f}). Trust the segment, not the topline average.",
                    ),
                )
            else:
                steps.insert(
                    0,
                    self._t(
                        "该变化在统计上不显著，建议拉长窗口或改为周同比再判断。",
                        "The change is not statistically significant; extend the window or use week-over-week.",
                    ),
                )
        lines += [f"{i + 1}. {s}" for i, s in enumerate(steps)]

        text = "\n".join(lines)
        steps_plain = self._strip_steps(lines)
        self.session["last_diagnosis"] = {
            "headline": headline,
            "driver_line": driver_line,
            "steps": steps_plain,
            "metric": metric.name,
        }
        self._record(
            self._t(
                f"{metric.name} 在{window.label}{'下降' if delta < 0 else '上升'} {_fmt_pct(rel)}（{_fmt_num(delta)}），"
                f"主因 {attribution['primary']['dimension']}={driver['segment']}（占变动 {driver['contribution_pct']:+.0%}）。",
                f"{metric.name} moved {'down' if delta < 0 else 'up'} {_fmt_pct(rel)} ({_fmt_num(delta)}); "
                f"main driver {attribution['primary']['dimension']}={driver['segment']} "
                f"({driver['contribution_pct']:+.0%} of the move).",
            ),
            Section(
                title=f"{metric.name}: {window.label} vs {comparison.label}",
                prose=self._plain(f"{headline} {driver_line}"),
                bullets=steps_plain,
                table=self._format_section_table(table_df, ("share_of_change", "growth", "w_base", "w_cur")),
                chart=chart_uri,
            ),
        )

        if episodes:
            top_ep = episodes[0]
            episode_df = pd.DataFrame(
                [
                    [
                        e["start"],
                        e["end"],
                        e["days"],
                        _episode_kind(e, self.lang),
                        _fmt_pct(e["impact_pct"]),
                        f"{e['p_value']:.4f}",
                        _severity(e["severity"], self.lang),
                    ]
                    for e in episodes[:5]
                ],
                columns=[
                    self._t("开始", "start"),
                    self._t("结束", "end"),
                    self._t("天数", "days"),
                    self._t("类型", "kind"),
                    self._t("偏离", "impact"),
                    "p",
                    self._t("严重度", "severity"),
                ],
            )
            episode_text = self._t(
                f"近90天共 {len(episodes)} 段异常区间，最大一段 {top_ep['start']}~{top_ep['end']}"
                f"（{top_ep['days']}天，{_episode_kind(top_ep, self.lang)}），偏离 {_fmt_pct(top_ep['impact_pct'])}，"
                f"p={top_ep['p_value']:.4f}。",
                f"{len(episodes)} episode(s) in 90 days; largest {top_ep['start']}~{top_ep['end']} "
                f"({top_ep['days']}d, {_episode_kind(top_ep, 'en')}), {_fmt_pct(top_ep['impact_pct'])}, "
                f"p={top_ep['p_value']:.4f}.",
            )
            self._record(
                episode_text,
                Section(
                    title=self._t("异常区间（近90天）", "Anomaly episodes (last 90 days)"),
                    prose=episode_text,
                    table=episode_df,
                    chart=chart_uri,
                ),
            )
        if tree:
            self._record(
                self._t(
                    f"根因决策树：最深下钻 {tree['depth_reached']} 层、{tree['n_nodes']} 个节点，"
                    f"首位路径 {tree['leaves'][0]['path_label'] if tree['leaves'] else '-'}。",
                    f"Root-cause tree: {tree['depth_reached']} levels, {tree['n_nodes']} nodes; "
                    f"top path {tree['leaves'][0]['path_label'] if tree['leaves'] else '-'}.",
                ),
                Section(
                    title=self._t("根因下钻决策树", "Root-cause drill-down tree"),
                    prose=self._plain_line(tree["narrative"]),
                    table=tree_table(tree, self.lang),
                    chart=charts.root_cause_tree_chart(tree["nodes"], title=f"{metric.name}: drill-down tree"),
                ),
            )
        self._record(
            verdict_line(conf["verdict"], self.lang),
            Section(
                title=self._t("置信度验证", "Confidence validation"),
                prose=verdict_line(conf["verdict"], self.lang)
                + " " + self._t(
                    "检查包括窗口显著性、基线回测质量、持续性、留一极端值依赖与分段一致性。",
                    "Checks cover window significance, baseline backtest quality, persistence, leave-one-out outlier dependence and segment agreement.",
                ),
                table=confidence_table(conf["checks"], self.lang, conf["verdict"]),
                chart=band_chart or None,
            ),
        )
        self.session.update({"last_metric": metric.name, "last_window": window, "last_attribution": attribution})

        return AgentReply(
            text=text,
            blocks=blocks,
            tool_calls=[
                {"tool": "detect_anomaly", "args": {"metric": metric.name, "days": 90}, "result": {"episodes": len(episodes), "level_shifts": len(detection["level_shifts"])}},
                {"tool": "attribute_change", "args": {"metric": metric.name, "dimensions": ordered_dims}, "result": {"primary_dimension": attribution["primary"]["dimension"], "primary_segment": driver["segment"]}},
            ],
            data={
                "metric": metric.name,
                "change": rel,
                "delta": delta,
                "attribution": attribution["narrative"],
                "dimension_scores": attribution["dimension_scores"],
                "episodes": episodes,
                "root_cause": (
                    {
                        "top_path": tree["leaves"][0]["path_label"] if tree["leaves"] else None,
                        "top_share": (
                            abs(float(tree["leaves"][0]["delta"])) / abs(float(tree["root_delta"]))
                            if tree["leaves"] and tree["root_delta"]
                            else None
                        ),
                        "n_nodes": tree["n_nodes"],
                        "depth_reached": tree["depth_reached"],
                    }
                    if tree
                    else None
                ),
                "confidence": {
                    "verdict": conf["verdict"],
                    "checks": [
                        {"name": c["name_zh"] if self.lang != "en" else c["name_en"], "status": c["status"]}
                        for c in conf["checks"]
                    ],
                },
                "baseline": {
                    "requested": baseline,
                    "method": chosen_method or "trailing",
                    "mase": (baseline_detail.get("accuracy") or {}).get("mase"),
                    "backtest": None if not isinstance(backtest, pd.DataFrame) else backtest.to_dict("records"),
                },
            },
        )

    # ------------------------------------------------------------- experiment
    def _handle_experiment_design(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        q = question
        if re.search(r"有哪些.*(实验|类型)|实验类型|experiment types|archetype", q, re.I):
            rows = []
            for key, spec in EXPERIMENT_TYPES.items():
                rows.append([key, spec["name_zh"], spec["name_en"], spec["randomization_unit"], spec["when"]])
            df = pd.DataFrame(rows, columns=["type", "中文", "English", "randomisation_unit", "when_to_use"])
            block = self._table_block(df, self._t("实验类型清单", "Experiment archetypes"), max_rows=20)
            block["columns"] = list(df.columns)
            block["rows"] = [[str(v) for v in r] for r in rows]
            text = self._t(
                f"当前内置 {len(EXPERIMENT_TYPES)} 种互联网常见实验类型，每种都带随机化单元、分析方法和常见坑。"
                "告诉我要验证什么（比如「双边市场动态定价」），我可以直接给出选型和设计。",
                f"{len(EXPERIMENT_TYPES)} archetypes are built in, each with its randomisation unit, analysis path and pitfalls. "
                "Describe your situation and I will pick one and size it.",
            )
            return AgentReply(text=text, blocks=[block], tool_calls=[{"tool": "list_experiment_types", "args": {}, "result": {"count": len(EXPERIMENT_TYPES)}}])

        recommendation = recommend_experiment_type(q)
        design_params = self._parse_design_params(q, slots)
        design = ExperimentDesign(
            goal=q[:120], experiment_type=recommendation["primary"], lang=self.lang, **design_params
        )
        result = design_experiment(design)

        sizing = result["sizing"]
        duration = result["duration"]
        blocks = [
            {
                "type": "kpi",
                "items": [
                    {"label": self._t("每臂样本", "per arm"), "value": f"{sizing['n_control']:,}"},
                    {"label": self._t("总样本", "total"), "value": f"{sizing['n_total']:,}"},
                    {"label": self._t("预计天数", "duration"), "value": f"{duration['recommended_days']}d"},
                    {"label": "MDE", "value": _fmt_pct((design.mde_rel or 0), 1) if design.mde_rel else _fmt_num(design.mde_abs, 2)},
                ],
            }
        ]
        from .stats.power import power_curve

        curve = power_curve(
            metric="proportion" if design.metric_kind == "proportion" else "mean",
            baseline=design.baseline,
        )
        chart_uri = charts.power_chart(curve, design.baseline)
        blocks.append({"type": "chart", "title": self._t("样本量曲线", "Sample size vs MDE"), "image": chart_uri})

        type_info = EXPERIMENT_TYPES[recommendation["primary"]]
        when_text = type_info.get("when_zh") if self.lang == "zh" else type_info["when"]
        lines = [
            self._t(
                f"**实验类型**：{type_info['name_zh']}（{recommendation['primary']}）——{when_text}",
                f"**Archetype**: {type_info['name_en']} ({recommendation['primary']}) - {when_text}",
            ),
            "",
            self._t(
                f"**样本量**：在基线 {design.baseline:.4g}、MDE {_fmt_pct(design.mde_rel) if design.mde_rel else _fmt_num(design.mde_abs,4)}、"
                f"α={design.alpha}、power={design.power} 下，每臂需要 **{sizing['n_control']:,}**，"
                f"合计 **{sizing['n_total']:,}**"
                + (f"（已按 {design.n_arms} 臂 + 聚类设计效应放大）" if design.n_arms > 2 or design.cluster_size > 1 else "（1:1 分流）")
                + "。",
                f"**Sizing**: baseline {design.baseline:.4g}, MDE {_fmt_pct(design.mde_rel) if design.mde_rel else _fmt_num(design.mde_abs,4)}, "
                f"alpha={design.alpha}, power={design.power} -> **{sizing['n_control']:,}** per arm, "
                f"**{sizing['n_total']:,}** total (allocation {design.allocation:.0%} applied).",
            ),
            self._t(
                f"**时效**：按每天 {duration['eligible_users_per_day']:,.0f} 个可分流入组，约 {duration['raw_days']} 天，"
                f"建议跑 **{duration['recommended_days']} 天**（覆盖完整自然周）。",
                f"**Duration**: at {duration['eligible_users_per_day']:,.0f} eligible users/day, ~{duration['raw_days']} days; "
                f"run **{duration['recommended_days']} days** to cover whole weeks.",
            ),
            "",
            self._t("**分析方案**", "**Analysis plan**"),
        ]
        lines += [f"- {p}" for p in result["analysis_plan"]]
        lines += ["", self._t("**护栏指标**", "**Guardrails**")] + [f"- {g}" for g in result["guardrails"]]
        lines += ["", self._t("**主要风险**", "**Risks**")] + [f"- {r}" for r in result["risks"][:4]]
        lines += ["", self._t("**上线判据**", "**Decision rule**"), result["decision_rule"]]
        if recommendation["secondary"]:
            lines += [
                "",
                self._t(
                    "**备选方案**：" + "、".join(f"{s}（{EXPERIMENT_TYPES[s]['name_zh']}）" for s in recommendation["secondary"]),
                    "**Alternatives**: " + ", ".join(f"{s} ({EXPERIMENT_TYPES[s]['name_en']})" for s in recommendation["secondary"]),
                ),
            ]

        text = "\n".join(lines)
        section = Section(
            title=self._t("实验设计", "Experiment design"),
            prose=result["markdown"].split("\n", 2)[-1],
            table=pd.DataFrame(
                [
                    [self._t("每臂样本", "per arm"), f"{sizing['n_control']:,}"],
                    [self._t("总样本", "total"), f"{sizing['n_total']:,}"],
                    [self._t("建议天数", "recommended days"), f"{duration['recommended_days']}"],
                    [self._t("随机化单元", "randomisation unit"), type_info["randomization_unit"]],
                ],
                columns=[self._t("项目", "item"), self._t("取值", "value")],
            ),
            chart=chart_uri,
        )
        self._record(f"Experiment design ({recommendation['primary']}): {sizing['n_total']:,} users, ~{duration['recommended_days']} days.", section)
        self.session["last_design"] = design

        return AgentReply(
            text=text,
            blocks=blocks,
            tool_calls=[
                {"tool": "recommend_experiment_type", "args": {"context": q[:80]}, "result": {"primary": recommendation["primary"]}},
                {"tool": "design_experiment", "args": design.__dict__, "result": {"n_total": sizing["n_total"], "days": duration["recommended_days"]}},
            ],
            data={"design": design.__dict__, "sizing": sizing, "duration": duration, "markdown": result["markdown"]},
        )

    def _parse_design_params(self, question: str, slots: Dict[str, Any]) -> Dict[str, Any]:
        q = question
        params: Dict[str, Any] = {}
        kind = "proportion"
        if re.search(r"时长|停留|使用时长|time spent|watch", q, re.I):
            kind = "mean"
        if re.search(r"人均|arpu|客单价|gmv|收入|revenue|每用户", q, re.I):
            kind = "mean"
        if re.search(r"订单数|次数|频次|count|订单量", q, re.I):
            kind = "count"
        if re.search(r"人均订单|每用户订单|orders per user", q, re.I):
            kind = "count"
        params["metric_kind"] = kind

        pcts = [(float(m.group(1)) / 100, m.start()) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", q)]
        baseline = None
        mde = None
        baseline_kw = ["基线", "基准", "当前", "现在", "目前", "baseline", "currently"]
        mde_kw = ["提升", "下降", "检测", "变化", "mde", "effect", "lift", "提高", "相对"]
        for value, pos in pcts:
            window = q[max(0, pos - 24): pos]
            # Pick the keyword that sits closest to the number: "基线3%，希望检测2%"
            # must not let "基线" claim the 2%.
            best_idx, best_kind = -1, None
            for kw in baseline_kw:
                idx = window.rfind(kw)
                if idx > best_idx:
                    best_idx, best_kind = idx, "baseline"
            for kw in mde_kw:
                idx = window.rfind(kw)
                if idx > best_idx:
                    best_idx, best_kind = idx, "mde"
            if best_kind == "baseline":
                baseline = value
            elif best_kind == "mde":
                mde = value
        if mde is not None and baseline is not None and mde == baseline:
            mde = None
        if mde is None and pcts:
            remaining = [v for v, _ in pcts if v != baseline]
            if remaining:
                candidate = remaining[0]
                mde = candidate if candidate < 0.5 else None
        if baseline is None:
            baseline = 0.03 if kind == "proportion" else (120.0 if kind == "mean" else 1.2)
        if mde is None:
            mde = 0.05 if kind == "proportion" else 0.03
        params["baseline"] = baseline
        params["mde_rel"] = mde

        traffic = re.search(r"(\d+(?:\.\d+)?)\s*(万|w|k)?\s*(?:日活|流量|用户|uv|dau|traffic)", q, re.I)
        if traffic:
            value = float(traffic.group(1))
            unit = (traffic.group(2) or "").lower()
            if unit in ("万", "w"):
                value *= 10_000
            elif unit == "k":
                value *= 1_000
            params["daily_traffic"] = value
        else:
            params["daily_traffic"] = 50_000.0

        if re.search(r"多组|多臂|a/b/n|三个版本|四个版本|several variants|multi-arm", q, re.I):
            params["n_arms"] = 3
        if re.search(r"1:2|1：2", q):
            params["allocation"] = 2.0
        if re.search(r"城市|地域|geo|cluster|聚类", q, re.I):
            params["cluster_size"] = 50.0
            params["icc"] = 0.02
        return params

    def _experiment_verdict(self, primary: Dict[str, Any], srm: Dict[str, Any]) -> str:
        """Turn the readout into a plain-language ship / no-ship call."""
        lift = primary["rel_lift"]
        direction_ok = (primary["abs_diff"] or 0) * (1 if primary["higher_is_better"] else -1) > 0
        if not srm["passed"]:
            return self._t(
                "**先别下结论**：SRM 未通过，分流或日志有问题，此时任何指标差异都不可信。",
                "**Do not conclude yet**: SRM failed, so randomisation or logging is broken and every metric difference is untrustworthy.",
            )
        if primary["significant_adjusted"] and direction_ok:
            return self._t(
                f"**可以上线**：{primary['metric']} 提升 {_fmt_pct(lift)}（BH 校正后 p={primary['p_adjusted']:.4f}，"
                f"置信区间不含 0），前提是护栏指标无回退、分群方向一致。",
                f"**Ship**: {primary['metric']} lifts {_fmt_pct(lift)} (BH-adjusted p={primary['p_adjusted']:.4f}, CI excludes 0), "
                "provided guardrails hold and segments agree in sign.",
            )
        if primary["significant_adjusted"]:
            return self._t(
                f"**不要上线**：{primary['metric']} 显著但方向为负（{_fmt_pct(lift)}），先定位原因再迭代。",
                f"**Do not ship**: {primary['metric']} is significant in the wrong direction ({_fmt_pct(lift)}). Find the cause first.",
            )
        return self._t(
            f"**结论不明确**：{primary['metric']} 变化 {_fmt_pct(lift)}（p={primary['p_value']:.3f}）。"
            "要么样本量不足以检测这么小的效应，要么效应确实接近 0；建议延长实验或用 CUPED 降方差后再看。",
            f"**Inconclusive**: {primary['metric']} moved {_fmt_pct(lift)} (p={primary['p_value']:.3f}). Either the test is "
            "underpowered for an effect this small, or the effect is genuinely ~0; extend the run or apply CUPED.",
        )

    def _handle_experiment_analysis(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        df = self.sql.run("SELECT * FROM experiment_users", max_rows=200000).dataframe
        metrics = [
            {"name": "conversion", "kind": "proportion", "numerator": "converted"},
            {"name": "revenue_per_user", "kind": "mean", "numerator": "revenue"},
            {"name": "revenue_per_visit", "kind": "ratio", "numerator": "revenue", "denominator": "visits"},
            {"name": "visits_per_user", "kind": "mean", "numerator": "visits", "higher_is_better": False},
        ]
        result = analyze_experiment(
            df, metrics, group_col="group", control="control", treatment="treatment",
            covariate_col="pre_revenue", segment_cols=("device",),
        )
        srm = result["srm"]
        primary = result["metrics"][0]
        # CUPED pays off on continuous metrics; report the metric where the
        # covariate actually buys variance reduction, not just the primary one.
        cuped_metric = max(
            (m for m in result["metrics"] if m.get("cuped")),
            key=lambda m: m["cuped"]["variance_reduction"],
            default=None,
        )
        best_cuped = cuped_metric["cuped"] if cuped_metric else None

        # Confidence validation: permutation cross-check on the primary metric
        # (exact for proportion/mean metrics; skipped for ratio estimands),
        # then the standardized check card.
        perm = None
        perm_chart = ""
        if primary["kind"] in ("proportion", "mean"):
            _perm_col = {"conversion": "converted", "revenue_per_user": "revenue", "visits_per_user": "visits"}
            col = _perm_col.get(primary["metric"])
            if col and col in df.columns:
                perm = permutation_test(
                    df[col].astype(float).to_numpy(),
                    (df["group"] == "treatment").astype(int).to_numpy(),
                    n_iter=500,
                )
                try:
                    perm_chart = charts.permutation_chart(perm["null"], perm["observed"])
                except Exception:  # noqa: BLE001
                    perm_chart = ""
        conf = experiment_confidence(srm, primary, best_cuped, cuped_metric, result["segments"], perm)

        blocks: List[Dict[str, Any]] = [
            {
                "type": "kpi",
                "items": [
                    {"label": "SRM", "value": "PASS" if srm["passed"] else "FAIL", "sub": f"p={srm['p_value']:.3f}", "good": srm["passed"]},
                    {
                        "label": self._t("主指标 lift", "primary lift"),
                        "value": _fmt_pct(primary["rel_lift"]),
                        "sub": f"p={primary['p_value']:.4f}",
                        "good": primary["rel_lift"] > 0,
                    },
                    {
                        "label": "CUPED",
                        "value": _fmt_pct(best_cuped["variance_reduction"], 0) if best_cuped else "-",
                        "sub": self._t("方差缩减（最佳指标）", "variance cut (best metric)"),
                    },
                ],
            }
        ]
        chart_uri = charts.effect_forest(result["metrics"])
        if chart_uri:
            blocks.append({"type": "chart", "title": self._t("各指标效应与置信区间", "Effects with 95% CI"), "image": chart_uri})

        rows = []
        for m in result["metrics"]:
            rows.append(
                [
                    m["metric"],
                    _fmt_num(m["control"], 4),
                    _fmt_num(m["treatment"], 4),
                    _fmt_pct(m["rel_lift"]),
                    f"[{_fmt_num(m['ci_abs'][0], 4)}, {_fmt_num(m['ci_abs'][1], 4)}]",
                    f"{m['p_value']:.4f}",
                    f"{m['p_adjusted']:.4f}",
                    self._t("显著", "yes") if m["significant_adjusted"] else self._t("不显著", "no"),
                ]
            )
        df_rows = pd.DataFrame(rows, columns=[self._t("指标", "metric"), "control", "treatment", "lift", "95% CI", "p", "p_BH", self._t("结论", "verdict")])
        blocks.append(self._table_block(df_rows, self._t("实验指标明细", "Metric readout"), pct_cols=("lift",)))

        seg_rows = [
            [s["segment_col"], s["segment"], _fmt_pct(s["rel_lift"]), f"{s['p_value']:.3f}", f"{s['n']:,}"]
            for s in result["segments"][:12]
        ]
        if seg_rows:
            seg_df = pd.DataFrame(seg_rows, columns=[self._t("维度", "dimension"), self._t("分群", "segment"), "lift", "p", "n"])
            blocks.append(self._table_block(seg_df, self._t("分群异质性（探索性）", "Segment heterogeneity (exploratory)")))

        if perm_chart:
            blocks.append(
                {
                    "type": "chart",
                    "title": self._t("置信度：置换检验零分布", "Confidence: permutation null"),
                    "image": perm_chart,
                }
            )
        blocks.append(
            self._table_block(
                confidence_table(conf["checks"], self.lang, conf["verdict"]),
                self._t("置信度验证", "Confidence validation"),
            )
        )

        # Two test batteries: the primary metric is a proportion (z / chi-square
        # / exact / SRM), while revenue is continuous (normality / variance /
        # Welch / rank / KS / bootstrap / permutation). Both are part of the
        # conclusion, so both are listed.
        panel = None
        try:
            ctrl = df[df["group"] == "control"]
            trt = df[df["group"] == "treatment"]
            panel_prop = test_panels.experiment_panel(
                ctrl["converted"].astype(float).to_numpy(), trt["converted"].astype(float).to_numpy(), kind="proportion"
            )
            panel_mean = test_panels.experiment_panel(
                ctrl["revenue"].astype(float).to_numpy(), trt["revenue"].astype(float).to_numpy(), kind="mean"
            )
            panel = {
                "tests": panel_prop["tests"] + panel_mean["tests"],
                "headline_zh": panel_prop["headline_zh"],
                "headline_en": panel_prop["headline_en"],
                "summary_zh": stat_tests.summary_line(panel_prop["tests"] + panel_mean["tests"], "zh", max_items=4),
                "summary_en": stat_tests.summary_line(panel_prop["tests"] + panel_mean["tests"], "en", max_items=4),
            }
            blocks.append(
                self._table_block(
                    stat_tests.to_frame(panel["tests"], self.lang),
                    self._t("统计检验（比例指标 + 金额指标）", "Statistical tests (proportion + revenue)"),
                    max_rows=18,
                )
            )
        except Exception:  # noqa: BLE001
            panel = None

        lines = [
            self._t(
                f"**SRM 检查**：{srm['message']}（observed={srm['observed']}, p={srm['p_value']:.3f}）。"
                + ("" if srm["passed"] else " **先别读指标，分流或日志有问题。**"),
                f"**SRM check**: {srm['message']} (observed={srm['observed']}, p={srm['p_value']:.3f})."
                + ("" if srm["passed"] else " **Do not read metrics until randomisation is fixed.**"),
            ),
            "",
            self._t(
                f"**主指标**（{primary['metric']}）：control {_fmt_num(primary['control'], 4)} vs treatment "
                f"{_fmt_num(primary['treatment'], 4)}，相对提升 **{_fmt_pct(primary['rel_lift'])}**，"
                f"95% CI [{_fmt_num(primary['ci_abs'][0], 4)}, {_fmt_num(primary['ci_abs'][1], 4)}]，"
                f"p={primary['p_value']:.4f}，BH 校正后 p={primary['p_adjusted']:.4f}。",
                f"**Primary metric** ({primary['metric']}): control {_fmt_num(primary['control'], 4)} vs treatment "
                f"{_fmt_num(primary['treatment'], 4)}, relative lift **{_fmt_pct(primary['rel_lift'])}**, "
                f"95% CI [{_fmt_num(primary['ci_abs'][0], 4)}, {_fmt_num(primary['ci_abs'][1], 4)}], "
                f"p={primary['p_value']:.4f}, BH-adjusted p={primary['p_adjusted']:.4f}.",
            ),
        ]
        if best_cuped and cuped_metric:
            lines.append(
                self._t(
                    f"**CUPED**（{cuped_metric['metric']}）：与实验前收入相关系数 {best_cuped['corr']:.2f}，"
                    f"方差缩减 {best_cuped['variance_reduction']:.0%}；校正后提升 {_fmt_pct(best_cuped['rel_lift'])}"
                    f"（p={best_cuped['p_value']:.4f}）。转化率这类 0/1 指标相关性天然低，CUPED 主要在金额类指标上省样本。",
                    f"**CUPED** ({cuped_metric['metric']}): corr with pre-period revenue {best_cuped['corr']:.2f}, "
                    f"variance cut {best_cuped['variance_reduction']:.0%}; adjusted lift {_fmt_pct(best_cuped['rel_lift'])} "
                    f"(p={best_cuped['p_value']:.4f}). Binary metrics have low correlation by nature - CUPED pays off on revenue-like metrics.",
                )
            )

        if perm is not None:
            lines.append(
                self._t(
                    f"**置换检验**：随机重排分组 500 次，观测差异落在零分布的 p={perm['p_perm']:.4f} 位置，"
                    f"与解析检验{'一致' if (perm['p_perm'] < 0.05) == primary['significant_adjusted'] else '不一致（需排查）'}。",
                    f"**Permutation test**: 500 label shuffles place the observed diff at p={perm['p_perm']:.4f} in the null, "
                    f"{'agreeing' if (perm['p_perm'] < 0.05) == primary['significant_adjusted'] else 'DISAGREEING'} with the analytic test.",
                )
            )
        if panel is not None:
            lines.append(
                self._t(
                    f"**统计检验**（检验方法 / 统计量 / p / 结论）：{panel['headline_zh']}{panel['summary_zh']}",
                    f"**Statistical tests** (method / statistic / p / conclusion): {panel['headline_en']} {panel['summary_en']}",
                )
            )
        lines.append(verdict_line(conf["verdict"], self.lang))

        verdict = self._experiment_verdict(primary, srm)
        lines += ["", self._t("**结论与建议**", "**Verdict**"), verdict]

        self._record(
            self._t(
                f"AB 实验：{primary['metric']} 提升 {_fmt_pct(primary['rel_lift'])}（p={primary['p_value']:.4f}），"
                f"SRM {'通过' if srm['passed'] else '未通过'}。",
                f"AB readout: {primary['metric']} lift {_fmt_pct(primary['rel_lift'])} (p={primary['p_value']:.4f}), "
                f"SRM {'pass' if srm['passed'] else 'fail'}.",
            ),
            Section(
                title=self._t("AB 实验结论", "AB test readout"),
                prose=self._t(
                    f"主指标 {primary['metric']}：control {_fmt_num(primary['control'], 4)} vs treatment "
                    f"{_fmt_num(primary['treatment'], 4)}，相对提升 {_fmt_pct(primary['rel_lift'])}"
                    f"（95% CI {_fmt_num(primary['ci_abs'][0], 4)} ~ {_fmt_num(primary['ci_abs'][1], 4)}，"
                    f"p={primary['p_value']:.4f}，BH 校正后 p={primary['p_adjusted']:.4f}）；"
                    f"SRM {'通过' if srm['passed'] else '未通过'}（p={srm['p_value']:.3f}）。",
                    result["narrative"],
                ),
                bullets=[verdict],
                table=df_rows,
                chart=chart_uri,
            ),
        )
        self._record(
            verdict_line(conf["verdict"], self.lang),
            Section(
                title=self._t("实验置信度验证", "Experiment confidence validation"),
                prose=verdict_line(conf["verdict"], self.lang)
                + " " + self._t(
                    "检查包括 SRM、BH 校正显著性、置换检验交叉验证、CUPED 方向一致性与分群反向显著。",
                    "Checks cover SRM, BH-adjusted significance, permutation cross-check, CUPED direction agreement and adverse segments.",
                ),
                table=confidence_table(conf["checks"], self.lang, conf["verdict"]),
                chart=perm_chart or None,
            ),
        )

        return AgentReply(
            text="\n".join(lines),
            blocks=blocks,
            tool_calls=[
                {"tool": "srm_check", "args": {}, "result": {"p_value": srm["p_value"], "passed": srm["passed"]}},
                {"tool": "analyze_experiment", "args": {"metrics": [m["name"] for m in metrics]}, "result": verdict},
            ],
            data={
                "readout": verdict,
                "metrics": result["metrics"],
                "confidence": {
                    "verdict": conf["verdict"],
                    "checks": [
                        {"name": c["name_zh"] if self.lang != "en" else c["name_en"], "status": c["status"]}
                        for c in conf["checks"]
                    ],
                },
            },
        )

    # ----------------------------------------------------------------- causal
    def _handle_causal(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        q = question
        method = "did"
        if re.search(r"合成控制|synthetic control|scm", q, re.I):
            method = "sc"
        elif re.search(r"倾向|psm|matching|ipw|aipw|匹配", q, re.I):
            method = "ps"

        if method == "ps":
            return self._causal_matching(q)
        if method == "sc":
            return self._causal_synthetic(q)
        return self._causal_did(q)

    def _causal_did(self, question: str) -> AgentReply:
        panel = self.sql.run("SELECT * FROM city_panel").dataframe
        did = difference_in_differences(panel, "gmv", "city", "week", "treated", "post")
        # Log specification: with a multiplicative trend the level ATT mixes the
        # treatment effect with group-size differences, the log version does not.
        did_log = difference_in_differences(panel, "gmv", "city", "week", "treated", "post", log_outcome=True)
        es = event_study(panel, "gmv", "city", "week", "treated", treat_time=26, max_lag=8, max_lead=6)
        chart_uri = charts.event_study_chart(es["coefficients"], title="Event study: policy effect by week")

        # Confidence validation: in-space placebo (permute the treated label
        # across cities and re-run TWFE) plus the standardized check card.
        placebo = did_placebo(panel, "gmv", "city", "week")
        conf = did_confidence(
            did, did_log, es, placebo, int(panel.loc[panel["treated"] == 1, "city"].nunique())
        )
        placebo_chart_uri = ""
        try:
            placebo_chart_uri = charts.placebo_chart(placebo["effects"], did["att"])
        except Exception:  # noqa: BLE001
            placebo_chart_uri = ""

        blocks = [
            {
                "type": "kpi",
                "items": [
                    {"label": "ATT (level)", "value": _fmt_num(did["att"]), "sub": f"p={did['p_value']:.4f}"},
                    {"label": self._t("相对效应 (log DiD)", "relative (log DiD)"), "value": _fmt_pct(did_log["relative_effect"]), "sub": self._t("推荐口径", "preferred")},
                    {"label": self._t("平行趋势", "parallel trends"), "value": "PASS" if es["parallel_trends_ok"] else "FAIL", "sub": f"p={es['pre_trend_joint_p']:.3f}", "good": es["parallel_trends_ok"]},
                ],
            },
            {"type": "chart", "title": self._t("事件研究系数", "Event study"), "image": chart_uri},
        ]
        if placebo_chart_uri:
            blocks.append(
                {
                    "type": "chart",
                    "title": self._t("置信度：安慰剂检验（假处理城市）", "Confidence: in-space placebo"),
                    "image": placebo_chart_uri,
                }
            )
        blocks.append(
            self._table_block(
                confidence_table(conf["checks"], self.lang, conf["verdict"]),
                self._t("置信度验证", "Confidence validation"),
            )
        )
        # DiD test panel: the joint F on pre-period leads is the identifying
        # assumption made testable; residual diagnostics decide whether the
        # cluster-robust SE is doing enough work.
        panel = None
        try:
            joint = stat_tests.wald_joint_f(
                es.get("pre_trend_joint_f", float("nan")),
                es["pre_trend_joint_p"],
                es.get("pre_trend_df", 0),
                did["n_units"],
            )
            panel = test_panels.regression_panel(did.get("diagnostics"), joint)
            blocks.append(
                self._table_block(
                    stat_tests.to_frame(panel["tests"], self.lang),
                    self._t("统计检验（平行趋势 + 残差诊断）", "Statistical tests (parallel trends + residual diagnostics)"),
                    max_rows=10,
                )
            )
        except Exception:  # noqa: BLE001
            panel = None
        rows = [[c["rel_period"], _fmt_num(c["coef"]), _fmt_num(c["se"]), f"{c['p_value']:.3f}"] for c in es["coefficients"]]
        df = pd.DataFrame(rows, columns=[self._t("相对期", "relative period"), "coef", "se", "p"])
        blocks.append(self._table_block(df, self._t("事件研究系数", "Event study coefficients"), max_rows=20))

        lines = [
            self._t(
                f"**双重差分（TWFE）**：政策对城市 GMV 的 ATT = **{_fmt_num(did['att'])}**，"
                f"95% CI [{_fmt_num(did['ci'][0])}, {_fmt_num(did['ci'][1])}]，p={did['p_value']:.4f}，"
                f"样本 {did['n_obs']:,} 个城市-周，{did['n_units']} 个城市。",
                f"**Two-way FE DiD**: ATT of the policy on city GMV = **{_fmt_num(did['att'])}**, "
                f"95% CI [{_fmt_num(did['ci'][0])}, {_fmt_num(did['ci'][1])}], p={did['p_value']:.4f}, "
                f"{did['n_obs']:,} city-weeks over {did['n_units']} cities.",
            ),
            "",
            self._t(
                f"**平行趋势检验**：{len([c for c in es['coefficients'] if c['rel_period'] < 0])} 个前置期联合检验 "
                f"p={es['pre_trend_joint_p']:.3f} → " + ("通过，DiD 可信。" if es["parallel_trends_ok"] else "未通过，ATT 有偏，建议改用合成控制/PSM。"),
                f"**Parallel trends**: joint test on pre-period leads p={es['pre_trend_joint_p']:.3f} -> "
                + ("passes, DiD is credible." if es["parallel_trends_ok"] else "fails, the ATT is biased; switch to synthetic control or PSM."),
            ),
            self._t(
                f"**口径提示**：ATT 为绝对量（约 {_fmt_num(did['att'])}，相对处理组反事实水平 {_fmt_pct(did['relative_effect'])}）。"
                f"趋势为乘法增长时，对数 DiD 更稳健，给出相对提升 **{_fmt_pct(did_log['relative_effect'])}**"
                f"（p={did_log['p_value']:.4f}）。两者差异就是“基数效应”，汇报时说明用了哪个口径。",
                f"**Note**: the ATT is absolute (~{_fmt_num(did['att'])}, i.e. {_fmt_pct(did['relative_effect'])} of the treated "
                f"counterfactual). With multiplicative growth the log specification is safer and gives a relative effect of "
                f"**{_fmt_pct(did_log['relative_effect'])}** (p={did_log['p_value']:.4f}). The gap between the two is the size effect - "
                "always state which scale you are reporting.",
            ),
        ]
        if placebo["n"] >= 20:
            tail = float(np.mean(np.abs(np.asarray(placebo["effects"])) >= abs(did["att"])))
            lines.append(
                self._t(
                    f"**安慰剂检验**：把处理标签随机换到 {placebo['n']} 组假处理城市重跑 DiD，"
                    f"真实 ATT 超过其中 {tail:.0%} 的安慰剂效应"
                    + ("——设计能区分政策与城市噪声。" if tail <= 0.05 else "——区分度不足，结论需谨慎。"),
                    f"**Placebo**: rerunning DiD with the treated label permuted onto {placebo['n']} fake city sets, "
                    f"the real ATT exceeds {tail:.0%} of placebo effects"
                    + (" - the design separates policy from city noise." if tail <= 0.05 else " - weak separation, be careful."),
                )
            )
        if panel is not None:
            lines.append(
                self._t(
                    f"**统计检验**（检验方法 / 统计量 / p / 结论）：{panel['headline_zh']}{panel['summary_zh']}",
                    f"**Statistical tests** (method / statistic / p / conclusion): {panel['headline_en']} {panel['summary_en']}",
                )
            )
        lines.append(verdict_line(conf["verdict"], self.lang))
        self._record(
            self._t(
                f"DiD：政策 ATT {_fmt_num(did['att'])}（p={did['p_value']:.3f}），相对效应 {_fmt_pct(did_log['relative_effect'])}；"
                f"平行趋势{'通过' if es['parallel_trends_ok'] else '不通过'}。",
                f"DiD ATT {_fmt_num(did['att'])} (p={did['p_value']:.3f}), relative {_fmt_pct(did_log['relative_effect'])}, "
                f"parallel trends {'ok' if es['parallel_trends_ok'] else 'violated'}.",
            ),
            Section(
                title=self._t("因果推断：双重差分", "Causal: difference-in-differences"),
                prose=self._t(
                    f"政策 ATT = {_fmt_num(did['att'])}（相对处理组反事实 {_fmt_pct(did['relative_effect'])}，"
                    f"p={did['p_value']:.4f}）；对数口径相对效应 {_fmt_pct(did_log['relative_effect'])}；"
                    f"平行趋势联合检验 p={es['pre_trend_joint_p']:.3f}，{'通过' if es['parallel_trends_ok'] else '未通过'}。",
                    did["narrative"],
                ),
                table=df,
                chart=chart_uri,
            ),
        )
        self._record(
            verdict_line(conf["verdict"], self.lang),
            Section(
                title=self._t("DiD 置信度验证", "DiD confidence validation"),
                prose=verdict_line(conf["verdict"], self.lang)
                + " " + self._t(
                    "检查包括平行趋势、前置期系数、安慰剂检验、口径一致性与处理组覆盖。",
                    "Checks cover parallel trends, pre-period leads, in-space placebo, specification agreement and treated coverage.",
                ),
                table=confidence_table(conf["checks"], self.lang, conf["verdict"]),
                chart=placebo_chart_uri or None,
            ),
        )
        return AgentReply(
            text="\n".join(lines),
            blocks=blocks,
            tool_calls=[
                {"tool": "difference_in_differences", "args": {"outcome": "gmv"}, "result": {"att": did["att"], "p": did["p_value"]}},
                {"tool": "event_study", "args": {"treat_time": 26}, "result": {"pre_trend_p": es["pre_trend_joint_p"]}},
            ],
            data={
                "did": {k: v for k, v in did.items() if k != "cells"},
                "event_study": es,
                "confidence": {
                    "verdict": conf["verdict"],
                    "checks": [
                        {"name": c["name_zh"] if self.lang != "en" else c["name_en"], "status": c["status"]}
                        for c in conf["checks"]
                    ],
                },
            },
        )

    def _causal_synthetic(self, question: str) -> AgentReply:
        panel = self.sql.run("SELECT * FROM city_panel").dataframe
        # Donor pool must exclude every treated unit, otherwise the synthetic
        # control absorbs part of the treatment and understates the effect.
        donors = sorted(panel.loc[panel["treated"] == 0, "city"].unique().tolist())
        sc = synthetic_control(
            panel, unit="city", time="week", outcome="gmv",
            treated_unit="city_00", treat_time=26, donors=donors,
        )
        chart_uri = charts.gap_chart(sc["gap_series"], sc["treat_time"], title="Synthetic control: city_00 vs synthetic")
        weights = pd.DataFrame(
            [[k, _fmt_num(v, 3)] for k, v in sorted(sc["weights"].items(), key=lambda kv: -kv[1])],
            columns=[self._t("捐赠城市", "donor"), self._t("权重", "weight")],
        )
        blocks = [
            {
                "type": "kpi",
                "items": [
                    {"label": self._t("平均处理效应", "average effect"), "value": _fmt_num(sc["post_avg_effect"]), "sub": self._t("政策后", "post period")},
                    {"label": self._t("拟合优度", "pre-fit R2"), "value": f"{sc['pre_fit_r2']:.3f}", "sub": f"RMSE {_fmt_num(sc['pre_rmse'])}"},
                    {"label": self._t("安慰剂 p", "placebo p"), "value": f"{sc['placebo_p_value']:.3f}", "sub": f"ratio {sc['post_pre_ratio']:.2f}", "good": sc["placebo_p_value"] < 0.1},
                ],
            },
            {"type": "chart", "title": self._t("实际 vs 合成", "Actual vs synthetic"), "image": chart_uri},
            self._table_block(weights, self._t("捐赠城市权重", "Donor weights"), max_rows=8),
        ]
        lines = [
            self._t(
                f"**合成控制**：捐赠池为 {sc['n_donors']} 个未受政策影响的城市（有效权重 {sc['n_effective_donors']} 个），"
                f"复现 city_00 的反事实。政策后平均缺口 **{_fmt_num(sc['post_avg_effect'])}**（相对 "
                f"{_fmt_pct(sc['relative_effect'])}），前置期拟合 RMSE {_fmt_num(sc['pre_rmse'])}（R²={sc['pre_fit_r2']:.3f}）。",
                f"**Synthetic control**: donor pool of {sc['n_donors']} untreated cities ({sc['n_effective_donors']} with weight) "
                f"reproduces city_00's counterfactual. Average post-period gap **{_fmt_num(sc['post_avg_effect'])}** "
                f"({_fmt_pct(sc['relative_effect'])} relative), pre-period RMSE {_fmt_num(sc['pre_rmse'])} "
                f"(R²={sc['pre_fit_r2']:.3f}).",
            ),
            self._t(
                f"**安慰剂检验**：post/pre RMSE 比值 {sc['post_pre_ratio']:.2f}，在 {sc['n_placebos']} 个安慰剂中的秩 "
                f"p={sc['placebo_p_value']:.3f}（该检验能取到的最小 p 为 {1 / (sc['n_placebos'] + 1):.3f}）→ "
                + ("效应显著强于随机分配。" if sc["placebo_p_value"] < 0.1 else "效应与安慰剂噪声同量级，不能下结论。"),
                f"**Placebo inference**: post/pre RMSE ratio {sc['post_pre_ratio']:.2f}, rank p={sc['placebo_p_value']:.3f} "
                f"across {sc['n_placebos']} placebos (floor is {1 / (sc['n_placebos'] + 1):.3f}) -> "
                + ("the effect beats random assignment." if sc["placebo_p_value"] < 0.1 else "indistinguishable from placebo noise."),
            ),
            self._t(
                "**注意**：合成控制没有标准误，靠安慰剂分布做推断；捐赠池越大结论越稳。",
                "**Caveat**: no standard error by construction; inference is rank-based on placebos, so a larger donor pool helps.",
            ),
        ]
        self._record(
            self._t(
                f"合成控制：政策后缺口 {_fmt_num(sc['post_avg_effect'])}（相对 {_fmt_pct(sc['relative_effect'])}），"
                f"安慰剂 p={sc['placebo_p_value']:.3f}。",
                f"Synthetic control effect {_fmt_num(sc['post_avg_effect'])} ({_fmt_pct(sc['relative_effect'])} relative), "
                f"placebo p={sc['placebo_p_value']:.3f}.",
            ),
            Section(
                title=self._t("因果推断：合成控制", "Causal: synthetic control"),
                prose=self._t(
                    f"政策后平均缺口 {_fmt_num(sc['post_avg_effect'])}（相对 {_fmt_pct(sc['relative_effect'])}），"
                    f"前置期拟合 R²={sc['pre_fit_r2']:.3f}，安慰剂 p={sc['placebo_p_value']:.3f}（捐赠池 "
                    f"{sc.get('n_donors', len(weights))} 个城市）。",
                    sc["narrative"],
                ),
                table=weights,
                chart=chart_uri,
            ),
        )
        return AgentReply(
            text="\n".join(lines),
            blocks=blocks,
            tool_calls=[{"tool": "synthetic_control", "args": {"treated_unit": "city_00", "treat_time": 26}, "result": {"effect": sc["post_avg_effect"], "placebo_p": sc["placebo_p_value"]}}],
            data={"synthetic_control": {k: v for k, v in sc.items() if k != "gap_series"}},
        )

    def _causal_matching(self, question: str) -> AgentReply:
        df = self.sql.run("SELECT * FROM promo_users", max_rows=200000).dataframe
        # log1p(pre_revenue) matters: the exposure mechanism is roughly linear in
        # log spend, so omitting it misspecifies the propensity model and biases IPW.
        df["log_pre_revenue"] = np.log1p(df["pre_revenue"])
        covariates = ["log_pre_revenue", "tenure_months", "sessions_30d"]
        naive = float(df.loc[df["promo"] == 1, "post_revenue"].mean() - df.loc[df["promo"] == 0, "post_revenue"].mean())
        estimates = compare_estimators(df, "promo", "post_revenue", covariates, n_bootstrap=120)
        atts = {k: v for k, v in estimates.items() if isinstance(v, dict) and "att" in v}
        ipw = estimates.get("ipw", {})
        aipw = estimates.get("aipw", {})

        rows = [["naive difference", _fmt_num(naive), "-", "-"]]
        for name, res in atts.items():
            rows.append([name.upper(), _fmt_num(res["att"]), f"[{_fmt_num(res['ci'][0])}, {_fmt_num(res['ci'][1])}]", f"{res['p_value']:.3f}"])
        if isinstance(ipw, dict) and "estimate" in ipw:
            rows.append(["IPW (ATE)", _fmt_num(ipw["estimate"]), f"[{_fmt_num(ipw['ci'][0])}, {_fmt_num(ipw['ci'][1])}]", f"{ipw['p_value']:.3f}"])
        if isinstance(aipw, dict) and "estimate" in aipw:
            rows.append(["AIPW (ATE)", _fmt_num(aipw["estimate"]), f"[{_fmt_num(aipw['ci'][0])}, {_fmt_num(aipw['ci'][1])}]", f"{aipw['p_value']:.3f}"])
        df_rows = pd.DataFrame(rows, columns=[self._t("估计方法", "estimator"), self._t("效应", "effect"), "95% CI", "p"])

        balance = pd.DataFrame(
            [
                [
                    b["covariate"],
                    _fmt_num(b["smd_before"], 3),
                    _fmt_num(b["smd_after"], 3),
                    f"{b['p_after']:.3f}" if b.get("p_after") is not None and np.isfinite(b.get("p_after", np.nan)) else "-",
                    "OK" if b["balanced"] else "IMBALANCED",
                ]
                for b in atts.get("psm", {}).get("balance", [])
            ],
            columns=[self._t("协变量", "covariate"), "SMD before", "SMD after", "p (after)", self._t("状态", "status")],
        )
        blocks = [
            {
                "type": "kpi",
                "items": [
                    {"label": self._t("朴素差异", "naive diff"), "value": _fmt_num(naive), "sub": self._t("有选择偏差", "confounded")},
                    {"label": "PSM ATT", "value": _fmt_num(atts.get("psm", {}).get("att")), "sub": f"pairs {atts.get('psm', {}).get('n_matched_pairs', '-')}"},
                    {"label": "AIPW ATE", "value": _fmt_num(aipw.get("estimate")), "sub": self._t("双稳健", "doubly robust")},
                ],
            },
            self._table_block(df_rows, self._t("多种估计方法对比", "Estimator comparison")),
        ]
        if len(balance):
            blocks.append(self._table_block(balance, self._t("协变量平衡性", "Covariate balance")))
        psm = atts.get("psm", {})
        # Confidence card: balance, overlap, estimator agreement, significance.
        conf = matching_confidence(psm, ipw if isinstance(ipw, dict) else {}, aipw if isinstance(aipw, dict) else {})
        if isinstance(psm.get("ps_overlap"), dict):
            try:
                blocks.append(
                    {
                        "type": "chart",
                        "title": self._t("倾向得分重叠性", "Propensity score overlap"),
                        "image": charts.ps_overlap_chart(psm["ps_overlap"]),
                    }
                )
            except Exception:  # noqa: BLE001 - a chart must never fail the answer
                pass
        blocks.append(
            self._table_block(
                confidence_table(conf["checks"], self.lang, conf["verdict"]),
                self._t("置信度验证", "Confidence validation"),
            )
        )

        # Test panel: balance tests on the matched sample plus a KS test on the
        # two propensity distributions (the overlap that matching needs).
        panel = None
        try:
            panel = test_panels.psm_panel(psm)
            blocks.append(
                self._table_block(
                    stat_tests.to_frame(panel["tests"], self.lang),
                    self._t("统计检验（匹配后平衡 + 重叠性）", "Statistical tests (post-match balance + overlap)"),
                    max_rows=8,
                )
            )
        except Exception:  # noqa: BLE001
            panel = None

        # Feature analysis AFTER the causal model: once the effect estimate is
        # validated, SHAP/permutation on outcome ~ covariates + treatment shows
        # which features drive the outcome — and how much the treatment adds
        # on top of them. Explanation follows identification, never precedes it.
        try:
            from .tools.modeling import feature_analysis as _feature_analysis
            from .tools.modeling import train_model as _train_model

            causal_df = df[[*covariates, "promo", "post_revenue"]].dropna()
            if len(causal_df) > 4000:
                causal_df = causal_df.sample(4000, random_state=0)
            cm_res = _train_model(causal_df[covariates + ["promo"]], causal_df["post_revenue"], task="regression")
            cm_fa = _feature_analysis(cm_res, max_features=8)
            engine = self._t("SHAP", "SHAP") if cm_fa["engine"] == "shap" else self._t("置换重要性", "permutation importance")
            blocks.append(
                {
                    "type": "chart",
                    "title": self._t(
                        f"因果模型之后的特征分析（{engine}，含处理变量 promo）",
                        f"Post-causal feature analysis ({engine}, treatment `promo` included)",
                    ),
                    "image": charts.importance_bar(cm_fa["importance"], title="Outcome drivers after causal estimation"),
                }
            )
        except Exception:  # noqa: BLE001 - explanation is additive, never blocking
            pass

        lines = [
            self._t(
                f"**为什么不能直接比均值**：促销曝光不是随机的（高价值用户更容易被选中），朴素差异 "
                f"{_fmt_num(naive)} 把选择偏差也算成了效果。",
                f"**Why the naive difference fails**: promo exposure is not random (high-value users are more likely to be targeted), "
                f"so the raw gap of {_fmt_num(naive)} mixes selection bias into the effect.",
            ),
            "",
            self._t(
                f"**倾向得分匹配**：匹配 {atts.get('psm', {}).get('n_matched_pairs', 0):,} 对，ATT = "
                f"{_fmt_num(atts.get('psm', {}).get('att'))}；**IPW** ATE = {_fmt_num(ipw.get('estimate'))}；"
                f"**AIPW** ATE = {_fmt_num(aipw.get('estimate'))}。",
                f"**PSM**: {atts.get('psm', {}).get('n_matched_pairs', 0):,} matched pairs, ATT = "
                f"{_fmt_num(atts.get('psm', {}).get('att'))}; **IPW** ATE = {_fmt_num(ipw.get('estimate'))}; "
                f"**AIPW** ATE = {_fmt_num(aipw.get('estimate'))}.",
            ),
            *(
                [
                    self._t(
                        f"**特征工程（建模前共线性处理）**：剔除与保留协变量高度共线的 "
                        f"{', '.join(psm['pruned_covariates'])}（成对 |r|>0.95），避免倾向得分模型病态——因果推断与回归共用同一套预处理。",
                        f"**Feature engineering (pre-model collinearity handling)**: dropped {', '.join(psm['pruned_covariates'])} "
                        f"(pairwise |r|>0.95 with a retained covariate) to keep the propensity model well-conditioned — "
                        f"causal inference reuses the exact same preprocessing as regression.",
                    )
                ]
                if psm.get("pruned_covariates")
                else []
            ),
            *(
                [
                    self._t(
                        f"**重叠性诊断**：共同支撑区间 [{support['low']:.2f}, {support['high']:.2f}]，"
                        f"{support['n_treated_outside']} 个处理组个体落在区间外（{support['share_outside']:.1%}）"
                        + (
                            "——这部分用户的效应无法由匹配识别，需谨慎外推。"
                            if support["n_treated_outside"]
                            else "，重叠良好。"
                        ),
                        f"**Overlap**: common support [{support['low']:.2f}, {support['high']:.2f}], "
                        f"{support['n_treated_outside']} treated units outside ({support['share_outside']:.1%})"
                        + (" — their effect is not identified by matching." if support["n_treated_outside"] else " — overlap is healthy."),
                    )
                    for support in [psm.get("common_support", {})]
                    if support
                ]
            ),
            self._t(
                "**稳健性**：三种方法给出的效应量级是否一致，比任何单一方法的 p 值都更值得相信。"
                "如果 PSM 与 AIPW 相差很大，通常是重叠性（overlap）不足。",
                "**Robustness**: agreement across PSM/IPW/AIPW is more convincing than any single p-value. "
                "Large disagreement usually signals poor overlap.",
            ),
            "",
            self._t("**识别假设**：条件可忽略性（给定协变量，曝光近似随机）+ 重叠性。这两个假设无法用数据检验，只能靠设计论证。",
                    "**Identifying assumptions**: conditional ignorability plus overlap. Neither is testable from data; argue them from design."),
        ]
        if panel is not None:
            lines.append(
                self._t(
                    f"**统计检验**（检验方法 / 统计量 / p / 结论）：{panel['headline_zh']}{panel['summary_zh']}",
                    f"**Statistical tests** (method / statistic / p / conclusion): {panel['headline_en']} {panel['summary_en']}",
                )
            )
        lines.append(verdict_line(conf["verdict"], self.lang))
        self._record(
            self._t(
                f"匹配估计：朴素差异 {_fmt_num(naive)}，PSM {_fmt_num(atts.get('psm', {}).get('att'))}，"
                f"AIPW {_fmt_num(aipw.get('estimate'))}（选择偏差已剔除）。",
                f"Causal matching: naive {_fmt_num(naive)} vs PSM {_fmt_num(atts.get('psm', {}).get('att'))} "
                f"vs AIPW {_fmt_num(aipw.get('estimate'))}.",
            ),
            Section(
                title=self._t("因果推断：匹配与加权", "Causal: matching and weighting"),
                prose=self._t(
                    f"朴素差异 {_fmt_num(naive)} 混入了选择偏差；PSM ATT = {_fmt_num(atts.get('psm', {}).get('att'))}"
                    f"（{atts.get('psm', {}).get('n_matched_pairs', 0):,} 对匹配），IPW ATE = {_fmt_num(ipw.get('estimate'))}，"
                    f"AIPW ATE = {_fmt_num(aipw.get('estimate'))}。",
                    atts.get("psm", {}).get("narrative", ""),
                ),
                table=df_rows,
            ),
        )
        self._record(
            verdict_line(conf["verdict"], self.lang),
            Section(
                title=self._t("匹配估计置信度验证", "Matching confidence validation"),
                prose=verdict_line(conf["verdict"], self.lang)
                + " " + self._t(
                    "检查包括协变量平衡、共同支撑、方法一致性与显著性；隐性混淆需敏感性分析补充。",
                    "Checks cover covariate balance, common support, estimator agreement and significance; hidden confounding needs a sensitivity pass.",
                ),
                table=confidence_table(conf["checks"], self.lang, conf["verdict"]),
            ),
        )
        return AgentReply(
            text="\n".join(lines),
            blocks=blocks,
            tool_calls=[{"tool": "compare_estimators", "args": {"treatment": "promo", "outcome": "post_revenue"}, "result": {k: v for k, v in estimates.items() if k == "summary"}}],
            data={"naive": naive, "estimates": {k: v for k, v in estimates.items()}},
        )

    # -------------------------------------------------- open-source cross-validation
    def _causal_oss(self, question: str, slots: Dict[str, Any] = None) -> AgentReply:
        """Re-estimate the promo effect with open-source engines and refute it.

        The point is *independent reproduction*: the same identification
        problem is handed to DoWhy (and, when installed, EconML's causal
        forest), its refutation suite runs, and the results are compared
        against the in-house estimators from ``_causal_matching``.
        """
        if not HAS_DOWHY:
            return AgentReply(
                text=self._t(
                    "本环境未安装 DoWhy。运行 `pip install dowhy` 后重试；当前可用引擎：" + str(available_engines()),
                    "DoWhy is not installed here. Run `pip install dowhy` and retry; available engines: " + str(available_engines()),
                )
            )

        df = self.sql.run("SELECT * FROM promo_users", max_rows=200000).dataframe
        df["log_pre_revenue"] = np.log1p(df["pre_revenue"])
        covariates = ["log_pre_revenue", "tenure_months", "sessions_30d"]

        # In-house reference: the matching handler's headline PSM estimate.
        inhouse = compare_estimators(df, "promo", "post_revenue", covariates, n_bootstrap=60)
        inhouse_psm = inhouse.get("psm", {})

        engines: List[Dict[str, Any]] = []
        ih_att = float(inhouse_psm.get("att", float("nan")))
        engines.append({"engine": "in-house PSM", "estimate": ih_att, "p": inhouse_psm.get("p_value")})

        oss = dowhy_estimate(df, "promo", "post_revenue", covariates, method="linear_regression")
        engines.append({"engine": "DoWhy linear", "estimate": oss["estimate"], "p": oss["p_value"]})
        try:
            oss_psm = dowhy_estimate(df, "promo", "post_revenue", covariates, method="psm")
            engines.append({"engine": "DoWhy PSM", "estimate": oss_psm["estimate"], "p": oss_psm["p_value"]})
        except Exception:  # noqa: BLE001 - PSM refusals (e.g. degenerate PS) must not kill the answer
            oss_psm = None

        refutations = dowhy_refutations(oss, num_simulations=20)
        conf = oss_confidence({"att": ih_att}, oss, refutations)

        rows = pd.DataFrame(
            [
                {
                    "engine": e["engine"],
                    self._t("效应", "effect"): _fmt_num(e["estimate"]),
                    "p": "-" if e["p"] != e["p"] else f"{e['p']:.4f}",
                }
                for e in engines
            ]
        )
        ref_rows = pd.DataFrame(
            [
                {
                    self._t("反驳检验", "refuter"): r["refuter"],
                    self._t("新效应", "new effect"): _fmt_num(r["new_effect"]),
                    self._t("结论", "verdict"): self._t("通过", "passed") if r["passed"] else self._t("未通过", "failed"),
                }
                for r in refutations
            ]
        )

        lines = [
            self._t(
                f"**开源引擎交叉验证**：自研 PSM ATT = {_fmt_num(ih_att)}；DoWhy 线性回归 = {_fmt_num(oss['estimate'])}"
                f"（p={oss['p_value']:.4f}）"
                + (f"，DoWhy PSM = {_fmt_num(oss_psm['estimate'])}。" if oss_psm else "。"),
                f"**Open-source cross-validation**: in-house PSM ATT = {_fmt_num(ih_att)}; DoWhy linear = "
                f"{_fmt_num(oss['estimate'])} (p={oss['p_value']:.4f})"
                + (f", DoWhy PSM = {_fmt_num(oss_psm['estimate'])}." if oss_psm else "."),
            ),
            verdict_line(conf["verdict"], self.lang),
        ]

        blocks: List[Dict[str, Any]] = [
            self._table_block(rows, self._t("跨引擎估计对比", "Cross-engine estimates")),
            {"type": "chart", "title": self._t("跨引擎效应对比", "Effect by engine"), "image": charts.engine_comparison_chart(engines)},
            self._table_block(ref_rows, self._t("DoWhy 反驳检验", "DoWhy refutation tests")),
            self._table_block(confidence_table(conf["checks"], self.lang, conf["verdict"]), self._t("置信度验证", "Confidence validation")),
        ]
        self._record(
            verdict_line(conf["verdict"], self.lang),
            Section(
                title=self._t("开源引擎交叉验证（DoWhy）", "Open-source cross-validation (DoWhy)"),
                prose=verdict_line(conf["verdict"], self.lang)
                + " "
                + self._t(
                    "同一识别问题交给 DoWhy 重估并运行安慰剂处理/随机共同原因反驳检验；自研与开源引擎一致是结果可信的重要证据。",
                    "The same identification problem is re-estimated by DoWhy with placebo/random-common-cause refutations; agreement across independent engines is key evidence.",
                ),
                table=rows,
            ),
        )
        return AgentReply(
            text="\n".join(lines),
            blocks=blocks,
            tool_calls=[{"tool": "dowhy_estimate", "args": {"method": "linear_regression"}, "result": {"estimate": oss["estimate"], "p": oss["p_value"]}}],
            data={"engines": engines, "refutations": refutations, "confidence": conf},
        )

    # ------------------------------------------------------ deep-learning anomaly
    def _handle_dl_anomaly(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """Autoencoder / isolation-forest anomaly detection with a confidence card."""
        table = semantic.default_table()
        metric: Metric = slots.get("metric") or table.metrics["gmv"]
        anchor: date = slots["anchor"]
        daily = self._series(metric, table, anchor - timedelta(days=89), anchor)
        values = daily["value"].to_numpy(dtype=float)
        dates = daily["date"]

        use_iso = bool(re.search(r"孤立森林|isolation", question, re.I))
        seasonal_flagged = None
        try:
            # Seasonal baseline runs alongside for the cross-check: agreement
            # between a learned detector and a classical one is real evidence.
            det = AnomalyDetector(AnomalyConfig(baseline="trailing"))
            seasonal = det.detect(daily["date"], daily["value"], label=metric.name)
            seasonal_flagged = seasonal["frame"]["flagged"].to_numpy(dtype=bool)
        except Exception:  # noqa: BLE001 - the cross-check is optional
            seasonal_flagged = None

        if use_iso:
            result = isolation_forest_scores(values)
            name = self._t("孤立森林异常检测", "Isolation-forest anomaly detection")
        else:
            result = autoencoder_scores(values, window=7)
            name = self._t("自编码器异常检测", "Autoencoder anomaly detection")
        conf = dl_anomaly_confidence(result, seasonal_flagged)

        frame = dl_anomaly_table(dates, result, self.lang)
        chart_uri = charts.dl_reconstruction_chart(result, dates, title=name)
        flagged_days = frame[frame[frame.columns[-1]] == True]  # noqa: E712 - bool column
        recent = flagged_days.tail(6)

        lines = [
            f"**{name}**（{result['backend']}）",
            self._t(
                f"检出 {int(result['flagged'].sum())} 个异常窗口"
                + (f"，最近的有：{', '.join(str(d)[:10] for d in recent[recent.columns[0]])}。" if len(recent) else "。"),
                f"{int(result['flagged'].sum())} anomalous windows"
                + (f"; most recent: {', '.join(str(d)[:10] for d in recent[recent.columns[0]])}." if len(recent) else "."),
            ),
            verdict_line(conf["verdict"], self.lang),
        ]
        blocks: List[Dict[str, Any]] = [
            {"type": "chart", "title": name, "image": chart_uri},
            self._table_block(confidence_table(conf["checks"], self.lang, conf["verdict"]), self._t("置信度验证", "Confidence validation")),
        ]
        if len(recent):
            blocks.insert(1, self._table_block(recent, self._t("最近的异常", "Most recent anomalies")))

        self._record(
            verdict_line(conf["verdict"], self.lang),
            Section(
                title=name,
                prose=verdict_line(conf["verdict"], self.lang)
                + " "
                + self._t(
                    "自编码器学习\"正常形态\"并用重建误差打分，孤立森林在多变量特征上隔离异常点；与经典季节基线交叉验证后给出置信度。",
                    "The autoencoder learns the normal shape and scores reconstruction error; isolation forests isolate anomalies in multivariate features; cross-checked against the classical seasonal baseline.",
                ),
                table=frame.tail(10),
            ),
        )
        return AgentReply(
            text="\n".join(lines),
            blocks=blocks,
            tool_calls=[{"tool": result["backend"], "args": {}, "result": {"n_flagged": int(result["flagged"].sum()), "threshold": result.get("threshold")}}],
            data={"dl_anomaly": {"backend": result["backend"], "flagged": result["flagged"].tolist(), "error": result["error"].tolist()}, "confidence": conf},
        )

    # ----------------------------------------------------------------- report
    def _handle_sampling(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """Sampling surveys: sizing, drawing, representativeness.

        Three modes, all grounded in the same theory (sampling distribution of
        the mean, CLT, standard error): "how many do I need" inverts the power
        equation; "draw me a sample" runs a reproducible SRS/systematic/
        stratified draw; "is it representative" tests the sample against the
        population with chi-square GOF (categorical) or Welch t + KS (numeric).
        """
        from .stats import sampling as smp

        q = question.lower()
        blocks: List[Dict[str, Any]] = []
        wants_draw = bool(re.search(r"抽样|抽取|抽\s*\d+|抽个|draw.{0,10}sample|sample.{0,6}of", q, re.I))
        wants_size = bool(re.search(r"样本量|样本大小|多少人|抽多少|需要多少|多大.{0,4}样本|sample size|how (many|large)", q, re.I))

        try:
            # ---------------------------------------------------- 1. sizing mode
            if wants_size and not wants_draw:
                # Mean mode: the question talks about a continuous metric
                # (standard deviation / sigma). MDE stays in original units
                # ("提升5元" -> 5), NOT divided by 100.
                mean_mode = bool(re.search(r"标准差|std\.?\s*dev|\bstd\b|\bsd\b|sigma|σ", q, re.I))
                noted_default = True
                if mean_mode:
                    sd_match = re.search(
                        r"标准差[^0-9]*(\d+(?:\.\d+)?)|std(?:\s*dev)?[^0-9]*(\d+(?:\.\d+)?)|\bsd\b[^0-9]*(\d+(?:\.\d+)?)",
                        q, re.I,
                    )
                    sd = 1.0
                    if sd_match:
                        sd = float(next(g for g in sd_match.groups() if g))
                    mde_match = re.search(
                        r"(?:检测|变化|差异|提升|下降|改变|detect|change|shift|difference)[^0-9]*(\d+(?:\.\d+)?)",
                        q, re.I,
                    )
                    mde = 1.0
                    if mde_match:
                        mde = float(mde_match.group(1))
                        noted_default = False
                    size = smp.sample_size_mean(sd, mde)
                else:
                    base_match = re.search(r"基线[^0-9%]*(\d+(?:\.\d+)?)\s*%|baseline[^0-9%]*(\d+(?:\.\d+)?)\s*%", q, re.I)
                    p0 = 0.70
                    if base_match:
                        p0 = float(base_match.group(1) or base_match.group(2)) / 100.0
                    rel = bool(re.search(r"相对|relative", q, re.I))
                    mde_match = re.search(
                        r"(?:检测|变化|差异|提升|下降|detect|change|shift|difference)[^0-9%]*(\d+(?:\.\d+)?)\s*(?:个)?\s*(?:百分点|pp|percentage points?)?",
                        q,
                        re.I,
                    )
                    mde = 0.03
                    noted_default = True
                    if mde_match:
                        raw = float(mde_match.group(1)) / 100.0
                        mde = p0 * raw if rel else raw
                        noted_default = False
                    size = smp.sample_size_proportion(p0, mde)

                # Sensitivity row: how n reacts to a tighter/looser MDE.
                sens_rows = []
                for factor in (0.5, 0.75, 1.0, 1.5, 2.0):
                    m = mde * factor
                    if size["kind"] == "proportion":
                        s = smp.sample_size_proportion(p0, m, alpha=size["alpha"], power=size["power"])
                    else:
                        s = smp.sample_size_mean(size["sd"], m, alpha=size["alpha"], power=size["power"])
                    sens_rows.append([f"δ={m:.4g}", f"{s['n_per_group']:,}"])
                default_note = (
                    self._t("（未识别到具体数字，以下用示例参数演示）", " (no numbers found; demo parameters shown)")
                    if noted_default
                    else ""
                )
                if size["kind"] == "mean":
                    text = self._t(
                        f"**样本量计算**{default_note}：连续指标（标准差 σ={size['sd']:g}），"
                        f"检测变化 δ={size['mde']:g}（α={size['alpha']}，功效 {size['power']:.0%}），"
                        f"每组需要 **{size['n_per_group']:,}** 个样本，两组共 {size['n_total']:,}。",
                        f"**Sample size**{default_note}: continuous metric (sd σ={size['sd']:g}), detect a shift of "
                        f"δ={size['mde']:g} (α={size['alpha']}, power {size['power']:.0%}) — "
                        f"**{size['n_per_group']:,}** per group, {size['n_total']:,} in total.",
                    )
                else:
                    text = self._t(
                        f"**样本量计算**{default_note}：基线 {size.get('p0', 0) * 100:.1f}%，"
                        f"检测绝对变化 {size['mde'] * 100:.2f} 个百分点（α={size['alpha']}，功效 {size['power']:.0%}），"
                        f"每组需要 **{size['n_per_group']:,}** 个样本，两组共 {size['n_total']:,}。",
                        f"**Sample size**{default_note}: baseline {size.get('p0', 0) * 100:.1f}%, detect an absolute change of "
                        f"{size['mde'] * 100:.2f}pp (α={size['alpha']}, power {size['power']:.0%}) — "
                        f"**{size['n_per_group']:,}** per group, {size['n_total']:,} in total.",
                    )
                text += "\n\n" + self._t(
                    f"**理论依据**：{size['formula']}，其中 z_{{1-α/2}}={size['z_alpha']}、z_{{1-β}}={size['z_power']}。"
                    "依据是中心极限定理：样本均值近似服从正态分布，标准误 σ/√n 随样本量平方根收缩，"
                    "所以精度翻倍（δ 减半）需要 4 倍样本。参考：Cochran, Sampling Techniques (1977)。",
                    f"**Theory**: {size['formula']} with z_{{1-α/2}}={size['z_alpha']}, z_{{1-β}}={size['z_power']}. "
                    "By the CLT the sample mean is approximately normal with standard error σ/√n, which shrinks "
                    "with the square root of n — halving the detectable effect costs 4x the sample. "
                    "Reference: Cochran, Sampling Techniques (1977).",
                )
                blocks.append(
                    {
                        "type": "kpi",
                        "items": [
                            {
                                "label": self._t("每组样本量", "n per group"),
                                "value": f"{size['n_per_group']:,}",
                                "sub": f"α={size['alpha']}, power={size['power']:.0%}",
                            },
                            {
                                "label": self._t("两组合计", "both groups"),
                                "value": f"{size['n_total']:,}",
                                "sub": self._t("按 1:1 分流", "1:1 allocation"),
                            },
                            {
                                "label": self._t("可检测效应", "detectable δ"),
                                "value": f"{size['mde'] * 100:.2f}pp",
                                "sub": self._t("双侧检验", "two-sided"),
                            },
                        ],
                    }
                )
                blocks.append(
                    self._table_block(
                        pd.DataFrame(sens_rows, columns=[self._t("检测效应", "effect size"), self._t("每组样本量", "n per group")]),
                        self._t("样本量对效应大小的敏感性（精度翻倍 = 样本 ×4）", "Sample size vs effect size (2x precision = 4x n)"),
                    )
                )
                return AgentReply(
                    text=text,
                    blocks=blocks,
                    tool_calls=[{"tool": "sample_size", "args": {"p0": size.get("p0"), "mde": size["mde"]}, "result": {"n_per_group": size["n_per_group"]}}],
                    data={"sampling": "size"},
                )

            # --------------------------------- 2. draw + representativeness mode
            pop = self.sql.run("SELECT user_id, channel, device, spend_30d, sessions_30d FROM uplift_users", max_rows=100000).dataframe
            n_request = 2000
            n_match = re.search(r"抽\s*(\d{2,6})|sample.{0,6}of\s*(\d{2,6})|(\d{2,6})\s*(?:个样本|rows)", q, re.I)
            if n_match:
                n_request = int(n_match.group(1) or n_match.group(2) or n_match.group(3))
            method = "stratified" if re.search(r"分层|stratif", q, re.I) else ("systematic" if re.search(r"系统抽样|systematic", q, re.I) else "srs")
            sample, draw_log = smp.draw_sample(pop, n_request, method=method, by="channel", seed=7)

            se_info = smp.standard_error(sample["spend_30d"])
            rep_cat = smp.representativeness(sample["channel"], pop["channel"])
            rep_num = smp.representativeness(sample["spend_30d"], pop["spend_30d"])
            all_tests = list(rep_cat["tests"]) + list(rep_num["tests"])

            method_label = {"srs": self._t("简单随机抽样", "simple random"), "systematic": self._t("系统抽样", "systematic"), "stratified": self._t("分层抽样（按渠道等比例）", "stratified (proportional by channel)")}[method]
            headline = self._t(
                f"**抽样完成**：{method_label}，从 {draw_log['population']:,} 个用户中抽出 {draw_log['fraction']:.2%}（n={len(sample):,}，seed={draw_log['seed']}，可复现）。"
                f"样本人均消费 {se_info['mean']:.2f}，标准误 {se_info['se']:.3f}，95% CI [{se_info['ci95'][0]:.2f}, {se_info['ci95'][1]:.2f}]。",
                f"**Sample drawn**: {method_label}, {len(sample):,} of {draw_log['population']:,} users ({draw_log['fraction']:.2%}, seed={draw_log['seed']}, reproducible). "
                f"Sample mean spend {se_info['mean']:.2f}, SE {se_info['se']:.3f}, 95% CI [{se_info['ci95'][0]:.2f}, {se_info['ci95'][1]:.2f}].",
            )
            headline += "\n\n" + self._t(
                "**理论依据**：中心极限定理保证大样本下样本均值近似正态，标准误 SE=s/√n；"
                "区间估计与代表性检验都建立在这个抽样分布上。",
                "**Theory**: the CLT makes the sample mean approximately normal with SE=s/√n; both the confidence "
                "interval and the representativeness tests below read off that sampling distribution.",
            )
            blocks.append(
                {
                    "type": "kpi",
                    "items": [
                        {"label": self._t("样本量 / 总体", "n / population"), "value": f"{len(sample):,}/{draw_log['population']:,}", "sub": f"{draw_log['fraction']:.2%} " + self._t("抽样比", "fraction")},
                        {"label": self._t("样本均值", "sample mean"), "value": f"{se_info['mean']:.2f}", "sub": f"SE={se_info['se']:.3f}"},
                        {"label": "95% CI", "value": f"[{se_info['ci95'][0]:.1f}, {se_info['ci95'][1]:.1f}]", "sub": "t(1-α/2)·SE"},
                    ],
                }
            )
            if method == "stratified":
                blocks.append(
                    self._table_block(
                        pd.DataFrame(
                            [
                                {"channel": k, "population": f"{v['population']:,}", "sampled": f"{v['sampled']:,}", "rate": f"{v['sampled'] / v['population']:.2%}"}
                                for k, v in draw_log["allocation"].items()
                            ]
                        ),
                        self._t("分层配额（等比例分配，小渠道保底 1 个）", "Stratum allocation (proportional, small strata keep ≥1)"),
                    )
                )
            blocks.append(
                self._table_block(
                    stat_tests.to_frame(all_tests, self.lang),
                    self._t("代表性检验（样本 vs 总体）", "Representativeness tests (sample vs population)"),
                    max_rows=6,
                )
            )
            share_rows = [
                {"channel": r["category"], "sample": f"{r['sample_share']:.2%}", "population": f"{r['population_share']:.2%}", "diff": f"{r['diff']:+.2%}"}
                for r in rep_cat.get("shares", [])[:5]
            ]
            if share_rows:
                blocks.append(self._table_block(pd.DataFrame(share_rows), self._t("渠道结构对比", "Channel mix: sample vs population")))
            agree = stat_tests.agreement(all_tests, [t["key"] for t in all_tests if t["status"] == "ok"])
            verdict = (
                self._t("各项检验一致：样本与总体分布无显著差异，代表性成立。", "All tests agree: the sample is statistically indistinguishable from the population — representative.")
                if agree is True
                else (
                    self._t("检验结论不一致或部分显著：建议检查抽样框与分层变量后再下结论。", "Tests disagree or some reject: revisit the sampling frame and stratification before concluding.")
                    if agree is False
                    else ""
                )
            )
            text = headline + ("\n\n" + self._t(f"**统计检验**：", "**Statistical tests**: ") + stat_tests.summary_line(all_tests, self.lang) if all_tests else "")
            if verdict:
                text += "\n\n" + verdict
            return AgentReply(
                text=text,
                blocks=blocks,
                tool_calls=[{"tool": "draw_sample", "args": {"n": len(sample), "method": method, "by": "channel"}, "result": {"tests": len(all_tests)}}],
                data={"sampling": "draw", "tests": [t["key"] for t in all_tests]},
            )
        except Exception as exc:  # noqa: BLE001 - surfaced with a next-step hint
            return AgentReply(
                text=self._t(
                    f"抽样分析没能完成：`{exc}`\n\n可以说得更具体，比如「满意度调查基线 70%，检测 3 个百分点变化，需要多少样本」或「从用户表分层抽样 2000 个并检验代表性」。",
                    f"Sampling analysis failed: `{exc}`\n\nTry e.g. \"survey with baseline 70%, detect a 3pp change - how many samples?\" "
                    f"or \"draw a stratified sample of 2000 users and test representativeness\".",
                )
            )

    def _handle_modeling(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """Train a classification or regression model on demo data + explain it.

        Classification: user conversion from behaviour features (uplift_users).
        Regression: daily GMV from calendar + traffic features (orders_daily).
        Task auto-selects from wording; every step is data the tools can repeat.
        """
        from .tools.modeling import engineer_features, feature_analysis, modeling_confidence, train_model

        q = question.lower()
        wants_classification = bool(re.search(r"分类|转化|流失|classification|churn|convert", q))
        wants_regression = bool(re.search(r"回归|预测.{0,6}(gmv|销量|值)|regression|forecast value", q))
        if wants_classification == wants_regression:  # both or neither: pick by data fit
            wants_classification, wants_regression = True, False

        blocks: List[Dict[str, Any]] = []
        try:
            if wants_classification:
                df = self.sql.run(
                    "SELECT tenure_days, spend_30d, sessions_30d, channel, device, converted "
                    "FROM uplift_users LIMIT 8000",
                    max_rows=100000,
                ).dataframe
                target = "converted"
                title = self._t("用户转化分类模型", "Conversion classification model")
            else:
                df = self.sql.run(
                    "SELECT date, SUM(gmv) AS gmv, SUM(sessions) AS sessions, SUM(orders) AS orders "
                    "FROM orders_daily GROUP BY 1 ORDER BY 1",
                    max_rows=100000,
                ).dataframe
                target = "gmv"
                title = self._t("每日 GMV 回归模型", "Daily GMV regression model")

            X, y, feature_names, fe_log = engineer_features(df, target=target, date_column=None if wants_classification else "date")

            # Stage reporting: preprocessing and feature engineering are shown
            # as first-class steps (they decide what the model may even see).
            pre_log = fe_log.get("preprocess", {})
            col_log = fe_log.get("collinearity", {})
            # Stage 1 always reports: the missing-value profile proves the
            # check ran, and the action column shows exactly what was done.
            try:
                from .tools.preprocess import missing_profile as _missing_profile

                actions = {
                    **{c: f"{info['strategy']} → {info['detail']}" for c, info in pre_log.get("imputed", {}).items()},
                    **{d["column"]: self._t(f"剔除（{d['reason']}）", f"dropped ({d['reason']})") for d in pre_log.get("dropped", [])},
                }
                none_needed = self._t("无需处理", "none needed")
                prof = _missing_profile(df)
                prof_rows = [
                    {
                        "column": r["column"],
                        "n_missing": str(r["n_missing"]),
                        "pct": f"{r['pct_missing']:.1f}%",
                        "action": actions.get(r["column"], none_needed),
                    }
                    for r in prof["columns"][:6]
                ]
                blocks.append(
                    self._table_block(
                        pd.DataFrame(prof_rows),
                        self._t(
                            f"① 数据预处理：缺失值 / NaN 档案（共 {prof['missing_cells']} 个缺失单元格，{prof['missing_pct']:.1f}%）",
                            f"1. Data preprocessing: missing/NaN profile ({prof['missing_cells']} missing cells, {prof['missing_pct']:.1f}%)",
                        ),
                    )
                )
            except Exception:  # noqa: BLE001 - profile is additive
                pass
            # Stage 2: collinearity handling — dropped features, or proof the
            # screen ran and found nothing above the thresholds.
            if col_log.get("dropped"):
                col_rows = [{"feature": d["column"], "reason": d["reason"]} for d in col_log["dropped"][:8]]
            else:
                col_rows = [
                    {
                        "feature": self._t("（无）", "(none)"),
                        "reason": self._t("所有特征成对 |r|<0.95 且 VIF<10，无需剔除", "all pairwise |r|<0.95 and VIF<10, nothing pruned"),
                    }
                ]
            blocks.append(
                self._table_block(
                    pd.DataFrame(col_rows),
                    self._t(
                        f"② 特征工程：共线性处理（保留 {col_log.get('n_features', len(feature_names))} 个特征）",
                        f"2. Feature engineering: collinearity handling ({col_log.get('n_features', len(feature_names))} features kept)",
                    ),
                )
            )

            result = train_model(X, y, task="classification" if wants_classification else "regression")
            fa = feature_analysis(result)
            conf = modeling_confidence(result)
            best = result["best_model"]
            metrics = result["metrics"][best]

            metric_lines = ", ".join(f"{k}={v}" for k, v in metrics.items())
            n_pre = len(pre_log.get("steps", []))
            n_col = len(col_log.get("dropped", []))
            text = self._t(
                f"**{title}**（{result['task']}，最优模型 `{best}`）：{metric_lines}；5折CV {result['cv_mean']:.3f}±{result['cv_std']:.3f}。\n\n"
                f"**建模前的处理**（同一套流程因果推断也在用）：① 数据预处理——{n_pre} 项清理（缺失值/NaN 填补、稀疏列与常数列剔除）；"
                f"② 特征工程——构造 + 共线性处理（剔除 {n_col} 个冗余特征，成对 |r| 与 VIF 双重把关）。"
                f"③ 建模之后做特征分析（SHAP/置换重要性），先有可信模型、再解释特征。",
                f"**{title}** ({result['task']}, best model `{best}`): {metric_lines}; 5-fold CV {result['cv_mean']:.3f}±{result['cv_std']:.3f}.\n\n"
                f"**Pre-modelling steps** (the same pipeline causal inference reuses): 1. preprocessing — {n_pre} cleanups "
                f"(missing/NaN imputation, sparse and constant columns dropped); 2. feature engineering — construction plus "
                f"collinearity handling ({n_col} redundant features removed via pairwise |r| and VIF). "
                f"3. feature analysis (SHAP/permutation) runs *after* the model is validated.",
            )

            # Explainability chart: SHAP (if installed) or permutation importance.
            # Deliberately positioned AFTER the model: features are interpreted
            # against a validated model, never before one exists.
            method = self._t(
                "④ 模型之后的特征分析（SHAP）" if fa["engine"] == "shap" else "④ 模型之后的特征分析（置换重要性",
                "4. Post-model feature analysis (SHAP)" if fa["engine"] == "shap" else "4. Post-model feature analysis (permutation importance)",
            )
            method += self._t("）", ")")
            blocks.append(
                {
                    "type": "chart",
                    "title": method,
                    "image": charts.importance_bar(fa["importance"], title=f"{title} · {method}"),
                }
            )

            # Task-specific evaluation chart.
            if result["task"] == "classification":
                from sklearn.metrics import roc_curve, confusion_matrix

                best_model = result["_fitted"][best]
                X_tr, X_te, y_tr, y_te = result["_split"]
                if hasattr(best_model, "predict_proba") and y.nunique() == 2:
                    proba = best_model.predict_proba(X_te)[:, 1]
                    fpr, tpr, _ = roc_curve(y_te, proba)
                    blocks.append(
                        {
                            "type": "chart",
                            "title": self._t("ROC 曲线", "ROC curve"),
                            "image": charts.roc_chart(fpr, tpr, metrics.get("roc_auc", 0.0)),
                        }
                    )
                    cm = confusion_matrix(y_te, best_model.predict(X_te))
                    blocks.append(
                        {
                            "type": "chart",
                            "title": self._t("混淆矩阵", "Confusion matrix"),
                            "image": charts.confusion_matrix_chart(cm.tolist(), labels=sorted(map(str, y.unique()))),
                        }
                    )
                else:
                    blocks.append({"type": "markdown", "markdown": self._t("`ROC/混淆矩阵` 需要二分类与概率输出。", "`ROC/confusion matrix` need binary labels and probabilistic output.")})
            else:
                _, X_te, _, y_te = result["_split"]
                pred = result["predictions"][best]
                blocks.append(
                    {
                        "type": "chart",
                        "title": self._t("预测 vs 实际", "Predicted vs actual"),
                        "image": charts.pred_vs_actual_chart(y_te, pred),
                    }
                )

            # Confidence card: overfit gap, CV stability, sample size.
            check_rows = [
                {"check": c["name"], "status": c["status"], "detail": c["detail"]} for c in conf["checks"]
            ]
            verdict = self._t(
                f"总体置信度：{conf['overall']}。模型结论用于决策前请先看过拟合缺口与 CV 稳定性。",
                f"Overall confidence: {conf['overall']}. Review the overfit gap and CV stability before using model output in decisions.",
            )
            blocks.append(
                {
                    "type": "table",
                    "title": self._t("置信度验证", "Confidence validation"),
                    "columns": ["check", "status", "detail"],
                    "rows": [[r["check"], r["status"], r["detail"]] for r in check_rows],
                }
            )
            blocks.append({"type": "markdown", "markdown": verdict})

            # Model diagnostics as tests: calibration for classifiers, residual
            # normality / autocorrelation / heteroskedasticity / VIF for regressions.
            panel = None
            try:
                panel = test_panels.model_panel(result)
                blocks.append(
                    self._table_block(
                        stat_tests.to_frame(panel["tests"], self.lang),
                        self._t("统计检验（模型诊断）", "Statistical tests (model diagnostics)"),
                        max_rows=8,
                    )
                )
                text += "\n\n" + self._t(
                    f"**统计检验**（检验方法 / 统计量 / p / 结论）：{panel['headline_zh']}{panel['summary_zh']}",
                    f"**Statistical tests** (method / statistic / p / conclusion): {panel['headline_en']} {panel['summary_en']}",
                )
            except Exception:  # noqa: BLE001
                panel = None

            tool_result = {"task": result["task"], "best_model": best, "n_train": result["n_train"], "n_test": result["n_test"], "confidence": conf["overall"]}
            return AgentReply(text=text, blocks=blocks, tool_calls=[{"tool": "train_model", "args": {"task": result["task"], "target": target}, "result": tool_result}], data={"modeling": tool_result})
        except Exception as exc:  # noqa: BLE001 - surfaced with a next-step hint
            return AgentReply(
                text=self._t(
                    f"建模没能完成：`{exc}`\n\n可以说得更具体，比如「训练一个预测用户是否转化的分类模型」。",
                    f"Modeling failed: `{exc}`\n\nTry e.g. \"train a classifier to predict conversion\".",
                )
            )

    def _handle_dashboard(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """One-click monitoring dashboard: standalone HTML in the artifacts dir."""
        from .tools.dashboard import build_dashboard

        try:
            out = build_dashboard(self, lang=self.lang)
        except Exception as exc:  # noqa: BLE001
            return AgentReply(
                text=self._t(
                    f"看板生成失败：`{exc}`",
                    f"Dashboard generation failed: `{exc}`",
                )
            )
        name = Path(out["path"]).name
        kpi_text = " · ".join(f"{k['label']} {k['value']}" for k in out["kpis"])
        text = self._t(
            f"看板已生成：**{name}**。核心 KPI（近14天环比）：{kpi_text}。点击下方链接打开（独立 HTML，图表内嵌，可直接分享/投屏）。",
            f"Dashboard ready: **{name}**. Key KPIs (last 14 days WoW): {kpi_text}. Open via the link below (standalone HTML, inline charts, share or cast directly).",
        )
        blocks: List[Dict[str, Any]] = [
            {
                "type": "kpi",
                "items": [
                    {"label": k["label"], "value": k["value"], "sub": "", "good": None if k["delta"] is None else k["delta"] >= 0}
                    for k in out["kpis"]
                ],
            },
            {"type": "markdown", "markdown": self._t("看板含：KPI 环比卡、GMV 趋势+均线、渠道/设备拆解、退款率、近期异动表。", "Includes: KPI cards with WoW, GMV trend + moving average, channel/device breakdowns, refund rate, recent anomalies.")},
        ]
        return AgentReply(
            text=text,
            blocks=blocks,
            artifacts=out["artifacts"] if "artifacts" in out else [out["path"]],
            tool_calls=[{"tool": "build_dashboard", "args": {}, "result": {"path": name, "kpis": len(out["kpis"])}}],
            data={"dashboard": name},
        )

    def _handle_report(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        if not self.findings:
            self._health_check()

        title = self._t("指标诊断与实验分析报告", "Metric diagnosis and experiment report")
        report = Report(
            title=title,
            subtitle=self._t("由 DS-Agent 自动生成：结论先行，全部数字可复现", "Generated by DS-Agent: answer first, every number reproducible"),
            lang=self.lang,
            meta={
                self._t("数据源", "Data source"): str(self.sql.warehouse.path.name),
                self._t("统计口径", "Metric definitions"): self._t("语义层统一口径，比率指标用分子/分母聚合", "governed semantic layer; ratios aggregated as numerator/denominator"),
                self._t("问题", "Question"): self.session.get("last_question", "-")[:140],
            },
        )
        summary = list(self.summary_bullets[:8])
        if len(self.summary_bullets) > len(summary):
            # Silent truncation is how a report ends up looking complete while
            # missing half the analysis; say what was left out instead.
            summary.append(
                self._t(
                    f"另有 {len(self.summary_bullets) - len(summary)} 条结论，见下方各章节。",
                    f"{len(self.summary_bullets) - len(summary)} more findings are in the sections below.",
                )
            )
        report.add_summary(summary)
        # Business context from the RAG knowledge base: pull the calibers and
        # known incidents relevant to the session's metrics so the report can
        # be read without the tribal knowledge.
        metrics_in_play = sorted({s.split()[0] for s in self.summary_bullets if s.split()})[:3]
        kb_text = self.kb_context(" ".join(metrics_in_play) + " 口径 已知问题", k=3)
        if kb_text:
            report.add_section(
                Section(
                    title=self._t("业务背景（来自知识库）", "Business context (from the knowledge base)"),
                    prose=kb_text,
                )
            )
        # Every recorded section goes in: a section exists because a question was
        # asked and answered, and dropping the tail silently is worse than a long
        # report.
        for section in self.findings:
            report.add_section(section)
        report.add_method(self._t("异常检测：周季节性基线（前28天中位数水平 × 星期指数，含局部增速修正）+ MAD 稳健 z（阈值 3.5）；持续位移用双侧 CUSUM 并做 t 检验校验。", "Anomaly detection: weekly-seasonal baseline (trailing 28-day level x weekday index, with a local growth correction) + MAD-robust z (threshold 3.5); sustained shifts via two-sided CUSUM, gated by a t-test."))
        report.add_method(self._t("贡献度：加法指标按分段增量精确分解；比率指标拆为率效应/结构效应/交互项，残差≈0。", "Contribution: exact additive decomposition; ratio metrics split into rate / mix / interaction with residual ~ 0."))
        report.add_method(self._t("实验：样本量由功效函数反解，分析含 SRM、CUPED、BH 多重比较。", "Experiments: sizing inverts the power function; readout includes SRM, CUPED, BH correction."))
        report.add_method(self._t("非标准实验设计：时间片轮转以时间片为推断单元并用 HAC 标准误；反转实验用 τ₁+τ₂ 检验漂移、用持续性衡量衰减；父子实验做正交性卡方 + 父层固定效应校正；双边实验从 2×2 四格边际拆直接效应与跨侧溢出；敏感性分析报符号稳定性而非单一 p 值。", "Non-standard designs: switchback infers at the time-block level with HAC errors; reversal tests drift with tau1+tau2 and measures decay by persistence; parent-child checks orthogonality by chi-square and adjusts for parent fixed effects; two-sided reads direct effects off the 2x2 marginals and reports cross-side spillover; sensitivity analysis reports sign stability rather than one p-value."))
        report.add_method(self._t("因果：DID 含平行趋势检验，合成控制用安慰剂分布推断，PSM/IPW/AIPW 交叉验证。", "Causal: DiD with parallel-trends test, synthetic control with placebo inference, PSM/IPW/AIPW cross-checks."))
        report.add_method(self._t("异质性与门槛：Uplift 用因果树/因果森林的 honest 切分估计 CATE，Qini 与十分位定向收益衡量可用性；断点回归用局部线性 + 密度操纵检验 + 协变量平衡，带宽敏感性不稳时不下结论。", "Heterogeneity and thresholds: uplift estimates CATE with honest splits in a causal tree/forest, with Qini and decile targeting to measure usefulness; RDD uses local linear fits with a density manipulation test and covariate balance, and refuses conclusions when the bandwidth sensitivity is unstable."))
        report.add_method(self._t("时序因果：ARIMA 中断时间序列在对数尺度上同时估计趋势、星期因子与干预项，并用安慰剂干预日与口径曲线校验；结构时序反事实用干预前一步拟合与滚动预测区间。两者都与未校正前后差对照，后者会把趋势算成效应。", "Time-series causal: ARIMA ITS estimates the trend, weekday terms and the post indicator jointly on the log scale, checked with placebo dates and a specification curve; the structural counterfactual uses a pre-period one-step fit and horizon-widening intervals. Both are shown against the naive pre/post difference, which bills the trend to the intervention."))
        report.add_method(self._t("所有统计结论基于观测数据，相关不等于因果；因果结论依赖明确的识别假设。", "All results are observational unless stated; causal claims rely on explicit identifying assumptions."))

        paths = report.save(self.artifact_dir, f"ds_report_{date.today()}")

        health_table = None
        blocks: List[Dict[str, Any]] = [
            {"type": "markdown", "markdown": self._t(
                f"报告已生成：共 {len(self.findings)} 个章节、{len(self.summary_bullets)} 条结论。",
                f"Report written: {len(self.findings)} sections, {len(self.summary_bullets)} summary bullets.",
            )}
        ]
        if self.findings and self.findings[0].chart:
            blocks.append({"type": "chart", "title": self.findings[0].title, "image": self.findings[0].chart})

        text = self._t(
            "**报告已生成**\n\n"
            f"- Markdown: `{paths['markdown']}`\n"
            f"- HTML（单文件，图表内嵌）: `{paths['html']}`\n\n"
            "结构：执行摘要 → 各分析章节（含图表与贡献度表）→ 方法论与局限。直接可贴到周报或分享给业务方。",
            "**Report written**\n\n"
            f"- Markdown: `{paths['markdown']}`\n"
            f"- HTML (single file, charts embedded): `{paths['html']}`\n\n"
            "Structure: executive summary -> analysis sections (charts + contribution tables) -> methodology and caveats.",
        )
        return AgentReply(
            text=text,
            blocks=blocks,
            tool_calls=[{"tool": "write_report", "args": {"sections": len(self.findings)}, "result": paths}],
            artifacts=[paths["html"], paths["markdown"]],
            data={"paths": paths},
        )

    def _health_check(self) -> None:
        """Default report content: a metric health sweep, then attribution on the biggest mover."""
        table = semantic.default_table()
        anchor = self.sql.warehouse.max_date(table.name, table.time_column) or date.today()
        window = TimeWindow(anchor - timedelta(days=6), anchor, "last 7 days", 7)
        comparison = TimeWindow(anchor - timedelta(days=13), anchor - timedelta(days=7), "previous 7 days", 7)
        localize_window(window, self.lang)
        localize_window(comparison, self.lang)

        rows = []
        movers = []
        for name in ("gmv", "orders", "conversion_rate", "aov", "refund_rate"):
            metric = table.metrics[name]
            cur = self._window_total(metric, window, table)
            prev = self._window_total(metric, comparison, table)
            change = (cur - prev) / prev if prev else float("nan")
            rows.append([
                metric.name,
                _fmt_num(prev),
                _fmt_num(cur),
                _fmt_pct(change),
                self._t("越高越好", "higher is better") if metric.higher_is_better else self._t("越低越好", "lower is better"),
            ])
            movers.append((abs(change) if np.isfinite(change) else 0, metric, change))
        df = pd.DataFrame(rows, columns=[self._t("指标", "metric"), self._t("上期", "prior"), self._t("本期", "current"), self._t("环比", "change"), self._t("方向", "direction")])
        self._record(
            self._t("近7天核心指标概览已生成（GMV/订单/转化率/客单价/退款率）。", "Core metrics for the last 7 days compared with the prior week."),
            Section(
                title=self._t("1. 核心指标健康度", "1. Core metric health"),
                prose=self._t(
                    f"窗口：{window.start} ~ {window.end}（对比 {comparison.start} ~ {comparison.end}）。",
                    f"Window: {window.start} ~ {window.end} vs {comparison.start} ~ {comparison.end}.",
                ),
                table=df,
            ),
        )

        movers.sort(key=lambda x: -x[0])
        _, metric, change = movers[0]
        reply = self._handle_diagnose(f"{metric.name} 近7天为什么变化", {"metric": metric, "anchor": anchor, "window": window, "comparison": comparison, "dimensions": []})
        # Re-label the auto-generated section so the report reads in order.
        if self.findings:
            self.findings[-1].title = self._t(f"2. 最大波动指标：{metric.name}（{_fmt_pct(change)}）", f"2. Largest mover: {metric.name} ({_fmt_pct(change)})")
        diagnosis = self.session.get("last_diagnosis", {})
        self._record(
            self._t(
                "综上：先看最大波动指标的量价拆解，再按分群定位到具体分段，最后核对埋点与对照指标。",
                "In short: decompose the biggest mover into rate vs mix, drill into the segment, then audit instrumentation against a paired metric.",
            ),
            Section(
                title=self._t("3. 结论与建议", "3. Conclusions and actions"),
                prose=self._plain(diagnosis.get("driver_line", "")),
                bullets=diagnosis.get("steps", [])[:4],
            ),
        )

    # ------------------------------------------------------------------- help
    def _help(self) -> AgentReply:
        lang_key = "zh" if self.lang == "zh" else "en"
        catalog = full_catalog()
        lines = [
            self._t(
                f"我能做 {len(catalog)} 类事，直接用自然语言问就行：",
                f"{len(catalog)} things I can do, all in natural language:",
            ),
            "",
        ]
        rows = []
        for name, spec in catalog.items():
            lines.append(f"**{name}** — {spec[lang_key]}")
            example = spec["example_zh" if lang_key == "zh" else "example_en"]
            lines.append(self._t(f"  例：`{example}`", f"  e.g. `{example}`"))
            rows.append([name, spec[lang_key], example])
        lines += [
            "",
            self._t(
                "数据：内置电商日粒度数仓（orders_daily）、用户级实验（experiment_users）、城市周面板（city_panel）、"
                "促销观测（promo_users）、时间片轮转日志（switchback_log）、反转实验（reverse_log）、父子实验（nested_users）、"
                "双边市场（two_sided_market）、断点样本（rdd_scores）、uplift 样本（uplift_users）、干预时序（its_series）。",
                "Data: bundled daily warehouse (orders_daily), user-level experiment (experiment_users), city-week panel "
                "(city_panel), observational promo (promo_users), switchback log (switchback_log), reversal log (reverse_log), "
                "parent-child users (nested_users), two-sided markets (two_sided_market), RDD scores (rdd_scores), "
                "uplift sample (uplift_users), intervention series (its_series).",
            ),
            self._t("追问会继承上下文：先问「GMV 为什么跌」，再问「按渠道拆」，我会沿用同一个指标和窗口。", "Follow-ups inherit context: ask 'why did GMV drop', then 'split by channel'."),
        ]
        df = pd.DataFrame(rows, columns=[self._t("能力", "capability"), self._t("说明", "what it does"), self._t("示例", "example")])
        return AgentReply(
            text="\n".join(lines),
            blocks=[self._table_block(df, self._t("能力清单", "Capabilities"), max_rows=20)],
        )
