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
        "zh": "Natural-language querying with a governed semantic layer and read-only SQL",
        "en": "Natural-language querying with a governed semantic layer and read-only SQL",
        "example_zh": "GMV by channel in the last 30 days",
        "example_en": "GMV by channel in the last 30 days",
    },
    "diagnose": {
        "zh": "Metric movement diagnosis: anomaly detection, contribution, attribution; baseline switchable to Holt-Winters / SARIMA / STL+ETS / auto-backtested",
        "en": "Metric movement diagnosis: anomaly detection, contribution, attribution; baseline switchable to Holt-Winters / SARIMA / STL+ETS / auto-backtested",
        "example_zh": "Why did GMV drop in the last 7 days? (or: check anomalies with holt-winters)",
        "example_en": "Why did GMV drop in the last 7 days? (or: check anomalies with holt-winters)",
    },
    "experiment_design": {
        "zh": "Experiment design: sizing, duration, randomisation unit, analysis plan across 15 archetypes",
        "en": "Experiment design: sizing, duration, randomisation unit, analysis plan across 15 archetypes",
        "example_zh": "Design a checkout conversion test: baseline 3%, detect a 2% relative lift, 50k daily traffic",
        "example_en": "Design a checkout conversion test: baseline 3%, detect a 2% relative lift, 50k daily traffic",
    },
    "experiment_analysis": {
        "zh": "Experiment readout: SRM, significance, CUPED, multiplicity, segments, ship decision",
        "en": "Experiment readout: SRM, significance, CUPED, multiplicity, segments, ship decision",
        "example_zh": "Read out the AB test results",
        "example_en": "Read out the AB test results",
    },
    "switchback": {
        "zh": "Switchback: time block as the inference unit, HAC standard errors, washout and carryover tests",
        "en": "Switchback: time block as the inference unit, HAC standard errors, washout and carryover tests",
        "example_zh": "Read out the switchback test",
        "example_en": "Read out the switchback test",
    },
    "reversal": {
        "zh": "Reversal experiment: swap the arms mid-flight, test drift with tau1+tau2 and persistence with |tau2|/|tau1|",
        "en": "Reversal experiment: swap the arms mid-flight, test drift with tau1+tau2 and persistence with |tau2|/|tau1|",
        "example_zh": "Run a reversal experiment to validate the effect",
        "example_en": "Run a reversal experiment to validate the effect",
    },
    "nested": {
        "zh": "Parent-child experiments: orthogonality check, parent fixed-effect adjustment, interaction test",
        "en": "Parent-child experiments: orthogonality check, parent fixed-effect adjustment, interaction test",
        "example_zh": "Check that my parent and child experiments do not interfere",
        "example_en": "Check that my parent and child experiments do not interfere",
    },
    "two_sided": {
        "zh": "Two-sided experiment: 2x2 buyer x seller cells separating direct effects, cross-side spillover and interaction",
        "en": "Two-sided experiment: 2x2 buyer x seller cells separating direct effects, cross-side spillover and interaction",
        "example_zh": "Two-sided test treating both buyers and sellers",
        "example_en": "Two-sided test treating both buyers and sellers",
    },
    "sensitivity": {
        "zh": "Sensitivity analysis: winsorise, trim, log, CUPED, segment FE, bootstrap, plus A/A placebo and jackknife",
        "en": "Sensitivity analysis: winsorise, trim, log, CUPED, segment FE, bootstrap, plus A/A placebo and jackknife",
        "example_zh": "Is the result robust? Run a sensitivity analysis",
        "example_en": "Is the result robust? Run a sensitivity analysis",
    },
    "uplift": {
        "zh": "Uplift modelling: causal tree, causal forest, T/S/X learners, Qini curve and targeting gain",
        "en": "Uplift modelling: causal tree, causal forest, T/S/X learners, Qini curve and targeting gain",
        "example_zh": "Which users respond most to the promo? Fit a causal forest",
        "example_en": "Which users respond most to the promo? Fit a causal forest",
    },
    "rdd": {
        "zh": "Regression discontinuity: local linear fits, bandwidth selection, manipulation test, fuzzy RDD",
        "en": "Regression discontinuity: local linear fits, bandwidth selection, manipulation test, fuzzy RDD",
        "example_zh": "Evaluate the membership threshold with a regression discontinuity",
        "example_en": "Evaluate the membership threshold with a regression discontinuity",
    },
    "ts_causal": {
        "zh": "Time-series causal: ARIMA interrupted time series plus a structural counterfactual, with placebo and specification checks",
        "en": "Time-series causal: ARIMA interrupted time series plus a structural counterfactual, with placebo and specification checks",
        "example_zh": "Estimate the launch effect on GMV with a time-series causal model",
        "example_en": "Estimate the launch effect on GMV with a time-series causal model",
    },
    "method_select": {
        "zh": "Method selection: recommends the right statistical or causal method from your scenario, with reasons and pitfalls",
        "en": "Method selection: recommends the right statistical or causal method from your scenario, with reasons and pitfalls",
        "example_zh": "Which method fits this scenario?",
        "example_en": "Which method fits this scenario?",
    },
    "tests": {
        "zh": "Statistical tests: 30+ built-in tests (normality/variance/location/proportion/time series/regression/resampling/multiplicity) with null hypothesis, formula, assumptions, decision rule and citation; every readout ships its own test table",
        "en": "Statistical tests: 30+ built-in tests (normality/variance/location/proportion/time series/regression/resampling/multiplicity) with null hypothesis, formula, assumptions, decision rule and citation; every readout ships its own test table",
        "example_zh": "Which tests does anomaly detection run? / What is a Welch t-test?",
        "example_en": "Which tests does anomaly detection run? / What is a Welch t-test?",
    },
    "sampling": {
        "zh": "Sampling surveys: sample-size determination (proportion/mean, power formula with citation), simple-random/systematic/stratified draws (reproducible), sampling distribution and standard errors, representativeness tests (chi-square GOF / Welch t / KS)",
        "en": "Sampling surveys: sample-size determination (proportion/mean, power formula with citation), simple-random/systematic/stratified draws (reproducible), sampling distribution and standard errors, representativeness tests (chi-square GOF / Welch t / KS)",
        "example_zh": "How large a sample do I need for a satisfaction survey, baseline 70%, detect a 3pp change?",
        "example_en": "How large a sample do I need for a satisfaction survey, baseline 70%, detect a 3pp change?",
    },
    "knowledge": {
        "zh": "Business knowledge Q&A: metric calibers, dimension semantics, data dictionary and known incidents, via RAG over the internal knowledge base",
        "en": "Business knowledge Q&A: metric calibers, dimension semantics, data dictionary and known incidents, via RAG over the internal knowledge base",
        "example_zh": "What is the GMV caliber and how does it differ from revenue?",
        "example_en": "What is the GMV caliber and how does it differ from revenue?",
    },
    "metric": {
        "zh": "Metric dictionary: register custom metrics (validated against the schema, activated into the semantic layer), list calibers, deprecate metrics",
        "en": "Metric dictionary: register custom metrics (validated against the schema, activated into the semantic layer), list calibers, deprecate metrics",
        "example_zh": "Register metric profit_rate = (gmv - refund_amount) / gmv",
        "example_en": "Register metric profit_rate = (gmv - refund_amount) / gmv",
    },
    "datasource": {
        "zh": "Data sources: attach enterprise databases (DuckDB/SQLite/Postgres/MySQL) and offline spreadsheets (CSV/Excel/Parquet/JSON); registered sources are directly queryable in natural language",
        "en": "Data sources: attach enterprise databases (DuckDB/SQLite/Postgres/MySQL) and offline spreadsheets (CSV/Excel/Parquet/JSON); registered sources are directly queryable in natural language",
        "example_zh": "Attach data source /Users/me/sales.csv as sales",
        "example_en": "Attach data source /Users/me/sales.csv as sales",
    },
    "causal": {
        "zh": "Causal inference: DiD/event study, PSM, IPW, AIPW, synthetic control with placebo tests",
        "en": "Causal inference: DiD/event study, PSM, IPW, AIPW, synthetic control with placebo tests",
        "example_zh": "Evaluate the subsidy policy effect with difference-in-differences",
        "example_en": "Evaluate the subsidy policy effect with difference-in-differences",
    },
    "causal_oss": {
        "zh": "Open-source cross-validation: re-estimate with DoWhy/EconML, run placebo/random-common-cause refutations, and compare against the in-house estimators",
        "en": "Open-source cross-validation: re-estimate with DoWhy/EconML, run placebo/random-common-cause refutations, and compare against the in-house estimators",
        "example_zh": "Cross-validate the promo effect with the DoWhy package",
        "example_en": "Cross-validate the promo effect with the DoWhy package",
    },
    "dl_anomaly": {
        "zh": "Deep-learning anomaly detection: autoencoder reconstruction error (window-shape sensitive) plus isolation-forest multivariate scoring, with a reconstruction/alert-rate/seasonal-cross-check confidence card",
        "en": "Deep-learning anomaly detection: autoencoder reconstruction error (window-shape sensitive) plus isolation-forest multivariate scoring, with a reconstruction/alert-rate/seasonal-cross-check confidence card",
        "example_zh": "Check GMV anomalies with an autoencoder / deep-learning detector",
        "example_en": "Check GMV anomalies with an autoencoder / deep-learning detector",
    },
    "modeling": {
        "zh": "Modelling as a four-stage pipeline shared by regression and causal inference: 1. data preprocessing (missing/NaN imputation, sparse and constant columns dropped) -> 2. feature engineering (datetime expansion / one-hot / collinearity handling via pairwise |r| and VIF pruning) -> 3. classification & regression (linear vs gradient boosting, 5-fold CV) -> 4. post-validation feature analysis (true SHAP if installed, else permutation+coefficients) + overfit/CV-stability confidence card",
        "en": "Modelling as a four-stage pipeline shared by regression and causal inference: 1. data preprocessing (missing/NaN imputation, sparse and constant columns dropped) -> 2. feature engineering (datetime expansion / one-hot / collinearity handling via pairwise |r| and VIF pruning) -> 3. classification & regression (linear vs gradient boosting, 5-fold CV) -> 4. post-validation feature analysis (true SHAP if installed, else permutation+coefficients) + overfit/CV-stability confidence card",
        "example_zh": "Train a classifier to predict conversion and show feature importance",
        "example_en": "Train a classifier to predict conversion and show feature importance",
    },
    "monitoring": {
        "zh": "Model monitoring: three post-launch decay checks — feature drift (banded PSI), data drift (KS tests), performance decay (cohort AUC trend) — with warn/critical thresholds and retrain advice",
        "en": "Model monitoring: three post-launch decay checks — feature drift (banded PSI), data drift (KS tests), performance decay (cohort AUC trend) — with warn/critical thresholds and retrain advice",
        "example_zh": "Check the conversion model for feature drift and performance decay",
        "example_en": "Check the conversion model for feature drift and performance decay",
    },
    "dashboard": {
        "zh": "Dashboard: one-click standalone HTML monitoring view (KPI cards with WoW, trend + moving average, channel/device breakdowns, refund rate, recent anomalies) — same source and calibers as agent answers",
        "en": "Dashboard: one-click standalone HTML monitoring view (KPI cards with WoW, trend + moving average, channel/device breakdowns, refund rate, recent anomalies) — same source and calibers as agent answers",
        "example_zh": "Generate a GMV monitoring dashboard",
        "example_en": "Generate a GMV monitoring dashboard",
    },
    "report": {
        "zh": "Report writing: turns findings into an answer-first, reproducible Markdown + HTML report",
        "en": "Report writing: turns findings into an answer-first, reproducible Markdown + HTML report",
        "example_zh": "Write this up as a weekly report",
        "example_en": "Write this up as a weekly report",
    },
}


def _episode_kind(episode: Dict[str, Any], lang: str = "zh") -> str:
    """Human label for how an episode was detected."""
    kind = episode.get("kind", "point")
    if lang == "zh":
        return "level shift" if kind == "level_shift" else "single-day spike"
    return "level shift" if kind == "level_shift" else "single-day spike"


_SEVERITY_ZH = {"critical": "critical", "high": "high", "medium": "medium", "low": "low"}

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
    "yesterday": "yesterday",
    "day before": "day before yesterday",
    "last week": "last week",
    "week before last": "two weeks ago",
    "this week": "this week",
    "last month": "last month",
    "month before last": "two months ago",
    "custom": "custom window",
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
        prefix = "last" if match.group(1) == "last" else "previous"
        unit = "days" if match.group(3) == "days" else "weeks"
        window.label = f"{prefix} {match.group(2)} {unit}"
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

        # Model monitoring: explicit post-launch checks. Must outrank generic
        # "模型" hits (which route to training).
        if re.search(r"模型监控|数据漂移|特征漂移|性能衰减|模型衰减|概念漂移|\bpsi\b|\bdrift\b|performance decay|model monitoring", q, re.I):
            scores["monitoring"] += 9

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
            "monitoring": self._handle_monitoring,
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
                tried_note = ", ".join(a["intent"] for a in attempts)
                reply = AgentReply(
                    text=self._t(
                        f"That step failed (tried: {tried_note}):\n"
                        + "\n".join(f"- `{a['intent']}`: {a['error']}" for a in attempts)
                        + "\n\nTry being more specific (e.g. \"GMV by channel, last 7 days vs prior\"), or type `help`.",
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
                                "Query result (LLM-generated SQL, read-only guard passed)",
                                "Query result (LLM-generated SQL, read-only guard passed)",
                            ),
                        )
                        text = self._t(
                            f"This goes beyond the semantic layer's preset metrics, so the LLM generated the SQL; "
                            f"it passed the read-only guard and returned {result.row_count} rows. {plan_sql['explanation']}",
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
            f"I could not match a preset metric. Available: {', '.join(m.name for m in semantic.default_table().metrics.values())}.",
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
                        f"First {result.row_count} row(s) of `{name}`. For aggregates, name a metric — or configure an LLM for free-form SQL.",
                        f"First {result.row_count} row(s) of `{name}`. For aggregates, name a metric — or configure an LLM for free-form SQL.",
                    ),
                    blocks=[self._table_block(result.dataframe, self._t(f"{name} rows", f"{name} rows"))],
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
                frame[dims + ["metric_value"]], self._t("Query result", "Query result")
            )
            total = self._window_total(metric, window, table)
            top = frame.iloc[0]
            share1 = top["metric_value"] / total if total else float("nan")
            share3 = frame["metric_value"].head(3).sum() / total if total else float("nan")
            text = self._t(
                f"**{metric.name}** ({window.label}): total {_fmt_num(total)}, split by `{'/'.join(dims)}` (top 15). "
                f"Leader {dims[0]}={top[dims[0]]} at {_fmt_num(top['metric_value'])} ({share1:.1%} share); "
                f"top-3 make up {share3:.1%}.",
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
                    f"**{metric.name}** daily trend ({window.label}): {_fmt_num(first)} on the first day → "
                    f"{_fmt_num(last)} on the last ({_fmt_pct(drift)}); average {_fmt_num(frame['metric_value'].mean())}.",
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
                                "label": self._t("vs prior", "vs prior"),
                                "value": _fmt_pct(change),
                                "sub": f"{arrow} {_fmt_num(total - prev)}",
                                "good": (change > 0) == metric.higher_is_better,
                            },
                        ],
                    }
                ]
                text = self._t(
                    f"**{metric.name}**: {window.label} = {_fmt_num(total)} vs {_fmt_num(prev)} prior, "
                    f"{_fmt_pct(change)} ({_fmt_num(total - prev)}).",
                    f"**{metric.name}**: {window.label} = {_fmt_num(total)} vs {_fmt_num(prev)} prior, "
                    f"{_fmt_pct(change)} ({_fmt_num(total - prev)}).",
                )
            table_block = None

        sql_text = self._explain_sql(metric, table, dims, window, grain)
        # RAG: surface the relevant business caliber straight in the answer, so
        # "查 GMV" also reminds the reader what GMV does and does not include.
        kb_note = self.kb_context(f"{metric.name} 口径 定义", k=1)
        if kb_note:
            text += "\n\n" + self._t("Business caliber (from the knowledge base):\n", "Business caliber (from the knowledge base):\n") + kb_note
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
                        f"Metric `{name or q}` is not in the dictionary. Use \"metric dictionary\" to list all.",
                        f"Metric `{name or q}` is not in the dictionary. Use \"metric dictionary\" to list all.",
                    )
                )
            if entry.get("status") == "deprecated":
                return AgentReply(
                    text=self._t(f"Metric `{entry['name']}` was already deprecated on {entry.get('updated_at')}.", f"Metric `{entry['name']}` was already deprecated on {entry.get('updated_at')}.")
                )
            md.deprecate(entry["name"], reason=q[:120])
            return AgentReply(
                text=self._t(
                    f"Metric **{entry['name']}** is deprecated (kept in the dictionary for lineage).\n\nRegister a v2 name for a new caliber instead of overwriting.",
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
                    text=self._t(f"Registration rejected: {exc}\n\nChecklist in docs/metric_dictionary.md.", f"Registration rejected: {exc}\n\nChecklist in docs/metric_dictionary.md."),
                    data={"limitation": False},
                )
            kind = self._t("ratio metric", "ratio metric") if entry.get("denominator") else self._t("additive metric", "additive metric")
            lines = [
                self._t(
                    f"Metric **{entry['name']}** registered and validated; it is live in the semantic layer — query it directly in natural language, one caliber everywhere.",
                    f"Metric **{entry['name']}** registered and validated; it is live in the semantic layer — query it directly in natural language, one caliber everywhere.",
                ),
                "",
                f"- formula: `{entry['formula_text']}` ({kind})",
                f"- table: `{entry['table']}`",
                f"- aliases: {', '.join(entry.get('aliases') or []) or '—'}",
                f"- unit: {entry.get('unit') or '—'}",
                f"- owner: {entry.get('owner') or '—'}",
            ]
            if kb_hits:
                lines += ["", self._t("**Caliber reminders** (from the metric-definition knowledge base):", "**Caliber reminders** (from the metric-definition knowledge base):"), kb_hits[:400]]
            df = pd.DataFrame(
                [[entry["name"], entry["formula_text"], entry["table"], entry.get("unit", ""), ", ".join(entry.get("aliases") or [])]],
                columns=[self._t("metric", "metric"), self._t("formula", "formula"), self._t("table", "table"), self._t("unit", "unit"), self._t("aliases", "aliases")],
            )
            return AgentReply(
                text="\n".join(lines),
                blocks=[self._table_block(df, self._t("Metric dictionary · registered", "Metric dictionary · registered"))],
                tool_calls=[{"tool": "metric_dict", "args": {"op": "register", "name": entry["name"]}, "result": {"formula": entry["formula_text"], "table": entry["table"]}}],
                data={"metric_dict": {"registered": entry["name"]}},
            )

        # --- list ----------------------------------------------------------
        if re.search(r"字典|列出|列举|查看|有哪些|清单|list|show|catalog", q, re.I):
            rows = []
            for table in semantic.tables.values():
                for m in table.metrics.values():
                    formula_text = f"{m.numerator}/{m.denominator}" if m.is_ratio else f"SUM({m.numerator})"
                    rows.append([m.name, formula_text, table.name, m.unit or "—", self._t("preset", "preset"), self._t("active", "active")])
            for entry in md.entries.values():
                rows.append([
                    entry["name"],
                    entry.get("formula_text", ""),
                    entry.get("table", ""),
                    entry.get("unit") or "—",
                    self._t("custom", "custom"),
                    self._t("active", "active") if entry.get("status") == "active" else self._t("deprecated", "deprecated"),
                ])
            df = pd.DataFrame(rows, columns=[self._t("metric", "metric"), self._t("formula", "formula"), self._t("table", "table"), self._t("unit", "unit"), self._t("source", "source"), self._t("status", "status")])
            return AgentReply(
                text=self._t(
                    f"The dictionary holds {len(df)} metrics (preset {len(semantic.all_metrics())} + custom {len(md.entries)}). Every tool resolves calibers here.",
                    f"The dictionary holds {len(df)} metrics (preset {len(semantic.all_metrics())} + custom {len(md.entries)}). Every tool resolves calibers here.",
                ),
                blocks=[self._table_block(df, self._t("Metric dictionary", "Metric dictionary"), max_rows=25)],
                tool_calls=[{"tool": "metric_dict", "args": {"op": "list"}, "result": {"count": len(df)}}],
                data={"metric_dict": {"count": len(df)}},
            )

        # --- guidance ------------------------------------------------------
        return AgentReply(
            text=self._t(
                "The metric dictionary supports:\n"
                "1. **Register**: `register metric: aov2 = gmv / orders` (validated against the warehouse, then live in the semantic layer)\n"
                "2. **List**: `metric dictionary`\n"
                "3. **Deprecate**: `deprecate metric x` (kept for lineage; use a v2 name for new calibers)\n\n"
                "Before registering, check dedup rules, time zone, filters and direction — see docs/metric_dictionary.md.",
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
                return AgentReply(text=self._t(f"Removal failed: {exc}", f"Removal failed: {exc}"))
            if removed is None:
                names = ", ".join(s["name"] for s in reg.list()) or self._t("(none)", "(none)")
                return AgentReply(
                    text=self._t(f"No source specified. Currently attached: {names}", f"No source specified. Currently attached: {names}")
                )
            return AgentReply(
                text=self._t(f"Data source **{removed['name']}** removed.", f"Data source **{removed['name']}** removed."),
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
                        f"Attachment failed: {exc}\n\nSupported: CSV/TSV/Parquet/JSON/Excel files; DuckDB files; sqlite:// paths; postgres:// or mysql:// URIs.",
                        f"Attachment failed: {exc}\n\nSupported: CSV/TSV/Parquet/JSON/Excel files; DuckDB files; sqlite:// paths; postgres:// or mysql:// URIs.",
                    )
                )
            lines = [
                self._t(
                    f"Data source **{info['name']}** attached ({info.get('format') or info.get('db_type')}).",
                    f"Data source **{info['name']}** attached ({info.get('format') or info.get('db_type')}).",
                )
            ]
            if info.get("columns"):
                lines.append(self._t(f"- columns: {', '.join(info['columns'][:12])}{' …' if len(info['columns']) > 12 else ''}", f"- columns: {', '.join(info['columns'][:12])}{' …' if len(info['columns']) > 12 else ''}"))
            if info.get("row_count") is not None:
                lines.append(self._t(f"- rows: {info['row_count']:,}", f"- rows: {info['row_count']:,}"))
            if info.get("tables"):
                lines.append(self._t(f"- tables: {', '.join(info['tables'][:10])}", f"- tables: {', '.join(info['tables'][:10])}"))
            lines.append(
                self._t(
                    "\nIt is queryable right away, e.g.: `show top rows of " + info["name"] + "`, or reference the table in any SQL / analysis question.",
                    "\nIt is queryable right away, e.g.: `show top rows of " + info["name"] + "`, or reference the table in any SQL / analysis question.",
                )
            )
            sample_df = None
            if info.get("sample"):
                sample_df = pd.DataFrame(info["sample"])
            blocks = [self._table_block(sample_df, self._t(f"Source {info['name']} preview", f"Source {info['name']} preview"), max_rows=5)] if sample_df is not None else []
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
                    self._t("yes", "yes") if s.get("stale") else self._t("ok", "ok"),
                ])
            df = pd.DataFrame(rows, columns=[self._t("name", "name"), self._t("kind", "kind"), self._t("format", "format"), self._t("location", "location"), self._t("status", "status")])
            return AgentReply(
                text=self._t(
                    f"{len(sources)} external source(s) attached; reference the table name directly (e.g. `daily summary of {sources[0]['name']}`).",
                    f"{len(sources)} external source(s) attached; reference the table name directly (e.g. `daily summary of {sources[0]['name']}`).",
                ),
                blocks=[self._table_block(df, self._t("Attached data sources", "Attached data sources"))],
                tool_calls=[{"tool": "datasource", "args": {"op": "list"}, "result": {"count": len(sources)}}],
                data={"datasource": {"count": len(sources)}},
            )
        return AgentReply(
            text=self._t(
                "No external sources attached yet. You can:\n"
                "- **Spreadsheets**: `attach data source /Users/me/sales.csv as sales` (CSV/TSV/Parquet/JSON/Excel)\n"
                "- **Databases**: `connect database postgres://user:pass@host:5432/dw` or `attach database /data/warehouse.duckdb` (DuckDB/SQLite/Postgres/MySQL)\n"
                "- **List**: `data sources`\n\n"
                "Attached sources are usable from NL querying, SQL, diagnosis, modeling and dashboards.",
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
                    "The knowledge base is unavailable (docs/business_knowledge.md missing).",
                    "The knowledge base is unavailable (docs/business_knowledge.md missing).",
                )
            )
        hits = kb.search(question, k=3)
        if not hits:
            return AgentReply(
                text=self._t(
                    "Nothing relevant found. Try metric calibers, dimension semantics, the data dictionary or known incidents.",
                    "Nothing relevant found. Try metric calibers, dimension semantics, the data dictionary or known incidents.",
                )
            )
        lines = [self._t(f"{len(hits)} relevant entries from the knowledge base:", f"{len(hits)} relevant entries from the knowledge base:"), ""]
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
                pd.DataFrame(rows, columns=[self._t("entry", "entry"), self._t("source", "source"), self._t("score", "score")]),
                self._t("Knowledge base hits", "Knowledge base hits"),
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
                [self._t("null hypothesis", "null hypothesis"), hit["h0_zh"] if zh else hit["h0_en"]],
                [self._t("test statistic", "test statistic"), f"{hit['stat_name']} · {hit['formula']}"],
                [self._t("assumptions", "assumptions"), hit["assume_zh"] if zh else hit["assume_en"]],
                [self._t("when to use", "when to use"), hit["use_zh"] if zh else hit["use_en"]],
                [self._t("how to read it", "how to read it"), hit["reject_zh"] if zh else hit["reject_en"]],
                [self._t("where this agent uses it", "where this agent uses it"), ", ".join(used_by) if zh else ", ".join(used_by)],
                [self._t("reference", "reference"), hit["ref"]],
            ]
            df = pd.DataFrame(rows, columns=[self._t("item", "item"), self._t("detail", "detail")])
            text = self._t(
                f"**{hit['name_zh']}** ({hit['name_en']}). H0: {hit['h0_en']}. Statistic {hit['stat_name']}; at alpha=0.05, "
                f"p<0.05 rejects H0 → {hit['reject_en']}, otherwise {hit['retain_en']}. "
                f"Assumptions: {hit['assume_en']}. Used for {hit['use_en']}. Reference: {hit['ref']}"
                + (f"\n\nThis agent attaches it to: {', '.join(used_by)}." if used_by else ""),
                f"**{hit['name_en']}**. H0: {hit['h0_en']}. Statistic {hit['stat_name']}; at alpha=0.05, "
                f"p<0.05 rejects H0 → {hit['reject_en']}, otherwise {hit['retain_en']}. "
                f"Assumptions: {hit['assume_en']}. Used for {hit['use_en']}. Reference: {hit['ref']}"
                + (f"\n\nThis agent attaches it to: {', '.join(used_by)}." if used_by else ""),
            )
            return AgentReply(
                text=text,
                blocks=[self._table_block(df, self._t("Test definition", "Test definition"), max_rows=12)],
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
            df = pd.DataFrame(rows, columns=[self._t("family", "family"), self._t("test", "test"), self._t("H0", "H0"), self._t("stat", "stat")])
            text = self._t(
                f"A '{spec['name_en']}' conclusion carries {len(rows)} tests. **Why these**: {spec['why_en']}"
                f"\n\nEach row reports the statistic, the p-value, the decision and its reference. "
                f"Ask 'what is a Welch t-test' for the theory of any single test.",
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
                            {"label": self._t("tests", "tests"), "value": str(len(rows))},
                            {"label": self._t("readout", "readout"), "value": spec["name_zh"] if zh else spec["name_en"]},
                            {"label": self._t("multiplicity", "multiplicity"), "value": "BH-FDR" if kind == "experiment" else "—"},
                        ],
                    },
                    self._table_block(df, self._t("Tests in this readout", "Tests in this readout"), max_rows=20),
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
        df = pd.DataFrame(rows, columns=[self._t("family", "family"), self._t("test", "test"), self._t("H0", "H0"), self._t("use", "use")])
        n_families = len({c["family_zh"] for c in stat_tests.catalogue()})
        text = self._t(
            f"**{len(rows)} statistical tests** across {n_families} families, each with its null hypothesis, formula, assumptions, decision rule and citation."
            f"\n\nReading rule: at alpha=0.05, p<alpha only means the result is unlikely under H0 — not that the effect is large. "
            f"That is why every conclusion also carries an effect size, a confidence interval and the assumption checks."
            f"\n\nEach readout attaches its own battery (anomaly, experiment, DiD, ITS, modeling, PSM); the full catalogue is in `docs/statistical_tests.md`."
            f"\n\nTry: 'which tests does anomaly detection run', 'what is a Welch t-test'.",
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
                        {"label": self._t("tests", "tests"), "value": str(len(rows))},
                        {"label": self._t("families", "families"), "value": str(n_families)},
                        {"label": self._t("readouts", "readouts"), "value": str(len(test_panels.PANEL_SPEC))},
                    ],
                },
                self._table_block(df, self._t("Statistical test catalogue", "Statistical test catalogue"), max_rows=40),
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
                        "label": self._t("change", "change"),
                        "value": _fmt_pct(rel),
                        "sub": _fmt_num(delta),
                        "good": is_good,
                    },
                    {
                        "label": self._t("significance", "significance"),
                        "value": f"p={sig['p_value']:.3f}" if sig else "-",
                        "sub": self._t("vs seasonal baseline", "vs seasonal baseline"),
                    },
                ],
            }
        )
        chart_frame = daily.copy()
        chart_frame["actual"] = chart_frame["value"]
        chart_frame["expected"] = detection["frame"]["expected"].to_numpy()
        chart_frame["flagged"] = detection["frame"]["flagged"].to_numpy()
        chart_uri = charts.anomaly_chart(chart_frame, title=f"{metric.name}: actual vs expected")
        blocks.append({"type": "chart", "title": self._t("Actual vs expected", "Actual vs expected"), "image": chart_uri})

        # When a forecasting baseline was asked for, say which model answered and
        # how well it backtested. A baseline with a bad MASE produces confident
        # nonsense, so its accuracy belongs next to the finding, not in a log.
        chosen_method = str(baseline_detail.get("method") or "")
        backtest = baseline_detail.get("backtest")
        if baseline != "trailing" and chosen_method:
            acc = baseline_detail.get("accuracy") or {}
            blocks[0]["items"].append(
                {
                    "label": self._t("baseline model", "baseline model"),
                    "value": str(baseline_detail.get("label", chosen_method)),
                    "sub": f"MASE {acc['mase']:.3f}" if acc.get("mase") else self._t("rolling backtest", "rolling backtest"),
                }
            )

        contrib_uri = charts.contribution_bar(
            attribution["primary"]["segments"], attribution["primary"]["dimension"],
            title=f"{metric.name}: contribution to change",
        )
        blocks.append({"type": "chart", "title": self._t("Contribution", "Contribution"), "image": contrib_uri})

        table_df = contribution_table(attribution["primary"], top_k=10)
        blocks.append(self._table_block(
            table_df, self._t("Segment contributions", "Segment contributions"), pct_cols=("share_of_change", "growth")
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
                    self._t("segment", "segment"),
                    self._t("window change", "window change"),
                    "p",
                    self._t("worst dev", "worst dev"),
                    self._t("worst date", "worst date"),
                    "|z|",
                    self._t("severity", "severity"),
                ],
            )
            blocks.append(self._table_block(scan_df, self._t("Segment anomaly scan", "Segment anomaly scan")))

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
                    "title": self._t("Root-cause drill-down tree", "Root-cause drill-down tree"),
                    "image": charts.root_cause_tree_chart(tree["nodes"], title=f"{metric.name}: drill-down tree"),
                }
            )
            blocks.append(self._table_block(tree_table(tree, self.lang), self._t("Root-cause paths", "Root-cause paths")))
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
            blocks.append({"type": "chart", "title": self._t("Confidence: noise band", "Confidence: noise band"), "image": band_chart})
        except Exception:  # noqa: BLE001
            band_chart = ""
        blocks.append(
            self._table_block(
                confidence_table(conf["checks"], self.lang, conf["verdict"]),
                self._t("Confidence validation", "Confidence validation"),
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
                    self._t("Statistical tests (method / statistic / p / conclusion)", "Statistical tests (method / statistic / p / conclusion)"),
                    max_rows=12,
                )
            )
        except Exception:  # noqa: BLE001 - the tests must never break the diagnosis
            panel = None

        if baseline != "trailing" and chosen_method and isinstance(backtest, pd.DataFrame) and len(backtest):
            cols = [c for c in ("label", "mae", "rmse", "mase", "bias") if c in backtest.columns]
            ranked = backtest.sort_values("mase")
            base_df = ranked[cols].rename(columns={"label": self._t("method", "method")}).reset_index(drop=True)
            base_df.insert(
                0,
                self._t("chosen", "chosen"),
                ["✓" if m == chosen_method else "" for m in ranked["method"]],
            )
            blocks.append(
                self._table_block(
                    base_df,
                    self._t("Baseline backtest on one rolling origin", "Baseline backtest on one rolling origin"),
                )
            )

        direction = self._t("down", "down") if delta < 0 else self._t("up", "up")
        headline = self._t(
            f"**{metric.name}** over {window.label} is {_fmt_num(total_cur)}, "
            f"{direction} {_fmt_num(abs(delta))} vs {comparison.label} ({_fmt_pct(rel)})"
            + (f"; vs its seasonal baseline p={sig['p_value']:.3f} ({'significant' if sig['significant'] else 'not significant'})." if sig else "."),
            f"**{metric.name}** over {window.label} is {_fmt_num(total_cur)}, "
            + ("down" if delta < 0 else "up")
            + f" {_fmt_num(abs(delta))} vs {comparison.label} ({_fmt_pct(rel)})"
            + (f"; vs its seasonal baseline p={sig['p_value']:.3f} ({'significant' if sig['significant'] else 'not significant'})." if sig else "."),
        )

        driver = attribution["primary"]["segments"].iloc[0]
        share_note = ""
        if abs(driver["contribution_pct"]) > 1.1:
            share_note = self._t(" (above 100% means other segments offset it)", " (above 100% means other segments offset it)")
        driver_line = self._t(
            f"Primary driver: `{attribution['primary']['dimension']}` = **{driver['segment']}**, contributing "
            f"{_fmt_num(driver['contribution'])} ({driver['contribution_pct']:+.0%} of the move{share_note}; "
            f"its own change {_fmt_pct(driver['growth'])}).",
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
                    f"One level deeper: inside **{driver['segment']}**, `{sub['dimension']}` = {sub_top['segment']} "
                    f"contributes {_fmt_num(sub_top['contribution'])} ({sub_top['contribution_pct']:+.0%}).",
                    f"One level deeper: inside **{driver['segment']}**, `{sub['dimension']}` = {sub_top['segment']} "
                    f"contributes {_fmt_num(sub_top['contribution'])} ({sub_top['contribution_pct']:+.0%}).",
                )
            )

        if metric.is_ratio:
            lines.append(
                self._t(
                    f"Rate vs mix: of the {_fmt_num(attribution['total_delta'], 4)} change, rate effect "
                    f"{_fmt_num(attribution['primary']['rate_effect_total'], 4)}, mix effect "
                    f"{_fmt_num(attribution['primary']['mix_effect_total'], 4)}, interaction "
                    f"{_fmt_num(attribution['primary']['interaction_total'], 4)}.",
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
                    "Other dimensions: " + "; ".join(f"{d['dimension']} (top1={d['top_segment']}, concentration {d['concentration_top1']:.0%})" for d in others),
                    "Other dimensions: " + "; ".join(f"{d['dimension']} (top1={d['top_segment']}, concentration {d['concentration_top1']:.0%})" for d in others),
                )
            )

        episodes = detection["episodes"]
        if episodes:
            top_ep = episodes[0]
            lines.append(
                self._t(
                    f"Anomaly scan (90d, weekly-seasonal baseline + robust z): {len(episodes)} episode(s); largest "
                    f"{top_ep['start']}~{top_ep['end']} ({top_ep['days']}d, {_episode_kind(top_ep, self.lang)}), actual "
                    f"{_fmt_num(top_ep['actual_sum'])} vs expected "
                    f"{_fmt_num(top_ep['expected_sum'])} ({_fmt_pct(top_ep['impact_pct'])}, p={top_ep['p_value']:.4f}), "
                    f"severity {_severity(top_ep['severity'], self.lang)}.",
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
                    f"Possible level shift around {shift['date']} ({_fmt_pct(shift['rel_change'])} in level); check releases or logging changes.",
                    f"Possible level shift around {shift['date']} ({_fmt_pct(shift['rel_change'])} in level); check releases or logging changes.",
                )
            )
        if baseline != "trailing" and chosen_method:
            acc = baseline_detail.get("accuracy") or {}
            margin = baseline_detail.get("selection_margin")
            margin_note = (
                self._t(f", {margin:.1%} better than the runner-up", f", {margin:.1%} better than the runner-up")
                if isinstance(margin, float) and np.isfinite(margin) and 0 < margin < 1
                else ""
            )
            lines.append(
                self._t(
                    f"Baseline: {baseline_detail.get('label', chosen_method)} "
                    f"({'picked by backtest' if baseline == 'auto' else 'as requested'})"
                    + (f", rolling MASE {acc['mase']:.3f}" if acc.get("mase") else "")
                    + f"{margin_note}. Every expectation uses only data before that day, so an incident cannot leak into its own baseline.",
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
                    f"; next path {second['path_label']} ({share2:.0%})",
                    f"; next path {second['path_label']} ({share2:.0%})",
                )
            lines.append(
                self._t(
                    f"**Root-cause tree** (up to 3 levels of drill-down): the {'decline' if delta < 0 else 'increase'} concentrates at "
                    f"**{path_txt}** (~{share:.0%} of the root move){more}. Tree chart and path table above.",
                    f"**Root-cause tree** (up to 3 levels of drill-down): the {'decline' if delta < 0 else 'increase'} concentrates at "
                    f"**{path_txt}** (~{share:.0%} of the root move){more}. Tree chart and path table above.",
                )
            )
        if panel is not None:
            lines.append(
                self._t(
                    f"**Statistical tests** (method / statistic / p / conclusion): {panel['headline_en']} {panel['summary_en']}",
                    f"**Statistical tests** (method / statistic / p / conclusion): {panel['headline_en']} {panel['summary_en']}",
                )
            )
        lines.append(verdict_line(conf["verdict"], self.lang))
        if scan["segments"]:
            top_scan = scan["segments"][0]
            sig_part = (
                self._t(f", window change {_fmt_pct(top_scan['window_change'])} (p={top_scan['p_value']:.4f})",
                        f", window change {_fmt_pct(top_scan['window_change'])} (p={top_scan['p_value']:.4f})")
                if top_scan["window_change"] is not None
                else ""
            )
            lines.append(
                self._t(
                    f"Segment scan (`{scan['dimension']}`, per-segment seasonal residual z): {scan['summary']}. Worst is "
                    f"**{top_scan['segment']}**{sig_part}, deviating {_fmt_pct(top_scan['worst_rel_dev'])} on "
                    f"{top_scan['worst_date']} (|z|={top_scan['max_abs_z']:.1f}, {_severity(top_scan['severity'], self.lang)}).",
                    f"Segment scan (`{scan['dimension']}`, per-segment seasonal residual z): {scan['summary']}. Worst is "
                    f"**{top_scan['segment']}**{sig_part}, deviating {_fmt_pct(top_scan['worst_rel_dev'])} on "
                    f"{top_scan['worst_date']} (|z|={top_scan['max_abs_z']:.1f}, {_severity(top_scan['severity'], self.lang)}).",
                )
            )

        lines += ["", self._t("**Suggested next steps**", "**Suggested next steps**")]
        steps = [
            self._t(
                f"Audit instrumentation for {driver['segment']} (numerator/denominator from the same source?).",
                f"Audit instrumentation for {driver['segment']} (numerator/denominator from the same source?).",
            ),
            self._t(
                "Cross-check with a paired metric (GMV down without an order drop usually means price/mix).",
                "Cross-check with a paired metric (GMV down without an order drop usually means price/mix).",
            ),
        ]
        if sig and not sig["significant"]:
            segment_sig = scan["segments"][0] if scan["segments"] else None
            if segment_sig and segment_sig.get("p_value") is not None and segment_sig["p_value"] < 0.05:
                steps.insert(
                    0,
                    self._t(
                        f"The topline change is not significant (p={sig['p_value']:.3f}) but the segment is: "
                        f"`{scan['dimension']}`={segment_sig['segment']} moved {_fmt_pct(segment_sig['window_change'])} "
                        f"(p={segment_sig['p_value']:.4f}). Trust the segment, not the topline average.",
                        f"The topline change is not significant (p={sig['p_value']:.3f}) but the segment is: "
                        f"`{scan['dimension']}`={segment_sig['segment']} moved {_fmt_pct(segment_sig['window_change'])} "
                        f"(p={segment_sig['p_value']:.4f}). Trust the segment, not the topline average.",
                    ),
                )
            else:
                steps.insert(
                    0,
                    self._t(
                        "The change is not statistically significant; extend the window or use week-over-week.",
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
                f"{metric.name} moved {'down' if delta < 0 else 'up'} {_fmt_pct(rel)} ({_fmt_num(delta)}); "
                f"main driver {attribution['primary']['dimension']}={driver['segment']} "
                f"({driver['contribution_pct']:+.0%} of the move).",
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
                    self._t("start", "start"),
                    self._t("end", "end"),
                    self._t("days", "days"),
                    self._t("kind", "kind"),
                    self._t("impact", "impact"),
                    "p",
                    self._t("severity", "severity"),
                ],
            )
            episode_text = self._t(
                f"{len(episodes)} episode(s) in 90 days; largest {top_ep['start']}~{top_ep['end']} "
                f"({top_ep['days']}d, {_episode_kind(top_ep, self.lang)}), {_fmt_pct(top_ep['impact_pct'])}, "
                f"p={top_ep['p_value']:.4f}.",
                f"{len(episodes)} episode(s) in 90 days; largest {top_ep['start']}~{top_ep['end']} "
                f"({top_ep['days']}d, {_episode_kind(top_ep, 'en')}), {_fmt_pct(top_ep['impact_pct'])}, "
                f"p={top_ep['p_value']:.4f}.",
            )
            self._record(
                episode_text,
                Section(
                    title=self._t("Anomaly episodes (last 90 days)", "Anomaly episodes (last 90 days)"),
                    prose=episode_text,
                    table=episode_df,
                    chart=chart_uri,
                ),
            )
        if tree:
            self._record(
                self._t(
                    f"Root-cause tree: {tree['depth_reached']} levels, {tree['n_nodes']} nodes; "
                    f"top path {tree['leaves'][0]['path_label'] if tree['leaves'] else '-'}.",
                    f"Root-cause tree: {tree['depth_reached']} levels, {tree['n_nodes']} nodes; "
                    f"top path {tree['leaves'][0]['path_label'] if tree['leaves'] else '-'}.",
                ),
                Section(
                    title=self._t("Root-cause drill-down tree", "Root-cause drill-down tree"),
                    prose=self._plain_line(tree["narrative"]),
                    table=tree_table(tree, self.lang),
                    chart=charts.root_cause_tree_chart(tree["nodes"], title=f"{metric.name}: drill-down tree"),
                ),
            )
        self._record(
            verdict_line(conf["verdict"], self.lang),
            Section(
                title=self._t("Confidence validation", "Confidence validation"),
                prose=verdict_line(conf["verdict"], self.lang)
                + " " + self._t(
                    "Checks cover window significance, baseline backtest quality, persistence, leave-one-out outlier dependence and segment agreement.",
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
            df = pd.DataFrame(rows, columns=["type", "Chinese", "English", "randomisation_unit", "when_to_use"])
            block = self._table_block(df, self._t("Experiment archetypes", "Experiment archetypes"), max_rows=20)
            block["columns"] = list(df.columns)
            block["rows"] = [[str(v) for v in r] for r in rows]
            text = self._t(
                f"{len(EXPERIMENT_TYPES)} archetypes are built in, each with its randomisation unit, analysis path and pitfalls. "
                "Describe your situation and I will pick one and size it.",
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
                    {"label": self._t("per arm", "per arm"), "value": f"{sizing['n_control']:,}"},
                    {"label": self._t("total", "total"), "value": f"{sizing['n_total']:,}"},
                    {"label": self._t("duration", "duration"), "value": f"{duration['recommended_days']}d"},
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
        blocks.append({"type": "chart", "title": self._t("Sample size vs MDE", "Sample size vs MDE"), "image": chart_uri})

        type_info = EXPERIMENT_TYPES[recommendation["primary"]]
        when_text = type_info.get("when_zh") if self.lang == "zh" else type_info["when"]
        lines = [
            self._t(
                f"**Archetype**: {type_info['name_en']} ({recommendation['primary']}) - {when_text}",
                f"**Archetype**: {type_info['name_en']} ({recommendation['primary']}) - {when_text}",
            ),
            "",
            self._t(
                f"**Sizing**: baseline {design.baseline:.4g}, MDE {_fmt_pct(design.mde_rel) if design.mde_rel else _fmt_num(design.mde_abs,4)}, "
                f"alpha={design.alpha}, power={design.power} -> **{sizing['n_control']:,}** per arm, "
                f"**{sizing['n_total']:,}** total (allocation {design.allocation:.0%} applied).",
                f"**Sizing**: baseline {design.baseline:.4g}, MDE {_fmt_pct(design.mde_rel) if design.mde_rel else _fmt_num(design.mde_abs,4)}, "
                f"alpha={design.alpha}, power={design.power} -> **{sizing['n_control']:,}** per arm, "
                f"**{sizing['n_total']:,}** total (allocation {design.allocation:.0%} applied).",
            ),
            self._t(
                f"**Duration**: at {duration['eligible_users_per_day']:,.0f} eligible users/day, ~{duration['raw_days']} days; "
                f"run **{duration['recommended_days']} days** to cover whole weeks.",
                f"**Duration**: at {duration['eligible_users_per_day']:,.0f} eligible users/day, ~{duration['raw_days']} days; "
                f"run **{duration['recommended_days']} days** to cover whole weeks.",
            ),
            "",
            self._t("**Analysis plan**", "**Analysis plan**"),
        ]
        lines += [f"- {p}" for p in result["analysis_plan"]]
        lines += ["", self._t("**Guardrails**", "**Guardrails**")] + [f"- {g}" for g in result["guardrails"]]
        lines += ["", self._t("**Risks**", "**Risks**")] + [f"- {r}" for r in result["risks"][:4]]
        lines += ["", self._t("**Decision rule**", "**Decision rule**"), result["decision_rule"]]
        if recommendation["secondary"]:
            lines += [
                "",
                self._t(
                    "**Alternatives**: " + ", ".join(f"{s} ({EXPERIMENT_TYPES[s]['name_en']})" for s in recommendation["secondary"]),
                    "**Alternatives**: " + ", ".join(f"{s} ({EXPERIMENT_TYPES[s]['name_en']})" for s in recommendation["secondary"]),
                ),
            ]

        text = "\n".join(lines)
        section = Section(
            title=self._t("Experiment design", "Experiment design"),
            prose=result["markdown"].split("\n", 2)[-1],
            table=pd.DataFrame(
                [
                    [self._t("per arm", "per arm"), f"{sizing['n_control']:,}"],
                    [self._t("total", "total"), f"{sizing['n_total']:,}"],
                    [self._t("recommended days", "recommended days"), f"{duration['recommended_days']}"],
                    [self._t("randomisation unit", "randomisation unit"), type_info["randomization_unit"]],
                ],
                columns=[self._t("item", "item"), self._t("value", "value")],
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
                "**Do not conclude yet**: SRM failed, so randomisation or logging is broken and every metric difference is untrustworthy.",
                "**Do not conclude yet**: SRM failed, so randomisation or logging is broken and every metric difference is untrustworthy.",
            )
        if primary["significant_adjusted"] and direction_ok:
            return self._t(
                f"**Ship**: {primary['metric']} lifts {_fmt_pct(lift)} (BH-adjusted p={primary['p_adjusted']:.4f}, CI excludes 0), "
                "provided guardrails hold and segments agree in sign.",
                f"**Ship**: {primary['metric']} lifts {_fmt_pct(lift)} (BH-adjusted p={primary['p_adjusted']:.4f}, CI excludes 0), "
                "provided guardrails hold and segments agree in sign.",
            )
        if primary["significant_adjusted"]:
            return self._t(
                f"**Do not ship**: {primary['metric']} is significant in the wrong direction ({_fmt_pct(lift)}). Find the cause first.",
                f"**Do not ship**: {primary['metric']} is significant in the wrong direction ({_fmt_pct(lift)}). Find the cause first.",
            )
        return self._t(
            f"**Inconclusive**: {primary['metric']} moved {_fmt_pct(lift)} (p={primary['p_value']:.3f}). Either the test is "
            "underpowered for an effect this small, or the effect is genuinely ~0; extend the run or apply CUPED.",
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
                        "label": self._t("primary lift", "primary lift"),
                        "value": _fmt_pct(primary["rel_lift"]),
                        "sub": f"p={primary['p_value']:.4f}",
                        "good": primary["rel_lift"] > 0,
                    },
                    {
                        "label": "CUPED",
                        "value": _fmt_pct(best_cuped["variance_reduction"], 0) if best_cuped else "-",
                        "sub": self._t("variance cut (best metric)", "variance cut (best metric)"),
                    },
                ],
            }
        ]
        chart_uri = charts.effect_forest(result["metrics"])
        if chart_uri:
            blocks.append({"type": "chart", "title": self._t("Effects with 95% CI", "Effects with 95% CI"), "image": chart_uri})

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
                    self._t("yes", "yes") if m["significant_adjusted"] else self._t("no", "no"),
                ]
            )
        df_rows = pd.DataFrame(rows, columns=[self._t("metric", "metric"), "control", "treatment", "lift", "95% CI", "p", "p_BH", self._t("verdict", "verdict")])
        blocks.append(self._table_block(df_rows, self._t("Metric readout", "Metric readout"), pct_cols=("lift",)))

        seg_rows = [
            [s["segment_col"], s["segment"], _fmt_pct(s["rel_lift"]), f"{s['p_value']:.3f}", f"{s['n']:,}"]
            for s in result["segments"][:12]
        ]
        if seg_rows:
            seg_df = pd.DataFrame(seg_rows, columns=[self._t("dimension", "dimension"), self._t("segment", "segment"), "lift", "p", "n"])
            blocks.append(self._table_block(seg_df, self._t("Segment heterogeneity (exploratory)", "Segment heterogeneity (exploratory)")))

        if perm_chart:
            blocks.append(
                {
                    "type": "chart",
                    "title": self._t("Confidence: permutation null", "Confidence: permutation null"),
                    "image": perm_chart,
                }
            )
        blocks.append(
            self._table_block(
                confidence_table(conf["checks"], self.lang, conf["verdict"]),
                self._t("Confidence validation", "Confidence validation"),
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
                    self._t("Statistical tests (proportion + revenue)", "Statistical tests (proportion + revenue)"),
                    max_rows=18,
                )
            )
        except Exception:  # noqa: BLE001
            panel = None

        lines = [
            self._t(
                f"**SRM check**: {srm['message']} (observed={srm['observed']}, p={srm['p_value']:.3f})."
                + ("" if srm["passed"] else " **Do not read metrics until randomisation is fixed.**"),
                f"**SRM check**: {srm['message']} (observed={srm['observed']}, p={srm['p_value']:.3f})."
                + ("" if srm["passed"] else " **Do not read metrics until randomisation is fixed.**"),
            ),
            "",
            self._t(
                f"**Primary metric** ({primary['metric']}): control {_fmt_num(primary['control'], 4)} vs treatment "
                f"{_fmt_num(primary['treatment'], 4)}, relative lift **{_fmt_pct(primary['rel_lift'])}**, "
                f"95% CI [{_fmt_num(primary['ci_abs'][0], 4)}, {_fmt_num(primary['ci_abs'][1], 4)}], "
                f"p={primary['p_value']:.4f}, BH-adjusted p={primary['p_adjusted']:.4f}.",
                f"**Primary metric** ({primary['metric']}): control {_fmt_num(primary['control'], 4)} vs treatment "
                f"{_fmt_num(primary['treatment'], 4)}, relative lift **{_fmt_pct(primary['rel_lift'])}**, "
                f"95% CI [{_fmt_num(primary['ci_abs'][0], 4)}, {_fmt_num(primary['ci_abs'][1], 4)}], "
                f"p={primary['p_value']:.4f}, BH-adjusted p={primary['p_adjusted']:.4f}.",
            ),
        ]
        if best_cuped and cuped_metric:
            lines.append(
                self._t(
                    f"**CUPED** ({cuped_metric['metric']}): corr with pre-period revenue {best_cuped['corr']:.2f}, "
                    f"variance cut {best_cuped['variance_reduction']:.0%}; adjusted lift {_fmt_pct(best_cuped['rel_lift'])} "
                    f"(p={best_cuped['p_value']:.4f}). Binary metrics have low correlation by nature - CUPED pays off on revenue-like metrics.",
                    f"**CUPED** ({cuped_metric['metric']}): corr with pre-period revenue {best_cuped['corr']:.2f}, "
                    f"variance cut {best_cuped['variance_reduction']:.0%}; adjusted lift {_fmt_pct(best_cuped['rel_lift'])} "
                    f"(p={best_cuped['p_value']:.4f}). Binary metrics have low correlation by nature - CUPED pays off on revenue-like metrics.",
                )
            )

        if perm is not None:
            lines.append(
                self._t(
                    f"**Permutation test**: 500 label shuffles place the observed diff at p={perm['p_perm']:.4f} in the null, "
                    f"{'agreeing' if (perm['p_perm'] < 0.05) == primary['significant_adjusted'] else 'DISAGREEING'} with the analytic test.",
                    f"**Permutation test**: 500 label shuffles place the observed diff at p={perm['p_perm']:.4f} in the null, "
                    f"{'agreeing' if (perm['p_perm'] < 0.05) == primary['significant_adjusted'] else 'DISAGREEING'} with the analytic test.",
                )
            )
        if panel is not None:
            lines.append(
                self._t(
                    f"**Statistical tests** (method / statistic / p / conclusion): {panel['headline_en']} {panel['summary_en']}",
                    f"**Statistical tests** (method / statistic / p / conclusion): {panel['headline_en']} {panel['summary_en']}",
                )
            )
        lines.append(verdict_line(conf["verdict"], self.lang))

        verdict = self._experiment_verdict(primary, srm)
        lines += ["", self._t("**Verdict**", "**Verdict**"), verdict]

        self._record(
            self._t(
                f"AB readout: {primary['metric']} lift {_fmt_pct(primary['rel_lift'])} (p={primary['p_value']:.4f}), "
                f"SRM {'pass' if srm['passed'] else 'fail'}.",
                f"AB readout: {primary['metric']} lift {_fmt_pct(primary['rel_lift'])} (p={primary['p_value']:.4f}), "
                f"SRM {'pass' if srm['passed'] else 'fail'}.",
            ),
            Section(
                title=self._t("AB test readout", "AB test readout"),
                prose=self._t(
                    f"Primary metric {primary['metric']}: control {_fmt_num(primary['control'], 4)} vs treatment "
                    f"{_fmt_num(primary['treatment'], 4)}, relative lift {_fmt_pct(primary['rel_lift'])} "
                    f"(95% CI {_fmt_num(primary['ci_abs'][0], 4)} ~ {_fmt_num(primary['ci_abs'][1], 4)}, "
                    f"p={primary['p_value']:.4f}, BH-adjusted p={primary['p_adjusted']:.4f}); "
                    f"SRM {'pass' if srm['passed'] else 'fail'} (p={srm['p_value']:.3f}).",
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
                title=self._t("Experiment confidence validation", "Experiment confidence validation"),
                prose=verdict_line(conf["verdict"], self.lang)
                + " " + self._t(
                    "Checks cover SRM, BH-adjusted significance, permutation cross-check, CUPED direction agreement and adverse segments.",
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
                    {"label": self._t("relative (log DiD)", "relative (log DiD)"), "value": _fmt_pct(did_log["relative_effect"]), "sub": self._t("preferred", "preferred")},
                    {"label": self._t("parallel trends", "parallel trends"), "value": "PASS" if es["parallel_trends_ok"] else "FAIL", "sub": f"p={es['pre_trend_joint_p']:.3f}", "good": es["parallel_trends_ok"]},
                ],
            },
            {"type": "chart", "title": self._t("Event study", "Event study"), "image": chart_uri},
        ]
        if placebo_chart_uri:
            blocks.append(
                {
                    "type": "chart",
                    "title": self._t("Confidence: in-space placebo", "Confidence: in-space placebo"),
                    "image": placebo_chart_uri,
                }
            )
        blocks.append(
            self._table_block(
                confidence_table(conf["checks"], self.lang, conf["verdict"]),
                self._t("Confidence validation", "Confidence validation"),
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
                    self._t("Statistical tests (parallel trends + residual diagnostics)", "Statistical tests (parallel trends + residual diagnostics)"),
                    max_rows=10,
                )
            )
        except Exception:  # noqa: BLE001
            panel = None
        rows = [[c["rel_period"], _fmt_num(c["coef"]), _fmt_num(c["se"]), f"{c['p_value']:.3f}"] for c in es["coefficients"]]
        df = pd.DataFrame(rows, columns=[self._t("relative period", "relative period"), "coef", "se", "p"])
        blocks.append(self._table_block(df, self._t("Event study coefficients", "Event study coefficients"), max_rows=20))

        lines = [
            self._t(
                f"**Two-way FE DiD**: ATT of the policy on city GMV = **{_fmt_num(did['att'])}**, "
                f"95% CI [{_fmt_num(did['ci'][0])}, {_fmt_num(did['ci'][1])}], p={did['p_value']:.4f}, "
                f"{did['n_obs']:,} city-weeks over {did['n_units']} cities.",
                f"**Two-way FE DiD**: ATT of the policy on city GMV = **{_fmt_num(did['att'])}**, "
                f"95% CI [{_fmt_num(did['ci'][0])}, {_fmt_num(did['ci'][1])}], p={did['p_value']:.4f}, "
                f"{did['n_obs']:,} city-weeks over {did['n_units']} cities.",
            ),
            "",
            self._t(
                f"**Parallel trends**: joint test on pre-period leads p={es['pre_trend_joint_p']:.3f} -> "
                + ("passes, DiD is credible." if es["parallel_trends_ok"] else "fails, the ATT is biased; switch to synthetic control or PSM."),
                f"**Parallel trends**: joint test on pre-period leads p={es['pre_trend_joint_p']:.3f} -> "
                + ("passes, DiD is credible." if es["parallel_trends_ok"] else "fails, the ATT is biased; switch to synthetic control or PSM."),
            ),
            self._t(
                f"**Note**: the ATT is absolute (~{_fmt_num(did['att'])}, i.e. {_fmt_pct(did['relative_effect'])} of the treated "
                f"counterfactual). With multiplicative growth the log specification is safer and gives a relative effect of "
                f"**{_fmt_pct(did_log['relative_effect'])}** (p={did_log['p_value']:.4f}). The gap between the two is the size effect - "
                "always state which scale you are reporting.",
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
                    f"**Placebo**: rerunning DiD with the treated label permuted onto {placebo['n']} fake city sets, "
                    f"the real ATT exceeds {tail:.0%} of placebo effects"
                    + (" - the design separates policy from city noise." if tail <= 0.05 else " - weak separation, be careful."),
                    f"**Placebo**: rerunning DiD with the treated label permuted onto {placebo['n']} fake city sets, "
                    f"the real ATT exceeds {tail:.0%} of placebo effects"
                    + (" - the design separates policy from city noise." if tail <= 0.05 else " - weak separation, be careful."),
                )
            )
        if panel is not None:
            lines.append(
                self._t(
                    f"**Statistical tests** (method / statistic / p / conclusion): {panel['headline_en']} {panel['summary_en']}",
                    f"**Statistical tests** (method / statistic / p / conclusion): {panel['headline_en']} {panel['summary_en']}",
                )
            )
        lines.append(verdict_line(conf["verdict"], self.lang))
        self._record(
            self._t(
                f"DiD ATT {_fmt_num(did['att'])} (p={did['p_value']:.3f}), relative {_fmt_pct(did_log['relative_effect'])}, "
                f"parallel trends {'ok' if es['parallel_trends_ok'] else 'violated'}.",
                f"DiD ATT {_fmt_num(did['att'])} (p={did['p_value']:.3f}), relative {_fmt_pct(did_log['relative_effect'])}, "
                f"parallel trends {'ok' if es['parallel_trends_ok'] else 'violated'}.",
            ),
            Section(
                title=self._t("Causal: difference-in-differences", "Causal: difference-in-differences"),
                prose=self._t(
                    f"Policy ATT = {_fmt_num(did['att'])} ({_fmt_pct(did['relative_effect'])} of the treated counterfactual, "
                    f"p={did['p_value']:.4f}); log specification relative effect {_fmt_pct(did_log['relative_effect'])}; "
                    f"parallel-trends joint test p={es['pre_trend_joint_p']:.3f}, {'passes' if es['parallel_trends_ok'] else 'fails'}.",
                    did["narrative"],
                ),
                table=df,
                chart=chart_uri,
            ),
        )
        self._record(
            verdict_line(conf["verdict"], self.lang),
            Section(
                title=self._t("DiD confidence validation", "DiD confidence validation"),
                prose=verdict_line(conf["verdict"], self.lang)
                + " " + self._t(
                    "Checks cover parallel trends, pre-period leads, in-space placebo, specification agreement and treated coverage.",
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
            columns=[self._t("donor", "donor"), self._t("weight", "weight")],
        )
        blocks = [
            {
                "type": "kpi",
                "items": [
                    {"label": self._t("average effect", "average effect"), "value": _fmt_num(sc["post_avg_effect"]), "sub": self._t("post period", "post period")},
                    {"label": self._t("pre-fit R2", "pre-fit R2"), "value": f"{sc['pre_fit_r2']:.3f}", "sub": f"RMSE {_fmt_num(sc['pre_rmse'])}"},
                    {"label": self._t("placebo p", "placebo p"), "value": f"{sc['placebo_p_value']:.3f}", "sub": f"ratio {sc['post_pre_ratio']:.2f}", "good": sc["placebo_p_value"] < 0.1},
                ],
            },
            {"type": "chart", "title": self._t("Actual vs synthetic", "Actual vs synthetic"), "image": chart_uri},
            self._table_block(weights, self._t("Donor weights", "Donor weights"), max_rows=8),
        ]
        lines = [
            self._t(
                f"**Synthetic control**: donor pool of {sc['n_donors']} untreated cities ({sc['n_effective_donors']} with weight) "
                f"reproduces city_00's counterfactual. Average post-period gap **{_fmt_num(sc['post_avg_effect'])}** "
                f"({_fmt_pct(sc['relative_effect'])} relative), pre-period RMSE {_fmt_num(sc['pre_rmse'])} "
                f"(R²={sc['pre_fit_r2']:.3f}).",
                f"**Synthetic control**: donor pool of {sc['n_donors']} untreated cities ({sc['n_effective_donors']} with weight) "
                f"reproduces city_00's counterfactual. Average post-period gap **{_fmt_num(sc['post_avg_effect'])}** "
                f"({_fmt_pct(sc['relative_effect'])} relative), pre-period RMSE {_fmt_num(sc['pre_rmse'])} "
                f"(R²={sc['pre_fit_r2']:.3f}).",
            ),
            self._t(
                f"**Placebo inference**: post/pre RMSE ratio {sc['post_pre_ratio']:.2f}, rank p={sc['placebo_p_value']:.3f} "
                f"across {sc['n_placebos']} placebos (floor is {1 / (sc['n_placebos'] + 1):.3f}) -> "
                + ("the effect beats random assignment." if sc["placebo_p_value"] < 0.1 else "indistinguishable from placebo noise."),
                f"**Placebo inference**: post/pre RMSE ratio {sc['post_pre_ratio']:.2f}, rank p={sc['placebo_p_value']:.3f} "
                f"across {sc['n_placebos']} placebos (floor is {1 / (sc['n_placebos'] + 1):.3f}) -> "
                + ("the effect beats random assignment." if sc["placebo_p_value"] < 0.1 else "indistinguishable from placebo noise."),
            ),
            self._t(
                "**Caveat**: no standard error by construction; inference is rank-based on placebos, so a larger donor pool helps.",
                "**Caveat**: no standard error by construction; inference is rank-based on placebos, so a larger donor pool helps.",
            ),
        ]
        self._record(
            self._t(
                f"Synthetic control effect {_fmt_num(sc['post_avg_effect'])} ({_fmt_pct(sc['relative_effect'])} relative), "
                f"placebo p={sc['placebo_p_value']:.3f}.",
                f"Synthetic control effect {_fmt_num(sc['post_avg_effect'])} ({_fmt_pct(sc['relative_effect'])} relative), "
                f"placebo p={sc['placebo_p_value']:.3f}.",
            ),
            Section(
                title=self._t("Causal: synthetic control", "Causal: synthetic control"),
                prose=self._t(
                    f"Average post-period gap {_fmt_num(sc['post_avg_effect'])} ({_fmt_pct(sc['relative_effect'])} relative), "
                    f"pre-period fit R²={sc['pre_fit_r2']:.3f}, placebo p={sc['placebo_p_value']:.3f} "
                    f"({sc.get('n_donors', len(weights))} donor cities).",
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
        df_rows = pd.DataFrame(rows, columns=[self._t("estimator", "estimator"), self._t("effect", "effect"), "95% CI", "p"])

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
            columns=[self._t("covariate", "covariate"), "SMD before", "SMD after", "p (after)", self._t("status", "status")],
        )
        blocks = [
            {
                "type": "kpi",
                "items": [
                    {"label": self._t("naive diff", "naive diff"), "value": _fmt_num(naive), "sub": self._t("confounded", "confounded")},
                    {"label": "PSM ATT", "value": _fmt_num(atts.get("psm", {}).get("att")), "sub": f"pairs {atts.get('psm', {}).get('n_matched_pairs', '-')}"},
                    {"label": "AIPW ATE", "value": _fmt_num(aipw.get("estimate")), "sub": self._t("doubly robust", "doubly robust")},
                ],
            },
            self._table_block(df_rows, self._t("Estimator comparison", "Estimator comparison")),
        ]
        if len(balance):
            blocks.append(self._table_block(balance, self._t("Covariate balance", "Covariate balance")))
        psm = atts.get("psm", {})
        # Confidence card: balance, overlap, estimator agreement, significance.
        conf = matching_confidence(psm, ipw if isinstance(ipw, dict) else {}, aipw if isinstance(aipw, dict) else {})
        if isinstance(psm.get("ps_overlap"), dict):
            try:
                blocks.append(
                    {
                        "type": "chart",
                        "title": self._t("Propensity score overlap", "Propensity score overlap"),
                        "image": charts.ps_overlap_chart(psm["ps_overlap"]),
                    }
                )
            except Exception:  # noqa: BLE001 - a chart must never fail the answer
                pass
        blocks.append(
            self._table_block(
                confidence_table(conf["checks"], self.lang, conf["verdict"]),
                self._t("Confidence validation", "Confidence validation"),
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
                    self._t("Statistical tests (post-match balance + overlap)", "Statistical tests (post-match balance + overlap)"),
                    max_rows=8,
                )
            )
        except Exception:  # noqa: BLE001
            panel = None

        # Robustness & sensitivity: fake-treatment refutation, Rosenbaum bounds
        # on hidden bias, and rule-based method switching when assumptions fail.
        from .tools.causal import method_selection_advice as _method_advice
        from .tools.causal import placebo_refutation as _placebo_refutation
        from .tools.causal import rosenbaum_bounds as _rosenbaum_bounds

        placebo = None
        try:
            placebo = _placebo_refutation(df, "promo", "post_revenue", covariates, n_placebo=8)
        except Exception:  # noqa: BLE001 - diagnostics must never fail the answer
            pass
        rosenbaum = []
        try:
            diffs_arr = np.asarray(psm.get("pair_diffs") or [], dtype=float)
            if diffs_arr.size:
                rosenbaum = _rosenbaum_bounds(diffs_arr)
        except Exception:  # noqa: BLE001
            rosenbaum = []
        advice = _method_advice(psm)

        if placebo and placebo.get("robust") is not None:
            blocks.append(
                self._table_block(
                    pd.DataFrame(
                        [
                            [self._t("Real ATT", "Real ATT"), _fmt_num(placebo["real_att"])],
                            [self._t("Placebo mean", "Placebo mean"), _fmt_num(placebo["placebo_mean"])],
                            [
                                self._t("Placebo 90% range", "Placebo 90% range"),
                                f"[{_fmt_num(placebo['placebo_ci'][0])}, {_fmt_num(placebo['placebo_ci'][1])}]",
                            ],
                            [
                                self._t("Verdict", "Verdict"),
                                self._t(
                                    "real effect outside the placebo band — refutation passed",
                                    "real effect outside the placebo band — refutation passed",
                                )
                                if placebo["robust"]
                                else self._t(
                                    "real effect INSIDE the placebo band — not separable from noise",
                                    "real effect INSIDE the placebo band — not separable from noise",
                                ),
                            ],
                        ],
                        columns=[self._t("Placebo refutation", "Placebo refutation"), self._t("Result", "Result")],
                    ),
                    self._t("Robustness: placebo refutation (shuffled treatment)", "Robustness: placebo refutation (shuffled treatment)"),
                )
            )
        if rosenbaum:
            blocks.append(
                self._table_block(
                    pd.DataFrame(
                        [
                            [f"Γ={r['gamma']}", f"{r['p_lower']:.4f}", f"{r['p_upper']:.4f}", "OK" if r["survives"] else "FAIL"]
                            for r in rosenbaum
                        ],
                        columns=[
                            self._t("Hidden bias Γ", "Hidden bias Γ"),
                            "p lower",
                            "p upper",
                            self._t("survives p<.05", "survives p<.05"),
                        ],
                    ),
                    self._t(
                        "Sensitivity: Rosenbaum bounds (unobserved confounding up to odds ratio Γ)",
                        "Sensitivity: Rosenbaum bounds (unobserved confounding up to odds ratio Γ)",
                    ),
                )
            )
        blocks.append(
            self._table_block(
                pd.DataFrame(
                    [[a["situation"], a["zh"] if self.lang == "zh" else a["en"]] for a in advice],
                    columns=[self._t("Assumption status", "Assumption status"), self._t("What to do instead", "What to do instead")],
                ),
                self._t("Method selection when assumptions fail", "Method selection when assumptions fail"),
            )
        )

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
            engine = self._t("SHAP", "SHAP") if cm_fa["engine"] == "shap" else self._t("permutation importance", "permutation importance")
            blocks.append(
                {
                    "type": "chart",
                    "title": self._t(
                        f"Post-causal feature analysis ({engine}, treatment `promo` included)",
                        f"Post-causal feature analysis ({engine}, treatment `promo` included)",
                    ),
                    "image": charts.importance_bar(cm_fa["importance"], title="Outcome drivers after causal estimation"),
                }
            )
        except Exception:  # noqa: BLE001 - explanation is additive, never blocking
            pass

        lines = [
            self._t(
                f"**Problem**: estimate the causal effect (ATT) of promo exposure on next-month revenue (post_revenue). "
                f"Exposure is not randomised, so selection bias must be removed. **Assumptions**: (1) conditional "
                f"ignorability given {', '.join(covariates)}; (2) overlap — every treated user has comparable controls; "
                f"(3) SUTVA — no spillovers. (1)/(3) are argued from design; (2) is diagnosable from data.",
                f"**Problem**: estimate the causal effect (ATT) of promo exposure on next-month revenue (post_revenue). "
                f"Exposure is not randomised, so selection bias must be removed. **Assumptions**: (1) conditional "
                f"ignorability given {', '.join(covariates)}; (2) overlap — every treated user has comparable controls; "
                f"(3) SUTVA — no spillovers. (1)/(3) are argued from design; (2) is diagnosable from data.",
            ),
            self._t(
                f"**Why the naive difference fails**: promo exposure is not random (high-value users are more likely to be targeted), "
                f"so the raw gap of {_fmt_num(naive)} mixes selection bias into the effect.",
                f"**Why the naive difference fails**: promo exposure is not random (high-value users are more likely to be targeted), "
                f"so the raw gap of {_fmt_num(naive)} mixes selection bias into the effect.",
            ),
            "",
            self._t(
                f"**PSM**: {atts.get('psm', {}).get('n_matched_pairs', 0):,} matched pairs, ATT = "
                f"{_fmt_num(atts.get('psm', {}).get('att'))}; **IPW** ATE = {_fmt_num(ipw.get('estimate'))}; "
                f"**AIPW** ATE = {_fmt_num(aipw.get('estimate'))}.",
                f"**PSM**: {atts.get('psm', {}).get('n_matched_pairs', 0):,} matched pairs, ATT = "
                f"{_fmt_num(atts.get('psm', {}).get('att'))}; **IPW** ATE = {_fmt_num(ipw.get('estimate'))}; "
                f"**AIPW** ATE = {_fmt_num(aipw.get('estimate'))}.",
            ),
            *(
                [
                    self._t(
                        f"**Feature engineering (pre-model collinearity handling)**: dropped {', '.join(psm['pruned_covariates'])} "
                        f"(pairwise |r|>0.95 with a retained covariate) to keep the propensity model well-conditioned — "
                        f"causal inference reuses the exact same preprocessing as regression.",
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
                        f"**Overlap**: common support [{support['low']:.2f}, {support['high']:.2f}], "
                        f"{support['n_treated_outside']} treated units outside ({support['share_outside']:.1%})"
                        + (" — their effect is not identified by matching." if support["n_treated_outside"] else " — overlap is healthy."),
                        f"**Overlap**: common support [{support['low']:.2f}, {support['high']:.2f}], "
                        f"{support['n_treated_outside']} treated units outside ({support['share_outside']:.1%})"
                        + (" — their effect is not identified by matching." if support["n_treated_outside"] else " — overlap is healthy."),
                    )
                    for support in [psm.get("common_support", {})]
                    if support
                ]
            ),
            self._t(
                "**Robustness**: agreement across PSM/IPW/AIPW is more convincing than any single p-value. "
                "Large disagreement usually signals poor overlap.",
                "**Robustness**: agreement across PSM/IPW/AIPW is more convincing than any single p-value. "
                "Large disagreement usually signals poor overlap.",
            ),
            "",
            self._t(
                "**Identifying assumptions**: conditional ignorability plus overlap. Neither is testable from data; argue them from design.",
                "**Identifying assumptions**: conditional ignorability plus overlap. Neither is testable from data; argue them from design.",
            ),
        ]
        if panel is not None:
            lines.append(
                self._t(
                    f"**Statistical tests** (method / statistic / p / conclusion): {panel['headline_en']} {panel['summary_en']}",
                    f"**Statistical tests** (method / statistic / p / conclusion): {panel['headline_en']} {panel['summary_en']}",
                )
            )
        lines.append(verdict_line(conf["verdict"], self.lang))

        # Conclusion with uncertainty + hidden-bias robustness (after all the
        # diagnostics above, so the numbers it quotes are the validated ones).
        gammas_ok = [r for r in rosenbaum if r["survives"]]
        gamma_max = max((r["gamma"] for r in gammas_ok), default=0)
        att_val = atts.get("psm", {}).get("att")
        ci_lo, ci_hi = atts.get("psm", {}).get("ci", (float("nan"), float("nan")))
        monthly = att_val * psm.get("n_treated", 0) if att_val is not None else None
        lines.append(
            self._t(
                f"**Conclusion & uncertainty**: promo lifts monthly revenue per exposed user by **{_fmt_num(att_val)}** "
                f"(bootstrap 95% CI [{_fmt_num(ci_lo)}, {_fmt_num(ci_hi)}], p={atts.get('psm', {}).get('p_value', float('nan')):.3f}). "
                f"Business read: ≈¥{format(round(monthly), ',') if monthly is not None else '-'} incremental revenue per month across "
                f"{psm.get('n_treated', 0):,} treated users; valid only within the common-support population.",
                f"**Conclusion & uncertainty**: promo lifts monthly revenue per exposed user by **{_fmt_num(att_val)}** "
                f"(bootstrap 95% CI [{_fmt_num(ci_lo)}, {_fmt_num(ci_hi)}], p={atts.get('psm', {}).get('p_value', float('nan')):.3f}). "
                f"Business read: ≈¥{format(round(monthly), ',') if monthly is not None else '-'} incremental revenue per month across "
                f"{psm.get('n_treated', 0):,} treated users; valid only within the common-support population.",
            )
        )
        if rosenbaum:
            lines.append(
                self._t(
                    f"**Hidden-bias robustness**: significance survives up to Γ={gamma_max:g} — an unobserved confounder "
                    f"would need to distort assignment odds by more than {gamma_max:g}x to overturn the result.",
                    f"**Hidden-bias robustness**: significance survives up to Γ={gamma_max:g} — an unobserved confounder "
                    f"would need to distort assignment odds by more than {gamma_max:g}x to overturn the result.",
                )
            )
        self._record(
            self._t(
                f"Causal matching: naive {_fmt_num(naive)} vs PSM {_fmt_num(atts.get('psm', {}).get('att'))} "
                f"vs AIPW {_fmt_num(aipw.get('estimate'))}.",
                f"Causal matching: naive {_fmt_num(naive)} vs PSM {_fmt_num(atts.get('psm', {}).get('att'))} "
                f"vs AIPW {_fmt_num(aipw.get('estimate'))}.",
            ),
            Section(
                title=self._t("Causal: matching and weighting", "Causal: matching and weighting"),
                prose=self._t(
                    f"The naive difference {_fmt_num(naive)} mixes in selection bias; PSM ATT = {_fmt_num(atts.get('psm', {}).get('att'))} "
                    f"({atts.get('psm', {}).get('n_matched_pairs', 0):,} matched pairs), IPW ATE = {_fmt_num(ipw.get('estimate'))}, "
                    f"AIPW ATE = {_fmt_num(aipw.get('estimate'))}.",
                    atts.get("psm", {}).get("narrative", ""),
                ),
                table=df_rows,
            ),
        )
        self._record(
            verdict_line(conf["verdict"], self.lang),
            Section(
                title=self._t("Matching confidence validation", "Matching confidence validation"),
                prose=verdict_line(conf["verdict"], self.lang)
                + " " + self._t(
                    "Checks cover covariate balance, common support, estimator agreement and significance; hidden confounding needs a sensitivity pass.",
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
                    "DoWhy is not installed here. Run `pip install dowhy` and retry; available engines: " + str(available_engines()),
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
                    self._t("effect", "effect"): _fmt_num(e["estimate"]),
                    "p": "-" if e["p"] != e["p"] else f"{e['p']:.4f}",
                }
                for e in engines
            ]
        )
        ref_rows = pd.DataFrame(
            [
                {
                    self._t("refuter", "refuter"): r["refuter"],
                    self._t("new effect", "new effect"): _fmt_num(r["new_effect"]),
                    self._t("verdict", "verdict"): self._t("passed", "passed") if r["passed"] else self._t("failed", "failed"),
                }
                for r in refutations
            ]
        )

        lines = [
            self._t(
                f"**Open-source cross-validation**: in-house PSM ATT = {_fmt_num(ih_att)}; DoWhy linear = "
                f"{_fmt_num(oss['estimate'])} (p={oss['p_value']:.4f})"
                + (f", DoWhy PSM = {_fmt_num(oss_psm['estimate'])}." if oss_psm else "."),
                f"**Open-source cross-validation**: in-house PSM ATT = {_fmt_num(ih_att)}; DoWhy linear = "
                f"{_fmt_num(oss['estimate'])} (p={oss['p_value']:.4f})"
                + (f", DoWhy PSM = {_fmt_num(oss_psm['estimate'])}." if oss_psm else "."),
            ),
            verdict_line(conf["verdict"], self.lang),
        ]

        blocks: List[Dict[str, Any]] = [
            self._table_block(rows, self._t("Cross-engine estimates", "Cross-engine estimates")),
            {"type": "chart", "title": self._t("Effect by engine", "Effect by engine"), "image": charts.engine_comparison_chart(engines)},
            self._table_block(ref_rows, self._t("DoWhy refutation tests", "DoWhy refutation tests")),
            self._table_block(confidence_table(conf["checks"], self.lang, conf["verdict"]), self._t("Confidence validation", "Confidence validation")),
        ]
        self._record(
            verdict_line(conf["verdict"], self.lang),
            Section(
                title=self._t("Open-source cross-validation (DoWhy)", "Open-source cross-validation (DoWhy)"),
                prose=verdict_line(conf["verdict"], self.lang)
                + " "
                + self._t(
                    "The same identification problem is re-estimated by DoWhy with placebo/random-common-cause refutations; agreement across independent engines is key evidence.",
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
            name = self._t("Isolation-forest anomaly detection", "Isolation-forest anomaly detection")
        else:
            result = autoencoder_scores(values, window=7)
            name = self._t("Autoencoder anomaly detection", "Autoencoder anomaly detection")
        conf = dl_anomaly_confidence(result, seasonal_flagged)

        frame = dl_anomaly_table(dates, result, self.lang)
        chart_uri = charts.dl_reconstruction_chart(result, dates, title=name)
        flagged_days = frame[frame[frame.columns[-1]] == True]  # noqa: E712 - bool column
        recent = flagged_days.tail(6)

        lines = [
            f"**{name}** ({result['backend']})",
            self._t(
                f"{int(result['flagged'].sum())} anomalous windows"
                + (f"; most recent: {', '.join(str(d)[:10] for d in recent[recent.columns[0]])}." if len(recent) else "."),
                f"{int(result['flagged'].sum())} anomalous windows"
                + (f"; most recent: {', '.join(str(d)[:10] for d in recent[recent.columns[0]])}." if len(recent) else "."),
            ),
            verdict_line(conf["verdict"], self.lang),
        ]
        blocks: List[Dict[str, Any]] = [
            {"type": "chart", "title": name, "image": chart_uri},
            self._table_block(confidence_table(conf["checks"], self.lang, conf["verdict"]), self._t("Confidence validation", "Confidence validation")),
        ]
        if len(recent):
            blocks.insert(1, self._table_block(recent, self._t("Most recent anomalies", "Most recent anomalies")))

        self._record(
            verdict_line(conf["verdict"], self.lang),
            Section(
                title=name,
                prose=verdict_line(conf["verdict"], self.lang)
                + " "
                + self._t(
                    "The autoencoder learns the normal shape and scores reconstruction error; isolation forests isolate anomalies in multivariate features; cross-checked against the classical seasonal baseline.",
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
                    self._t(" (no numbers found; demo parameters shown)", " (no numbers found; demo parameters shown)")
                    if noted_default
                    else ""
                )
                if size["kind"] == "mean":
                    text = self._t(
                        f"**Sample size**{default_note}: continuous metric (sd σ={size['sd']:g}), detect a shift of "
                        f"δ={size['mde']:g} (α={size['alpha']}, power {size['power']:.0%}) — "
                        f"**{size['n_per_group']:,}** per group, {size['n_total']:,} in total.",
                        f"**Sample size**{default_note}: continuous metric (sd σ={size['sd']:g}), detect a shift of "
                        f"δ={size['mde']:g} (α={size['alpha']}, power {size['power']:.0%}) — "
                        f"**{size['n_per_group']:,}** per group, {size['n_total']:,} in total.",
                    )
                else:
                    text = self._t(
                        f"**Sample size**{default_note}: baseline {size.get('p0', 0) * 100:.1f}%, detect an absolute change of "
                        f"{size['mde'] * 100:.2f}pp (α={size['alpha']}, power {size['power']:.0%}) — "
                        f"**{size['n_per_group']:,}** per group, {size['n_total']:,} in total.",
                        f"**Sample size**{default_note}: baseline {size.get('p0', 0) * 100:.1f}%, detect an absolute change of "
                        f"{size['mde'] * 100:.2f}pp (α={size['alpha']}, power {size['power']:.0%}) — "
                        f"**{size['n_per_group']:,}** per group, {size['n_total']:,} in total.",
                    )
                text += "\n\n" + self._t(
                    f"**Theory**: {size['formula']} with z_{{1-α/2}}={size['z_alpha']}, z_{{1-β}}={size['z_power']}. "
                    "By the CLT the sample mean is approximately normal with standard error σ/√n, which shrinks "
                    "with the square root of n — halving the detectable effect costs 4x the sample. "
                    "Reference: Cochran, Sampling Techniques (1977).",
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
                                "label": self._t("n per group", "n per group"),
                                "value": f"{size['n_per_group']:,}",
                                "sub": f"α={size['alpha']}, power={size['power']:.0%}",
                            },
                            {
                                "label": self._t("both groups", "both groups"),
                                "value": f"{size['n_total']:,}",
                                "sub": self._t("1:1 allocation", "1:1 allocation"),
                            },
                            {
                                "label": self._t("detectable δ", "detectable δ"),
                                "value": f"{size['mde'] * 100:.2f}pp",
                                "sub": self._t("two-sided", "two-sided"),
                            },
                        ],
                    }
                )
                blocks.append(
                    self._table_block(
                        pd.DataFrame(sens_rows, columns=[self._t("effect size", "effect size"), self._t("n per group", "n per group")]),
                        self._t("Sample size vs effect size (2x precision = 4x n)", "Sample size vs effect size (2x precision = 4x n)"),
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

            method_label = {"srs": self._t("simple random", "simple random"), "systematic": self._t("systematic", "systematic"), "stratified": self._t("stratified (proportional by channel)", "stratified (proportional by channel)")}[method]
            headline = self._t(
                f"**Sample drawn**: {method_label}, {len(sample):,} of {draw_log['population']:,} users ({draw_log['fraction']:.2%}, seed={draw_log['seed']}, reproducible). "
                f"Sample mean spend {se_info['mean']:.2f}, SE {se_info['se']:.3f}, 95% CI [{se_info['ci95'][0]:.2f}, {se_info['ci95'][1]:.2f}].",
                f"**Sample drawn**: {method_label}, {len(sample):,} of {draw_log['population']:,} users ({draw_log['fraction']:.2%}, seed={draw_log['seed']}, reproducible). "
                f"Sample mean spend {se_info['mean']:.2f}, SE {se_info['se']:.3f}, 95% CI [{se_info['ci95'][0]:.2f}, {se_info['ci95'][1]:.2f}].",
            )
            headline += "\n\n" + self._t(
                "**Theory**: the CLT makes the sample mean approximately normal with SE=s/√n; both the confidence "
                "interval and the representativeness tests below read off that sampling distribution.",
                "**Theory**: the CLT makes the sample mean approximately normal with SE=s/√n; both the confidence "
                "interval and the representativeness tests below read off that sampling distribution.",
            )
            blocks.append(
                {
                    "type": "kpi",
                    "items": [
                        {"label": self._t("n / population", "n / population"), "value": f"{len(sample):,}/{draw_log['population']:,}", "sub": f"{draw_log['fraction']:.2%} " + self._t("fraction", "fraction")},
                        {"label": self._t("sample mean", "sample mean"), "value": f"{se_info['mean']:.2f}", "sub": f"SE={se_info['se']:.3f}"},
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
                        self._t("Stratum allocation (proportional, small strata keep ≥1)", "Stratum allocation (proportional, small strata keep ≥1)"),
                    )
                )
            blocks.append(
                self._table_block(
                    stat_tests.to_frame(all_tests, self.lang),
                    self._t("Representativeness tests (sample vs population)", "Representativeness tests (sample vs population)"),
                    max_rows=6,
                )
            )
            share_rows = [
                {"channel": r["category"], "sample": f"{r['sample_share']:.2%}", "population": f"{r['population_share']:.2%}", "diff": f"{r['diff']:+.2%}"}
                for r in rep_cat.get("shares", [])[:5]
            ]
            if share_rows:
                blocks.append(self._table_block(pd.DataFrame(share_rows), self._t("Channel mix: sample vs population", "Channel mix: sample vs population")))
            agree = stat_tests.agreement(all_tests, [t["key"] for t in all_tests if t["status"] == "ok"])
            verdict = (
                self._t("All tests agree: the sample is statistically indistinguishable from the population — representative.", "All tests agree: the sample is statistically indistinguishable from the population — representative.")
                if agree is True
                else (
                    self._t("Tests disagree or some reject: revisit the sampling frame and stratification before concluding.", "Tests disagree or some reject: revisit the sampling frame and stratification before concluding.")
                    if agree is False
                    else ""
                )
            )
            text = headline + ("\n\n" + self._t("**Statistical tests**: ", "**Statistical tests**: ") + stat_tests.summary_line(all_tests, self.lang) if all_tests else "")
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
                    f"Sampling analysis failed: `{exc}`\n\nTry e.g. \"survey with baseline 70%, detect a 3pp change - how many samples?\" "
                    f"or \"draw a stratified sample of 2000 users and test representativeness\".",
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
                title = self._t("Conversion classification model", "Conversion classification model")
            else:
                df = self.sql.run(
                    "SELECT date, SUM(gmv) AS gmv, SUM(sessions) AS sessions, SUM(orders) AS orders "
                    "FROM orders_daily GROUP BY 1 ORDER BY 1",
                    max_rows=100000,
                ).dataframe
                target = "gmv"
                title = self._t("Daily GMV regression model", "Daily GMV regression model")

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
                    **{d["column"]: self._t(f"dropped ({d['reason']})", f"dropped ({d['reason']})") for d in pre_log.get("dropped", [])},
                }
                none_needed = self._t("none needed", "none needed")
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
                            f"1. Data preprocessing: missing/NaN profile ({prof['missing_cells']} missing cells, {prof['missing_pct']:.1f}%)",
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
                        "feature": self._t("(none)", "(none)"),
                        "reason": self._t("all pairwise |r|<0.95 and VIF<10, nothing pruned", "all pairwise |r|<0.95 and VIF<10, nothing pruned"),
                    }
                ]
            blocks.append(
                self._table_block(
                    pd.DataFrame(col_rows),
                    self._t(
                        f"2. Feature engineering: collinearity handling ({col_log.get('n_features', len(feature_names))} features kept)",
                        f"2. Feature engineering: collinearity handling ({col_log.get('n_features', len(feature_names))} features kept)",
                    ),
                )
            )

            result = train_model(X, y, task="classification" if wants_classification else "regression")
            best = result["best_model"]

            # ③ Label definition + leakage screen: what is being predicted and
            # whether any feature could only exist after the label was written.
            y_ser = pd.Series(y)
            if wants_classification:
                label_rows = [[target, self._t("binary", "binary"), f"{y_ser.mean():.1%}", f"n={len(y_ser):,}"]]
            else:
                label_rows = [
                    [target, self._t("regression", "regression"), f"{y_ser.min():.4g} ~ {y_ser.max():.4g}", f"n={len(y_ser):,}"]
                ]
            blocks.append(
                self._table_block(
                    pd.DataFrame(
                        label_rows,
                        columns=[self._t("label", "label"), self._t("type", "type"), self._t("positives / range", "positives / range"), "n"],
                    ),
                    self._t("3. Label definition", "3. Label definition"),
                )
            )
            leak_rows = [
                {"feature": f["feature"], "reasons": "; ".join(f["reasons"]), "action": f["action"]}
                for f in result.get("leakage", [])
            ][:6]
            if not leak_rows:
                leak_rows = [
                    {
                        "feature": self._t("(none)", "(none)"),
                        "reasons": self._t(
                            "no ID keys, target proxies (|r|>0.95) or post-outcome names",
                            "no ID keys, target proxies (|r|>0.95) or post-outcome names",
                        ),
                        "action": "-",
                    }
                ]
            blocks.append(
                self._table_block(
                    pd.DataFrame(leak_rows),
                    self._t("3. Feature-leakage screen", "3. Feature-leakage screen"),
                )
            )

            # ④ Strong baseline: a candidate model is only interesting if it
            # beats the dumbest sensible predictor, by a stated margin.
            bl = result.get("baseline", {})
            bl_val = bl.get("accuracy", bl.get("rmse", "-"))
            blocks.append(
                self._table_block(
                    pd.DataFrame(
                        [
                            [bl.get("model", "-"), bl_val, bl.get("prevalence", "-") if wants_classification else bl.get("r2_vs_test_mean", "-")],
                            [self._t("best model", "best model"), best, self._t("see metrics below", "see metrics below")],
                        ],
                        columns=[self._t("model", "model"), self._t("primary", "primary"), self._t("note", "note")],
                    ),
                    self._t("4. Strong baseline", "4. Strong baseline"),
                )
            )

            fa = feature_analysis(result)
            conf = modeling_confidence(result)
            best = result["best_model"]
            metrics = result["metrics"][best]

            metric_lines = ", ".join(f"{k}={v}" for k, v in metrics.items())
            n_pre = len(pre_log.get("steps", []))
            n_col = len(col_log.get("dropped", []))
            n_leak = len(result.get("leakage", []))
            bl = result.get("baseline", {})
            bl_cmp = (
                self._t(
                    f"baseline (majority class) accuracy {bl.get('accuracy', '-')} — the model must beat it first",
                    f"baseline (majority class) accuracy {bl.get('accuracy', '-')} — the model must beat it first",
                )
                if wants_classification
                else self._t(
                    f"baseline (mean predictor) RMSE {bl.get('rmse', '-')}",
                    f"baseline (mean predictor) RMSE {bl.get('rmse', '-')}",
                )
            )
            cal = result.get("calibration", {})
            cal_line = (
                self._t(
                    f"5. calibration: Brier={cal.get('brier', '-')}, ECE={cal.get('ece', '-')}; ",
                    f"5. calibration: Brier={cal.get('brier', '-')}, ECE={cal.get('ece', '-')}; ",
                )
                if cal
                else ""
            )
            text = self._t(
                f"**{title}** ({result['task']}, best model `{best}`): {metric_lines}; stratified 5-fold CV (AUC) "
                f"{result['cv_mean']:.3f}±{result['cv_std']:.3f}.\n\n"
                f"**End-to-end loop**: 1. label definition (positive rate {pd.Series(y).mean():.1%}) → "
                f"2. preprocessing & feature engineering ({n_pre} cleanups, {n_col} redundant features pruned) → "
                f"3. leakage screen ({n_leak} suspects) → 4. {bl_cmp} → training (stratified CV + held-out test) → "
                f"{cal_line}6. post-model feature analysis (SHAP/permutation) — explain only after the model is validated.",
                f"**{title}** ({result['task']}, best model `{best}`): {metric_lines}; stratified 5-fold CV (AUC) "
                f"{result['cv_mean']:.3f}±{result['cv_std']:.3f}.\n\n"
                f"**End-to-end loop**: 1. label definition (positive rate {pd.Series(y).mean():.1%}) → "
                f"2. preprocessing & feature engineering ({n_pre} cleanups, {n_col} redundant features pruned) → "
                f"3. leakage screen ({n_leak} suspects) → 4. {bl_cmp} → training (stratified CV + held-out test) → "
                f"{cal_line}6. post-model feature analysis (SHAP/permutation) — explain only after the model is validated.",
            )

            # Explainability chart: SHAP (if installed) or permutation importance.
            # Deliberately positioned AFTER the model: features are interpreted
            # against a validated model, never before one exists.
            method = self._t(
                "6. Post-model feature analysis (SHAP)" if fa["engine"] == "shap" else "6. Post-model feature analysis (permutation importance)",
                "6. Post-model feature analysis (SHAP)" if fa["engine"] == "shap" else "6. Post-model feature analysis (permutation importance)",
            )
            method += self._t(")", ")")
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
                            "title": self._t("ROC curve", "ROC curve"),
                            "image": charts.roc_chart(fpr, tpr, metrics.get("roc_auc", 0.0)),
                        }
                    )
                    # ⑤ Calibration: ranking quality (AUC) and probability
                    # quality (Brier/ECE) are different virtues; thresholds
                    # built on miscalibrated scores misroute every user.
                    cal = result.get("calibration")
                    if cal:
                        blocks.append(
                            self._table_block(
                                pd.DataFrame(cal["bins"]),
                                self._t(
                                    f"5. Probability calibration (Brier={cal['brier']:.3f}, ECE={cal['ece']:.3f}, "
                                    f"{'well calibrated' if cal['well_calibrated'] else 'drifted'})",
                                    f"5. Probability calibration (Brier={cal['brier']:.3f}, ECE={cal['ece']:.3f}, "
                                    f"{'well calibrated' if cal['well_calibrated'] else 'drifted'})",
                                ),
                            )
                        )
                    cm = confusion_matrix(y_te, best_model.predict(X_te))
                    blocks.append(
                        {
                            "type": "chart",
                            "title": self._t("Confusion matrix", "Confusion matrix"),
                            "image": charts.confusion_matrix_chart(cm.tolist(), labels=sorted(map(str, y.unique()))),
                        }
                    )
                else:
                    blocks.append({"type": "markdown", "markdown": self._t("`ROC/confusion matrix` need binary labels and probabilistic output.", "`ROC/confusion matrix` need binary labels and probabilistic output.")})
            else:
                _, X_te, _, y_te = result["_split"]
                pred = result["predictions"][best]
                blocks.append(
                    {
                        "type": "chart",
                        "title": self._t("Predicted vs actual", "Predicted vs actual"),
                        "image": charts.pred_vs_actual_chart(y_te, pred),
                    }
                )

            # Confidence card: overfit gap, CV stability, sample size.
            check_rows = [
                {"check": c["name"], "status": c["status"], "detail": c["detail"]} for c in conf["checks"]
            ]
            verdict = self._t(
                f"Overall confidence: {conf['overall']}. Review the overfit gap and CV stability before using model output in decisions.",
                f"Overall confidence: {conf['overall']}. Review the overfit gap and CV stability before using model output in decisions.",
            )
            blocks.append(
                {
                    "type": "table",
                    "title": self._t("Confidence validation", "Confidence validation"),
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
                        self._t("Statistical tests (model diagnostics)", "Statistical tests (model diagnostics)"),
                        max_rows=8,
                    )
                )
                text += "\n\n" + self._t(
                    f"**Statistical tests** (method / statistic / p / conclusion): {panel['headline_en']} {panel['summary_en']}",
                    f"**Statistical tests** (method / statistic / p / conclusion): {panel['headline_en']} {panel['summary_en']}",
                )
            except Exception:  # noqa: BLE001
                panel = None

            tool_result = {"task": result["task"], "best_model": best, "n_train": result["n_train"], "n_test": result["n_test"], "confidence": conf["overall"]}
            return AgentReply(text=text, blocks=blocks, tool_calls=[{"tool": "train_model", "args": {"task": result["task"], "target": target}, "result": tool_result}], data={"modeling": tool_result})
        except Exception as exc:  # noqa: BLE001 - surfaced with a next-step hint
            return AgentReply(
                text=self._t(
                    f"Modeling failed: `{exc}`\n\nTry e.g. \"train a classifier to predict conversion\".",
                    f"Modeling failed: `{exc}`\n\nTry e.g. \"train a classifier to predict conversion\".",
                )
            )

    def _handle_monitoring(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """Post-launch model monitoring: drift, decay, alert thresholds.

        Reference population = the training snapshot (older cohorts); current
        = the newest cohort the model now scores. Performance history comes
        from re-scoring the validated conversion model on tenure-ordered
        cohort slices, oldest first — the same shape a production monitor
        would see week by week.
        """
        from .tools.monitoring import PSI_CRITICAL, monitor_report

        blocks: List[Dict[str, Any]] = []
        try:
            df = self.sql.run(
                "SELECT tenure_days, spend_30d, sessions_30d, channel, device, converted "
                "FROM uplift_users LIMIT 8000",
                max_rows=100000,
            ).dataframe
            df = df.dropna(subset=["converted"])
            q60 = df["tenure_days"].quantile(0.6)
            reference = df[df["tenure_days"] >= q60].drop(columns=["converted", "tenure_days"])
            current = df[df["tenure_days"] < q60].drop(columns=["converted", "tenure_days"])
            # tenure_days is the cohort split itself — comparing it across
            # cohorts is tautological, so it stays out of the drift screen.

            # Cohort AUC history: fit on the training snapshot, score the
            # remaining cohorts oldest -> newest (deployment order).
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score

            feat_cols = ["spend_30d", "sessions_30d", "tenure_days"]
            X_all = pd.get_dummies(
                df[feat_cols + ["channel", "device"]], columns=["channel", "device"], drop_first=True
            )
            tr_mask = df["tenure_days"] >= q60
            model = LogisticRegression(max_iter=2000).fit(X_all[tr_mask], df.loc[tr_mask, "converted"])
            ev = df[~tr_mask]
            ev_X = X_all[~tr_mask]
            slices = pd.qcut(ev["tenure_days"], 4, labels=False, duplicates="drop")
            auc_history = []
            for s in sorted(pd.unique(slices), reverse=True):  # oldest first
                mask = (slices == s).to_numpy()
                lo, hi = ev.loc[mask, "tenure_days"].min(), ev.loc[mask, "tenure_days"].max()
                if mask.sum() >= 60 and ev.loc[mask, "converted"].nunique() == 2:
                    auc_history.append(
                        {
                            "slice": self._t(f"tenure {int(lo)}–{int(hi)}d", f"tenure {int(lo)}–{int(hi)}d"),
                            "auc": float(roc_auc_score(ev.loc[mask, "converted"], model.predict_proba(ev_X[mask])[:, 1])),
                        }
                    )

            rep = monitor_report(reference, current, auc_history)
            drift = rep["feature_drift"]
            ks = rep["data_drift"]
            perf = rep["performance"]

            verdict_zh = {"ok": "healthy", "warn": "warning", "critical": "retrain", "insufficient_history": "insufficient history"}
            verdict_en = {"ok": "healthy", "warn": "warning", "critical": "retrain", "insufficient_history": "insufficient history"}
            blocks.append(
                {
                    "type": "kpi",
                    "items": [
                        {
                            "label": self._t("Monitoring verdict", "Monitoring verdict"),
                            "value": verdict_zh.get(rep["overall"], rep["overall"]),
                            "sub": self._t(f"{len(rep['alerts'])} alert(s)", f"{len(rep['alerts'])} alert(s)"),
                        },
                        {
                            "label": self._t("Max PSI", "Max PSI"),
                            "value": f"{drift['max_psi']:.3f}",
                            "sub": self._t(f"threshold {PSI_CRITICAL}", f"threshold {PSI_CRITICAL}"),
                        },
                        {
                            "label": self._t("AUC decay", "AUC decay"),
                            "value": f"-{perf.get('drop', 0):.3f}" if perf.get("drop") is not None else "-",
                            "sub": f"{perf.get('baseline_auc', '-')} → {perf.get('recent_auc', '-')}",
                        },
                    ],
                }
            )
            if drift["columns"]:
                blocks.append(
                    self._table_block(
                        pd.DataFrame(
                            [[r["feature"], f"{r['psi']:.4f}", r["severity"]] for r in drift["columns"][:8]],
                            columns=[
                                self._t("feature", "feature"),
                                "PSI",
                                self._t(f"severity (warn>{PSI_CRITICAL - 0.15:.2f}, critical>{PSI_CRITICAL})", f"severity (warn>0.10, critical>{PSI_CRITICAL})"),
                            ],
                        ),
                        self._t("Feature drift (PSI vs training distribution)", "Feature drift (PSI vs training distribution)"),
                    )
                )
                try:
                    blocks.append(
                        {
                            "type": "chart",
                            "title": self._t("PSI ranking", "PSI ranking"),
                            "image": charts.importance_bar(
                                [{"feature": r["feature"], "value": r["psi"]} for r in drift["columns"][:8]],
                                title="Feature drift (PSI)",
                            ),
                        }
                    )
                except Exception:  # noqa: BLE001
                    pass
            if ks["tests"]:
                blocks.append(
                    self._table_block(
                        pd.DataFrame(
                            [[r["feature"], f"{r['ks']:.4f}", f"{r['p_value']:.4f}", "DRIFT" if r["drifted"] else "ok"] for r in ks["tests"][:8]],
                            columns=[self._t("feature", "feature"), "KS", "p", self._t("dist", "dist")],
                        ),
                        self._t("Data drift (two-sample KS, p<0.05)", "Data drift (two-sample KS, p<0.05)"),
                    )
                )
            if perf.get("aucs"):
                blocks.append(
                    self._table_block(
                        pd.DataFrame(
                            [[h["slice"], f"{h['auc']:.4f}"] for h in perf["aucs"]],
                            columns=[self._t("cohort", "cohort"), "AUC"],
                        ),
                        self._t(
                            f"Performance decay (baseline {perf['baseline_auc']:.3f} → recent {perf['recent_auc']:.3f}, verdict {perf['verdict']})",
                            f"Performance decay (baseline {perf['baseline_auc']:.3f} → recent {perf['recent_auc']:.3f}, verdict {perf['verdict']})",
                        ),
                    )
                )
            if rep["alerts"]:
                blocks.append(
                    self._table_block(
                        pd.DataFrame(
                            [[a["level"], a["check"], a["message"]] for a in rep["alerts"]],
                            columns=["level", "check", self._t("alert", "alert")],
                        ),
                        self._t("Alerts", "Alerts"),
                    )
                )
            advice = {
                "ok": self._t("All checks pass; keep monitoring weekly.", "All checks pass; keep monitoring weekly."),
                "warn": self._t(
                    "Moderate drift: no retrain yet, tighten monitoring to daily and audit the drifting features' calibers.",
                    "Moderate drift: no retrain yet, tighten monitoring to daily and audit the drifting features' calibers.",
                ),
                "critical": self._t(
                    "Drift/decay crossed the retrain threshold: retrain on recent data, rerun calibration and leakage screens before shipping, and shadow-deploy the old model meanwhile.",
                    "Drift/decay crossed the retrain threshold: retrain on recent data, rerun calibration and leakage screens before shipping, and shadow-deploy the old model meanwhile.",
                ),
            }
            text = self._t(
                f"**Model monitoring** (reference = training distribution, current = newest cohort): overall "
                f"**{verdict_zh.get(rep['overall'])}**. Max feature PSI {drift['max_psi']:.3f}; {ks.get('n_drifted', 0)} KS-drifted "
                f"features; AUC {perf.get('baseline_auc', '-')} → {perf.get('recent_auc', '-')} (drop {perf.get('drop', 0):.3f}). "
                f"{advice.get(rep['overall'], advice['ok'])}",
                f"**Model monitoring** (reference = training distribution, current = newest cohort): overall "
                f"**{verdict_en.get(rep['overall'])}**. Max feature PSI {drift['max_psi']:.3f}; {ks.get('n_drifted', 0)} KS-drifted "
                f"features; AUC {perf.get('baseline_auc', '-')} → {perf.get('recent_auc', '-')} (drop {perf.get('drop', 0):.3f}). "
                f"{advice.get(rep['overall'], advice['ok'])}",
            )
            return AgentReply(
                text=text,
                blocks=blocks,
                tool_calls=[{"tool": "monitor_report", "args": {}, "result": {"overall": rep["overall"], "n_alerts": len(rep["alerts"])}}],
                data={"monitoring": {"overall": rep["overall"], "alerts": rep["alerts"]}},
            )
        except Exception as exc:  # noqa: BLE001 - surfaced with a next-step hint
            return AgentReply(
                text=self._t(
                    f"Model monitoring failed: `{exc}`",
                    f"Model monitoring failed: `{exc}`",
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
                    f"Dashboard generation failed: `{exc}`",
                    f"Dashboard generation failed: `{exc}`",
                )
            )
        name = Path(out["path"]).name
        kpi_text = " · ".join(f"{k['label']} {k['value']}" for k in out["kpis"])
        text = self._t(
            f"Dashboard ready: **{name}**. Key KPIs (last 14 days WoW): {kpi_text}. Open via the link below (standalone HTML, inline charts, share or cast directly).",
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
            {"type": "markdown", "markdown": self._t("Includes: KPI cards with WoW, GMV trend + moving average, channel/device breakdowns, refund rate, recent anomalies.", "Includes: KPI cards with WoW, GMV trend + moving average, channel/device breakdowns, refund rate, recent anomalies.")},
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

        title = self._t("Metric diagnosis and experiment report", "Metric diagnosis and experiment report")
        report = Report(
            title=title,
            subtitle=self._t("Generated by DS-Agent: answer first, every number reproducible", "Generated by DS-Agent: answer first, every number reproducible"),
            lang=self.lang,
            meta={
                self._t("Data source", "Data source"): str(self.sql.warehouse.path.name),
                self._t("Metric definitions", "Metric definitions"): self._t("governed semantic layer; ratios aggregated as numerator/denominator", "governed semantic layer; ratios aggregated as numerator/denominator"),
                self._t("Question", "Question"): self.session.get("last_question", "-")[:140],
            },
        )
        summary = list(self.summary_bullets[:8])
        if len(self.summary_bullets) > len(summary):
            # Silent truncation is how a report ends up looking complete while
            # missing half the analysis; say what was left out instead.
            summary.append(
                self._t(
                    f"{len(self.summary_bullets) - len(summary)} more findings are in the sections below.",
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
                    title=self._t("Business context (from the knowledge base)", "Business context (from the knowledge base)"),
                    prose=kb_text,
                )
            )
        # Every recorded section goes in: a section exists because a question was
        # asked and answered, and dropping the tail silently is worse than a long
        # report.
        for section in self.findings:
            report.add_section(section)
        report.add_method(self._t("Anomaly detection: weekly-seasonal baseline (trailing 28-day level x weekday index, with a local growth correction) + MAD-robust z (threshold 3.5); sustained shifts via two-sided CUSUM, gated by a t-test.", "Anomaly detection: weekly-seasonal baseline (trailing 28-day level x weekday index, with a local growth correction) + MAD-robust z (threshold 3.5); sustained shifts via two-sided CUSUM, gated by a t-test."))
        report.add_method(self._t("Contribution: exact additive decomposition; ratio metrics split into rate / mix / interaction with residual ~ 0.", "Contribution: exact additive decomposition; ratio metrics split into rate / mix / interaction with residual ~ 0."))
        report.add_method(self._t("Experiments: sizing inverts the power function; readout includes SRM, CUPED, BH correction.", "Experiments: sizing inverts the power function; readout includes SRM, CUPED, BH correction."))
        report.add_method(self._t("Non-standard designs: switchback infers at the time-block level with HAC errors; reversal tests drift with tau1+tau2 and measures decay by persistence; parent-child checks orthogonality by chi-square and adjusts for parent fixed effects; two-sided reads direct effects off the 2x2 marginals and reports cross-side spillover; sensitivity analysis reports sign stability rather than one p-value.", "Non-standard designs: switchback infers at the time-block level with HAC errors; reversal tests drift with tau1+tau2 and measures decay by persistence; parent-child checks orthogonality by chi-square and adjusts for parent fixed effects; two-sided reads direct effects off the 2x2 marginals and reports cross-side spillover; sensitivity analysis reports sign stability rather than one p-value."))
        report.add_method(self._t("Causal: DiD with parallel-trends test, synthetic control with placebo inference, PSM/IPW/AIPW cross-checks.", "Causal: DiD with parallel-trends test, synthetic control with placebo inference, PSM/IPW/AIPW cross-checks."))
        report.add_method(self._t("Heterogeneity and thresholds: uplift estimates CATE with honest splits in a causal tree/forest, with Qini and decile targeting to measure usefulness; RDD uses local linear fits with a density manipulation test and covariate balance, and refuses conclusions when the bandwidth sensitivity is unstable.", "Heterogeneity and thresholds: uplift estimates CATE with honest splits in a causal tree/forest, with Qini and decile targeting to measure usefulness; RDD uses local linear fits with a density manipulation test and covariate balance, and refuses conclusions when the bandwidth sensitivity is unstable."))
        report.add_method(self._t("Time-series causal: ARIMA ITS estimates the trend, weekday terms and the post indicator jointly on the log scale, checked with placebo dates and a specification curve; the structural counterfactual uses a pre-period one-step fit and horizon-widening intervals. Both are shown against the naive pre/post difference, which bills the trend to the intervention.", "Time-series causal: ARIMA ITS estimates the trend, weekday terms and the post indicator jointly on the log scale, checked with placebo dates and a specification curve; the structural counterfactual uses a pre-period one-step fit and horizon-widening intervals. Both are shown against the naive pre/post difference, which bills the trend to the intervention."))
        report.add_method(self._t("All results are observational unless stated; causal claims rely on explicit identifying assumptions.", "All results are observational unless stated; causal claims rely on explicit identifying assumptions."))

        paths = report.save(self.artifact_dir, f"ds_report_{date.today()}")

        health_table = None
        blocks: List[Dict[str, Any]] = [
            {"type": "markdown", "markdown": self._t(
                f"Report written: {len(self.findings)} sections, {len(self.summary_bullets)} summary bullets.",
                f"Report written: {len(self.findings)} sections, {len(self.summary_bullets)} summary bullets.",
            )}
        ]
        if self.findings and self.findings[0].chart:
            blocks.append({"type": "chart", "title": self.findings[0].title, "image": self.findings[0].chart})

        text = self._t(
            "**Report written**\n\n"
            f"- Markdown: `{paths['markdown']}`\n"
            f"- HTML (single file, charts embedded): `{paths['html']}`\n\n"
            "Structure: executive summary -> analysis sections (charts + contribution tables) -> methodology and caveats.",
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
                self._t("higher is better", "higher is better") if metric.higher_is_better else self._t("lower is better", "lower is better"),
            ])
            movers.append((abs(change) if np.isfinite(change) else 0, metric, change))
        df = pd.DataFrame(rows, columns=[self._t("metric", "metric"), self._t("prior", "prior"), self._t("current", "current"), self._t("change", "change"), self._t("direction", "direction")])
        self._record(
            self._t("Core metrics for the last 7 days compared with the prior week.", "Core metrics for the last 7 days compared with the prior week."),
            Section(
                title=self._t("1. Core metric health", "1. Core metric health"),
                prose=self._t(
                    f"Window: {window.start} ~ {window.end} vs {comparison.start} ~ {comparison.end}.",
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
            self.findings[-1].title = self._t(f"2. Largest mover: {metric.name} ({_fmt_pct(change)})", f"2. Largest mover: {metric.name} ({_fmt_pct(change)})")
        diagnosis = self.session.get("last_diagnosis", {})
        self._record(
            self._t(
                "In short: decompose the biggest mover into rate vs mix, drill into the segment, then audit instrumentation against a paired metric.",
                "In short: decompose the biggest mover into rate vs mix, drill into the segment, then audit instrumentation against a paired metric.",
            ),
            Section(
                title=self._t("3. Conclusions and actions", "3. Conclusions and actions"),
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
                f"{len(catalog)} things I can do, all in natural language:",
                f"{len(catalog)} things I can do, all in natural language:",
            ),
            "",
        ]
        rows = []
        for name, spec in catalog.items():
            lines.append(f"**{name}** — {spec[lang_key]}")
            example = spec["example_zh" if lang_key == "zh" else "example_en"]
            lines.append(self._t(f"  e.g. `{example}`", f"  e.g. `{example}`"))
            rows.append([name, spec[lang_key], example])
        lines += [
            "",
            self._t(
                "Data: bundled daily warehouse (orders_daily), user-level experiment (experiment_users), city-week panel "
                "(city_panel), observational promo (promo_users), switchback log (switchback_log), reversal log (reverse_log), "
                "parent-child users (nested_users), two-sided markets (two_sided_market), RDD scores (rdd_scores), "
                "uplift sample (uplift_users), intervention series (its_series).",
                "Data: bundled daily warehouse (orders_daily), user-level experiment (experiment_users), city-week panel "
                "(city_panel), observational promo (promo_users), switchback log (switchback_log), reversal log (reverse_log), "
                "parent-child users (nested_users), two-sided markets (two_sided_market), RDD scores (rdd_scores), "
                "uplift sample (uplift_users), intervention series (its_series).",
            ),
            self._t("Follow-ups inherit context: ask 'why did GMV drop', then 'split by channel'.", "Follow-ups inherit context: ask 'why did GMV drop', then 'split by channel'."),
        ]
        df = pd.DataFrame(rows, columns=[self._t("capability", "capability"), self._t("what it does", "what it does"), self._t("example", "example")])
        return AgentReply(
            text="\n".join(lines),
            blocks=[self._table_block(df, self._t("Capabilities", "Capabilities"), max_rows=20)],
        )
