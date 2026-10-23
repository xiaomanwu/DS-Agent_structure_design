"""Semantic layer: the agent's shared understanding of tables, metrics and dimensions.

Every downstream tool (SQL, anomaly, contribution, report) reads metric
semantics from here, so "gmv" is always SUM(gmv) and "conversion rate" is
always SUM(orders) / SUM(sessions). This removes the most common class of
inconsistency in ad-hoc analytics: two answers using two different definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class Metric:
    """A metric definition expressed as SQL fragments."""

    name: str
    numerator: str
    denominator: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    unit: str = ""
    higher_is_better: bool = True
    description: str = ""
    table: str = ""

    @property
    def is_ratio(self) -> bool:
        return self.denominator is not None

    @property
    def kind(self) -> str:
        return "ratio" if self.is_ratio else "additive"

    def sql(self) -> str:
        """Aggregate expression for this metric over any grouped slice."""
        if self.is_ratio:
            return f"SUM({self.numerator}) / NULLIF(SUM({self.denominator}), 0)"
        return f"SUM({self.numerator})"

    def numerator_sql(self) -> str:
        return f"SUM({self.numerator})"

    def denominator_sql(self) -> str:
        return f"SUM({self.denominator})" if self.denominator else ""


@dataclass
class Table:
    """A table plus the role of each column."""

    name: str
    time_column: str
    dimensions: Dict[str, List[str]]          # column -> aliases
    component_columns: List[str]              # raw additive columns (orders, gmv...)
    metrics: Dict[str, Metric]
    id_columns: List[str] = field(default_factory=list)
    description: str = ""
    value_labels: Dict[str, Dict[object, str]] = field(default_factory=dict)

    def dimension_aliases(self) -> Dict[str, str]:
        """Flatten alias -> canonical column, including the column name itself."""
        mapping: Dict[str, str] = {}
        for col, aliases in self.dimensions.items():
            mapping[col.lower()] = col
            for alias in aliases:
                mapping[alias.lower()] = col
        return mapping

    def dimension_expr(self, col: str) -> str:
        """SQL expression for a dimension, mapping codes to readable labels.

        Keeps human-readable labels in every downstream artefact (tables,
        reports, narratives) instead of leaking 0/1 codes to stakeholders.
        """
        labels = self.value_labels.get(col)
        if not labels:
            return col
        cases = " ".join(f"WHEN {col} = {key!r} THEN {value!r}" for key, value in labels.items())
        return f"CASE {cases} ELSE CAST({col} AS VARCHAR) END"


def _demo_tables() -> Dict[str, Table]:
    """Definitions for the bundled demo warehouse (e-commerce daily aggregate)."""

    orders_daily = Table(
        name="orders_daily",
        time_column="date",
        description="Daily aggregated e-commerce performance by channel/device/region/category.",
        dimensions={
            "channel": ["渠道", "流量渠道", "来源"],
            "device": ["设备", "终端", "机型"],
            "region": ["地区", "区域", "省份", "城市"],
            "category": ["品类", "类目", "商品类目"],
            "is_new_user": ["新老客", "新客", "用户类型"],
        },
        component_columns=["orders", "gmv", "buyers", "sessions", "refund_orders", "refund_amount"],
        value_labels={"is_new_user": {0: "returning", 1: "new"}},
        metrics={
            "gmv": Metric(
                name="gmv", numerator="gmv", aliases=["成交额", "销售额", "交易额", "营收", "revenue", "sales"],
                unit="CNY", table="orders_daily", description="Gross merchandise value.",
            ),
            "orders": Metric(
                name="orders", numerator="orders", aliases=["订单量", "订单数", "单量", "order_cnt"],
                unit="order", table="orders_daily", description="Number of paid orders.",
            ),
            "buyers": Metric(
                name="buyers", numerator="buyers", aliases=["买家数", "购买人数", "uv_buyer"],
                unit="user", table="orders_daily", description="Distinct paying users.",
            ),
            "sessions": Metric(
                name="sessions", numerator="sessions", aliases=["访问量", "会话数", "流量", "pv", "visits"],
                unit="session", table="orders_daily", description="Visits / sessions.",
            ),
            "refund_amount": Metric(
                name="refund_amount", numerator="refund_amount", aliases=["退款金额", "退款额"],
                unit="CNY", higher_is_better=False, table="orders_daily", description="Refunded amount.",
            ),
            "refund_orders": Metric(
                name="refund_orders", numerator="refund_orders", aliases=["退款单量", "退款订单"],
                unit="order", higher_is_better=False, table="orders_daily", description="Refunded orders.",
            ),
            "aov": Metric(
                name="aov", numerator="gmv", denominator="orders",
                aliases=["客单价", "平均订单金额", "average order value"],
                unit="CNY", table="orders_daily", description="GMV per order.",
            ),
            "conversion_rate": Metric(
                name="conversion_rate", numerator="orders", denominator="sessions",
                aliases=["转化率", "cvr", "支付转化率", "conversion"],
                unit="%", table="orders_daily", description="Orders per session.",
            ),
            "refund_rate": Metric(
                name="refund_rate", numerator="refund_orders", denominator="orders",
                aliases=["退款率", "退货率"],
                unit="%", higher_is_better=False, table="orders_daily", description="Refunded orders share.",
            ),
            "gmv_per_session": Metric(
                name="gmv_per_session", numerator="gmv", denominator="sessions",
                aliases=["人均gmv", "每会话gmv"],
                unit="CNY", table="orders_daily", description="GMV per session.",
            ),
        },
    )

    experiment_users = Table(
        name="experiment_users",
        time_column="dt",
        description="User-level rows of a checkout AB test (one row per user).",
        dimensions={"group": ["分组", "实验组", "arm"], "device": ["设备", "终端"]},
        component_columns=["converted", "revenue", "visits", "pre_revenue"],
        id_columns=["user_id"],
        metrics={
            "converted": Metric(
                name="converted", numerator="converted", aliases=["转化", "conversion"],
                unit="%", table="experiment_users", description="1 if the user converted.",
            ),
            "revenue": Metric(
                name="revenue", numerator="revenue", aliases=["收入", "gmv"],
                unit="CNY", table="experiment_users", description="Revenue per user.",
            ),
            "pre_revenue": Metric(
                name="pre_revenue", numerator="pre_revenue", aliases=["实验前收入"],
                unit="CNY", table="experiment_users", description="Pre-period revenue, used for CUPED.",
            ),
        },
    )

    return {t.name: t for t in (orders_daily, experiment_users)}


class SemanticLayer:
    """Registry of tables/metrics with fuzzy name resolution."""

    def __init__(self, tables: Optional[Dict[str, Table]] = None) -> None:
        self.tables: Dict[str, Table] = tables or _demo_tables()

    # ------------------------------------------------------------------ lookup
    def default_table(self, name: Optional[str] = None) -> Table:
        if name and name in self.tables:
            return self.tables[name]
        return self.tables["orders_daily"]

    def resolve_metric(self, text: str, table: Optional[str] = None) -> Optional[Metric]:
        """Fuzzy-match a metric from free text (longest alias wins)."""
        if not text:
            return None
        needle = text.strip().lower()
        pool: Iterable[Table] = [self.default_table(table)] if table else self.tables.values()
        best: Optional[Metric] = None
        best_len = 0
        for tbl in pool:
            for metric in tbl.metrics.values():
                candidates = [metric.name.lower()] + [a.lower() for a in metric.aliases]
                for cand in candidates:
                    if cand == needle:
                        return metric
                    if cand in needle and len(cand) > best_len:
                        best, best_len = metric, len(cand)
        return best

    def resolve_dimension(self, text: str, table: Optional[str] = None) -> Optional[str]:
        """Fuzzy-match a dimension column from free text."""
        if not text:
            return None
        needle = text.strip().lower()
        pool: Iterable[Table] = [self.default_table(table)] if table else self.tables.values()
        best: Optional[str] = None
        best_len = 0
        for tbl in pool:
            for alias, col in tbl.dimension_aliases().items():
                if alias == needle:
                    return col
                if alias in needle and len(alias) > best_len:
                    best, best_len = col, len(alias)
        return best

    def all_metrics(self, table: Optional[str] = None) -> List[Metric]:
        return list(self.default_table(table).metrics.values())

    def all_dimensions(self, table: Optional[str] = None) -> List[str]:
        return list(self.default_table(table).dimensions.keys())

    # ------------------------------------------------------------------ prompt
    def describe_for_prompt(self) -> str:
        """Compact textual schema used inside the LLM system prompt."""
        lines: List[str] = []
        for tbl in self.tables.values():
            lines.append(f"TABLE {tbl.name} (time column: {tbl.time_column}) -- {tbl.description}")
            lines.append("  dimensions: " + ", ".join(tbl.dimensions.keys()))
            if tbl.id_columns:
                lines.append("  keys: " + ", ".join(tbl.id_columns))
            for m in tbl.metrics.values():
                ratio = f" = {m.numerator}/{m.denominator}" if m.is_ratio else ""
                lines.append(f"  metric {m.name}{ratio}: {m.description}")
        return "\n".join(lines)


# Shared singleton used by all tools.
semantic = SemanticLayer()
