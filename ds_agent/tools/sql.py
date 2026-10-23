"""SQL tool: a guarded DuckDB executor plus a template-based NL-to-SQL fallback.

The executor is intentionally paranoid: only a single read-only SELECT/WITH
statement is allowed, and every result is row-capped. The NL-to-SQL path is
deterministic and rule based (metric + dimension + time window), so the agent
keeps working without an LLM; when an LLM is configured it produces the SQL and
this module still validates it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd

from ..config import settings
from ..semantic import Metric, Table, semantic

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|install|load|"
    r"pragma|export|import|truncate|replace|vacuum|set|call|grant|revoke)\b",
    re.IGNORECASE,
)
_COMMENT = re.compile(r"(--[^\n]*(?:\n|$))|(/\*.*?\*/)", re.S)


class UnsafeQueryError(ValueError):
    """Raised when generated SQL fails the read-only guard."""


def sanitize_sql(sql: str) -> str:
    """Strip comments and reject anything that is not a single SELECT statement."""
    cleaned = _COMMENT.sub(" ", sql).strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeQueryError("empty SQL")
    lowered = cleaned.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise UnsafeQueryError("only SELECT / WITH statements are allowed")
    if ";" in cleaned:
        raise UnsafeQueryError("multiple statements are not allowed")
    if _FORBIDDEN.search(cleaned):
        raise UnsafeQueryError(f"forbidden keyword in SQL: {_FORBIDDEN.search(cleaned).group(0)}")
    return cleaned


@dataclass
class QueryResult:
    """Tabular result plus the metadata the report layer needs."""

    sql: str
    dataframe: pd.DataFrame
    row_count: int
    truncated: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sql": self.sql,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "columns": list(self.dataframe.columns),
            "rows": self.dataframe.head(50).to_dict(orient="records"),
            "preview_markdown": self.dataframe.head(20).to_string(index=False),
        }


class Warehouse:
    """Thin wrapper around a DuckDB connection."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or settings.warehouse)
        self._con: Optional[duckdb.DuckDBPyConnection] = None
        self._read_only = True

    def _open(self, read_only: bool) -> duckdb.DuckDBPyConnection:
        self._read_only = read_only
        self._con = duckdb.connect(str(self.path), read_only=read_only)
        return self._con

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            if not self.path.exists():
                raise FileNotFoundError(
                    f"warehouse not found: {self.path}. Run `python examples/00_build_demo_data.py` first."
                )
            # Prefer read-write: attaching external sources and other governed
            # ingestion paths need DDL. Fall back to read-only on read-only
            # filesystems / locked files.
            try:
                return self._open(read_only=False)
            except duckdb.Error:
                return self._open(read_only=True)
        return self._con

    def writable_con(self) -> duckdb.DuckDBPyConnection:
        """A read-write connection (the default open mode).

        Raises a readable error when the file could only be opened read-only
        (read-only filesystem or locked by another process).
        """
        con = self.con
        if self._read_only:
            raise RuntimeError(
                f"{self.path} is open in read-only mode (read-only filesystem or locked by another process) — "
                "cannot register external sources"
            )
        return con

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def tables(self) -> List[str]:
        rows = self.con.execute("SHOW TABLES").fetchall()
        return [r[0] for r in rows]

    def schema(self, table: str) -> pd.DataFrame:
        return self.con.execute(f"DESCRIBE {table}").df()

    def max_date(self, table: str, time_column: str) -> Optional[date]:
        try:
            value = self.con.execute(f"SELECT MAX({time_column}) FROM {table}").fetchone()[0]
        except Exception:
            return None
        if value is None:
            return None
        return value if isinstance(value, date) else pd.Timestamp(value).date()

    def run(self, sql: str, max_rows: Optional[int] = None) -> QueryResult:
        """Execute a sanitised SELECT with a row cap."""
        safe_sql = sanitize_sql(sql)
        cap = max_rows or settings.sql_max_rows
        # Wrap instead of editing the query: guarantees the cap even with an existing LIMIT.
        wrapped = f"SELECT * FROM ({safe_sql}) AS _q LIMIT {cap + 1}"
        df = self.con.execute(wrapped).df()
        truncated = len(df) > cap
        if truncated:
            df = df.head(cap)
        return QueryResult(sql=safe_sql, dataframe=df, row_count=len(df), truncated=truncated)


# --------------------------------------------------------------------------- #
# Time window parsing
# --------------------------------------------------------------------------- #
@dataclass
class TimeWindow:
    start: Optional[date]
    end: Optional[date]
    label: str
    days: Optional[int] = None

    def clause(self, time_column: str) -> str:
        if self.start is None or self.end is None:
            return "TRUE"
        return f"{time_column} BETWEEN DATE '{self.start}' AND DATE '{self.end}'"


_REL_DAYS = re.compile(r"(?:最近|近|过去|last|past|previous)\s*(\d{1,3})\s*(?:天|日|days?|d)(?![0-9a-zA-Z])", re.I)
_REL_WEEKS = re.compile(r"(?:最近|近|过去|last|past)\s*(\d{1,2})\s*(?:周|个?星期|weeks?|w)(?![0-9a-zA-Z])", re.I)


def parse_time_window(text: str, anchor: date) -> Tuple[TimeWindow, Optional[TimeWindow]]:
    """Parse a natural-language time window relative to ``anchor``.

    Returns (current_window, comparison_window). The comparison window defaults
    to the equally long period immediately before the current one.
    """
    text = text or ""
    if m := _REL_DAYS.search(text):
        n = int(m.group(1))
        end, start = anchor, anchor - timedelta(days=n - 1)
        cur = TimeWindow(start, end, f"last {n} days", n)
        cmp_end = start - timedelta(days=1)
        cmp_ = TimeWindow(cmp_end - timedelta(days=n - 1), cmp_end, f"previous {n} days", n)
        return cur, cmp_
    if m := _REL_WEEKS.search(text):
        n = int(m.group(1))
        days = n * 7
        end, start = anchor, anchor - timedelta(days=days - 1)
        cur = TimeWindow(start, end, f"last {n} weeks", days)
        cmp_end = start - timedelta(days=1)
        cmp_ = TimeWindow(cmp_end - timedelta(days=days - 1), cmp_end, f"previous {n} weeks", days)
        return cur, cmp_
    if re.search(r"昨天|昨日|yesterday", text, re.I):
        d = anchor - timedelta(days=1)
        return TimeWindow(d, d, "yesterday", 1), TimeWindow(d - timedelta(days=1), d - timedelta(days=1), "day before", 1)
    if re.search(r"上周|last week", text, re.I):
        end = anchor - timedelta(days=anchor.weekday() + 1)
        start = end - timedelta(days=6)
        cmp_ = TimeWindow(start - timedelta(days=7), end - timedelta(days=7), "week before last", 7)
        return TimeWindow(start, end, "last week", 7), cmp_
    if re.search(r"本周|this week", text, re.I):
        start = anchor - timedelta(days=anchor.weekday())
        cur = TimeWindow(start, anchor, "this week", (anchor - start).days + 1)
        cmp_ = TimeWindow(start - timedelta(days=7), start - timedelta(days=1), "last week", 7)
        return cur, cmp_
    if re.search(r"上月|last month", text, re.I):
        first_this = anchor.replace(day=1)
        end = first_this - timedelta(days=1)
        start = end.replace(day=1)
        cmp_end = start - timedelta(days=1)
        return TimeWindow(start, end, "last month", (end - start).days + 1), TimeWindow(
            cmp_end.replace(day=1), cmp_end, "month before last", (cmp_end - cmp_end.replace(day=1)).days + 1
        )
    if re.search(r"本月|this month", text, re.I):
        start = anchor.replace(day=1)
        cur = TimeWindow(start, anchor, "this month", (anchor - start).days + 1)
        prev_start = (start - timedelta(days=1)).replace(day=1)
        return cur, TimeWindow(prev_start, start - timedelta(days=1), "last month", 7)
    # Default: last 7 days vs previous 7 days.
    end = anchor
    start = anchor - timedelta(days=6)
    return (
        TimeWindow(start, end, "last 7 days", 7),
        TimeWindow(start - timedelta(days=7), start - timedelta(days=1), "previous 7 days", 7),
    )


# --------------------------------------------------------------------------- #
# Template-based natural language to SQL
# --------------------------------------------------------------------------- #
def build_metric_query(
    metric: Metric,
    table: Table,
    time_window: TimeWindow,
    dimensions: Optional[List[str]] = None,
    grain: Optional[str] = None,
    order_limit: int = 20,
) -> str:
    """Assemble a grouped metric query from structured slots."""
    dims = list(dimensions or [])
    select_parts = []
    group_parts = []

    if grain in ("day", "week", "month"):
        date_trunc = {"day": "day", "week": "week", "month": "month"}[grain]
        alias = {"day": "date", "week": "week_start", "month": "month_start"}[grain]
        select_parts.append(f"DATE_TRUNC('{date_trunc}', {table.time_column}) AS {alias}")
        group_parts.append(alias)
        order_col = alias
    else:
        order_col = "metric_value"

    for d in dims:
        select_parts.append(d)
        group_parts.append(d)

    select_parts.append(f"{metric.sql()} AS metric_value")

    sql = f"SELECT {', '.join(select_parts)} FROM {table.name} WHERE {time_window.clause(table.time_column)}"
    if group_parts:
        sql += f" GROUP BY {', '.join(group_parts)}"
    if grain:
        sql += f" ORDER BY {order_col} ASC"
    else:
        sql += f" ORDER BY metric_value DESC NULLS LAST LIMIT {order_limit}"
    return sql


def infer_grain(text: str) -> Optional[str]:
    """Detect a requested time grain from free text."""
    if re.search(r"按月|每月|月度|monthly|by month|month by month", text, re.I):
        return "month"
    if re.search(r"按周|每周|周维度|weekly|by week", text, re.I):
        return "week"
    if re.search(r"按天|每天|每日|日维度|趋势|daily|by day|trend", text, re.I):
        return "day"
    return None


def infer_dimensions(text: str, table: Table) -> List[str]:
    """Detect every dimension column mentioned in the question."""
    found: List[str] = []
    lowered = (text or "").lower()
    for col, aliases in table.dimensions.items():
        tokens = [col.lower()] + [a.lower() for a in aliases]
        if any(tok in lowered for tok in tokens):
            found.append(col)
    return found


class SQLTool:
    """Facade used by the agent: natural language in, QueryResult out."""

    def __init__(self, warehouse: Optional[Warehouse] = None) -> None:
        self.warehouse = warehouse or Warehouse()

    # -- direct execution ---------------------------------------------------
    def run(self, sql: str, max_rows: Optional[int] = None) -> QueryResult:
        return self.warehouse.run(sql, max_rows=max_rows)

    # -- natural language ---------------------------------------------------
    def translate(self, question: str, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Rule-based NL -> SQL. Returns SQL plus the parsed slots for transparency."""
        table = semantic.default_table(table_name)
        metric = semantic.resolve_metric(question, table.name)
        if metric is None:
            raise ValueError("could not identify a metric in the question; fall back to a custom SELECT")
        anchor = self.warehouse.max_date(table.name, table.time_column) or date.today()
        current, comparison = parse_time_window(question, anchor)
        dims = infer_dimensions(question, table)
        grain = infer_grain(question)
        sql = build_metric_query(metric, table, current, dims, grain)
        return {
            "sql": sql,
            "metric": metric,
            "table": table.name,
            "time_window": current,
            "comparison_window": comparison,
            "dimensions": dims,
            "grain": grain,
            "anchor_date": anchor,
        }

    def ask(self, question: str, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Translate then execute; returns SQL, slots and a preview."""
        parsed = self.translate(question, table_name)
        result = self.run(parsed["sql"])
        payload = result.to_dict()
        payload.update(
            {
                "metric": parsed["metric"].name,
                "dimensions": parsed["dimensions"],
                "grain": parsed["grain"],
                "time_window": parsed["time_window"].label,
                "window_start": str(parsed["time_window"].start),
                "window_end": str(parsed["time_window"].end),
            }
        )
        return payload

    def metric_series(
        self,
        metric_name: str,
        table_name: str = "orders_daily",
        grain: str = "day",
        start: Optional[date] = None,
        end: Optional[date] = None,
        dimensions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fetch a full metric series; used by anomaly detection and attribution."""
        table = semantic.default_table(table_name)
        metric = table.metrics[metric_name]
        window = TimeWindow(start, end, "custom")
        sql = build_metric_query(metric, table, window, dimensions or [], grain, order_limit=100000)
        return self.run(sql, max_rows=200000).dataframe
