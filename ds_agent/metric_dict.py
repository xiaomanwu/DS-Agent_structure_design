"""Metric dictionary: user-registered metric definitions with governance.

The semantic layer holds the *preset* metrics every downstream tool shares.
This module is the *user-facing* registry on top of it:

- register a new metric from a formula (``客单价 = gmv / 订单数``), validated
  against the live warehouse schema (numerator/denominator columns must exist);
- once registered, the metric is activated into the semantic layer, so NL
  querying, diagnosis, attribution and reports immediately use one caliber;
- definitions persist to JSON (survives restarts) and can be deprecated (kept
  for lineage) rather than deleted.

The methodology behind the rules enforced here lives in
``docs/metric_dictionary.md`` — the same document the RAG knowledge base
retrieves for caliber questions.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from .semantic import Metric, SemanticLayer
from .tools.sql import Warehouse


class MetricDefinitionError(ValueError):
    """Raised when a metric definition fails validation."""


_FORMULA_RE = re.compile(r"(?P<name>[A-Za-z][A-Za-z0-9_ ]*|[一-龥][一-龥0-9]*)\s*[=＝]\s*(?P<body>.+?)\s*$")


class MetricDictionary:
    """CRUD + governance for user-defined metrics, persisted as JSON."""

    def __init__(
        self,
        path: Optional[Path] = None,
        semantic: Optional[SemanticLayer] = None,
        warehouse: Optional[Warehouse] = None,
    ) -> None:
        self.path = Path(path) if path else Path(__file__).resolve().parent.parent / "data" / "metric_dictionary.json"
        self.semantic = semantic
        self.warehouse = warehouse
        self.entries: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.entries = {}

    # ------------------------------------------------------------------ persist
    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, ensure_ascii=False, indent=2), encoding="utf-8")

    # ----------------------------------------------------------------- validate
    def _table_exists(self, table: str) -> bool:
        if self.semantic is not None and table in self.semantic.tables:
            return True
        if self.warehouse is not None:
            try:
                return table in self.warehouse.tables()
            except Exception:  # noqa: BLE001
                return False
        return True  # nothing to validate against

    def _expr_valid(self, table: str, expr: str) -> bool:
        """Probe ``SELECT <expr> FROM <table> LIMIT 1`` on the live warehouse.

        Accepts bare columns and SQL expressions (``gmv - refund_amount``);
        the connection is read-only, so this cannot mutate anything.
        """
        if self.warehouse is None:
            return True
        try:
            self.warehouse.con.execute(f"SELECT {expr} FROM {table} LIMIT 1")
            return True
        except Exception:  # noqa: BLE001 - any SQL error means the caliber is wrong
            return False

    def validate(self, table: str, numerator: str, denominator: Optional[str]) -> None:
        if not self._table_exists(table):
            raise MetricDefinitionError(
                f"table `{table}` is unknown — the definition must be bound to a real table"
            )
        for label, expr in (("numerator", numerator), ("denominator", denominator)):
            if expr and not self._expr_valid(table, expr):
                raise MetricDefinitionError(
                    f"`{expr}` ({label}) is not a valid column or expression on table `{table}` — "
                    "check the caliber against the data dictionary before registering"
                )

    def _resolve_operand(self, operand: str, table: str) -> str:
        """Map a human operand (``订单数``) to a physical column (``orders``).

        Order of resolution: preset metric numerator -> dimension alias ->
        raw column name. Raises with a helpful message on failure.
        """
        operand = operand.strip().strip("`'\"")
        if self.semantic is not None:
            metric = self.semantic.resolve_metric(operand, table)
            if metric is not None and metric.table == table:
                return metric.numerator
        return operand

    # ----------------------------------------------------------------- register
    def register(
        self,
        name: str,
        numerator: str,
        denominator: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        unit: str = "",
        higher_is_better: bool = True,
        description: str = "",
        table: str = "orders_daily",
        owner: str = "",
        formula_text: str = "",
        activate: bool = True,
    ) -> Dict[str, Any]:
        """Validate and register a metric; returns the stored entry.

        With ``activate`` (default) the metric also joins the semantic layer,
        so every downstream tool immediately uses the same definition.
        """
        name = name.strip().lower().replace(" ", "_")
        if not re.fullmatch(r"[a-z一-龥][a-z0-9_一-龥]*", name):
            raise MetricDefinitionError(
                f"metric name `{name}` is invalid: use lowercase letters/digits/underscores (Chinese names allowed), "
                "starting with a letter or CJK character"
            )
        numerator = self._resolve_operand(numerator, table)
        if denominator:
            denominator = self._resolve_operand(denominator, table)
        self.validate(table, numerator, denominator)
        if numerator == denominator:
            raise MetricDefinitionError("numerator and denominator must differ")
        existing = self.entries.get(name)
        if existing and existing.get("status") == "active":
            raise MetricDefinitionError(
                f"metric `{name}` already exists (registered {existing.get('created_at')}) — "
                "use a v2 name and deprecate the old one instead of overwriting the caliber"
            )

        entry = {
            "name": name,
            "table": table,
            "numerator": numerator,
            "denominator": denominator,
            "aliases": aliases or [],
            "unit": unit or ("%" if denominator else ""),
            "higher_is_better": higher_is_better,
            "description": description,
            "owner": owner,
            "formula_text": formula_text or self._formula_text(numerator, denominator),
            "status": "active",
            "created_at": date.today().isoformat(),
            "updated_at": date.today().isoformat(),
            "is_custom": True,
        }
        self.entries[name] = entry
        if activate:
            self.activate(name)
        self._save()
        return entry

    @staticmethod
    def _formula_text(numerator: str, denominator: Optional[str]) -> str:
        return f"{numerator} / {denominator}" if denominator else f"SUM({numerator})"

    def activate(self, name: str) -> None:
        """Push a dictionary entry into the live semantic layer."""
        entry = self.entries.get(name)
        if entry is None or self.semantic is None:
            return
        table = self.semantic.default_table(entry.get("table"))
        table.metrics[name] = Metric(
            name=name,
            numerator=entry["numerator"],
            denominator=entry.get("denominator"),
            aliases=list(entry.get("aliases", [])),
            unit=entry.get("unit", ""),
            higher_is_better=entry.get("higher_is_better", True),
            description=entry.get("description", ""),
            table=entry.get("table", ""),
        )

    # --------------------------------------------------------------- lifecycle
    def deprecate(self, name: str, reason: str = "") -> Dict[str, Any]:
        entry = self.entries.get(name)
        if entry is None:
            raise MetricDefinitionError(f"metric `{name}` is not in the dictionary")
        entry["status"] = "deprecated"
        entry["updated_at"] = date.today().isoformat()
        if reason:
            entry["deprecation_reason"] = reason
        if self.semantic is not None:
            self.semantic.default_table(entry.get("table")).metrics.pop(name, None)
        self._save()
        return entry

    # ------------------------------------------------------------------ lookup
    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self.entries.get(name.strip().lower())

    def resolve(self, text: str) -> Optional[Dict[str, Any]]:
        """Alias-aware lookup over custom entries (longest alias wins)."""
        needle = (text or "").strip().lower()
        if not needle:
            return None
        best, best_len = None, 0
        for entry in self.entries.values():
            for cand in [entry["name"]] + list(entry.get("aliases", [])):
                cand = cand.lower()
                if cand == needle:
                    return entry
                if cand in needle and len(cand) > best_len:
                    best, best_len = entry, len(cand)
        return best

    def list(self, status: str = "active") -> List[Dict[str, Any]]:
        return [e for e in self.entries.values() if e.get("status", "active") == status]

    # ------------------------------------------------------------------ parse
    def parse_formula(self, text: str) -> Optional[Dict[str, str]]:
        """Extract ``name = numerator [ / denominator ]`` from free text.

        Returns ``{"name", "numerator", "denominator"}`` or None. The name
        keeps its original script (Chinese names are converted on register).
        """
        match = _FORMULA_RE.search(text or "")
        if not match:
            return None
        body = match.group("body").strip()
        # The leftmost name match can swallow the command prefix ("register
        # metric profit_rate = ...") because the name class allows spaces;
        # strip the leading verbs so the metric is named what follows them.
        name = re.sub(
            r"^(?:register|define|create|add|new|注册|新增|创建|登记|定义)\s+(?:metric|指标)?\s*",
            "",
            match.group("name").strip(),
            flags=re.I,
        ).strip() or match.group("name").strip()
        if "/" in body:
            numerator, _, denominator = body.partition("/")
            return {"name": name, "numerator": numerator.strip(), "denominator": denominator.strip()}
        return {"name": name, "numerator": body, "denominator": None}
