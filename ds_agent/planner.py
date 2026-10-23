"""LLM planner: the optional large-model interface for questions and tools.

Four hooks, all degrading gracefully to the deterministic pipeline:

- ``plan(question)``            -> intent + optional task chain for a question.
- ``generate_sql(question)``    -> flexible read-only SQL when the semantic
                                   layer cannot resolve the metric. The SQL is
                                   ALWAYS re-validated by ``sanitize_sql`` and
                                   executed through the guarded SQLTool.
- ``narrate(question, digest)`` -> final-report narrative on top of completed
                                   tool results. The deterministic reply is
                                   always kept as the fallback.
- ``assess_limitation(question)`` -> an honest limitation statement when the
                                   request is out of scope, with the closest
                                   available capabilities as alternatives.

The planner never executes anything itself: it only picks intents, writes SQL
that the guard re-checks, and rewrites prose. All computation stays inside the
audited tools.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .llm import LLMClient

PLANNER_VERSION = "llm-planner-2"

_MAX_CHAIN = 3


class LLMPlanner:
    """Maps user questions to agent workflows, SQL, narratives and limits."""

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm or LLMClient()

    @property
    def available(self) -> bool:
        return self.llm.available

    # ------------------------------------------------------------------ plan
    def plan(self, question: str, capabilities: Dict[str, Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Return {"intent", "reason", "chain"?} or None when unusable.

        ``chain`` is an optional ordered list of extra intents to run after the
        primary one (e.g. ``["report"]`` for "diagnose it, then write it up").
        ``capabilities`` is ``CAPABILITIES``; only real intent keys are
        selectable, so the LLM cannot invent a handler.
        """
        if not self.available:
            return None
        catalog = "\n".join(
            f"- {key}: {spec.get('zh', '')} / {spec.get('en', '')}" for key, spec in capabilities.items()
        )
        system = (
            "You are the workflow planner of a data-science agent. Pick exactly "
            "one primary intent key from the catalog for the user question. "
            "Optionally list extra intents (max 2) that should run AFTER the "
            "primary one as a task chain (e.g. analyze then report). Answer "
            "ONLY with a JSON object: {\"intent\": \"<key>\", \"reason\": "
            "\"<short>\", \"chain\": [\"<key>\", ...]}. Do not invent keys; "
            "omit \"chain\" when a single tool is enough. If nothing fits, use "
            "\"sql\" for data lookups or \"help\" for meta questions.\n\n"
            "Catalog:\n" + catalog
        )
        payload = self._ask_json(system, question)
        if payload is None:
            return None
        intent = str(payload.get("intent", "")).strip()
        if intent not in capabilities:
            return None
        chain = payload.get("chain")
        chain = [str(c).strip() for c in chain] if isinstance(chain, list) else []
        chain = [c for c in chain if c in capabilities and c != intent][:_MAX_CHAIN - 1]
        plan: Dict[str, Any] = {
            "intent": intent,
            "reason": str(payload.get("reason", ""))[:200],
            "engine": PLANNER_VERSION,
        }
        if chain:
            plan["chain"] = chain
        return plan

    # ---------------------------------------------------------- generate sql
    def generate_sql(self, question: str, schema_text: str, lang: str = "zh") -> Optional[Dict[str, str]]:
        """Flexible NL-to-SQL for requests the semantic layer cannot resolve.

        Returns {"sql", "explanation"} or None. The SQL is a suggestion only:
        the caller must pass it through ``sanitize_sql`` and the guarded
        SQLTool before anything touches the warehouse.
        """
        if not self.available:
            return None
        system = (
            "You are the SQL generator of a read-only analytics agent over "
            "DuckDB. Given the schema below, write ONE read-only SELECT/WITH "
            "statement that answers the user question. Rules: aggregate with "
            "GROUP BY where useful; filter on the time column by explicit "
            "dates when possible, otherwise relative to CURRENT_DATE; never "
            "write DML/DDL. Answer ONLY with JSON: {\"sql\": \"<query>\", "
            "\"explanation\": \"<one sentence in "
            + ("Chinese" if lang == "zh" else "English")
            + ">\"}.\n\nSchema:\n" + schema_text
        )
        payload = self._ask_json(system, question)
        if payload is None:
            return None
        sql = str(payload.get("sql", "")).strip()
        if not sql:
            return None
        return {
            "sql": sql,
            "explanation": str(payload.get("explanation", ""))[:300],
        }

    # --------------------------------------------------------------- narrate
    def narrate(self, question: str, digest: Dict[str, Any], lang: str = "zh") -> Optional[str]:
        """Turn a compact result digest into a polished answer, or None.

        ``digest`` is small (intent + headline numbers + verdicts) so the call
        stays cheap; the full deterministic reply remains authoritative.
        """
        if not self.available:
            return None
        system = (
            "You are the narrative layer of a data-science agent. The JSON below "
            "is the verified output of deterministic statistical tools. Write a "
            "concise, decision-oriented answer (max 6 sentences, plain text, "
            "Markdown bold for key numbers allowed). Never invent numbers that "
            "are not in the JSON. Reply in "
            + ("Chinese" if lang == "zh" else "English")
            + "."
        )
        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(digest, ensure_ascii=False, default=str)},
                ],
                temperature=0.2,
            )
        except Exception:  # noqa: BLE001
            return None
        text = (raw or "").strip()
        return text or None

    # ----------------------------------------------------- assess limitation
    def assess_limitation(self, question: str, capabilities: Dict[str, Dict[str, str]], reason: str = "", lang: str = "zh") -> Optional[str]:
        """Honest limitation assessment for out-of-scope requests, or None."""
        if not self.available:
            return None
        catalog = "\n".join(
            f"- {key}: {spec.get('zh', '')} / {spec.get('en', '')}" for key, spec in capabilities.items()
        )
        system = (
            "You are the self-assessment layer of a data-science agent. The "
            "user's request could not be mapped to any capability. Write a "
            "short, honest reply (max 5 sentences, reply in "
            + ("Chinese" if lang == "zh" else "English")
            + "): (1) state clearly that this is beyond the agent's current "
            "capabilities and why; (2) suggest the closest capability from the "
            "catalog, rephrased as a concrete example question the user could "
            "ask. Never promise capabilities that are not in the catalog, and "
            "never fabricate data or results.\n\nCatalog:\n" + catalog
            + ("\n\nWhy it failed: " + reason if reason else "")
        )
        try:
            raw = self.llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": question}],
                temperature=0.2,
            )
        except Exception:  # noqa: BLE001
            return None
        text = (raw or "").strip()
        return text or None

    # ------------------------------------------------------------------ util
    def _ask_json(self, system: str, user: str) -> Optional[Dict[str, Any]]:
        """chat(json_mode) with fence-tolerant parsing; None on any failure."""
        try:
            raw = self.llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                json_mode=True,
            )
        except Exception:  # noqa: BLE001 - network/auth errors fall back to rules
            return None
        payload = LLMClient.extract_json(raw)
        return payload if isinstance(payload, dict) else None
