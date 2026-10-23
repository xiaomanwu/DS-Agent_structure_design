"""Zero-dependency MCP (Model Context Protocol) server over stdio.

Exposes the DS-Agent methodology engine — the deterministic statistical
tools, not an LLM — to any MCP client (Claude Desktop, Qwen CLI, IDEs…).
The host LLM orchestrates; this server guarantees that every number comes
from the tested stats layer (308 tests lock behaviour).

Protocol: JSON-RPC 2.0, one message per line on stdin, one response per
line on stdout. Implements the subset of the MCP spec that tool servers
need — ``initialize``, ``notifications/initialized``, ``ping``,
``tools/list``, ``tools/call`` — with no third-party SDK, keeping the
project's "determinism first, zero required dependencies" ethos.

Run::

    python -m ds_agent.mcp_server            # stdio server
    # or, after ``pip install -e .``:
    ds-agent-mcp

Environment: ``DS_AGENT_WAREHOUSE`` / ``DS_AGENT_ARTIFACT_DIR`` as usual.
If the default demo warehouse is missing it is built on first use.

Tools exposed
-------------
- ``ds_agent_ask``            full NL pipeline (routing → analysis → evidence)
- ``sample_size_proportion``  Cochran sizing for proportion metrics
- ``sample_size_mean``        Cochran sizing for continuous metrics
- ``knowledge_search``        TF-IDF RAG over the business-knowledge corpus
- ``metric_list``             active metric calibers from the dictionary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, IO, List, Optional

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "ds-agent", "version": "0.1.0"}

# ---------------------------------------------------------------------------
# Tool implementations (thin wrappers; all maths lives in the tested layers)
# ---------------------------------------------------------------------------

_agent = None  # lazily built DSQueryAgent (expensive: opens DuckDB)


def _get_agent():
    """Build the agent on first use, building the demo warehouse if missing."""
    global _agent
    if _agent is None:
        from ds_agent.config import settings
        from ds_agent.data.synthetic import build_demo_warehouse

        if not Path(settings.warehouse).exists():
            build_demo_warehouse(Path(settings.warehouse), force=True)
        from ds_agent.agent import DSQueryAgent

        _agent = DSQueryAgent()
    return _agent


def tool_ask(args: Dict[str, Any]) -> Dict[str, Any]:
    question = str(args.get("question", "")).strip()
    if not question:
        raise ValueError("question is required")
    reply = _get_agent().ask(question)
    blocks = [{"type": b.get("type"), "title": b.get("title", "")} for b in reply.blocks]
    return {
        "text": reply.text,
        "blocks": blocks,
        "capability": reply.data.get("intent", "unknown"),
    }


def tool_sample_size_proportion(args: Dict[str, Any]) -> Dict[str, Any]:
    from ds_agent.stats.sampling import sample_size_proportion

    return sample_size_proportion(
        p0=float(args["baseline"]),
        mde=float(args["mde"]),
        alpha=float(args.get("alpha", 0.05)),
        power=float(args.get("power", 0.8)),
    )


def tool_sample_size_mean(args: Dict[str, Any]) -> Dict[str, Any]:
    from ds_agent.stats.sampling import sample_size_mean

    return sample_size_mean(
        sd=float(args["sd"]),
        mde=float(args["mde"]),
        alpha=float(args.get("alpha", 0.05)),
        power=float(args.get("power", 0.8)),
    )


def tool_knowledge_search(args: Dict[str, Any]) -> Dict[str, Any]:
    query = str(args.get("query", "")).strip()
    if not query:
        raise ValueError("query is required")
    kb = _get_agent().knowledge
    if kb is None:
        return {"query": query, "hits": [], "note": "knowledge corpus not available"}
    hits = kb.search(query, k=int(args.get("k", 3)))
    return {"query": query, "hits": hits}


def tool_metric_list(args: Dict[str, Any]) -> Dict[str, Any]:
    rows = _get_agent().metric_dict.list(status="active")
    return {
        "metrics": [
            {"name": r.get("name"), "formula": r.get("formula"), "table": r.get("table")}
            for r in rows
        ]
    }


# ---------------------------------------------------------------------------
# MCP plumbing
# ---------------------------------------------------------------------------

# JSON-Schema subset per tool (type/object/properties/required only).
_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "ds_agent_ask",
        "description": (
            "Run the DS-Agent natural-language analytics pipeline: intent routing, "
            "governed SQL, anomaly diagnosis, attribution, experiments, causal "
            "inference, sampling, modelling, dashboards, metric dictionary and "
            "reports. Every number comes from deterministic, tested statistics."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    {
        "name": "sample_size_proportion",
        "description": "Per-group sample size for detecting an absolute change in a proportion (Cochran normal approximation).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "baseline": {"type": "number", "description": "baseline proportion p0, 0<p0<1"},
                "mde": {"type": "number", "description": "absolute detectable change"},
                "alpha": {"type": "number"},
                "power": {"type": "number"},
            },
            "required": ["baseline", "mde"],
        },
    },
    {
        "name": "sample_size_mean",
        "description": "Per-group sample size for detecting a shift in a continuous metric (known standard deviation).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sd": {"type": "number"},
                "mde": {"type": "number"},
                "alpha": {"type": "number"},
                "power": {"type": "number"},
            },
            "required": ["sd", "mde"],
        },
    },
    {
        "name": "knowledge_search",
        "description": "Search the business-knowledge / metric-caliber corpus (offline TF-IDF RAG).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "metric_list",
        "description": "List active metric calibers registered in the metric dictionary.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

_HANDLERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "ds_agent_ask": tool_ask,
    "sample_size_proportion": tool_sample_size_proportion,
    "sample_size_mean": tool_sample_size_mean,
    "knowledge_search": tool_knowledge_search,
    "metric_list": tool_metric_list,
}


def handle_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle one JSON-RPC request; returns a response dict or None for notifications."""
    method = msg.get("method", "")
    msg_id = msg.get("id")
    is_notification = "id" not in msg

    def result(payload: Any) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "result": payload}

    def err(code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

    try:
        if method == "initialize":
            return result(
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                }
            )
        if method in ("notifications/initialized", "initialized"):
            return None
        if method == "ping":
            return result({})
        if method == "tools/list":
            return result({"tools": _TOOLS})
        if method == "tools/call":
            name = str(msg.get("params", {}).get("name", ""))
            args = msg.get("params", {}).get("arguments", {}) or {}
            handler = _HANDLERS.get(name)
            if handler is None:
                return err(-32602, f"unknown tool: {name}")
            try:
                payload = handler(args)
            except Exception as exc:  # tool-level failure -> isError result
                return result(
                    {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                     "isError": True}
                )
            return result({"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, default=str)}]})
        if is_notification:
            return None
        return err(-32601, f"method not found: {method}")
    except Exception as exc:  # server-level failure
        if is_notification:
            return None
        return err(-32603, f"{type(exc).__name__}: {exc}")


def serve(stdin: IO[str] = sys.stdin, stdout: IO[str] = sys.stdout) -> None:
    """Read line-delimited JSON-RPC from stdin, write responses to stdout."""
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            resp: Optional[Dict[str, Any]] = {
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": "parse error"},
            }
        else:
            resp = handle_message(msg)
        if resp is not None:
            stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            stdout.flush()


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
