"""MCP stdio server: protocol handshake, tool listing and tool calls."""

import io
import json

from ds_agent.mcp_server import handle_message, serve


def _req(method, params=None, id=1):
    msg = {"jsonrpc": "2.0", "id": id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


# ------------------------------------------------------------------ protocol
def test_initialize_handshake():
    resp = handle_message(_req("initialize", {"protocolVersion": "2024-11-05"}))
    assert resp["result"]["serverInfo"]["name"] == "ds-agent"
    assert "tools" in resp["result"]["capabilities"]
    assert resp["result"]["protocolVersion"] == "2024-11-05"


def test_notifications_return_none():
    assert handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert handle_message({"jsonrpc": "2.0", "method": "unknown/notification"}) is None


def test_ping_and_unknown_method():
    assert handle_message(_req("ping"))["result"] == {}
    resp = handle_message(_req("no/such/method"))
    assert resp["error"]["code"] == -32601


def test_malformed_json_line_yields_parse_error():
    out = io.StringIO()
    serve(io.StringIO("not-json\n"), out)
    lines = out.getvalue().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["error"]["code"] == -32700


# -------------------------------------------------------------------- tools
def test_tools_list_has_five_documented_tools():
    resp = handle_message(_req("tools/list"))
    names = [t["name"] for t in resp["result"]["tools"]]
    assert names == [
        "ds_agent_ask",
        "sample_size_proportion",
        "sample_size_mean",
        "knowledge_search",
        "metric_list",
    ]
    for tool in resp["result"]["tools"]:
        assert "description" in tool and "inputSchema" in tool


def test_tool_call_unknown_name_is_error():
    resp = handle_message(_req("tools/call", {"name": "nope", "arguments": {}}))
    assert resp["error"]["code"] == -32602


def test_sample_size_proportion_tool_matches_stats_layer():
    from ds_agent.stats.sampling import sample_size_proportion

    resp = handle_message(
        _req("tools/call", {"name": "sample_size_proportion",
                            "arguments": {"baseline": 0.7, "mde": 0.03}})
    )
    assert "error" not in resp
    payload = json.loads(resp["result"]["content"][0]["text"])
    assert payload == sample_size_proportion(0.7, 0.03)
    assert payload["n_per_group"] == 2452


def test_sample_size_tool_rejects_bad_args_as_iserror():
    resp = handle_message(
        _req("tools/call", {"name": "sample_size_proportion",
                            "arguments": {"baseline": 1.5, "mde": 0.03}})
    )
    assert resp["result"]["isError"] is True
    assert "ValueError" in resp["result"]["content"][0]["text"]


# ------------------------------------------------- end-to-end (needs warehouse)
def test_stdio_roundtrip_ask_and_metrics(monkeypatch):
    from ds_agent.data.synthetic import build_demo_warehouse

    base = __import__("pathlib").Path(__file__).parent / ".tmp" / "mcp_roundtrip"
    base.mkdir(parents=True, exist_ok=True)
    wh = base / "wh.duckdb"
    build_demo_warehouse(wh, force=True)
    monkeypatch.setenv("DS_AGENT_WAREHOUSE", str(wh))
    monkeypatch.setenv("DS_AGENT_ARTIFACT_DIR", str(base / "art"))

    inp = io.StringIO(
        json.dumps(_req("tools/list", id=10)) + "\n"
        + json.dumps(_req("tools/call",
                          {"name": "ds_agent_ask",
                           "arguments": {"question": "满意度调查基线 70%，检测 3 个百分点变化，需要多少样本量？"}},
                          id=11)) + "\n"
        + json.dumps(_req("tools/call", {"name": "metric_list", "arguments": {}}, id=12)) + "\n"
    )
    out = io.StringIO()
    serve(inp, out)
    lines = [json.loads(x) for x in out.getvalue().strip().splitlines()]
    assert [r["id"] for r in lines] == [10, 11, 12]

    ask_payload = json.loads(lines[1]["result"]["content"][0]["text"])
    assert ask_payload["capability"] == "sampling"
    assert "2,452" in ask_payload["text"] or "2452" in ask_payload["text"]

    metrics = json.loads(lines[2]["result"]["content"][0]["text"])
    assert isinstance(metrics["metrics"], list)
