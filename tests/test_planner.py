"""LLM planner: question routing and narration hooks, with rule fallback."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from ds_agent.agent import CAPABILITIES, DSQueryAgent
from ds_agent.llm import LLMClient
from ds_agent.planner import LLMPlanner


class FakeLLM:
    """Stands in for LLMClient; replays canned chat responses."""

    def __init__(self, responses: List[str], available: bool = True, fail: bool = False) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []
        self._available = available
        self._fail = fail

    @property
    def available(self) -> bool:
        return self._available

    def chat(self, messages, temperature: float = 0.0, json_mode: bool = False) -> str:
        self.calls.append({"messages": messages, "json_mode": json_mode})
        if self._fail:
            raise RuntimeError("boom")
        if not self.responses:
            raise RuntimeError("no canned response")
        return self.responses.pop(0)


def test_plan_returns_valid_intent():
    fake = FakeLLM(['{"intent": "diagnose", "reason": "user asks why"}'])
    plan = LLMPlanner(fake).plan("why did gmv drop last week", CAPABILITIES)
    assert plan is not None
    assert plan["intent"] == "diagnose"
    assert plan["engine"].startswith("llm-planner")
    assert fake.calls[0]["json_mode"] is True


def test_plan_rejects_unknown_intent():
    fake = FakeLLM(['{"intent": "make_coffee"}'])
    assert LLMPlanner(fake).plan("hello", CAPABILITIES) is None


def test_plan_falls_back_on_error():
    fake = FakeLLM([], fail=True)
    assert LLMPlanner(fake).plan("hello", CAPABILITIES) is None


def test_plan_garbage_json_is_none():
    fake = FakeLLM(["not json at all {"])
    assert LLMPlanner(fake).plan("hello", CAPABILITIES) is None


def test_unavailable_client_short_circuits():
    fake = FakeLLM(["anything"], available=False)
    planner = LLMPlanner(fake)
    assert planner.available is False
    assert planner.plan("hello", CAPABILITIES) is None
    assert planner.narrate("hello", {"intent": "sql"}) is None
    assert fake.calls == []


def test_narrate_returns_text():
    fake = FakeLLM(["GMV fell 12% week over week; the driver is channel X."])
    out = LLMPlanner(fake).narrate("why did gmv drop", {"intent": "diagnose", "delta": -0.12})
    assert out is not None
    assert "GMV" in out


def test_plan_chain_keeps_only_valid_steps():
    fake = FakeLLM(['{"intent": "diagnose", "chain": ["report", "make_coffee", "diagnose"]}'])
    plan = LLMPlanner(fake).plan("diagnose then report", CAPABILITIES)
    assert plan is not None
    assert plan["chain"] == ["report"]


def test_generate_sql_returns_parsed_query():
    fake = FakeLLM(['{"sql": "SELECT channel, SUM(gmv) AS gmv FROM orders GROUP BY 1", "explanation": "agg"}'])
    out = LLMPlanner(fake).generate_sql("gmv by channel", "TABLE orders ...")
    assert out is not None
    assert out["sql"].upper().startswith("SELECT")
    assert out["explanation"] == "agg"


def test_generate_sql_rejects_empty():
    fake = FakeLLM(['{"sql": "", "explanation": "none"}'])
    assert LLMPlanner(fake).generate_sql("gmv", "schema") is None


def test_assess_limitation_returns_text():
    fake = FakeLLM(["该需求超出当前能力范围，建议改问「近7天 GMV 为什么跌了」。"])
    out = LLMPlanner(fake).assess_limitation("帮我预测明年股价", CAPABILITIES, reason="no match")
    assert out is not None
    assert "超出" in out


def test_agent_generates_llm_sql_for_unknown_metric(agent_warehouse_pair):
    """Requests outside the semantic layer go through guarded LLM SQL."""
    agent, _ = agent_warehouse_pair
    fake = FakeLLM(
        [
            '{"sql": "SELECT channel, SUM(gmv) AS gmv FROM orders_daily WHERE date >= CURRENT_DATE - INTERVAL 7 DAY GROUP BY 1 ORDER BY gmv DESC", "explanation": "weekly gmv by channel"}',
        ]
    )
    agent.llm = fake
    try:
        # Direct fallback call: the semantic layer has no matching metric.
        reply = agent._llm_sql_fallback("每用户平均会话时长明细")
        llm_calls = [tc for tc in reply.tool_calls if tc.get("tool") == "llm_sql"]
        assert llm_calls, "expected the llm_sql tool call"
        assert llm_calls[0]["result"]["row_count"] > 0
        assert "SELECT" in reply.blocks[-1]["markdown"].upper()
    finally:
        agent.llm = LLMClient()


def test_agent_limitation_note_without_llm_sql(agent_warehouse_pair):
    """When LLM SQL cannot answer either, the reply flags a limitation."""
    agent, _ = agent_warehouse_pair
    fake = FakeLLM(
        [
            '{"sql": "SELECT channel FROM nonexistent_table_xyz", "explanation": "bad table"}',
            "这一需求超出了当前能力范围。",
        ]
    )
    agent.llm = fake
    try:
        reply = agent._llm_sql_fallback("每用户平均会话时长明细")
        assert reply.data.get("limitation") is True
        assert "超出" in reply.text
    finally:
        agent.llm = LLMClient()


def test_agent_runs_llm_task_chain(agent_warehouse_pair):
    """plan.chain executes extra tools after the primary one (diagnose+report)."""
    agent, _ = agent_warehouse_pair
    fake = FakeLLM(['{"intent": "help", "reason": "capability question"}'])
    agent.llm = fake
    try:
        # Directly exercise the merge helper: diagnose output + report output.
        from ds_agent.reply import AgentReply

        primary = AgentReply(text="诊断完成", blocks=[{"type": "markdown", "markdown": "a"}])
        extra = AgentReply(text="报告已生成", artifacts=["artifacts/x.html"])
        merged = DSQueryAgent._merge_replies(primary, extra, "report")
        assert "诊断完成" in merged.text and "报告已生成" in merged.text
        assert merged.blocks == [{"type": "markdown", "markdown": "a"}]
        assert merged.artifacts == ["artifacts/x.html"]
    finally:
        agent.llm = LLMClient()


def test_help_handler_adds_limitation_for_unmatched_question(agent_warehouse_pair):
    agent, _ = agent_warehouse_pair
    fake = FakeLLM(["这一需求超出当前能力范围，可以试试「近7天 GMV 为什么跌了」。"])
    agent.llm = fake
    try:
        reply = agent._handle_help("帮我算一下公司明年的估值", {})
        assert reply.data.get("limitation") is True
        assert reply.text.startswith("这一需求超出当前能力范围")
    finally:
        agent.llm = LLMClient()


def test_agent_uses_llm_planner_then_rules_fallback(agent_warehouse_pair):
    """The planner picks the intent; slot filling still comes from the rules."""
    agent, warehouse = agent_warehouse_pair
    fake = FakeLLM(['{"intent": "sql", "reason": "ranking request"}'])
    agent.llm = fake
    try:
        reply = agent.ask("近30天各渠道 GMV 排名")
        assert reply.data["planner"].startswith("llm-planner")
        assert reply.data["intent"] == "sql"
        # A failing LLM silently degrades to the deterministic router.
        agent.llm = FakeLLM([], fail=True)
        reply2 = agent.ask("近30天各渠道 GMV 排名")
        assert reply2.data["planner"] == "rules"
        assert reply2.data["intent"] == "sql"
    finally:
        agent.llm = LLMClient()  # unavailable default: rules mode again


def test_agent_narration_prepends_summary(agent_warehouse_pair):
    agent, _ = agent_warehouse_pair
    fake = FakeLLM(
        [
            '{"intent": "sql", "reason": "ranking request"}',
            "Top channel holds the largest GMV share in the window.",
        ]
    )
    agent.llm = fake
    try:
        reply = agent.ask("近30天各渠道 GMV 排名")
        assert reply.data.get("narrated") is True
        assert reply.data.get("deterministic_text")  # original answer preserved
        assert reply.text.startswith("Top channel")
    finally:
        agent.llm = LLMClient()  # unavailable default: rules mode again


@pytest.fixture
def agent_warehouse_pair(warehouse, workdir):
    from ds_agent.agent import DSQueryAgent

    artifacts = workdir / "artifacts_planner"
    artifacts.mkdir(parents=True, exist_ok=True)
    instance = DSQueryAgent(warehouse=warehouse, artifact_dir=artifacts)
    yield instance, warehouse
    instance.close()
