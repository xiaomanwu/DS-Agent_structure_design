"""Metric dictionary: registration, validation, semantic activation, lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest

from ds_agent.agent import DSQueryAgent
from ds_agent.metric_dict import MetricDefinitionError, MetricDictionary
from ds_agent.semantic import semantic
from ds_agent.tools.sql import Warehouse


@pytest.fixture()
def setup(workdir: Path, warehouse: Path):
    """Fresh dictionary + agent on a private warehouse copy (no cross-test pollution)."""
    wh = workdir / "warehouse_metric_dict.duckdb"
    if not wh.exists() or wh.stat().st_size == 0:
        import shutil

        shutil.copy(warehouse, wh)
    md = MetricDictionary(path=workdir / "metric_dict_test.json", semantic=semantic, warehouse=Warehouse(wh))
    md.entries = {}
    agent = DSQueryAgent(warehouse=wh, artifact_dir=workdir / "artifacts_md", metric_dict=md)
    yield agent, md
    agent.close()
    # Deactivate custom metrics so other tests see the pristine semantic layer.
    for entry in list(md.entries.values()):
        semantic.default_table(entry.get("table")).metrics.pop(entry["name"], None)
    md.entries = {}


def test_register_validates_and_activates(setup):
    agent, md = setup
    entry = md.register("profit_rate", "gmv - refund_amount", "gmv", aliases=["利润率"], table="orders_daily")
    assert entry["status"] == "active"
    assert entry["unit"] == "%"
    # Activated into the semantic layer: resolvable by name and by alias.
    assert semantic.resolve_metric("利润率") is not None
    assert semantic.resolve_metric("profit_rate") is not None
    # And queryable end-to-end through the agent.
    reply = agent.ask("近7天各渠道 利润率 排名")
    assert reply.data["intent"] == "sql"
    assert "profit_rate" in reply.text


def test_register_rejects_unknown_column(setup):
    _, md = setup
    with pytest.raises(MetricDefinitionError):
        md.register("bad_metric", "gmv", "column_that_does_not_exist", table="orders_daily")


def test_register_rejects_duplicate_active_name(setup):
    _, md = setup
    md.register("dup_metric", "gmv", table="orders_daily")
    with pytest.raises(MetricDefinitionError):
        md.register("dup_metric", "orders", table="orders_daily")
    # After deprecation the name frees up (v2 workflow).
    md.deprecate("dup_metric")
    entry = md.register("dup_metric", "orders", table="orders_daily")
    assert entry["status"] == "active"


def test_formula_parsing_and_alias_resolution(setup):
    _, md = setup
    parsed = md.parse_formula("注册指标：客单价2 = gmv / 订单量")
    assert parsed == {"name": "客单价2", "numerator": "gmv", "denominator": "订单量"}
    # The human operand "订单量" resolves to the physical column `orders`.
    entry = md.register(parsed["name"], parsed["numerator"], parsed["denominator"], table="orders_daily")
    assert entry["denominator"] == "orders"


def test_deprecate_removes_from_semantic_but_keeps_lineage(setup):
    _, md = setup
    md.register("old_metric", "gmv", table="orders_daily")
    md.deprecate("old_metric", reason="口径变更")
    assert semantic.default_table("orders_daily").metrics.get("old_metric") is None
    assert md.entries["old_metric"]["status"] == "deprecated"
    assert md.entries["old_metric"]["deprecation_reason"] == "口径变更"
    assert md.list() == []  # active list is empty


def test_agent_registers_via_natural_language(setup):
    agent, md = setup
    reply = agent.ask("注册指标：利润率 = (gmv - refund_amount) / gmv")
    assert reply.data["intent"] == "metric"
    assert "利润率" in reply.text
    assert md.get("利润率") is not None
    # Invalid caliber is rejected with a reason, not executed.
    reply2 = agent.ask("注册指标：坏指标 = gmv / 不存在的列")
    assert "Registration rejected" in reply2.text
    assert md.get("坏指标") is None
    # Dictionary listing includes preset + custom entries.
    reply3 = agent.ask("指标字典")
    assert "dictionary holds" in reply3.text


def test_dictionary_persists_to_json(setup, workdir: Path):
    _, md = setup
    md.register("persisted_metric", "gmv", table="orders_daily")
    reloaded = MetricDictionary(path=workdir / "metric_dict_test.json", semantic=semantic)
    assert reloaded.get("persisted_metric") is not None
    semantic.default_table("orders_daily").metrics.pop("persisted_metric", None)


def test_metric_dictionary_doc_in_knowledge_base():
    """docs/metric_dictionary.md is part of the RAG corpus."""
    from ds_agent.knowledge import KnowledgeBase

    kb = KnowledgeBase()
    if not len(kb):  # docs missing in this checkout: skip quietly
        pytest.skip("docs corpus unavailable")
    hits = kb.search("比率指标 分子分母 聚合口径", k=3)
    assert any("指标" in h["title"] or "指标" in h["text"] for h in hits)


def test_parse_formula_strips_register_prefix(setup):
    """\"Register metric profit_rate = ...\" must be named profit_rate, not
    register_metric_profit_rate (the old leftmost-match bug)."""
    _, md = setup
    parsed = md.parse_formula("Register metric profit_rate = (gmv - refund_amount) / gmv")
    assert parsed is not None
    assert parsed["name"].lower() == "profit_rate"
    assert parsed["denominator"] == "gmv"
    parsed_zh = md.parse_formula("注册指标：利润率 = gmv - refund_amount")
    assert parsed_zh is not None and parsed_zh["name"] == "利润率"
