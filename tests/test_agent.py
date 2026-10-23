"""Agent routing, end-to-end answers and the SQL guard."""

from __future__ import annotations

import pytest

from ds_agent.tools.sql import UnsafeQueryError, parse_time_window, sanitize_sql


def test_router_maps_questions_to_capabilities(agent):
    cases = {
        "近7天 GMV 为什么跌了？": "diagnose",
        "哪个渠道贡献了这次下滑": "diagnose",
        "近30天各渠道 GMV 排名": "sql",
        "看下这个 AB 实验的结果": "experiment_analysis",
        "设计一个支付转化率实验，基线3%，检测2%提升": "experiment_design",
        "满意度调查基线70%，检测3个百分点变化，需要多少样本？": "sampling",
        "How many samples do I need for a 70%-baseline survey to detect a 3pp change?": "sampling",
        "用双重差分评估补贴政策效果": "causal",
        "把分析写成周报": "report",
    }
    for question, expected in cases.items():
        intent, _, _ = agent.route(question)
        assert intent == expected, f"{question} -> {intent}, expected {expected}"


def test_diagnose_produces_charts_and_tables(agent):
    reply = agent.ask("近7天 GMV 为什么跌了？")
    kinds = [b["type"] for b in reply.blocks]
    assert "kpi" in kinds and "chart" in kinds and "table" in kinds
    assert "iOS" in reply.text  # the planted payment-SDK regression
    assert reply.data["attribution"]


def test_default_baseline_is_the_decomposition(agent):
    reply = agent.ask("近7天 GMV 为什么跌了？")
    assert reply.data["baseline"]["requested"] == "trailing"
    assert reply.data["baseline"]["method"] == "trailing"


def test_diagnose_can_use_holt_winters(agent):
    reply = agent.ask("用 holt-winters 看下近7天 GMV 为什么跌了")
    baseline = reply.data["baseline"]
    assert baseline["requested"] == "holt_winters"
    assert baseline["method"] == "holt_winters"
    # An explicit model still reports its own out-of-sample error.
    assert 0 < baseline["mase"] < 1
    assert "Holt-Winters" in reply.text
    assert any(i["label"] == "基线模型" for i in reply.blocks[0]["items"])


def test_auto_baseline_backtests_the_candidates(agent):
    reply = agent.ask("用时序基线对比一下近7天 GMV 的异常")
    baseline = reply.data["baseline"]
    assert baseline["requested"] == "auto"
    assert baseline["backtest"] and len(baseline["backtest"]) >= 5
    chosen = [r for r in baseline["backtest"] if r["method"] == baseline["method"]]
    assert chosen, "the chosen model must appear in the comparison table"
    assert min(r["mase"] for r in baseline["backtest"]) == pytest.approx(chosen[0]["mase"])
    # The comparison table is what makes `auto` auditable rather than magic.
    assert sum(1 for b in reply.blocks if b["type"] == "table") >= 3


def test_follow_up_question_keeps_context(agent):
    agent.ask("近7天 GMV 为什么跌了？")
    reply = agent.ask("按渠道拆一下")
    assert reply.blocks, "follow-up should still produce output"


def test_experiment_readout_reports_a_verdict(agent):
    reply = agent.ask("看下这个 AB 实验的结果")
    assert "SRM" in reply.text
    assert any(b["type"] == "table" for b in reply.blocks)
    assert reply.data["readout"]


def test_experiment_design_parses_baseline_and_mde(agent):
    reply = agent.ask("设计一个支付转化率实验，基线3%，希望检测2%的相对提升，每天5万流量")
    sizing = reply.data["sizing"]
    assert reply.data["design"]["baseline"] == pytest.approx(0.03)
    assert reply.data["design"]["mde_rel"] == pytest.approx(0.02)
    assert sizing["n_total"] > 0
    assert reply.data["duration"]["recommended_days"] >= 7


def test_causal_replies_are_structured(agent):
    reply = agent.ask("用双重差分评估补贴政策对城市 GMV 的影响")
    assert "ATT" in reply.text
    assert reply.data["did"]["att"] > 0


def test_report_writes_both_formats(agent, workdir):
    from ds_agent.agent import DSQueryAgent

    local = DSQueryAgent(warehouse=agent.sql.warehouse.path, artifact_dir=workdir / "report_out")
    try:
        local.ask("近7天 GMV 为什么跌了？")
        reply = local.ask("把分析写成周报")
    finally:
        local.close()
    assert len(reply.artifacts) == 2
    html_path = reply.artifacts[0]
    text = open(html_path, encoding="utf-8").read()
    assert "<img src=\"data:image/png;base64," in text
    assert "执行摘要" in text


def test_help_lists_capabilities(agent):
    from ds_agent.agent import CAPABILITIES

    reply = agent.ask("help")
    for capability in CAPABILITIES:
        assert capability in reply.text, f"{capability} missing from help"
    # Every capability carries an example, and the label follows the language.
    assert reply.text.count("e.g.") == len(CAPABILITIES)
    assert "例：" not in reply.text

    chinese = agent.ask("你能做什么？")
    assert "例：" in chinese.text
    assert "e.g." not in chinese.text


def test_unknown_metric_degrades_gracefully(agent):
    reply = agent.ask("查一下营收")
    assert reply.text


def test_sql_guard_blocks_writes():
    for bad in ("DROP TABLE orders_daily", "SELECT 1; DROP TABLE x", "UPDATE t SET a=1", "/*x*/ delete from t"):
        with pytest.raises(UnsafeQueryError):
            sanitize_sql(bad)


def test_sql_guard_allows_select():
    assert sanitize_sql("SELECT 1 AS a -- comment") == "SELECT 1 AS a"
    assert sanitize_sql("with x as (select 1) select * from x").startswith("with")


def test_time_window_parsing():
    from datetime import date

    anchor = date(2026, 10, 20)
    window, comparison = parse_time_window("近30天 GMV 趋势", anchor)
    assert window.days == 30
    assert window.start == date(2026, 9, 21)
    assert comparison.end == date(2026, 9, 20)

    week, _ = parse_time_window("上周的转化率", anchor)
    assert week.days == 7
    assert week.end == date(2026, 10, 18)  # Sunday of the previous week
