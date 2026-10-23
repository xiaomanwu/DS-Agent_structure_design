"""Advanced readouts: routing, block shape, and the number each handler must report.

Every handler is exercised through ``agent.ask`` rather than called directly,
because the contract the user sees is the reply - text, KPI blocks, charts,
tables, and the ``data`` payload that a follow-up report consumes.
"""

from __future__ import annotations

import pytest

from ds_agent.agent import DSQueryAgent

ADVANCED_CASES = {
    "看下时间片轮转实验的结果": "switchback",
    "这个反转实验的效应能持续吗": "reversal",
    "父子实验怎么验证两层互不干扰": "nested",
    "双边实验，买家侧和卖家侧同时干预": "two_sided",
    "这个实验结论稳健吗，做个敏感性分析": "sensitivity",
    "用因果森林看哪类用户对补贴最敏感": "uplift",
    "用断点回归评估会员门槛的效果": "rdd",
    "这次改版对 GMV 的影响，用时序因果评估": "ts_causal",
}


@pytest.fixture(scope="module")
def local(warehouse, workdir):
    """A private agent: these tests add sections, and ordering must not matter."""
    instance = DSQueryAgent(warehouse=warehouse, artifact_dir=workdir / "readout_artifacts")
    yield instance
    instance.close()


@pytest.mark.parametrize("question,intent", ADVANCED_CASES.items())
def test_advanced_questions_route_to_their_handler(local, question, intent):
    routed, _slots, confidence = local.route(question)
    assert routed == intent, f"{question} -> {routed}, expected {intent}"
    assert confidence > 0


@pytest.mark.parametrize("question,intent", ADVANCED_CASES.items())
def test_every_advanced_reply_has_blocks_and_a_section(local, question, intent):
    before = len(local.findings)
    reply = local.ask(question)
    assert reply.text and len(reply.text) > 40
    kinds = [b["type"] for b in reply.blocks]
    assert "kpi" in kinds, f"{intent} should lead with KPIs"
    assert "table" in kinds, f"{intent} should show its estimates"
    assert reply.tool_calls, "the reply should name the tool it called"
    assert len(local.findings) > before, "a section must be recorded for the report"


def test_switchback_reports_the_block_level_estimate(local):
    data = local.ask("看下时间片轮转实验的结果").data["switchback"]
    assert 0.0 < data["relative"] < 0.12
    assert data["p_value"] < 0.05
    assert data["n_blocks"] >= 10
    # The paired-block estimator is the robustness reading of the same design.
    assert "se" in data["paired_blocks"]


def test_switchback_washout_can_be_disabled(local):
    with_washout = local.ask("看下时间片轮转实验的结果").data["switchback"]
    without = local.ask("看下时间片轮转实验的结果，不清洗").data["switchback"]
    assert without["washout_days"] == 0
    assert without["relative"] != pytest.approx(with_washout["relative"], abs=1e-9)


def test_reversal_reports_both_phases_and_persistence(local):
    data = local.ask("这个反转实验的效应能持续吗").data["reversal"]
    assert data["tau1"] > 0
    assert data["tau2"] < 0
    assert data["reversal_detected"] is True
    assert data["infrastructure_valid"] is True, "a sum test near zero means no drift in the pipeline"
    assert data["persistence"] == pytest.approx(0.73, abs=0.25)
    assert data["pooled"]["relative"] == pytest.approx(0.05, abs=0.03)


def test_nested_reports_orthogonality_and_child_effect(local):
    data = local.ask("父子实验怎么验证两层互不干扰").data["nested"]
    assert data["orthogonal"] is True
    # The generator plants +6% on the child layer; the parent-FE estimate must find it.
    assert data["adjusted"]["relative"] == pytest.approx(0.06, abs=0.03)
    assert data["adjusted"]["p_value"] < 0.05
    assert data["interaction"]["detected"] is False


def test_two_sided_separates_the_two_roles(local):
    data = local.ask("双边实验，买家侧和卖家侧同时干预").data["two_sided"]
    buyer = data["results"]["buyer"]
    seller = data["results"]["seller"]
    assert buyer["buyer_direct_relative"] == pytest.approx(0.08, abs=0.03)
    assert seller["seller_direct_relative"] == pytest.approx(0.06, abs=0.03)
    # Cross-side ratios are computed per metric, and the direct effect dominates.
    assert data["spillover"]["buyer"]["ratio"] < data["spillover"]["seller"]["ratio"] * 10
    assert data["cells"] == {"00": 60, "01": 60, "10": 60, "11": 60}


def test_sensitivity_keeps_the_sign_across_specs(local):
    data = local.ask("这个实验结论稳健吗，做个敏感性分析").data["sensitivity"]
    assert data["n_specs"] >= 3
    assert data["sign_stable"] is True
    assert data["placebo_false_positive_rate"] <= 0.1
    low, high = data["lift_range"]
    assert 0 < low <= high < 0.15


def test_uplift_is_heterogeneous_and_targetable(local):
    data = local.ask("用因果森林看哪类用户对补贴最敏感").data["causal_forest"]
    assert data["ate"] == pytest.approx(12.5, abs=4.0)
    assert data["qini"] > 0.1
    # Targeting a smaller share must capture a disproportionate share of the gain.
    targeting = data["targeting"]
    assert targeting[0]["uplift_lift"] > targeting[-1]["uplift_lift"] > 1.0
    assert data["best_decile_uplift"] > 2 * data["ate"]


def test_uplift_tree_branch_takes_the_tree_path(local):
    reply = local.ask("用因果树看哪类用户对补贴最敏感")
    assert "causal_tree" in reply.data
    assert reply.data["causal_tree"]["ate"] == pytest.approx(12.5, abs=5.0)
    assert reply.tool_calls[-1]["tool"] == "causal_tree"
    assert reply.data["causal_tree"]["n_leaves"] >= 2


def test_rdd_reports_the_jump_and_the_falsification_checks(local):
    data = local.ask("用断点回归评估会员门槛的效果").data["rdd"]
    assert data["effect"] == pytest.approx(22.0, abs=8.0)
    assert data["p_value"] < 0.05
    assert data["covariate_balance_ok"] is True
    assert data["density"]["manipulation"] is False


def test_rdd_cutoff_can_be_overridden_in_the_question(local):
    data = local.ask("用断点回归看一下 620 分这个门槛的效果").data["rdd"]
    assert data["cutoff"] == pytest.approx(620.0)


def test_ts_causal_reports_three_readings(local):
    data = local.ask("这次改版对 GMV 的影响，用时序因果评估").data
    its, impact = data["its"], data["impact"]
    assert its["relative"] == pytest.approx(0.12, abs=0.05)
    assert impact["relative"] == pytest.approx(0.12, abs=0.06)
    assert impact["pre_fit_r2"] > 0.8
    assert {"frame", "history"}.isdisjoint(its.keys() | impact.keys()), "payloads stay light"


def test_english_question_gets_english_output(warehouse, workdir):
    instance = DSQueryAgent(warehouse=warehouse, artifact_dir=workdir / "readout_en")
    try:
        reply = instance.ask("Read out the switchback test results")
    finally:
        instance.close()
    assert "Switchback" in reply.text
    assert "相对效应" not in reply.text


def test_advanced_sections_reach_the_report(local):
    local.ask("看下时间片轮转实验的结果")
    local.ask("用断点回归评估会员门槛的效果")
    local.ask("这次改版对 GMV 的影响，用时序因果评估")
    reply = local.ask("把上面的分析写成周报")
    assert len(reply.artifacts) == 2
    html = open(reply.artifacts[0], encoding="utf-8").read()
    for title in ("Switchback experiment", "Regression discontinuity", "Time-series causal inference"):
        assert title in html


def test_report_keeps_every_section(warehouse, workdir):
    """A long session must not silently lose the tail of the analysis."""
    from ds_agent.tools.report import Section

    instance = DSQueryAgent(warehouse=warehouse, artifact_dir=workdir / "report_cap")
    try:
        instance.findings = [Section(title=f"章节 {i} 标题", prose=f"第 {i} 节正文") for i in range(1, 15)]
        instance.summary_bullets = [f"结论 {i}" for i in range(1, 15)]
        reply = instance.ask("把上面的分析写成周报")
    finally:
        instance.close()
    markdown = next(p for p in reply.artifacts if str(p).endswith(".md"))
    text = open(markdown, encoding="utf-8").read()
    for i in (1, 10, 11, 14):
        assert f"章节 {i} 标题" in text, f"section {i} was dropped from the report"
    # The executive summary stays short, but it says what it left out.
    assert "more findings are in the sections below." in text
