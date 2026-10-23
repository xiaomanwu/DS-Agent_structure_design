"""Method advisor: scenario parsing must route to the right estimator family."""

from __future__ import annotations

import pytest

from ds_agent.tools.method_advisor import CATALOGUE, parse_scenario, recommend_methods


@pytest.mark.parametrize(
    "question,expected",
    [
        ("设计一个双边实验，买家侧和卖家侧同时发券", "two_sided"),
        ("时间片轮转实验怎么设计推断单元", "switchback"),
        ("父子实验两层同时跑怎么估", "nested"),
        ("没有实验，评估某天上线的改版对 GMV 的影响", "its"),
        ("会员发放按 600 分门槛定的，评估效果", "rdd"),
        ("多个城市推行新政策，评估对城市 GMV 的影响", "did"),
        ("促销是运营手动选人发的，评估真实效果", "psm"),
        ("想看补贴对哪类用户最有效，好定向投放", "uplift"),
        ("标准 AB 实验怎么设计，每天5万流量", "ab_test"),
    ],
)
def test_scenarios_pick_the_expected_primary(question, expected):
    result = recommend_methods(question)
    assert result["primary"] is not None
    assert result["primary"]["method"] == expected, [
        (r["method"], r["score"]) for r in result["recommendations"]
    ]


def test_every_recommendation_is_executable():
    # Each recommended method must name a tool that exists in the catalogue
    # (i.e. is actually shipped), not generic advice.
    for question in ("评估改版影响", "设计实验", "评估补贴效果"):
        for rec in recommend_methods(question)["recommendations"]:
            assert rec["tool"] in {m["tool"] for m in CATALOGUE.values()}
            assert rec["reason"] and rec["assumption"] and rec["pitfall"]


def test_experiment_branch_prefers_designs_over_matching():
    result = recommend_methods("我们做了随机分流实验，评估效果")
    methods = [r["method"] for r in result["recommendations"]]
    assert "ab_test" in methods or "switchback" in methods
    assert "psm" not in methods[:1]


def test_features_are_parsed():
    f = parse_scenario("时间片轮转 + 双边市场，没有做实验")
    assert f["switchback"] is True
    assert f["two_sided"] is True
    assert f["observational"] is True
    assert f["experiment"] is False


def test_checklist_and_narrative_present():
    result = recommend_methods("多个城市推行新政策")
    assert len(result["checklist"]) >= 4
    assert "DiD" in result["narrative"] or "双重差分" in result["narrative"]
    # bilingual output
    en = recommend_methods("evaluate a policy across cities", lang="en")
    assert en["primary"]["method"] == "did"
    assert en["primary"]["name"] != "双重差分（DiD）"


def test_unknown_scenario_still_returns_something():
    result = recommend_methods("随便看看")
    assert result["recommendations"], "should fall back to a default method"
    assert result["primary"] is not None
