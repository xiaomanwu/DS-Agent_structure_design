"""Experiment sizing maths and readout behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from ds_agent.data.synthetic import build_experiment_users
from ds_agent.stats import inference as inf
from ds_agent.stats import power as pw
from ds_agent.tools.experiment import EXPERIMENT_TYPES, ExperimentDesign, analyze_experiment, design_experiment, recommend_experiment_type


def test_sample_size_monotone_in_effect_size():
    small = pw.sample_size_proportion(0.03, mde_rel=0.02)
    large = pw.sample_size_proportion(0.03, mde_rel=0.10)
    assert small["n_control"] > large["n_control"] > 0


def test_power_inverts_sample_size():
    sizing = pw.sample_size_proportion(0.05, mde_rel=0.10, power=0.8)
    # Power at the returned sample size must be at least the target.
    achieved = sizing["achieved_power"]
    assert 0.79 <= achieved <= 0.82


def test_mde_matches_sample_size():
    sizing = pw.sample_size_proportion(0.04, mde_rel=0.10)
    mde = pw.mde_proportion(sizing["n_control"], 0.04)
    assert mde["mde_rel"] == pytest.approx(0.10, abs=0.005)


def test_ratio_metric_sizing_runs():
    result = pw.sample_size_ratio_metric(mean_numerator=200, mean_denominator=1.0, mde_rel=0.03)
    assert result["n_control"] > 0
    assert result["metric_type"] == "ratio"


def test_cluster_design_effect_inflates_sample():
    deff = pw.cluster_design_effect(cluster_size=50, icc=0.02)
    assert deff["design_effect"] == pytest.approx(1 + 49 * 0.02)


def test_srm_flags_broken_split():
    ok = inf.srm_check([50000, 50000])
    bad = inf.srm_check([50000, 47000])
    assert ok["passed"] and not bad["passed"]


def test_cuped_reduces_variance():
    rng = np.random.default_rng(2)
    pre = rng.lognormal(3.5, 0.8, 8000)
    post = 0.8 * pre + rng.normal(50, 30, 8000)
    result = inf.cuped_adjust(post, pre)
    assert result["variance_reduction"] > 0.5
    assert 0.7 < result["theta"] < 0.9  # true coefficient is 0.8


def test_bh_correction_monotone():
    adjusted = inf.bh_correction([0.001, 0.01, 0.04, 0.2])
    assert [a["p_adjusted"] for a in adjusted] == sorted(a["p_adjusted"] for a in adjusted)
    assert adjusted[0]["significant"] is True


def test_design_document_contains_the_essentials():
    design = ExperimentDesign(
        goal="checkout conversion",
        metric_kind="proportion",
        baseline=0.03,
        mde_rel=0.05,
        daily_traffic=60_000,
        lang="zh",
    )
    result = design_experiment(design)
    assert result["sizing"]["n_total"] > 0
    assert result["duration"]["recommended_days"] % 7 == 0
    assert any("SRM" in p for p in result["analysis_plan"])
    assert result["decision_rule"]
    assert "实验设计" in result["markdown"]


def test_recommend_experiment_type_matches_context():
    assert recommend_experiment_type("双边市场的动态定价，骑手和用户耦合")["primary"] == "switchback"
    assert recommend_experiment_type("城市级别的地推投放，无法做用户分流")["primary"] in ("cluster", "quasi")
    assert recommend_experiment_type("一个普通的小改动，用户之间独立")["primary"] in ("ab", "cuped")


def test_catalogue_is_complete():
    for key, spec in EXPERIMENT_TYPES.items():
        for field in ("name_en", "name_zh", "when", "when_zh", "randomization_unit", "analysis", "pitfalls"):
            assert spec.get(field), f"{key} missing {field}"


def test_analyze_recovers_the_planted_lift():
    df = build_experiment_users(n_per_arm=40_000, true_lift=0.08, seed=5)
    metrics = [{"name": "conversion", "kind": "proportion", "numerator": "converted"}]
    result = analyze_experiment(df, metrics, covariate_col="pre_revenue")
    lift = result["metrics"][0]["rel_lift"]
    assert 0.04 < lift < 0.12
    assert result["srm"]["passed"]
    assert result["recommendation"]


def test_analysis_detects_srm_when_traffic_is_lopsided():
    df = build_experiment_users(n_per_arm=20000, seed=9)
    broken = df.iloc[:28000]  # drop ~20% of treatment users
    metrics = [{"name": "conversion", "kind": "proportion", "numerator": "converted"}]
    result = analyze_experiment(broken, metrics, covariate_col=None)
    assert not result["srm"]["passed"]


def test_ratio_metric_uses_delta_method_not_a_binomial_guess():
    """Regression: a ratio above 1 used to produce a ~1e-300 p-value."""
    import numpy as np

    from ds_agent.stats.inference import ratio_metric_test, ratio_metric_test_from_samples

    rng = np.random.default_rng(0)
    n = 20_000
    visits = rng.poisson(3, n).astype(float)
    revenue = rng.gamma(2.0, 6.0, n) * (visits > 0)

    same = ratio_metric_test_from_samples(revenue, visits, revenue, visits)
    assert np.isfinite(same["se"]) and same["se"] > 0
    assert same["p_value"] == pytest.approx(1.0, abs=0.05)

    shifted = ratio_metric_test_from_samples(revenue, visits, revenue * 1.05, visits)
    assert 0 < shifted["p_value"] < 0.05
    assert shifted["rel_lift"] == pytest.approx(0.05, rel=0.2)

    # The aggregate helper must refuse to guess when only partial moments exist.
    partial = ratio_metric_test(100.0, 10.0, 110.0, 10.0)
    assert partial["p_value"] != partial["p_value"]  # NaN, not a fake certainty
