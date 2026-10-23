"""Tests for the pre-modelling pipeline and the sampling toolkit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ds_agent.stats.sampling import (
    draw_sample,
    representativeness,
    sample_size_mean,
    sample_size_proportion,
    standard_error,
)
from ds_agent.tools.preprocess import (
    correlation_pairs,
    drop_collinear,
    missing_profile,
    preprocess_table,
    prepare_features,
)


# ---------------------------------------------------------------- sampling math
def test_sample_size_proportion_matches_closed_form():
    # Baseline 3%, detect 1pp: (1.96+1.2816)^2 * .03*.97 / .01^2 = 3057.5...
    out = sample_size_proportion(0.03, 0.01)
    expected = np.ceil(((1.959964 + 1.281552) ** 2) * 0.03 * 0.97 / 0.0001)
    assert out["n_per_group"] == expected
    assert out["n_total"] == 2 * out["n_per_group"]


def test_sample_size_mean_halving_mde_quadruples_n():
    base = sample_size_mean(50, 5)
    tight = sample_size_mean(50, 2.5)
    assert tight["n_per_group"] == pytest.approx(4 * base["n_per_group"], rel=0.02)


def test_sample_size_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        sample_size_mean(0, 5)
    with pytest.raises(ValueError):
        sample_size_proportion(1.5, 0.01)


def test_draw_sample_sizes_and_reproducibility():
    rng = np.random.default_rng(0)
    pop = pd.DataFrame(
        {
            "channel": rng.choice(["a", "b", "c"], 900, p=[0.6, 0.3, 0.1]),
            "x": rng.normal(size=900),
        }
    )
    s1, log1 = draw_sample(pop, 300, method="srs", seed=7)
    s2, _ = draw_sample(pop, 300, method="srs", seed=7)
    assert len(s1) == 300
    assert s1.index.tolist() == s2.index.tolist()  # same seed -> same draw
    sys_s, slog = draw_sample(pop, 300, method="systematic", seed=3)
    assert len(sys_s) == 300 and slog["k"] == 3
    strat, stlog = draw_sample(pop, 300, method="stratified", by="channel", seed=5)
    assert len(strat) == 300
    # every stratum keeps proportional representation (within rounding)
    for stratum, alloc in stlog["allocation"].items():
        share = alloc["sampled"] / 300
        assert share == pytest.approx(alloc["population"] / 900, abs=0.01)


def test_standard_error_interval_covers_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(100, 15, 500)
    out = standard_error(x)
    assert out["ci95"][0] < out["mean"] < out["ci95"][1]
    assert out["se"] == pytest.approx(x.std(ddof=1) / np.sqrt(500), rel=1e-3)


def test_representativeness_passes_on_true_sample_and_flags_bias():
    rng = np.random.default_rng(2)
    pop = pd.DataFrame(
        {
            "channel": rng.choice(["a", "b"], 5000, p=[0.7, 0.3]),
            "gmv": rng.normal(100, 20, 5000),
        }
    )
    fair, _ = draw_sample(pop, 500, method="srs", seed=1)
    ok_cat = representativeness(fair["channel"], pop["channel"])
    assert ok_cat["tests"][0]["p_value"] > 0.05
    ok_num = representativeness(fair["gmv"], pop["gmv"])
    assert all(t["p_value"] > 0.01 for t in ok_num["tests"])

    biased = pop[pop["channel"] == "a"].sample(500, random_state=2)
    bad = representativeness(biased["channel"], pop["channel"])
    assert bad["tests"][0]["decision"] == "reject"  # 100% channel-a is not representative


# ----------------------------------------------------------------- preprocessing
def test_missing_profile_counts_and_sorts():
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0], "b": ["x", "y", "z", "w"]})
    prof = missing_profile(df)
    assert prof["columns"][0]["column"] == "a"
    assert prof["columns"][0]["pct_missing"] == 50.0
    assert prof["missing_cells"] == 2


def test_preprocess_imputes_drops_and_logs():
    df = pd.DataFrame(
        {
            "num": [1.0, np.nan, 3.0, 4.0, 5.0],
            "cat": ["a", None, "b", "a", "b"],
            "sparse": [np.nan] * 4 + [1.0],
            "const": [7.0] * 5,
        }
    )
    clean, log = preprocess_table(df, numeric_strategy="median", categorical_strategy="missing")
    assert "sparse" in [d["column"] for d in log["dropped"]]
    assert "const" in [d["column"] for d in log["dropped"]]
    assert clean["num"].isna().sum() == 0
    assert clean["num"].iloc[1] == 3.5  # median of the observed values
    assert "missing" in set(clean["cat"])  # NaN became its own category
    assert log["imputed"]["num"]["strategy"] == "median"


def test_preprocess_strategies_are_respected():
    df = pd.DataFrame({"num": [1.0, np.nan, 3.0]})
    meaned, log = preprocess_table(df, numeric_strategy="mean")
    assert meaned["num"].iloc[1] == 2.0 and log["imputed"]["num"]["strategy"] == "mean"
    dropped, log2 = preprocess_table(df, numeric_strategy="drop")
    assert "num" not in dropped.columns
    with pytest.raises(ValueError):
        preprocess_table(df, numeric_strategy="nope")


def test_drop_collinear_pairwise_and_vif():
    rng = np.random.default_rng(0)
    n = 800
    x1 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "x1": x1,
            "x2": x1 * 2 + rng.normal(0, 0.05, n),  # r~1 with x1
            "x3": x3,
            "x4": 0.5 * x1 + 0.5 * x3 + rng.normal(0, 0.05, n),  # multi-way VIF
        }
    )
    pruned, log = drop_collinear(frame)
    assert list(pruned.columns) == ["x1", "x3"]
    reasons = {d["column"]: d["reason"] for d in log["dropped"]}
    assert "r=" in reasons["x2"] and "VIF=" in reasons["x4"]
    assert max(log["vif_after"].values()) < 10


def test_correlation_pairs_threshold():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    frame["c"] = frame["a"] * 3 + rng.normal(0, 0.01, 300)
    pairs = correlation_pairs(frame, threshold=0.9)
    assert {(p["a"], p["b"]) for p in pairs} == {("a", "c")}


def test_prepare_features_end_to_end():
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "cat": rng.choice(["x", "y"], n),
            "sparse": np.where(rng.random(n) < 0.9, np.nan, 1.0),
            "const": ["k"] * n,
        }
    )
    df["num1_dup"] = df["num1"] * 2 + rng.normal(0, 0.01, n)
    df["y"] = rng.integers(0, 2, n)
    X, y, log = prepare_features(df, target="y")
    assert "sparse" not in X.columns and "const" not in X.columns and "num1_dup" not in X.columns
    assert "num1" in X.columns and any(c.startswith("cat_") for c in X.columns)
    assert y is not None and len(y) == n
    assert log["n_features"] == X.shape[1]


# ------------------------------------------------------- agent integration
def test_agent_sampling_size_mode():
    from ds_agent.agent import DSQueryAgent

    agent = DSQueryAgent()
    reply = agent.ask("满意度调查基线 70%，检测 3 个百分点变化，需要多少样本量？")
    assert reply.data.get("intent") == "sampling"
    assert "2,452" in reply.text or "2452" in reply.text
    assert "Cochran" in reply.text
    kinds = [b.get("type") for b in reply.blocks]
    assert "kpi" in kinds and "table" in kinds


def test_agent_sampling_size_mode_without_liang_suffix():
    """「需要多少样本？」(no 量) must also route to sampling and parse 70%/3pp.

    Regression: the router used to require 样本量/样本大小 verbatim, so the
    sidebar example「满意度调查基线70%，检测3个百分点变化，需要多少样本？」
    fell through to another handler and was never answered.
    """
    from ds_agent.agent import DSQueryAgent

    agent = DSQueryAgent()
    for question in (
        "满意度调查基线70%，检测3个百分点变化，需要多少样本？",
        "How many samples do I need for a satisfaction survey with a 70% baseline to detect a 3pp change?",
    ):
        reply = agent.ask(question)
        assert reply.data.get("intent") == "sampling", question
        assert "2,452" in reply.text or "2452" in reply.text, question
        assert "70.0%" in reply.text, question
        assert "3.00" in reply.text, question


def test_agent_sampling_size_mean_mode():
    """Questions mentioning a standard deviation must size in mean mode.

    Regression: 「人均消费提升5元（标准差80）」 used to be parsed as a
    proportion question (baseline 0%, mde 0.05), returning a nonsense
    26.9M sample size. The MDE must stay in original units (5, not 0.05).
    """
    from ds_agent.agent import DSQueryAgent

    agent = DSQueryAgent()
    reply = agent.ask("要检验人均消费提升5元（标准差80），每组需要多少样本？")
    assert reply.data.get("intent") == "sampling"
    assert "sd σ=80" in reply.text
    assert "δ=5 (" in reply.text  # NOT 0.05 — original units preserved
    assert "2,690" in reply.text
    assert "26,899,004" not in reply.text


def test_agent_sampling_draw_and_representativeness():
    from ds_agent.agent import DSQueryAgent

    agent = DSQueryAgent()
    reply = agent.ask("从用户表按渠道分层抽样 2000 个，检验样本代表性")
    assert reply.data.get("intent") == "sampling"
    text = reply.text
    assert "chi_square_gof" in reply.data.get("tests", []) or "卡方" in text
    kinds = [b.get("type") for b in reply.blocks]
    assert kinds.count("table") >= 2  # allocation + test table (+ share table)


def test_agent_modeling_reports_preprocess_and_collinearity_stages():
    from ds_agent.agent import DSQueryAgent

    agent = DSQueryAgent()
    reply = agent.ask("训练分类模型预测用户是否转化，看特征重要性")
    titles = [str(b.get("title", "")) for b in reply.blocks if b.get("type") == "table"]
    assert any("①" in t or "preprocessing" in t.lower() for t in titles)
    assert any("②" in t or "collinearity" in t.lower() for t in titles)
    # SHAP/permutation analysis is explicitly positioned after the model.
    chart_titles = [str(b.get("title", "")) for b in reply.blocks if b.get("type") == "chart"]
    assert any("之后" in t or "Post-model" in t or "post" in t.lower() for t in chart_titles)


def test_psm_reuses_preprocessing_for_collinear_covariates():
    from ds_agent.tools.causal import propensity_score_matching

    rng = np.random.default_rng(4)
    n = 1200
    spend = rng.lognormal(3, 0.5, n)
    df = pd.DataFrame(
        {
            "treat": (rng.random(n) < 0.4).astype(int),
            "outcome": rng.normal(10, 2, n),
            "x1": spend,
            "x2": spend * 2 + rng.normal(0, 0.01, n),  # r~1 with x1
        }
    )
    out = propensity_score_matching(df, "treat", "outcome", ["x1", "x2"], n_bootstrap=40)
    assert out["pruned_covariates"] == ["x2"]
    assert out["covariates_used"] == ["x1"]
    assert [b["covariate"] for b in out["balance"]] == ["x1"]
