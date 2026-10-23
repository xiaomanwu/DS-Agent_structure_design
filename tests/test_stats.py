"""Tests for the statistical test battery, its panels, and the docs contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ds_agent.stats import panels, tests as T


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #
def test_catalogue_is_complete_and_unique():
    keys = [c["key"] for c in T.catalogue()]
    assert len(keys) == len(set(keys))
    required = {
        "key", "name_zh", "name_en", "family_zh", "family_en", "h0_zh", "h0_en",
        "stat_name", "formula", "assume_zh", "assume_en", "use_zh", "use_en",
        "reject_zh", "reject_en", "retain_zh", "retain_en", "ref", "short_ref",
    }
    for entry in T.catalogue():
        missing = required - set(entry)
        assert not missing, f"{entry['key']} missing {missing}"
        assert entry["ref"] and entry["formula"], f"{entry['key']} has no theory attached"


def test_catalogue_covers_every_family():
    families = {c["family_zh"] for c in T.catalogue()}
    for expected in ("正态性", "方差", "位置", "比例", "时序", "回归", "重抽样", "多重检验"):
        assert expected in families


def test_find_by_text_matches_chinese_and_english():
    assert T.find_by_text("什么是 Welch t 检验")["key"] == "welch_t"
    assert T.find_by_text("run a mann-whitney test")["key"] == "mann_whitney"
    assert T.find_by_text("How do you check autocorrelation with Ljung-Box")["key"] == "ljung_box"
    assert T.find_by_text("随机森林") is None


# --------------------------------------------------------------------------- #
# Individual tests
# --------------------------------------------------------------------------- #
def test_normality_tests_separate_normal_from_skewed():
    rng = np.random.default_rng(3)
    normal = rng.normal(size=300)
    skewed = rng.exponential(size=300)
    assert T.shapiro_wilk(normal)["decision"] == "retain"
    assert T.shapiro_wilk(skewed)["decision"] == "reject"
    # Above 5,000 the primary test switches to D'Agostino and Shapiro reports why.
    big = rng.normal(size=6000)
    assert T.shapiro_wilk(big)["status"] == "skipped"
    assert T.dagostino_k2(big)["status"] == "ok"


def test_welch_and_student_differ_when_variances_do():
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 60)
    b = rng.normal(0.6, 6, 30)
    welch = T.welch_t(a, b)
    student = T.student_t(a, b)
    assert welch["status"] == student["status"] == "ok"
    assert welch["df"] != student["df"]
    assert T.levene(a, b)["decision"] == "reject"


def test_levene_flags_zero_median_degeneracy():
    x = np.r_[np.zeros(80), np.random.default_rng(1).exponential(1.0, 20)]
    y = np.r_[np.zeros(70), np.random.default_rng(2).exponential(2.0, 30)]
    res = T.levene(x, y)
    assert "median is 0" in res["note_en"] or "退化为位置比较" in res["note_zh"]


def test_proportion_and_contingency_tests():
    z = T.two_proportion_z(100, 1000, 130, 1000)
    assert z["decision"] == "reject"
    assert z["extra"]["rel_lift"] > 0
    table = [[100, 900], [130, 870]]
    chi = T.chi_square_independence(table)
    assert chi["decision"] == "reject"
    assert chi["extra"]["cramers_v"] > 0
    # Fisher on a sparse table: exact p, odds ratio as the statistic.
    fisher = T.fisher_exact([[1, 9], [8, 2]])
    assert fisher["status"] == "ok"
    assert 0 <= fisher["p_value"] <= 1


def test_srm_gof_and_mcnemar():
    assert T.chi_square_gof([5000, 5000])["decision"] == "retain"
    assert T.chi_square_gof([5400, 4600])["decision"] == "reject"
    assert T.mcnemar(10, 40)["decision"] == "reject"
    assert T.mcnemar(0, 0)["status"] == "skipped"


def test_time_series_tests():
    rng = np.random.default_rng(11)
    white = rng.normal(size=120)
    ar1 = np.zeros(120)
    for i in range(1, 120):
        ar1[i] = 0.9 * ar1[i - 1] + rng.normal(0, 0.3)
    assert T.ljung_box(white)["decision"] == "retain"
    assert T.ljung_box(ar1)["decision"] == "reject"
    assert 0 <= T.durbin_watson(white)["statistic"] <= 4
    # A random walk is non-stationary: ADF should not reject, KPSS should.
    walk = np.cumsum(rng.normal(size=200))
    assert T.adf(walk)["decision"] == "retain"
    assert T.kpss(walk)["decision"] == "reject"


def test_regression_diagnostics():
    rng = np.random.default_rng(5)
    n = 300
    x = rng.uniform(0, 3, n)
    # Variance linear in x: exactly what Breusch-Pagan is designed to detect.
    y = 1 + 2 * x + rng.normal(0, 0.2 + x, n)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    assert T.breusch_pagan(resid, X)["decision"] == "reject"
    collinear = np.column_stack([x, x * 2 + rng.normal(0, 0.01, n)])
    assert T.variance_inflation(collinear, ["a", "b"])["statistic"] > 10
    independent = np.column_stack([x, rng.normal(size=n)])
    assert T.variance_inflation(independent, ["a", "b"])["statistic"] < 5


def test_hosmer_lemeshow_detects_miscalibration():
    rng = np.random.default_rng(9)
    n = 2000
    p = rng.uniform(0.05, 0.95, n)
    y_cal = rng.binomial(1, p)
    y_bad = rng.binomial(1, np.clip(p + 0.25, 0, 1))
    assert T.hosmer_lemeshow(y_cal, p)["decision"] == "retain"
    assert T.hosmer_lemeshow(y_bad, p)["decision"] == "reject"


def test_resampling_and_equivalence():
    rng = np.random.default_rng(13)
    a = rng.normal(10, 1, 200)
    b = rng.normal(10.4, 1, 200)
    perm = T.permutation_test(np.r_[a, b], np.r_[np.zeros(200), np.ones(200)], n_iter=400)
    assert perm["decision"] == "reject"
    assert len(perm["extra"]["null"]) == 400
    boot = T.bootstrap_ci(a, b, n_boot=400)
    low, high = boot["extra"]["ci"]
    assert low < 0.4 < high
    # An effect well inside the margin is equivalent; a wide margin is not.
    assert T.tost(a, b, margin=2.0)["decision"] == "reject"
    assert T.tost(a, b, margin=0.05)["decision"] == "retain"


@pytest.mark.parametrize("fn,args", [
    (T.shapiro_wilk, ([],)),
    (T.welch_t, ([1.0], [2.0])),
    (T.paired_t, ([1.0, 2.0], [1.0, 2.0])),
    (T.ljung_box, ([1.0, 2.0],)),
    (T.two_proportion_z, (0, 0, 0, 0)),
    (T.hosmer_lemeshow, ([0, 1], [0.5, 0.5])),
])
def test_degenerate_input_returns_skipped_never_raises(fn, args):
    out = fn(*args)
    assert out["status"] == "skipped"
    assert out["decision"] == "skipped"
    assert out["conclusion_zh"].startswith("未执行")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_frame_and_summary_carry_the_reasoning():
    rng = np.random.default_rng(17)
    a, b = rng.normal(size=100), rng.normal(0.5, size=100)
    battery = [T.welch_t(a, b), T.mann_whitney(a, b), T.levene(a, b)]
    frame = T.to_frame(battery, "zh")
    assert list(frame.columns) == ["检验", "统计量", "p", "结论", "参考"]
    assert len(frame) == 3
    line_zh = T.summary_line(battery, "zh")
    assert "Welch" in line_zh and "p=" in line_zh
    line_en = T.summary_line(battery, "en")
    assert "tests accompany this conclusion" in line_en


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def test_anomaly_panel_finds_a_real_incident():
    rng = np.random.default_rng(19)
    expected = 1000 + 40 * rng.normal(size=14)
    values = expected.copy()
    values[-7:] *= 0.85  # a 15% drop for a week
    panel = panels.anomaly_panel(values, expected, series=np.r_[expected * 1.02, values])
    assert panel["agreement"] is True
    assert panel["tests"][2]["decision"] == "reject"  # paired t against the baseline
    assert "检验" in panel["summary_zh"]


def test_experiment_panel_proportion_and_mean():
    rng = np.random.default_rng(23)
    ctrl_bin = rng.binomial(1, 0.10, 4000)
    trt_bin = rng.binomial(1, 0.12, 4000)
    prop = panels.experiment_panel(ctrl_bin, trt_bin, kind="proportion")
    keys = [t["key"] for t in prop["tests"]]
    assert "two_proportion_z" in keys and "chi_square_independence" in keys
    assert prop["tests"][0]["decision"] == "reject"

    ctrl = rng.normal(10, 2, 800)
    trt = rng.normal(11, 2, 800)
    mean = panels.experiment_panel(ctrl, trt)
    decisions = {t["decision"] for t in mean["tests"] if t["status"] == "ok" and t["key"] in ("welch_t", "mann_whitney", "permutation")}
    assert decisions == {"reject"}
    assert mean["agreement"] is True


def test_model_panel_regression_and_classification(agent):
    from ds_agent.tools.modeling import engineer_features, train_model

    df = agent.sql.run("SELECT * FROM uplift_users LIMIT 4000").dataframe
    X, y, _, _ = engineer_features(df, target="converted", drop=["user_id", "group", "true_tau", "revenue"])
    result = train_model(X, y, task="classification")
    panel = panels.model_panel(result)
    assert any(t["key"] == "hosmer_lemeshow" for t in panel["tests"])
    assert any(t["key"] == "mcnemar" for t in panel["tests"])

    daily = agent.sql.run("SELECT * FROM orders_daily").dataframe
    Xr, yr, _, _ = engineer_features(daily, target="gmv", date_column="date", drop=["buyers", "orders"])
    reg = train_model(Xr, yr, task="regression")
    rpanel = panels.model_panel(reg)
    keys = {t["key"] for t in rpanel["tests"]}
    assert {"jarque_bera", "breusch_pagan", "variance_inflation"} <= keys


def test_psm_panel_uses_the_estimator_tests():
    psm = {
        "n_matched_pairs": 1200,
        "att_test": {"t": 4.2, "p_value": 0.0001},
        "balance": [
            {"covariate": "tenure", "smd_after": 0.02, "t_after": 0.3, "p_after": 0.76},
            {"covariate": "spend", "smd_after": 0.06, "t_after": 0.9, "p_after": 0.37},
        ],
    }
    panel = panels.psm_panel(psm)
    assert [t["key"] for t in panel["tests"]] == ["paired_t", "welch_t"]
    assert "ATT" in panel["tests"][0]["note_zh"] and "spend" in panel["tests"][1]["note_zh"]


# --------------------------------------------------------------------------- #
# Agent integration
# --------------------------------------------------------------------------- #
def test_router_sends_test_questions_to_the_tests_capability(agent):
    assert agent.ask("用了哪些统计检验？").data["intent"] == "tests"
    assert agent.ask("什么是 Welch t 检验").data["intent"] == "tests"
    assert agent.ask("异常检测用了哪些检验").data["intent"] == "tests"
    # A normal experiment question must NOT be captured by the meta-question rule.
    assert agent.ask("看下这个 AB 实验的结果").data["intent"] == "experiment_analysis"


def test_tests_capability_explains_any_catalogue_entry(agent):
    reply = agent.ask("什么是 Mann-Whitney U 检验")
    assert "原假设" in reply.text
    assert "Mann" in reply.text and "1947" in reply.text
    assert reply.blocks and reply.blocks[0]["type"] == "table"


def test_diagnose_answer_ships_the_test_table(agent):
    reply = agent.ask("最近7天 GMV 为什么跌了？")
    titles = [b.get("title", "") for b in reply.blocks]
    assert any("统计检验" in t for t in titles)
    table = next(b for b in reply.blocks if "统计检验" in b.get("title", ""))
    assert "结论" in table["columns"]
    assert "**统计检验**" in reply.text


def test_experiment_answer_ships_proportion_and_mean_tests(agent):
    reply = agent.ask("看下这个 AB 实验的结果")
    table = next(b for b in reply.blocks if "统计检验" in b.get("title", ""))
    names = [row[0] for row in table["rows"]]
    assert any("两比例" in n for n in names)
    assert any("Welch" in n for n in names)


# --------------------------------------------------------------------------- #
# Documentation contract
# --------------------------------------------------------------------------- #
def test_generated_test_reference_matches_the_code():
    doc = Path(__file__).resolve().parent.parent / "docs" / "statistical_tests.md"
    assert doc.exists(), "run scripts/make_test_reference.py"
    text = doc.read_text(encoding="utf-8")
    for entry in T.catalogue():
        assert entry["name_zh"] in text, f"{entry['key']} missing from the reference doc"
        assert entry["short_ref"] in text, f"{entry['key']} citation missing from the reference doc"
    for spec in panels.PANEL_SPEC.values():
        assert spec["name_zh"] in text, f"panel {spec['name_zh']} missing from the reference doc"
