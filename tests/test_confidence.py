"""Confidence validation: check builders, verdicts and chart inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ds_agent.confidence import (
    PASS,
    WARN,
    anomaly_band,
    anomaly_confidence,
    confidence_table,
    did_confidence,
    did_placebo,
    experiment_confidence,
    matching_confidence,
    permutation_test,
    ts_causal_confidence,
    verdict,
    verdict_line,
)
from ds_agent import charts


# ---------------------------------------------------------------- anomaly
def test_anomaly_confidence_all_green():
    values = np.array([100.0, 98, 60, 62, 58, 61, 63])
    expected = np.full(7, 99.0)
    res = anomaly_confidence(
        sig={"p_value": 0.003},
        window_values=values,
        window_expected=expected,
        window_z=None,
        chosen_method="holt_winters",
        mase=0.85,
        episodes=[{"days": 11}],
        scan_top={"window_change": -0.30, "segment": "iOS"},
        total_delta=-40.0,
        sigma_rel=0.03,
    )
    assert res["verdict"]["label_zh"] == "high confidence"
    assert res["verdict"]["n_fail"] == 0
    names = [c["name_zh"] for c in res["checks"]]
    assert "window significance" in names and "outlier dependence (leave-one-out)" in names


def test_anomaly_confidence_flags_insignificant_window():
    res = anomaly_confidence(
        sig={"p_value": 0.4},
        window_values=np.array([100.0, 99, 101, 100, 99]),
        window_expected=np.array([100.0, 100, 100, 100, 100]),
        window_z=None,
        chosen_method=None,
        mase=None,
        episodes=[],
        scan_top=None,
        total_delta=1.0,
        sigma_rel=0.02,
    )
    assert res["verdict"]["label_en"] == "low confidence"
    assert any(c["status"] == "FAIL" for c in res["checks"])


def test_anomaly_band_frame_and_chart():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=6),
            "actual": [100, 101, 60, 102, 99, 30],
            "expected": [100] * 6,
            "flagged": [False, False, True, False, False, True],
        }
    )
    band = anomaly_band(frame, sigma_rel=0.03)
    assert (band["band_high"] >= band["band_low"]).all()
    uri = charts.confidence_band_chart(band)
    assert uri.startswith("data:image/png;base64,")


# ------------------------------------------------------------ permutation
def test_permutation_test_detects_real_effect():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0, 1, 400), rng.normal(0.6, 1, 400)])
    groups = np.array([0] * 400 + [1] * 400)
    perm = permutation_test(values, groups, n_iter=300)
    assert perm["p_perm"] < 0.01
    assert len(perm["null"]) == 300
    uri = charts.permutation_chart(perm["null"], perm["observed"])
    assert uri.startswith("data:image/png;base64,")


def test_permutation_test_null_stays_null():
    rng = np.random.default_rng(1)
    values = rng.normal(0, 1, 400)
    groups = np.array([0] * 200 + [1] * 200)
    perm = permutation_test(values, groups, n_iter=300)
    assert perm["p_perm"] > 0.05


def test_experiment_confidence_card():
    srm = {"passed": True, "p_value": 0.6, "message": "ok"}
    primary = {
        "significant_adjusted": True,
        "p_value": 0.002,
        "p_adjusted": 0.008,
        "ci_abs": (0.01, 0.03),
        "rel_lift": 0.06,
    }
    perm = {"p_perm": 0.004}
    cuped = {"rel_lift": 0.058, "variance_reduction": 0.41}
    res = experiment_confidence(srm, primary, cuped, {"metric": "revenue_per_user"}, [], perm)
    assert res["verdict"]["label_zh"] == "high confidence"
    # Ratio metrics get a WARN note instead of a fake permutation check.
    res2 = experiment_confidence(srm, primary, None, None, [], None)
    assert any("permutation" in c["name_zh"] and c["status"] == WARN for c in res2["checks"])


# ---------------------------------------------------------------- DiD
@pytest.fixture(scope="module")
def did_panel():
    # Additive growth with a constant absolute weekly increment, so pre-trends
    # are parallel across cities (the multiplicative variant would — rightly —
    # fail the parallel-trends check the card performs).
    rng = np.random.default_rng(10)
    rows = []
    for city in range(10):
        treated = city < 4
        base = 100 + city * 5
        for week in range(1, 41):
            effect = 15 if (treated and week >= 26) else 0.0
            rows.append(
                {
                    "city": f"c{city:02d}",
                    "week": week,
                    "treated": int(treated),
                    "post": int(week >= 26),
                    "gmv": base + 0.5 * week + effect + rng.normal(0, 1.0),
                }
            )
    return pd.DataFrame(rows)


def test_did_placebo_and_confidence(did_panel):
    from ds_agent.tools.causal import difference_in_differences, event_study

    did = difference_in_differences(did_panel, "gmv", "city", "week", "treated", "post")
    did_log = difference_in_differences(did_panel, "gmv", "city", "week", "treated", "post", log_outcome=True)
    es = event_study(did_panel, "gmv", "city", "week", "treated", treat_time=26, max_lag=4, max_lead=4)
    placebo = did_placebo(did_panel, "gmv", "city", "week", n_iter=30)
    assert placebo["n"] >= 20
    res = did_confidence(did, did_log, es, placebo, n_treated_units=4)
    assert res["verdict"]["n_fail"] == 0
    assert res["verdict"]["label_en"] == "high confidence"
    uri = charts.placebo_chart(placebo["effects"], did["att"])
    assert uri.startswith("data:image/png;base64,")


# -------------------------------------------------------------- matching
def test_matching_confidence_assembles_checks():
    psm = {
        "att": 11.8,
        "p_value": 0.001,
        "balance": [{"covariate": "x", "smd_after": 0.04}],
        "common_support": {"share_outside": 0.01},
    }
    aipw = {"estimate": 12.2}
    res = matching_confidence(psm, {}, aipw)
    names = [c["name_zh"] for c in res["checks"]]
    assert "post-match covariate balance" in names and "hidden-confounding caveat" in names
    assert res["verdict"]["n_fail"] == 0


# ------------------------------------------------------------- ts causal
def test_ts_causal_confidence():
    its = {"ljung_box_p": 0.4, "placebo_p": 0.2, "relative": 0.12, "spec_table": pd.DataFrame({"relative": [0.12, 0.11], "p_value": [1e-6, 1e-5]})}
    impact = {"placebo_p": 0.3, "relative": 0.128, "pre_fit_r2": 0.95}
    res = ts_causal_confidence(its, impact, agree=True)
    assert res["verdict"]["n_fail"] == 0
    res2 = ts_causal_confidence({**its, "ljung_box_p": 0.01}, impact, agree=False)
    assert res2["verdict"]["label_en"] in ("medium confidence", "low confidence")


# -------------------------------------------------------------- verdicts
def test_verdict_and_table_helpers():
    checks = [
        {"name_zh": "a", "name_en": "a", "status": PASS, "detail_zh": "", "detail_en": ""},
        {"name_zh": "b", "name_en": "b", "status": PASS, "detail_zh": "", "detail_en": ""},
        {"name_zh": "c", "name_en": "c", "status": "FAIL", "detail_zh": "", "detail_en": ""},
    ]
    v = verdict(checks)
    assert v["n_fail"] == 1 and v["label_en"] == "low confidence"
    df = confidence_table(checks, "en", v)
    assert list(df.columns) == ["check", "result", "detail"]
    assert df.iloc[-1]["result"] == "low confidence"
    assert "Confidence validation" in verdict_line(v, "en")
