"""Causal estimators must recover the effects planted in the demo data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ds_agent.data.synthetic import build_city_panel, build_promo_users
from ds_agent.tools.causal import (
    aipw_effect,
    compare_estimators,
    difference_in_differences,
    event_study,
    ipw_effect,
    propensity_score_matching,
    synthetic_control,
)


@pytest.fixture(scope="module")
def panel():
    return build_city_panel()


def test_did_recovers_the_relative_effect(panel):
    level = difference_in_differences(panel, "gmv", "city", "week", "treated", "post")
    log = difference_in_differences(panel, "gmv", "city", "week", "treated", "post", log_outcome=True)
    assert level["att"] > 0
    assert level["p_value"] < 0.05
    # The generator applies a 9% lift; the log specification identifies it.
    assert log["relative_effect"] == pytest.approx(0.09, abs=0.03)


def test_event_study_flat_pre_trends(panel):
    result = event_study(panel, "gmv", "city", "week", "treated", treat_time=26, max_lag=6, max_lead=6)
    assert result["parallel_trends_ok"], result["narrative"]
    post = [c for c in result["coefficients"] if c["rel_period"] >= 0]
    assert all(c["coef"] > 0 for c in post)


def test_psm_and_aipw_recover_the_promo_effect():
    df = build_promo_users(n=30_000)
    df["log_pre_revenue"] = np.log1p(df["pre_revenue"])
    covariates = ["log_pre_revenue", "tenure_months", "sessions_30d"]

    naive = df.loc[df.promo == 1, "post_revenue"].mean() - df.loc[df.promo == 0, "post_revenue"].mean()
    psm = propensity_score_matching(df, "promo", "post_revenue", covariates, n_bootstrap=80)
    aipw = aipw_effect(df, "promo", "post_revenue", covariates)

    # True effect is 12 CNY; the naive difference is badly confounded.
    assert naive > 25
    assert psm["att"] == pytest.approx(12, abs=4)
    assert aipw["estimate"] == pytest.approx(12, abs=4)
    assert psm["n_matched_pairs"] > 1000


def test_psm_balance_improves():
    df = build_promo_users(n=20_000)
    df["log_pre_revenue"] = np.log1p(df["pre_revenue"])
    result = propensity_score_matching(
        df, "promo", "post_revenue", ["log_pre_revenue", "tenure_months", "sessions_30d"], n_bootstrap=40
    )
    for row in result["balance"]:
        assert abs(row["smd_after"]) <= abs(row["smd_before"]) + 1e-9


def test_psm_common_support_and_overlap_payload():
    df = build_promo_users(n=15_000)
    df["log_pre_revenue"] = np.log1p(df["pre_revenue"])
    result = propensity_score_matching(
        df, "promo", "post_revenue", ["log_pre_revenue", "tenure_months", "sessions_30d"], n_bootstrap=40
    )
    support = result["common_support"]
    assert 0.0 <= support["low"] < support["high"] <= 1.0
    assert support["n_treated_outside"] >= 0
    assert support["share_outside"] == pytest.approx(
        support["n_treated_outside"] / max(result["n_treated"], 1)
    )
    overlap = result["ps_overlap"]
    edges = np.asarray(overlap["edges"])
    assert len(edges) >= 20
    assert np.all(np.diff(edges) > 0)
    # Histograms share the bin grid and together count every unit.
    assert len(overlap["treated"]) == len(overlap["control"]) == len(edges) - 1
    assert sum(overlap["treated"]) == result["n_treated"]
    assert sum(overlap["control"]) > 0
    # The narrative reports the support interval.
    assert "Common support" in result["narrative"]


def test_psm_extreme_confounding_breaks_support():
    # One covariate perfectly separates the arms: propensity is ~0/1 and the
    # common support collapses — matching must fail loudly, not guess.
    rng = np.random.default_rng(7)
    n = 2_000
    df = pd.DataFrame(
        {
            "promo": (np.arange(n) % 2),  # deterministic from the covariate below
            "post_revenue": rng.normal(100, 10, n),
            "x": np.where(np.arange(n) % 2 == 1, 10.0, -10.0),
        }
    )
    with pytest.raises(ValueError, match="caliper|matched"):
        propensity_score_matching(df, "promo", "post_revenue", ["x"], caliper=0.01, n_bootstrap=10)


def test_ipw_trimming_keeps_effective_sample():
    df = build_promo_users(n=20_000)
    df["log_pre_revenue"] = np.log1p(df["pre_revenue"])
    result = ipw_effect(df, "promo", "post_revenue", ["log_pre_revenue", "tenure_months"], n_bootstrap=40)
    assert 0 < result["effective_sample_size"] <= len(df)


def test_synthetic_control_finds_the_policy(panel):
    donors = sorted(panel.loc[panel.treated == 0, "city"].unique().tolist())
    result = synthetic_control(panel, "city", "week", "gmv", "city_00", 26, donors=donors)
    assert result["relative_effect"] == pytest.approx(0.09, abs=0.05)
    assert result["placebo_p_value"] <= 0.10
    assert abs(sum(result["weights"].values()) - 1) < 0.05


def test_compare_estimators_reports_spread():
    df = build_promo_users(n=15_000)
    df["log_pre_revenue"] = np.log1p(df["pre_revenue"])
    out = compare_estimators(df, "promo", "post_revenue", ["log_pre_revenue", "tenure_months"], n_bootstrap=30)
    assert out["summary"]["estimates"]
    assert out["summary"]["robust"] is True


# ------------------------------------------------ robustness & sensitivity
def test_placebo_refutation_shreds_fake_treatment():
    """Shuffling the treatment must collapse the effect toward zero."""
    from ds_agent.tools.causal import placebo_refutation

    df = build_promo_users(n=8_000)
    df["log_pre_revenue"] = np.log1p(df["pre_revenue"])
    out = placebo_refutation(
        df, "promo", "post_revenue", ["log_pre_revenue", "tenure_months"],
        n_placebo=4, seed=0,
    )
    assert out["robust"] is True
    assert abs(out["placebo_mean"]) < abs(out["real_att"]) / 4
    assert abs(out["real_att"]) > out["placebo_max_abs"]


def test_rosenbaum_bounds_monotone_and_covers_wilcoxon():
    """p_upper must grow with Gamma; Gamma=1 must match the plain signed-rank p."""
    import scipy.stats as st

    from ds_agent.tools.causal import rosenbaum_bounds

    rng = np.random.default_rng(7)
    diffs = rng.normal(1.0, 1.0, size=60)
    rows = rosenbaum_bounds(diffs)
    assert [r["gamma"] for r in rows] == [1.0, 1.5, 2.0, 2.5, 3.0]
    uppers = [r["p_upper"] for r in rows]
    assert all(b >= a for a, b in zip(uppers, uppers[1:]))
    assert rows[0]["survives"] is True          # Γ=1 is the real test
    t, p = st.wilcoxon(diffs, alternative="greater")
    assert rows[0]["p_upper"] == pytest.approx(p, rel=0.35)  # normal approx


def test_method_selection_advice_flags_unbalance_and_no_support():
    from ds_agent.tools.causal import method_selection_advice

    ok = method_selection_advice(
        {"common_support": {"share_outside": 0.0},
         "balance": [{"balanced": True}, {"balanced": True}]}
    )
    assert any("hold" in a["situation"] for a in ok)
    bad = method_selection_advice(
        {"common_support": {"share_outside": 0.2},
         "balance": [{"balanced": False}, {"balanced": True}]}
    )
    joined = " ".join(a["situation"] for a in bad)
    assert "common support" in joined and "unbalanced" in joined
    assert any("en" in a for a in bad) and all(a.get("zh") and a.get("en") for a in bad)
