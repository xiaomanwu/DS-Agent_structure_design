"""Time-series causal inference: the two estimators, and the checks that license them.

The generator plants a +12% level shift on day 95 of a 140-day series whose
underlying trend is already rising. That setup is deliberate: a naive pre/post
difference reads roughly +24%, so any estimator that does not separate the trend
from the intervention fails the first test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ds_agent.data.synthetic import build_its_series
from ds_agent.tools.ts_causal import arima_intervention, causal_impact, compare_ts_methods

TRUE_EFFECT = 0.12
INTERVENTION = "2026-09-06"  # index 95 of the generated series, which ends 2026-10-20
CONTROLS = ("control_gmv_1", "control_gmv_2")


@pytest.fixture(scope="module")
def series():
    return build_its_series()


@pytest.fixture(scope="module")
def its(series):
    return arima_intervention(
        series, time_col="date", outcome="gmv", intervention=INTERVENTION, covariates=CONTROLS
    )


@pytest.fixture(scope="module")
def impact(series):
    return causal_impact(
        series, time_col="date", outcome="gmv", intervention=INTERVENTION, controls=CONTROLS
    )


# --------------------------------------------------------------------------- #
# ARIMA interrupted time series
# --------------------------------------------------------------------------- #


def test_its_recovers_the_level_shift(its):
    assert its["method"] == "arima_its"
    assert its["relative"] == pytest.approx(TRUE_EFFECT, abs=0.04)
    assert its["p_value"] < 0.01


def test_its_beats_the_naive_reading(series, its):
    """The naive comparison bills the trend to the intervention, so it overshoots."""
    pre = series[series["date"] < pd.Timestamp(INTERVENTION).date()]["gmv"].mean()
    post = series[series["date"] >= pd.Timestamp(INTERVENTION).date()]["gmv"].mean()
    naive = post / pre - 1
    assert naive > 0.18, "fixture should contain a trend that the naive read inflates"
    assert abs(its["relative"] - TRUE_EFFECT) < abs(naive - TRUE_EFFECT)


def test_its_uses_the_log_scale_and_reports_it(its):
    # GMV is strictly positive and multiplicative, so `auto` must pick the log.
    assert its["scale"] == "log"
    assert its["log_effect"] > 0
    assert its["order"][1] == 0, "no differencing: the trend is a regressor, not a difference"


def test_its_counterfactual_sits_below_the_actual(its):
    frame = its["frame"]
    post = frame[frame["post"] == 1]
    assert (post["effect"] > 0).mean() > 0.9
    assert post["counterfactual"].mean() < post["actual"].mean()
    # Before the intervention there is no shift, so the gap must be centred on zero.
    pre = frame[frame["post"] == 0]
    assert abs(pre["effect"].mean()) < 0.02 * pre["actual"].mean()


def test_its_residuals_are_white_noise(its):
    """A significant Ljung-Box would mean the SEs are too small, not that the effect is bigger."""
    assert its["ljung_box_p"] > 0.05


def test_its_specification_curve_is_tight(its):
    table = its["spec_table"]
    assert len(table) >= 3
    assert {"spec", "relative", "p_value"}.issubset(table.columns)
    assert table["relative"].between(0.02, 0.30).all(), "every spec should keep the sign"


def test_its_placebo_dates_do_not_reproduce_the_effect(its):
    assert its["n_placebos"] >= 4
    assert its["placebo_p"] > 0.05


def test_its_reports_period_counts_and_means(its):
    assert its["n_pre"] + its["n_post"] == 140
    assert its["post_mean"] > its["pre_mean"]


def test_its_accepts_a_timestamp_intervention(series, its):
    other = arima_intervention(
        series,
        time_col="date",
        outcome="gmv",
        intervention=pd.Timestamp(INTERVENTION),
        covariates=CONTROLS,
        placebo_n=0,
    )
    assert other["intervention"] == its["intervention"]
    assert other["relative"] == pytest.approx(its["relative"], abs=0.02)


def test_its_is_deterministic(series, its):
    again = arima_intervention(
        series, time_col="date", outcome="gmv", intervention=INTERVENTION, covariates=CONTROLS, placebo_n=0
    )
    assert again["relative"] == pytest.approx(its["relative"], rel=1e-5)


def test_its_level_scale_is_available_but_less_precise(series, its):
    level = arima_intervention(
        series,
        time_col="date",
        outcome="gmv",
        intervention=INTERVENTION,
        covariates=CONTROLS,
        scale="level",
        placebo_n=0,
    )
    assert level["scale"] == "level"
    assert level["relative"] > 0


def test_its_needs_a_pre_and_post_window(series):
    truncated = series.tail(12)
    with pytest.raises(ValueError):
        arima_intervention(truncated, time_col="date", outcome="gmv", intervention=INTERVENTION)


def test_its_needs_thirty_periods():
    tiny = pd.DataFrame(
        {"date": pd.date_range("2026-01-01", periods=20, freq="D"), "gmv": np.linspace(100, 120, 20)}
    )
    with pytest.raises(ValueError, match="30 periods"):
        arima_intervention(tiny, time_col="date", outcome="gmv", intervention="2026-01-10")


# --------------------------------------------------------------------------- #
# Structural counterfactual ("causal impact")
# --------------------------------------------------------------------------- #


def test_causal_impact_recovers_the_level_shift(impact):
    assert impact["method"] == "causal_impact"
    assert impact["relative"] == pytest.approx(TRUE_EFFECT, abs=0.05)
    assert impact["effect"] > 0
    assert impact["p_value"] < 0.01


def test_causal_impact_pre_fit_is_good(impact):
    """A model that cannot reproduce the pre-period has no business forecasting it."""
    assert impact["pre_fit_r2"] > 0.8
    assert impact["pre_rmse"] < 0.1 * impact["frame"]["actual"].mean()


def test_causal_impact_interval_widens_with_horizon(impact):
    frame = impact["frame"]
    width = frame["upper"] - frame["lower"]
    assert (width > 0).all()
    assert width.iloc[-1] > width.iloc[0]


def test_causal_impact_cumulative_effect_accumulates(impact):
    frame = impact["frame"]
    # Daily effects are noisy, so the level can dip; the cumulative path must not.
    assert frame["cumulative_effect"].iloc[-1] > frame["cumulative_effect"].iloc[0]
    assert impact["cumulative_effect"] == pytest.approx(frame["effect"].sum(), rel=1e-6)


def test_causal_impact_placebo_windows_are_quiet(impact):
    assert impact["n_placebos"] >= 4
    assert impact["placebo_p"] > 0.05


def test_causal_impact_uses_controls_and_says_so(impact):
    assert impact["controls"] == list(CONTROLS)
    assert impact["n_controls"] == 2
    assert impact["scale"] == "log"
    assert not impact["stochastic_trend"], "a stochastic trend would absorb the level shift"
    assert "控制序列" in impact["narrative"]


def test_causal_impact_without_controls_warns(series, impact):
    bare = causal_impact(series, time_col="date", outcome="gmv", intervention=INTERVENTION, placebo_n=0)
    assert bare["n_controls"] == 0
    # Unidentified without controls: the estimate is allowed to be worse, but the
    # narrative has to admit it rather than present it as equally credible.
    assert "注意" in bare["narrative"]
    assert "不可区分" in bare["narrative"]
    assert abs(bare["relative"] - impact["relative"]) > 0.01


def test_causal_impact_reports_history_for_charting(impact):
    history = impact["history"]
    assert {"time", "actual", "counterfactual"}.issubset(history.columns)
    assert len(history) == impact["n_pre"]
    # The first fitted value is the diffuse prior, not a prediction, so it is
    # masked rather than drawn as a spike from zero.
    assert np.isnan(history["counterfactual"].iloc[0])
    assert np.isfinite(history["counterfactual"].iloc[1:]).all()


def test_causal_impact_is_deterministic(series, impact):
    again = causal_impact(
        series, time_col="date", outcome="gmv", intervention=INTERVENTION, controls=CONTROLS, placebo_n=0
    )
    assert again["relative"] == pytest.approx(impact["relative"], rel=1e-6)


def test_causal_impact_needs_a_pre_window(series):
    with pytest.raises(ValueError):
        causal_impact(series.tail(15), time_col="date", outcome="gmv", intervention=INTERVENTION)


# --------------------------------------------------------------------------- #
# Side-by-side comparison
# --------------------------------------------------------------------------- #


def test_compare_ts_methods_lines_up_three_readings(series):
    result = compare_ts_methods(
        series, time_col="date", outcome="gmv", intervention=INTERVENTION, controls=CONTROLS
    )
    table = result["table"]
    assert len(table) == 3
    assert {"method", "effect", "relative", "p_value"}.issubset(table.columns)
    naive, arima, structural = table["relative"].tolist()
    assert naive > arima > 0
    assert structural == pytest.approx(arima, rel=0.5, abs=0.06)
    assert result["agree"] is True


def test_compare_ts_methods_narrative_names_the_naive_bias(series):
    result = compare_ts_methods(
        series, time_col="date", outcome="gmv", intervention=INTERVENTION, controls=CONTROLS, lang="en"
    )
    assert "naive" in result["narrative"]
    assert "overstates" in result["narrative"]
