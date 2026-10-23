"""Forecasters used as anomaly baselines: causality, accuracy, graceful failure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ds_agent.tools.forecast import (
    FORECAST_METHODS,
    accuracy,
    compare_methods,
    fit_forecast,
    rolling_forecast,
    select_method,
)


def _seasonal(n: int = 120, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    weekly = np.array([1.00, 1.02, 0.98, 0.99, 1.05, 1.13, 1.07])[t % 7]
    return 1000 * (1 + 0.0015 * t) * weekly * rng.normal(1.0, 0.008, n)


def test_rolling_forecast_is_causal():
    """A forecast for day t must not move when only the future changes.

    This is the property the whole anomaly baseline rests on: if the
    "expectation" can see the incident, it absorbs it and nothing is flagged.
    The tolerance is loose because statsmodels' optimiser is only reproducible
    to ~1e-8; a future leak would move these numbers by percent, not by ppm.
    """
    values = _seasonal()
    base = rolling_forecast(values, method="holt_winters", period=7)
    shifted = values.copy()
    shifted[80:] *= 0.6  # rewrite the future only
    after = rolling_forecast(shifted, method="holt_winters", period=7)
    head = slice(30, 80)  # skip the warm-up where everything is NaN
    np.testing.assert_allclose(base[head], after[head], rtol=1e-4)

    # And the leak would be obvious if it existed: the tail must move.
    assert not np.allclose(base[85:], after[85:], rtol=1e-3)


def test_holt_winters_beats_seasonal_naive_on_seasonal_data():
    values = _seasonal()
    table = compare_methods(values, methods=("holt_winters", "seasonal_naive"), period=7)
    mase = table.set_index("method")["mase"]
    assert mase["holt_winters"] < mase["seasonal_naive"]


def test_select_method_returns_a_candidate():
    values = _seasonal()
    chosen = select_method(values, methods=("holt_winters", "seasonal_naive", "ewma"))
    assert chosen["method"] in ("holt_winters", "seasonal_naive", "ewma")
    assert np.isfinite(chosen["margin"])
    assert not chosen["table"].empty


def test_perfect_forecast_has_zero_error():
    values = _seasonal(60)
    metrics = accuracy(values, values.copy(), period=7)
    assert metrics["mae"] == 0
    assert metrics["mase"] == 0


def test_every_method_produces_finite_forecasts():
    values = _seasonal(90)
    for method in FORECAST_METHODS:
        preds = rolling_forecast(values, method=method, period=7, refit_every=7)
        assert len(preds) == len(values)
        assert np.isfinite(preds[30:]).all(), f"{method} produced non-finite forecasts"


def test_short_and_constant_series_do_not_raise():
    """Fit must degrade, never raise: it runs inside a monitoring job."""
    assert np.isfinite(fit_forecast([5.0, 6.0, 5.5], horizon=2, method="holt_winters")).all()
    assert np.isfinite(fit_forecast(np.full(40, 7.0), horizon=3, method="arima")).all()


def test_returns_dataframe_ready_table():
    values = _seasonal(80)
    table = compare_methods(values, period=7)
    assert set(["method", "mase", "mae", "rmse", "n"]).issubset(table.columns)
    assert (table["n"] > 0).all()
    assert isinstance(table, pd.DataFrame)
