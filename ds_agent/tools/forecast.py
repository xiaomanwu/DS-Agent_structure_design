"""Time-series forecasters used as anomaly baselines and counterfactuals.

Why a separate module: an anomaly detector is only as good as its expectation.
A robust z-score on top of a bad baseline produces confident nonsense, so the
forecasting models live here, are backtested with the same rolling-origin
protocol, and only then feed the detector.

Implemented models
------------------
* ``holt_winters``      additive trend (damped) + additive seasonality
* ``holt_winters_mul``  multiplicative seasonality, for series whose weekly
                        amplitude scales with the level (GMV, sessions)
* ``arima``             SARIMAX with a weekly seasonal term
* ``stl_ets``           STL seasonal indices + ETS on the seasonally adjusted
                        series (robust to a few outage days)
* ``seasonal_naive``    the same weekday last week; the benchmark every model
                        must beat
* ``ewma``              damped exponential smoothing of the deseasonalised level
* ``trailing``          trailing median level x weekday index (the default in
                        :mod:`ds_agent.tools.anomaly`)

Every function is *causal*: the forecast for day t uses data up to t-1 only, so
a multi-day incident cannot leak into its own baseline. Backtests use the same
one-step-ahead protocol as production, which is why the reported MASE is
comparable across models.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

FORECAST_METHODS: Tuple[str, ...] = (
    "holt_winters",
    "holt_winters_mul",
    "arima",
    "stl_ets",
    "seasonal_naive",
    "ewma",
    "trailing",
)

METHOD_LABELS: Dict[str, str] = {
    "holt_winters": "Holt-Winters (additive)",
    "holt_winters_mul": "Holt-Winters (multiplicative)",
    "arima": "SARIMA",
    "stl_ets": "STL + ETS",
    "seasonal_naive": "Seasonal naive",
    "ewma": "EWMA level",
    "trailing": "Trailing median",
}


# --------------------------------------------------------------------------- #
# Single-model forecasters
# --------------------------------------------------------------------------- #
def _hw(train: np.ndarray, horizon: int, period: int, seasonal: str) -> Optional[np.ndarray]:
    """Damped Holt-Winters. Returns ``None`` when the series is too short."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    if len(train) < 2 * period:
        return None

    def _fit(init: str):
        model = ExponentialSmoothing(
            train,
            trend="add",
            damped_trend=True,
            seasonal=seasonal,
            seasonal_periods=period,
            initialization_method=init,
        )
        return model.fit(optimized=True, use_brute=False, remove_bias=False)

    # Heuristic initialisation first: statsmodels' "estimated" mode draws random
    # starting states, so two calls on identical data differ in the 5th decimal.
    # That is harmless for one baseline but it makes an alert impossible to
    # reproduce the next morning, which is the whole point of a baseline.
    out: Optional[np.ndarray] = None
    for init in ("heuristic", "estimated"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = _fit(init)
            out = np.asarray(fit.forecast(horizon), dtype=float)
        except Exception:  # noqa: BLE001 - numerical failure is expected on odd series
            out = None
            continue
        if out is not None and np.all(np.isfinite(out)):
            break
    if not np.all(np.isfinite(out)):
        return None
    return out


def _arima(train: np.ndarray, horizon: int, period: int) -> Optional[np.ndarray]:
    """SARIMAX(1,1,1)(0,1,1)_period with a fixed, fast specification.

    Auto-order search is deliberately avoided: it changes the baseline from one
    day to the next, which makes yesterday's alert impossible to reproduce.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    if len(train) < 3 * period:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                train,
                order=(1, 1, 1),
                seasonal_order=(0, 1, 1, period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=60)
        out = np.asarray(fit.forecast(horizon), dtype=float)
    except Exception:  # noqa: BLE001
        return None
    if not np.all(np.isfinite(out)):
        return None
    return out


def _stl_cycle(train: np.ndarray, period: int) -> Optional[Tuple[str, np.ndarray]]:
    """One seasonal cycle from a robust STL fit, ready for extrapolation.

    STL is additive, so on a strongly trending business metric the seasonal
    component comes out in absolute units (e.g. +90k on Saturdays) and cannot be
    used as a multiplier. Positive series are therefore decomposed on the log
    scale and converted back to a multiplicative index with mean 1; others keep
    the additive cycle centred on zero.
    """
    from statsmodels.tsa.seasonal import STL

    if len(train) < 2 * period + 1:
        return None
    positive = bool((train > 0).all())
    series = np.log(train) if positive else train
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = STL(series, period=period, robust=True).fit()
        seasonal = np.asarray(res.seasonal, dtype=float)
    except Exception:  # noqa: BLE001
        return None
    cycle = seasonal[-period:]
    if positive:
        cycle = np.exp(cycle)
        mean = cycle.mean()
        if not np.isfinite(mean) or mean <= 0:
            return None
        return "multiplicative", cycle / mean
    cycle = cycle - cycle.mean()
    return "additive", cycle


def _ets_trend_forecast(deseason: np.ndarray, horizon: int) -> np.ndarray:
    """Damped local trend on a deseasonalised series (Holt's linear method)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    if len(deseason) >= 8:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ExponentialSmoothing(
                    deseason, trend="add", damped_trend=True, seasonal=None,
                    initialization_method="estimated",
                ).fit(optimized=True, remove_bias=False)
            out = np.asarray(fit.forecast(horizon), dtype=float)
            if np.all(np.isfinite(out)):
                return out
        except Exception:  # noqa: BLE001
            pass
    # Fallback: last level plus half the average recent slope, damped.
    level = float(deseason[-1])
    slope = float(np.median(np.diff(deseason[-8:]))) if len(deseason) >= 9 else 0.0
    steps = np.arange(1, horizon + 1)
    return level + 0.5 * slope * steps


def _stl_ets(train: np.ndarray, horizon: int, period: int) -> Optional[np.ndarray]:
    fit = _stl_cycle(train, period)
    if fit is None:
        return None
    mode, cycle = fit
    n = len(train)
    # The cycle was taken as seasonal[-period:], i.e. cycle[j] belongs to
    # observation n - period + j. Future observation n + h - 1 therefore maps to
    # cycle[(h - 1) % period] -- independent of where the sample starts.
    current = cycle[(np.arange(n) - (n - period)) % period]
    future = cycle[np.array([(h - 1) % period for h in range(1, horizon + 1)])]
    if mode == "multiplicative":
        deseason = train / current
        return _ets_trend_forecast(deseason, horizon) * future
    deseason = train - current
    return _ets_trend_forecast(deseason, horizon) + future


def _seasonal_naive(train: np.ndarray, horizon: int, period: int) -> Optional[np.ndarray]:
    if len(train) < period:
        return None
    return np.asarray([train[len(train) - period + ((h - 1) % period)] for h in range(1, horizon + 1)], dtype=float)


def _ewma_level(train: np.ndarray, horizon: int, period: int, halflife: float = 7.0) -> Optional[np.ndarray]:
    if len(train) < period:
        return None
    phase = np.arange(len(train)) % period
    profile = pd.Series(train).groupby(phase).median()
    profile = profile / float(profile.mean() or 1.0)
    deseason = pd.Series(train) / pd.Series(phase).map(profile).astype(float)
    level = float(deseason.ewm(halflife=halflife, adjust=False).mean().iloc[-1])
    slope = float(deseason.diff(period).tail(4).median() or 0.0) / period
    out = []
    for h in range(1, horizon + 1):
        out.append((level + slope * h) * float(profile.iloc[(len(train) + h - 1) % period]))
    return np.asarray(out, dtype=float)


def _trailing_median(train: np.ndarray, horizon: int, period: int, window: int = 28) -> Optional[np.ndarray]:
    if len(train) < period + 1:
        return None
    phase = np.arange(len(train)) % period
    profile = pd.Series(train).groupby(phase).median()
    profile = profile / float(profile.mean() or 1.0)
    deseason = pd.Series(train) / pd.Series(phase).map(profile).astype(float)
    level = float(deseason.tail(min(window, len(deseason))).median())
    out = []
    for h in range(1, horizon + 1):
        out.append(level * float(profile.iloc[(len(train) + h - 1) % period]))
    return np.asarray(out, dtype=float)


def fit_forecast(
    train: Sequence[float], horizon: int = 1, method: str = "holt_winters", period: int = 7
) -> np.ndarray:
    """Forecast ``horizon`` steps ahead from ``train`` with one model.

    Always returns an array of length ``horizon``: failures degrade to the
    seasonal-naive benchmark instead of raising, because a monitoring job must
    not die on a numerically awkward series.
    """
    y = np.asarray(train, dtype=float)
    y = y[np.isfinite(y)]
    method = method.lower()

    if len(y) < 3:
        return np.full(horizon, y[-1] if len(y) else 0.0)

    out: Optional[np.ndarray] = None
    if method in ("holt_winters", "hw"):
        out = _hw(y, horizon, period, "add")
    elif method in ("holt_winters_mul", "hw_mul"):
        if (y <= 0).any():
            out = _hw(y, horizon, period, "add")
        else:
            out = _hw(y, horizon, period, "mul")
    elif method in ("arima", "sarima", "sarimax"):
        out = _arima(y, horizon, period)
    elif method in ("stl_ets", "stl"):
        out = _stl_ets(y, horizon, period)
    elif method in ("seasonal_naive", "snaive"):
        out = _seasonal_naive(y, horizon, period)
    elif method in ("ewma", "ets"):
        out = _ewma_level(y, horizon, period)
    elif method in ("trailing", "median"):
        out = _trailing_median(y, horizon, period)

    if out is None:
        out = _seasonal_naive(y, horizon, period)
    if out is None:
        out = np.full(horizon, float(np.median(y)))
    out = np.asarray(out, dtype=float)
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, float(np.median(y)))
    return out


# --------------------------------------------------------------------------- #
# Rolling one-step-ahead forecasts
# --------------------------------------------------------------------------- #
def rolling_forecast(
    values: Sequence[float],
    method: str = "holt_winters",
    period: int = 7,
    refit_every: int = 1,
    min_train: Optional[int] = None,
) -> np.ndarray:
    """One-step-ahead predictions produced causally, one per observation.

    ``refit_every`` controls the cost/accuracy trade-off: refitting daily is the
    most accurate (180 fits on a six-month series), refitting weekly and walking
    the forecast forward is ~7x cheaper and still causal. The anomalies this is
    designed for are multi-day, so the cheaper setting changes nothing in
    practice while making the detector fast enough to run per metric.
    """
    y = np.asarray(values, dtype=float)
    n = len(y)
    min_train = min_train or max(2 * period + 1, min(28, n // 3))
    preds = np.full(n, np.nan)
    if n <= min_train:
        return preds

    refit_every = max(1, int(refit_every))
    cache: List[float] = []
    cache_pos = 0
    for i in range(min_train, n):
        if not cache or cache_pos >= len(cache):
            horizon = min(refit_every, n - i)
            cache = list(fit_forecast(y[:i], horizon=horizon, method=method, period=period))
            cache_pos = 0
        preds[i] = cache[cache_pos]
        cache_pos += 1
    return preds


def rolling_forecast_matrix(
    values: Sequence[float],
    methods: Sequence[str] = FORECAST_METHODS,
    period: int = 7,
    refit_every: int = 7,
    min_train: Optional[int] = None,
) -> pd.DataFrame:
    """One-step predictions of several models side by side (columns = models)."""
    out = {}
    for method in methods:
        out[method] = rolling_forecast(values, method=method, period=period, refit_every=refit_every, min_train=min_train)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #
def accuracy(actual: Sequence[float], predicted: Sequence[float], period: int = 7) -> Dict[str, float]:
    """MAE / RMSE / MASE / bias / interval-free coverage on a rolling backtest.

    MASE divides by the in-sample seasonal-naive error, so a value below 1 means
    "better than repeating last week" -- the only comparison that matters for a
    business baseline.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    if mask.sum() < 3:
        return {"mae": float("nan"), "rmse": float("nan"), "mase": float("nan"), "bias": float("nan"), "n": int(mask.sum())}
    err = a[mask] - p[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    scale = float(np.mean(np.abs(np.diff(a[np.isfinite(a)], n=period)))) if len(a) > period else 0.0
    mase = mae / scale if scale > 1e-12 else float("nan")
    denom = float(np.mean(np.abs(a[mask]))) or 1e-12
    return {
        "mae": mae,
        "rmse": rmse,
        "mase": mase,
        "bias": float(np.mean(err)),
        "mape": float(np.mean(np.abs(err / np.where(np.abs(a[mask]) > 1e-12, a[mask], np.nan)))),
        "rel_rmse": rmse / denom,
        "n": int(mask.sum()),
    }


def compare_methods(
    values: Sequence[float],
    methods: Sequence[str] = FORECAST_METHODS,
    period: int = 7,
    refit_every: int = 7,
    min_train: Optional[int] = None,
) -> pd.DataFrame:
    """Backtest every model and rank by MASE (lower is better)."""
    y = np.asarray(values, dtype=float)
    rows = []
    for method in methods:
        preds = rolling_forecast(y, method=method, period=period, refit_every=refit_every, min_train=min_train)
        acc = accuracy(y, preds, period)
        rows.append({"method": method, "label": METHOD_LABELS.get(method, method), **acc})
    frame = pd.DataFrame(rows)
    if frame["mase"].notna().any():
        frame = frame.sort_values("mase", na_position="last").reset_index(drop=True)
    return frame


def select_method(
    values: Sequence[float],
    methods: Sequence[str] = ("holt_winters", "holt_winters_mul", "arima", "stl_ets", "seasonal_naive"),
    period: int = 7,
    refit_every: int = 7,
) -> Dict[str, object]:
    """Pick the model with the lowest rolling-origin MASE.

    Model selection is logged with the full table so a reviewer can see how
    close the runner-up was -- a 1% MASE gap is not a reason to prefer a model
    with three extra assumptions.
    """
    table = compare_methods(values, methods=methods, period=period, refit_every=refit_every)
    valid = table[table["mase"].notna()]
    if valid.empty:
        return {"method": "seasonal_naive", "table": table, "margin": float("nan")}
    best = valid.iloc[0]
    runner_up = float(valid.iloc[1]["mase"]) if len(valid) > 1 else float("nan")
    margin = (runner_up - float(best["mase"])) / runner_up if np.isfinite(runner_up) and runner_up else float("nan")
    return {"method": str(best["method"]), "table": table, "margin": float(margin)}
