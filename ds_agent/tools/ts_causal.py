"""Time-series causal inference: interrupted time series and causal impact.

When an intervention happens at a known date and randomisation is impossible, the
counterfactual is a forecast. Two families are implemented:

* **ARIMA interrupted time series** (``arima_intervention``) - one model on the
  whole series with a post-period indicator. The coefficient on that indicator is
  the level shift, with an ARMA error structure that absorbs autocorrelation.
  This is the right tool when the effect is a step change and the series is long
  enough to estimate the error process.
* **Causal impact / structural time series** (``causal_impact``) - a local level
  state-space model with control series as regressors, fitted on the pre-period
  only and forecast forward. Effect = actual - counterfactual, with the
  state-space forecast variance providing the interval.

Three design choices carry most of the weight, and all three are mistakes people
actually make:

1. **Scale.** Revenue-like metrics are multiplicative: the weekly pattern and the
   shock both scale with the level, so modelling them additively lets the trend
   masquerade as an effect. Strictly positive series are therefore modelled on the
   log scale by default, and the coefficient is converted back to a level effect.
2. **Trend and seasonality.** A post indicator is almost collinear with a rising
   trend and with the day-of-week pattern. Leaving either out of the design makes
   the estimate absorb them - on the demo series that is the difference between
   +12% (correct) and +24% (naive pre/post difference).
3. **Inference.** Residual autocorrelation inflates the t-statistic by an order of
   magnitude, so the error structure is estimated rather than assumed, and every
   estimate is checked against placebo interventions where nothing happened.

``compare_ts_methods`` runs the naive difference, the ARIMA ITS and the structural
model side by side, because the gap between them is usually the most persuasive
number in the room.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

_WEEK = 7


def _tr(lang: str, zh: str, en: str) -> str:
    return zh if lang == "zh" else en


def _prepare(
    df: pd.DataFrame, time_col: str, outcome: str, extra: Sequence[str] = ()
) -> pd.DataFrame:
    """Sort, de-duplicate and type-cast the series, keeping the columns we need."""
    keep = [time_col, outcome] + [c for c in extra if c in df.columns and c != outcome]
    keep = list(dict.fromkeys(keep))
    work = df[keep].copy()
    work[time_col] = pd.to_datetime(work[time_col])
    for col in keep[1:]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = (
        work.dropna()
        .sort_values(time_col)
        .drop_duplicates(time_col)
        .reset_index(drop=True)
    )
    if len(work) < 30:
        raise ValueError("time-series causal inference needs at least 30 periods")
    return work


def _use_log(y: np.ndarray, scale: str) -> bool:
    """Decide whether to model on the log scale.

    ``auto`` means: log for a strictly positive series whose minimum is not
    negligible next to its median, because that is what makes a multiplicative
    weekly pattern and a proportional shock additive. Pass ``scale="level"`` to
    force the linear specification.
    """
    y = np.asarray(y, dtype=float)
    if scale == "log":
        return True
    if scale == "level":
        return False
    if not np.all(np.isfinite(y)) or y.min() <= 0:
        return False
    # A series that touches zero cannot be logged safely.
    return bool(y.min() > 0.01 * np.median(y))


def _dow_dummies(dates: pd.Series) -> Tuple[np.ndarray, List[str]]:
    """Day-of-week dummies (Monday dropped) for a daily series.

    Deterministic seasonality as regressors beats seasonal differencing here: it
    keeps the post indicator comparable across weekdays and does not throw away
    the first week of the pre-period.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    empty = (np.zeros((len(idx), 0)), [])
    if len(idx) < 2 * _WEEK:
        return empty
    spacing = np.median(np.diff(idx.values).astype("timedelta64[D]").astype(float))
    if spacing > 1.5:  # not daily data, weekly dummies are meaningless
        return empty
    dummies = pd.get_dummies(idx.dayofweek, prefix="dow", drop_first=True).astype(float)
    return dummies.to_numpy(), list(dummies.columns)


def _design(
    work: pd.DataFrame,
    cutoff: pd.Timestamp,
    time_col: str,
    outcome_col: str,
    trend: bool,
    seasonal_dummies: bool,
    covariates: Sequence[str],
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Build the regression design: post indicator, intercept, trend, season, controls.

    The intercept is not optional. With ``d=0`` the ARMA error process has zero
    unconditional mean, so without a constant the series *level* has nowhere to go
    and gets loaded onto ``_trend`` and ``_post`` instead - which is how the same
    specification produced +12% and -13.5% on consecutive runs.
    """
    post = (work[time_col] >= cutoff).to_numpy().astype(float)
    cols: List[np.ndarray] = [post, np.ones(len(work))]
    names: List[str] = ["_post", "_const"]
    if trend:
        cols.append(np.linspace(0.0, 1.0, len(work)))
        names.append("_trend")
    if seasonal_dummies:
        matrix, dow_names = _dow_dummies(work[time_col])
        for i, name in enumerate(dow_names):
            cols.append(matrix[:, i])
            names.append(name)
    for col in covariates:
        if col in work.columns:
            cols.append(work[col].to_numpy(dtype=float))
            names.append(col)
    return np.column_stack(cols), names, post


def _fit_sarimax(y, X, order, seasonal_order, maxiter: int = 300):
    """Fit a SARIMAX, degrading gracefully instead of raising at the user.

    Seasonal orders are the usual failure on short or very noisy series; a plain
    ARMA on the same regressors is a legitimate fallback and is reported back so
    the caller can say which specification actually ran.

    Starting values are taken from OLS on the same regressors rather than from
    statsmodels' defaults: the default start puts the regression coefficients at
    zero, and on a log-scale revenue series that is far enough from the optimum
    for the optimizer to settle in a different, wrong, local mode.
    """
    attempts: List[Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int, int]]]] = [
        (order, seasonal_order)
    ]
    if seasonal_order:
        attempts.append((order, None))
    attempts.extend([((1, 0, 0), None), ((0, 0, 0), None)])
    last_error: Optional[Exception] = None
    for od, so in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sm.tsa.SARIMAX(
                    y,
                    exog=X,
                    order=od,
                    seasonal_order=so or (0, 0, 0, 0),
                    trend="n",  # the intercept is an explicit column of the design
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                start = _ols_start(np.asarray(y, dtype=float), X, od)
                try:
                    res = model.fit(disp=False, maxiter=maxiter, start_params=start)
                except Exception:  # noqa: BLE001 - a bad start is better than no fit
                    res = model.fit(disp=False, maxiter=maxiter)
            return res, od, so
        except Exception as exc:  # noqa: BLE001 - try the next, simpler, spec
            last_error = exc
    raise RuntimeError(f"all SARIMAX specifications failed: {last_error}")


def _ols_start(y: np.ndarray, X, order: Tuple[int, int, int]) -> Optional[np.ndarray]:
    """OLS coefficients for the regressors, zeros for the ARMA part.

    Returns ``None`` when the design is singular, in which case statsmodels'
    default start is used instead.
    """
    if X is None:
        return None
    x = np.asarray(X, dtype=float)
    try:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(beta)):
        return None
    resid = y - x @ beta
    p, _, q = order
    scale = float(np.var(resid)) or 1.0
    return np.concatenate([beta, np.full(p, 0.1), np.full(q, 0.1), [scale]])



def _relative(effect: float, base: float, use_log: bool) -> float:
    if use_log:
        return float(np.exp(effect) - 1.0)
    return float(effect / base) if base else float("nan")


def _to_level(effect: float, base: float, use_log: bool) -> float:
    """Convert an effect to level units so revenue and rate metrics are comparable."""
    if use_log:
        return float(base * (np.exp(effect) - 1.0))
    return float(effect)


def arima_intervention(
    df: pd.DataFrame,
    time_col: str,
    outcome: str,
    intervention,
    order: Tuple[int, int, int] = (1, 0, 1),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    covariates: Sequence[str] = (),
    trend: bool = True,
    seasonal_dummies: bool = True,
    scale: str = "auto",
    placebo_n: int = 8,
    alpha: float = 0.05,
    lang: str = "zh",
) -> Dict[str, object]:
    """Interrupted time series with ARMA errors.

    The model is ``f(y_t) = beta * post_t + gamma' X_t + u_t`` with ``u_t`` an
    ARMA process and ``f`` the identity or the log. ``X`` holds a linear trend,
    day-of-week dummies and any control series, so that ``beta`` is not asked to
    explain the trend or the weekly shape as well as the intervention.

    Regressing the post indicator with OLS instead is the classic mistake here:
    the residual autocorrelation inflates the t-statistic by an order of
    magnitude on a weekly series, and the missing trend makes the coefficient
    absorb the growth that was happening anyway.
    """
    work = _prepare(df, time_col, outcome, extra=tuple(covariates))
    cov = [c for c in covariates if c in work.columns]
    cutoff = pd.Timestamp(intervention)

    y_raw = work[outcome].to_numpy(dtype=float)
    use_log = _use_log(y_raw, scale)
    y = np.log(y_raw) if use_log else y_raw

    X, names, post = _design(work, cutoff, time_col, outcome, trend, seasonal_dummies, cov)
    n_pre = int((post == 0).sum())
    n_post = int((post == 1).sum())
    if n_pre < 15 or n_post < 3:
        raise ValueError("need at least 15 pre-period and 3 post-period observations")

    exog = pd.DataFrame(X, columns=names)
    fit, used_order, used_seasonal = _fit_sarimax(y, exog, order, seasonal_order)

    beta = float(fit.params["_post"])
    se = float(fit.bse["_post"])
    p_value = float(fit.pvalues["_post"])
    z = float(stats.norm.ppf(1 - alpha / 2))

    pre_mean = float(y_raw[post == 0].mean())
    post_mean = float(y_raw[post == 1].mean())
    effect = _to_level(beta, pre_mean, use_log)
    relative = _relative(effect=beta, base=pre_mean, use_log=use_log)
    ci = (
        _to_level(beta - z * se, pre_mean, use_log),
        _to_level(beta + z * se, pre_mean, use_log),
    )

    # ---- specification curve: is the answer an artefact of these choices? ----
    spec_rows: List[Dict[str, object]] = [
        {
            "spec": _tr(lang, "主口径", "main specification"),
            "relative": relative,
            "p_value": p_value,
            "note": f"order={used_order}, seasonal={used_seasonal or '-'}, log={use_log}",
        }
    ]

    def _variant(label_zh: str, label_en: str, **overrides) -> None:
        """Re-fit with one design choice changed; identical designs are skipped."""
        t = overrides.get("trend", trend)
        s = overrides.get("seasonal_dummies", seasonal_dummies)
        sc = overrides.get("scale", scale)
        od = overrides.get("order", order)
        so = overrides.get("seasonal_order", seasonal_order)
        if (t, s, _use_log(y_raw, sc), tuple(od), so) == (
            trend,
            seasonal_dummies,
            use_log,
            tuple(order),
            seasonal_order,
        ):
            return
        try:
            yy = np.log(y_raw) if _use_log(y_raw, sc) else y_raw
            XX, nn, _ = _design(work, cutoff, time_col, outcome, t, s, cov)
            alt, _, _ = _fit_sarimax(yy, pd.DataFrame(XX, columns=nn), od, so, maxiter=200)
            b = float(alt.params["_post"])
            spec_rows.append(
                {
                    "spec": _tr(lang, label_zh, label_en),
                    "relative": float(np.exp(b) - 1) if _use_log(y_raw, sc) else float(b / pre_mean),
                    "p_value": float(alt.pvalues["_post"]),
                    "note": f"order={od}, seasonal={so or '-'}, log={_use_log(y_raw, sc)}",
                }
            )
        except Exception:  # noqa: BLE001 - a spec that will not converge is not reported
            return

    _variant("去掉线性趋势", "no linear trend", trend=False)
    _variant("去掉星期因子", "no day-of-week terms", seasonal_dummies=False)
    _variant(
        "原值尺度季节ARIMA",
        "seasonal ARIMA on the raw scale",
        order=(1, 1, 1),
        seasonal_order=(0, 1, 1, _WEEK),
        trend=False,
        seasonal_dummies=False,
        scale="level",
    )
    _variant("另一尺度", "other scale", scale="level" if use_log else "log")
    spec_table = pd.DataFrame(spec_rows)

    # ---- placebo interventions: how big a "step" appears where nothing happened? ----
    placebo_effects: List[float] = []
    pre_dates = work.loc[post == 0, time_col].to_numpy()
    if placebo_n > 0 and len(pre_dates) > 30:
        for frac in np.linspace(0.4, 0.85, placebo_n):
            fake = pd.Timestamp(pre_dates[int(len(pre_dates) * frac)])
            sub = work[work[time_col] < cutoff].reset_index(drop=True)
            if (sub[time_col] >= fake).sum() < 3:
                continue
            try:
                Xf, nf, _ = _design(sub, fake, time_col, outcome, trend, seasonal_dummies, cov)
                yf = np.log(sub[outcome].to_numpy(dtype=float)) if use_log else sub[outcome].to_numpy(dtype=float)
                fake_fit, _, _ = _fit_sarimax(yf, pd.DataFrame(Xf, columns=nf), used_order, used_seasonal, maxiter=120)
                placebo_effects.append(abs(float(fake_fit.params["_post"])))
            except Exception:  # noqa: BLE001 - a failed placebo is simply skipped
                continue
    placebo_p = (
        max(
            float(np.mean(np.array(placebo_effects) >= abs(beta))),
            1.0 / (len(placebo_effects) + 1),
        )
        if placebo_effects
        else float("nan")
    )

    try:
        lb = sm.stats.acorr_ljungbox(
            fit.resid, lags=[min(10, max(2, len(work) // 5))], return_df=True
        )
        ljung_p = float(lb["lb_pvalue"].iloc[0])
    except Exception:  # noqa: BLE001
        ljung_p = float("nan")

    # Per-period counterfactual for charting: the fitted path with the post shift
    # removed. Residual + beta averages to beta, so the chart is consistent with
    # the headline number instead of being a different model.
    fitted = np.asarray(fit.fittedvalues, dtype=float)
    cf = fitted - beta * post
    if use_log:
        cf_level = np.exp(cf)
        fit_level = np.exp(fitted)
    else:
        cf_level = cf
        fit_level = fitted
    frame = pd.DataFrame(
        {
            "time": work[time_col].to_numpy(),
            "actual": y_raw,
            "model_fit": fit_level,
            "counterfactual": cf_level,
            "effect": y_raw - cf_level,
            "post": post.astype(int),
        }
    )

    spread = spec_table["relative"].astype(float)
    narrative = _tr(
        lang,
        f"中断时间序列（{'对数' if use_log else '原值'}尺度，{'趋势' if trend else '无趋势'}"
        f"{'+星期因子' if seasonal_dummies else ''}，ARMA{used_order}）：干预后水平位移 "
        f"{effect:+,.0f}（相对 {relative:+.2%}，95% CI {ci[0]:+,.0f}..{ci[1]:+,.0f}，p={p_value:.2e}）；"
        f"实际均值 {post_mean:,.0f} vs 干预前 {pre_mean:,.0f}。"
        f"口径敏感性：{len(spec_table)} 种设定下相对效应 {spread.min():+.1%}..{spread.max():+.1%}；"
        f"残差 Ljung-Box p={ljung_p:.3f}（{'残差已白噪声' if ljung_p > 0.05 else '残差仍有自相关，标准误偏小'}）；"
        f"安慰剂检验 p={placebo_p:.3f}（{len(placebo_effects)} 个假干预日）。",
        f"ARIMA ITS ({'log' if use_log else 'level'} scale, {'with' if trend else 'no'} trend"
        f"{', day-of-week terms' if seasonal_dummies else ''}, ARMA{used_order}): post-period level shift "
        f"{effect:+,.0f} ({relative:+.2%} relative, 95% CI {ci[0]:+,.0f}..{ci[1]:+,.0f}, p={p_value:.2e}); "
        f"actual mean {post_mean:,.0f} vs pre-period {pre_mean:,.0f}. "
        f"Specification sensitivity: {spread.min():+.1%}..{spread.max():+.1%} across {len(spec_table)} designs; "
        f"residual Ljung-Box p={ljung_p:.3f} "
        f"({'residuals are white noise' if ljung_p > 0.05 else 'residual autocorrelation remains, SE is too small'}); "
        f"placebo p={placebo_p:.3f} ({len(placebo_effects)} fake intervention dates).",
    )

    return {
        "method": "arima_its",
        "effect": effect,
        "log_effect": beta,
        "se": se,
        "p_value": p_value,
        "ci": ci,
        "relative": relative,
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "n_pre": n_pre,
        "n_post": n_post,
        "scale": "log" if use_log else "level",
        "order": used_order,
        "seasonal_order": used_seasonal,
        "intervention": str(cutoff.date()),
        "spec_table": spec_table,
        "ljung_box_p": ljung_p,
        "placebo_p": placebo_p,
        "n_placebos": len(placebo_effects),
        "aic": float(fit.aic),
        "frame": frame,
        "residuals": pd.Series(np.asarray(fit.resid)),
        "assumption": "No concurrent shock at the intervention date; the ARMA error structure is stable across it.",
        "narrative": narrative,
    }


def _fit_structural(y_pre, X_pre, trend: bool):
    """Local level (optionally with a stochastic trend) plus regressors."""
    kwargs = dict(
        level=True,
        irregular=True,
        stochastic_level=True,
        exog=X_pre,
    )
    if trend:
        kwargs.update(trend=True, stochastic_trend=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.tsa.UnobservedComponents(y_pre, **kwargs)
        return model.fit(disp=False, maxiter=300)


def causal_impact(
    df: pd.DataFrame,
    time_col: str,
    outcome: str,
    intervention,
    controls: Sequence[str] = (),
    trend: bool = False,
    seasonal_dummies: bool = True,
    scale: str = "auto",
    placebo_n: int = 8,
    alpha: float = 0.05,
    lang: str = "zh",
) -> Dict[str, object]:
    """Structural time-series counterfactual ("causal impact").

    A local level state-space model is fitted on the pre-period with the control
    series and day-of-week dummies as regressors, then forecast over the
    post-period. The point forecast is the counterfactual; its state-space
    variance gives the interval, which grows with the horizon instead of
    pretending the counterfactual is known exactly.

    A stochastic trend is *off* by default on purpose: fitted on a trending series
    it extrapolates the trend and eats a genuine level shift (on the demo data it
    halves a +12% effect). Add ``trend=True`` only when the pre-period really is
    accelerating and the post window is short.
    """
    work = _prepare(df, time_col, outcome, extra=tuple(controls))
    cutoff = pd.Timestamp(intervention)
    control_cols = [c for c in controls if c in work.columns]

    y_raw = work[outcome].to_numpy(dtype=float)
    use_log = _use_log(y_raw, scale)
    level = y_raw if not use_log else np.log(y_raw)
    work["_y"] = level

    pre_mask = (work[time_col] < cutoff).to_numpy()
    post_mask = ~pre_mask
    n_pre, n_post = int(pre_mask.sum()), int(post_mask.sum())
    if n_pre < 20 or n_post < 3:
        raise ValueError("need at least 20 pre-period and 3 post-period observations")

    # Standardise the controls on the pre-period only: using the full series would
    # leak post-period information into the counterfactual scaling.
    exog_parts: List[np.ndarray] = []
    names: List[str] = []
    control_scaling: Dict[str, Dict[str, float]] = {}
    for col in control_cols:
        values = work[col].to_numpy(dtype=float)
        mu = float(values[pre_mask].mean())
        sd = float(values[pre_mask].std(ddof=0)) or 1.0
        exog_parts.append((values - mu) / sd)
        names.append(f"_c_{col}")
        control_scaling[col] = {"mean": mu, "sd": sd}
    if seasonal_dummies:
        matrix, dow_names = _dow_dummies(work[time_col])
        for i, name in enumerate(dow_names):
            exog_parts.append(matrix[:, i])
            names.append(name)
    X = np.column_stack(exog_parts) if exog_parts else None

    y_pre = level[pre_mask]
    X_pre = X[pre_mask] if X is not None else None
    X_post = X[post_mask] if X is not None else None

    fit = _fit_structural(y_pre, X_pre, trend)
    forecast = fit.get_forecast(steps=n_post, exog=X_post)
    predicted = np.asarray(forecast.predicted_mean, dtype=float).ravel()
    ci_array = np.asarray(forecast.conf_int(alpha=alpha), dtype=float)
    if ci_array.ndim == 1:  # defensive: some statsmodels builds squeeze it
        ci_array = ci_array.reshape(-1, 2)
    lower, upper = ci_array[:, 0], ci_array[:, 1]
    se_steps = np.asarray(forecast.se_mean, dtype=float).ravel()

    # ---- pre-period fit: one-step-ahead predictions ----
    # statsmodels returns fitted values where element 0 is the diffuse prior, so
    # the first observation has no prediction. Dropping it is the difference
    # between R2 = -5.9 (nonsense) and R2 = 0.99 on the same model.
    fitted_pre = np.asarray(fit.fittedvalues, dtype=float).ravel()
    y_fit = y_pre[1:]
    pred_fit = fitted_pre[1:]
    ss_res = float(np.sum((y_fit - pred_fit) ** 2))
    ss_tot = float(np.sum((y_fit - y_fit.mean()) ** 2)) or 1.0
    pre_r2 = 1 - ss_res / ss_tot

    # Back-transform to level units with a smearing factor estimated on the
    # pre-period, which removes both the Jensen gap and any structural offset.
    if use_log:
        smear = float(np.mean(y_raw[pre_mask][1:]) / max(np.mean(np.exp(pred_fit)), 1e-12))
        cf_level = smear * np.exp(predicted)
        lo_level, hi_level = smear * np.exp(lower), smear * np.exp(upper)
        fit_level = smear * np.exp(fitted_pre)
        log_effect = level[post_mask] - predicted
        relative = float(np.exp(log_effect.mean()) - 1)
    else:
        cf_level = predicted
        lo_level, hi_level = lower, upper
        fit_level = fitted_pre
        log_effect = level[post_mask] - predicted
        relative = (
            float(log_effect.mean() / predicted.mean()) if predicted.mean() else float("nan")
        )
    # Element 0 of the fitted path is the diffuse prior, not a prediction (same
    # reason the R2 diagnostic drops it). Plotted as a number it draws a spike
    # from zero, so it is masked out and the line simply starts at day 2.
    fit_level[0] = np.nan

    actual = y_raw[post_mask]
    effect = actual - cf_level
    avg_effect = float(effect.mean())
    cumulative = float(effect.sum())
    pre_rmse = float(np.sqrt(np.mean((y_raw[pre_mask][1:] - fit_level[1:]) ** 2)))

    se_mean = float(se_steps.mean())
    p_value = (
        float(2 * (1 - stats.norm.cdf(abs(log_effect.mean() / se_mean))))
        if se_mean > 0
        else float("nan")
    )

    # ---- placebo windows in the pre-period ----
    # The state-space interval assumes the model is right; a placebo distribution
    # asks a different question - how often does a window this long, elsewhere in
    # the same series, look like this "effect"?
    placebo_effects: List[float] = []
    window = int(min(n_post, 21))
    if placebo_n > 0 and n_pre - window >= 20:
        for k in np.linspace(20, n_pre - window, placebo_n).astype(int):
            fake_cut = pd.Timestamp(work[time_col].iloc[int(k)])
            sub_mask = np.zeros(len(work), dtype=bool)
            sub_mask[: int(k)] = True
            win_mask = np.zeros(len(work), dtype=bool)
            win_mask[int(k) : int(k) + window] = True
            try:
                fit_f = _fit_structural(
                    level[sub_mask],
                    X[sub_mask] if X is not None else None,
                    trend,
                )
                fc_f = fit_f.get_forecast(
                    steps=window, exog=X[win_mask] if X is not None else None
                )
                pred_f = np.asarray(fc_f.predicted_mean, dtype=float).ravel()
                placebo_effects.append(float(np.mean(level[win_mask] - pred_f)))
            except Exception:  # noqa: BLE001 - a failed placebo is simply skipped
                continue
        del fake_cut
    scale_ref = abs(float(log_effect[:window].mean())) or 1e-12
    placebo_p = (
        max(
            float(np.mean(np.abs(placebo_effects) >= scale_ref)),
            1.0 / (len(placebo_effects) + 1),
        )
        if placebo_effects
        else float("nan")
    )

    frame = pd.DataFrame(
        {
            "time": work.loc[post_mask, time_col].to_numpy(),
            "actual": actual,
            "counterfactual": cf_level,
            "lower": lo_level,
            "upper": hi_level,
            "effect": effect,
            "cumulative_effect": np.cumsum(effect),
        }
    )
    history = pd.DataFrame(
        {
            "time": work.loc[pre_mask, time_col].to_numpy(),
            "actual": y_raw[pre_mask],
            "counterfactual": fit_level,
            "lower": np.nan,
            "upper": np.nan,
            "effect": np.nan,
            "cumulative_effect": np.nan,
        }
    )

    caveat = _tr(
        lang,
        "" if control_cols else "注意：无控制序列时，干预日附近的一次持续性冲击与真实效应观测量上不可区分，请优先补齐对照序列。",
        "" if control_cols else " Caveat: without control series a persistent shock near the cutoff is observationally equivalent to the effect - add controls before trusting this.",
    )
    narrative = _tr(
        lang,
        f"时序因果（结构时间序列，{'对数' if use_log else '原值'}尺度，"
        f"{'含' if control_cols else '无'}控制序列）：干预后平均效应 {avg_effect:+,.0f}"
        f"（相对 {relative:+.2%}，模型 p={p_value:.2e}，安慰剂 p={placebo_p:.3f}），累计 {cumulative:+,.0f}；"
        f"干预前一步拟合 R²={pre_r2:.3f}、RMSE={pre_rmse:,.0f}。{caveat}",
        f"Causal impact (structural model, {'log' if use_log else 'level'} scale, "
        f"{'with' if control_cols else 'without'} controls): post-period average effect "
        f"{avg_effect:+,.0f} ({relative:+.2%}, model p={p_value:.2e}, placebo p={placebo_p:.3f}), "
        f"cumulative {cumulative:+,.0f}; pre-period one-step fit R²={pre_r2:.3f}, RMSE={pre_rmse:,.0f}.{caveat}",
    )

    return {
        "method": "causal_impact",
        "effect": avg_effect,
        "cumulative_effect": cumulative,
        "relative": relative,
        "se": se_mean,
        "p_value": p_value,
        "placebo_p": placebo_p,
        "n_placebos": len(placebo_effects),
        "ci": (float(avg_effect - 1.96 * se_mean * (cf_level.mean() if use_log else 1.0)),
               float(avg_effect + 1.96 * se_mean * (cf_level.mean() if use_log else 1.0))),
        "pre_fit_r2": float(pre_r2),
        "pre_rmse": pre_rmse,
        "n_pre": n_pre,
        "n_post": n_post,
        "controls": control_cols,
        "n_controls": len(control_cols),
        "scale": "log" if use_log else "level",
        "stochastic_trend": bool(trend),
        "intervention": str(cutoff.date()),
        "frame": frame,
        "history": history,
        "assumption": "The control series and the local level model span the counterfactual; no other shock at the cutoff.",
        "narrative": narrative,
    }


def compare_ts_methods(
    df: pd.DataFrame,
    time_col: str,
    outcome: str,
    intervention,
    controls: Sequence[str] = (),
    lang: str = "zh",
) -> Dict[str, object]:
    """Run the three readings of the same intervention side by side.

    The naive pre/post difference is included on purpose: it is what a dashboard
    shows, it ignores the trend, and the gap between it and the model-based
    estimates is usually the most persuasive number in the room.
    """
    work = _prepare(df, time_col, outcome, extra=tuple(controls))
    cutoff = pd.Timestamp(intervention)
    pre = work[work[time_col] < cutoff][outcome]
    post = work[work[time_col] >= cutoff][outcome]
    naive = float(post.mean() - pre.mean())
    naive_relative = naive / float(pre.mean()) if pre.mean() else float("nan")

    its = arima_intervention(work, time_col, outcome, intervention, covariates=controls, lang=lang)
    impact = causal_impact(work, time_col, outcome, intervention, controls=controls, lang=lang)

    rows = [
        {
            "method": _tr(lang, "前后均值差（未校正）", "naive pre/post difference"),
            "effect": naive,
            "relative": naive_relative,
            "p_value": float("nan"),
        },
        {
            "method": _tr(lang, "ARIMA 中断时间序列", "ARIMA interrupted time series"),
            "effect": its["effect"],
            "relative": its["relative"],
            "p_value": its["p_value"],
        },
        {
            "method": _tr(lang, "时序因果（结构模型）", "causal impact (structural)"),
            "effect": impact["effect"],
            "relative": impact["relative"],
            "p_value": impact["p_value"],
        },
    ]
    table = pd.DataFrame(rows)
    agree = bool(
        np.sign(its["relative"]) == np.sign(impact["relative"])
        and abs(its["relative"] - impact["relative"]) < 0.5 * max(abs(impact["relative"]), 1e-9)
    )
    return {
        "method": "ts_comparison",
        "table": table,
        "naive": {"effect": naive, "relative": naive_relative},
        "arima_its": its,
        "causal_impact": impact,
        "agree": agree,
        "narrative": _tr(
            lang,
            f"三种口径：未校正前后差 {naive:+,.0f}（{naive_relative:+.1%}），ARIMA ITS {its['effect']:+,.0f}"
            f"（{its['relative']:+.1%}），结构时序 {impact['effect']:+,.0f}（{impact['relative']:+.1%}）；"
            f"两个模型{'一致' if agree else '不一致，需要检查控制序列与干预日'}。"
            f"未校正口径把趋势算作效应，通常明显高估。",
            f"Three readings: naive {naive:+,.0f} ({naive_relative:+.1%}), ARIMA ITS {its['effect']:+,.0f} "
            f"({its['relative']:+.1%}), structural {impact['effect']:+,.0f} ({impact['relative']:+.1%}); "
            f"the two models {'agree' if agree else 'disagree - inspect the controls and the intervention date'}. "
            f"The naive reading bills the underlying trend to the intervention, so it overstates by construction.",
        ),
    }
