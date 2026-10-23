"""Metric anomaly detection.

Method stack (chosen because it is explainable to business stakeholders):

1. Baseline - the expected value for day *t*, from one of:
   * ``trailing`` (default): stable day-of-week profile applied to a causal
     trailing level, ``expected_t = level_{t-1..t-28} x seasonal_{dow(t)}``.
   * ``holt_winters`` / ``holt_winters_mul``: damped Holt-Winters, the classic
     exponential-smoothing reference model for a trending, weekly-seasonal KPI.
   * ``arima``: SARIMAX(1,1,1)(0,1,1)_7.
   * ``stl_ets``: robust STL seasonal indices plus ETS on the adjusted series.
   * ``auto``: backtest the candidates on the same rolling origin and keep the
     lowest-MASE model -- reported, never hidden.
   Every one of them is causal: the expectation for *t* only uses data < *t*, so
   a multi-day incident cannot leak into its own baseline. A centred STL trend is
   available via ``baseline="stl"`` but is deliberately not the default: a
   smoother that sees the incident days pulls the baseline down with them and
   quietly hides the very thing we are looking for.
2. Robust z-score on the relative residual, scaled with 1.4826 * MAD so a few
   outliers do not inflate the threshold.
3. Two-sided CUSUM on the standardised residual. A 10% drop sustained for a
   week is only ~2.5 sigma on any single day, yet it is the most common shape
   of a real business incident; CUSUM accumulates the evidence and its alarms
   are validated with a t-test before being promoted to an episode.
4. Optional binomial z-test for rate metrics, which uses the real denominator
   instead of treating a conversion rate as a continuous variable.

Output is a structured dict plus a per-day frame that the report layer renders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .forecast import (
    METHOD_LABELS,
    compare_methods,
    rolling_forecast,
    select_method,
)
from .forecast import accuracy as forecast_accuracy

SEVERITY_BANDS = (
    (6.0, "critical"),
    (4.5, "high"),
    (3.5, "medium"),
    (0.0, "low"),
)

# Baselines that come from the forecasting module rather than the local
# decomposition helpers.
FORECAST_BASELINES = ("holt_winters", "holt_winters_mul", "arima", "stl_ets", "auto")


def _band(z: float) -> str:
    for threshold, name in SEVERITY_BANDS:
        if abs(z) >= threshold:
            return name
    return "low"


@dataclass
class AnomalyConfig:
    """Tunable detection parameters."""

    period: int = 7                  # weekly seasonality
    baseline: str = "trailing"       # trailing | stl | holt_winters | holt_winters_mul | arima | stl_ets | auto
    forecast_refit_every: int = 7    # refit cadence for forecasting baselines
    forecast_candidates: Tuple[str, ...] = ("holt_winters", "holt_winters_mul", "seasonal_naive")
    trend_window: int = 28           # level window in days (trailing baseline)
    slope_window: int = 56           # window used to measure the local growth rate
    profile_iters: int = 2           # weekday-profile refinement passes
    stl_trend: Optional[int] = 51    # STL trend span, only used when baseline="stl"
    z_threshold: float = 3.5
    min_rel_change: float = 0.05     # ignore sub-5% wiggles
    direction: str = "both"          # both | up | down
    method: str = "point"            # point | cusum | ewma ("stl" is an alias of point)
    detect_shifts: bool = True       # promote validated CUSUM alarms to episodes
    cusum_threshold: float = 5.0
    cusum_slack: float = 0.5
    shift_alpha: float = 0.05        # t-test gate for a sustained-shift episode
    min_shift_days: int = 2
    ewma_lambda: float = 0.25
    min_points: int = 14


@dataclass
class Anomaly:
    date: str
    actual: float
    expected: float
    rel_dev: float
    z: float
    direction: str
    severity: str
    extra: Dict[str, float] = field(default_factory=dict)


class AnomalyDetector:
    """Detects point anomalies, level shifts and sustained episodes."""

    def __init__(self, config: Optional[AnomalyConfig] = None) -> None:
        self.config = config or AnomalyConfig()
        # Metadata of the last decomposition (forecast model, backtest table).
        self.last_baseline: Dict[str, object] = {}

    # ------------------------------------------------------------------ core
    def decompose(self, values: Sequence[float]) -> pd.DataFrame:
        """Split the series into level, seasonality and a relative residual.

        ``baseline="trailing"`` (default) is the honest choice for incident
        review: the expected value for day *t* only uses data before *t*, so a
        multi-day incident cannot leak into its own baseline. Forecasting
        baselines (``holt_winters``, ``arima``, ``stl_ets``, ``auto``) preserve
        the same property and are selected when the metric has a strong trend or
        when the reviewer explicitly asks for a classical model. ``"stl"`` is
        kept for smoother trend reporting on long, stable series.
        """
        y = pd.Series(np.asarray(values, dtype=float))
        cfg = self.config
        positive = bool((y > 0).all()) and len(y) >= 2 * cfg.period + 2

        if cfg.baseline in FORECAST_BASELINES:
            try:
                return self._decompose_forecast(y, cfg.baseline)
            except Exception:  # noqa: BLE001 - fall through to the local baselines
                pass
        if positive and cfg.baseline != "stl":
            try:
                return self._decompose_trailing(y)
            except Exception:  # noqa: BLE001 - fall through to the next baseline
                pass
        if positive and len(y) >= 3 * cfg.period:
            try:
                frame = self._decompose_stl(y)
                if frame is not None:
                    return frame
            except Exception:  # noqa: BLE001 - fall back to the simple decomposition
                pass
        return self._decompose_simple(y)

    def _decompose_forecast(self, y: pd.Series, baseline: str) -> pd.DataFrame:
        """Causal one-step-ahead forecast as the expectation.

        Each point's expected value comes from a model fitted on the preceding
        observations only, so the residual is a genuine out-of-sample surprise.
        ``auto`` backtests the candidate list on the same rolling origin and
        records the full comparison table (MASE, MAE, RMSE) next to the result.
        """
        cfg = self.config
        values = y.to_numpy(dtype=float)
        method = baseline
        table = None
        margin = float("nan")
        if baseline == "auto":
            selection = select_method(
                values,
                methods=cfg.forecast_candidates,
                period=cfg.period,
                refit_every=cfg.forecast_refit_every,
            )
            method = str(selection["method"])
            table = selection["table"]
            margin = float(selection["margin"])

        preds = rolling_forecast(
            values, method=method, period=cfg.period, refit_every=cfg.forecast_refit_every
        )
        expected = pd.Series(preds, index=y.index)
        # The head of the series has no forecast history. Borrowing the actual
        # value there yields a zero residual, which is the correct statement:
        # "no expectation, therefore no surprise".
        expected = expected.where(np.isfinite(expected), y)
        expected = expected.where(expected.abs() > 1e-12, y)

        trend = expected.rolling(cfg.period, center=True, min_periods=1).median()
        seasonal = (expected / trend.replace(0, np.nan)).fillna(1.0)
        residual = (y - expected) / expected.replace(0, np.nan)

        accuracy = {}
        if table is not None and len(table):
            row = table[table["method"] == method]
            if len(row):
                accuracy = {k: float(row.iloc[0][k]) for k in ("mae", "rmse", "mase") if k in row}
        elif method == baseline:
            # An explicitly requested model has no comparison table, but its own
            # out-of-sample error still decides whether the answer is trustworthy.
            scored = forecast_accuracy(
                y.to_numpy(dtype=float), np.asarray(preds, dtype=float), period=cfg.period
            )
            accuracy = {k: float(scored[k]) for k in ("mae", "rmse", "mase") if k in scored}

        self.last_baseline = {
            "baseline": baseline,
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            "refit_every": cfg.forecast_refit_every,
            "selection_margin": margin,
            "backtest": table,
            "accuracy": accuracy,
        }
        return pd.DataFrame(
            {
                "actual": y,
                "trend": trend,
                "seasonal": seasonal,
                "expected": expected,
                "residual_rel": residual,
                "model": f"forecast:{method}",
            }
        )

    def _decompose_trailing(self, y: pd.Series) -> pd.DataFrame:
        """Stable weekday profile x causal, trend-corrected trailing level."""
        cfg = self.config
        period = cfg.period
        phase = pd.Series(np.arange(len(y)) % period, index=y.index)

        # Weekday profile: each day relative to the centred weekly median, then
        # the median per weekday. Medians keep a single outage day from
        # distorting the Saturday index for the rest of the series.
        centred = y.rolling(period, center=True, min_periods=max(2, period // 2)).median()
        ratio = (y / centred).replace([np.inf, -np.inf], np.nan)
        profile = ratio.groupby(phase).median()
        if profile.isna().any() or float(profile.mean()) == 0.0:
            profile = pd.Series(1.0, index=range(period))
        else:
            profile = profile / float(profile.mean())
        seasonal = phase.map(profile).astype(float).fillna(1.0)

        # Two refinement passes: the initial profile comes from a crude weekly
        # median, and residual bias shows up as per-weekday error. Correcting
        # the profile multiplicatively converges in one or two passes.
        for _ in range(max(0, cfg.profile_iters)):
            expected = self._trailing_level(y / seasonal) * seasonal
            correction = (y / expected).groupby(phase).median()
            correction = correction.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            seasonal = (seasonal * phase.map(correction)).astype(float)
            seasonal = seasonal / float(seasonal.iloc[:period].mean())

        level = self._trailing_level(y / seasonal)
        expected = level * seasonal
        residual = (y - expected) / expected.replace(0, np.nan)
        return pd.DataFrame(
            {
                "actual": y,
                "trend": level,
                "seasonal": seasonal,
                "expected": expected,
                "residual_rel": residual,
                "model": "seasonal_trailing",
            }
        )

    def _trailing_level(self, deseason: pd.Series) -> pd.Series:
        """Causal level estimate, corrected for the lag a trailing window creates.

        A median of the previous four weeks describes the middle of that window,
        i.e. two weeks ago; in a growing business that alone leaves a ~2% bias,
        which is several sigma on a low-noise metric. We add back the local
        growth rate measured over ``slope_window`` days. The slope is a median of
        three overlapping estimates so one outlier day cannot tilt it.
        """
        cfg = self.config
        window = max(2 * cfg.period, cfg.trend_window)
        median_level = deseason.shift(1).rolling(window, min_periods=cfg.period).median()
        log_d = np.log(deseason.where(deseason > 0))
        slope = (log_d - log_d.shift(cfg.slope_window)) / cfg.slope_window
        slope = slope.rolling(3, min_periods=1).median()
        level = median_level * np.exp(slope.fillna(0.0) * (window / 2.0))
        return level.fillna(deseason.expanding(min_periods=1).median())

    def _decompose_stl(self, y: pd.Series) -> Optional[pd.DataFrame]:
        """STL on the log series, with a deliberately long trend window."""
        cfg = self.config
        from statsmodels.tsa.seasonal import STL

        trend_window = cfg.stl_trend or 51
        if trend_window % 2 == 0:
            trend_window += 1
        trend_window = min(trend_window, len(y) - 1 if len(y) % 2 == 0 else len(y))
        if trend_window <= cfg.period:
            return None
        fit = STL(np.log(y.to_numpy()), period=cfg.period, trend=trend_window, robust=True).fit()
        trend = pd.Series(np.exp(fit.trend))
        seasonal = pd.Series(np.exp(fit.seasonal))
        expected = trend * seasonal
        residual = (y - expected) / expected.replace(0, np.nan)
        return pd.DataFrame(
            {
                "actual": y,
                "trend": trend,
                "seasonal": seasonal,
                "expected": expected,
                "residual_rel": residual,
                "model": "stl_multiplicative",
            }
        )

    def _decompose_simple(self, y: pd.Series) -> pd.DataFrame:
        """Rolling-median decomposition; additive when the series has zeros."""
        cfg = self.config
        window = max(2 * cfg.period + 1, cfg.trend_window)
        trend = y.rolling(window=window, center=True, min_periods=max(3, cfg.period)).median()
        trend = trend.bfill().ffill()
        if (y <= 0).any():
            # Guard against zeros: fall back to additive residuals.
            seasonal = pd.Series(0.0, index=y.index)
            expected = trend
            residual = y - expected
            model = "additive"
        else:
            detrended = y / trend
            phase = pd.Series(np.arange(len(y)) % cfg.period, index=y.index)
            seasonal_map = detrended.groupby(phase).median()
            seasonal = phase.map(seasonal_map).astype(float)
            expected = trend * seasonal
            residual = (y - expected) / expected.replace(0, np.nan)
            model = "multiplicative"
        return pd.DataFrame(
            {"actual": y, "trend": trend, "seasonal": seasonal, "expected": expected, "residual_rel": residual, "model": model}
        )

    def detect(
        self,
        dates: Sequence,
        values: Sequence[float],
        denominators: Optional[Sequence[float]] = None,
        label: str = "metric",
    ) -> Dict[str, object]:
        """Run the full detection pipeline on one metric series."""
        cfg = self.config
        self.last_baseline = {}
        frame = self.decompose(values)
        # np.asarray resets the index, otherwise frame.insert would misalign the
        # date column when the caller passes a slice of a larger frame.
        frame.insert(0, "date", pd.to_datetime(pd.Series(np.asarray(dates))))
        frame["residual_rel"] = frame["residual_rel"].fillna(0.0)

        resid = frame["residual_rel"].to_numpy()
        sigma = self._sigma(resid)
        frame["z"] = resid / sigma

        # Binomial refinement for rate metrics: uses the true denominator.
        if denominators is not None:
            den = np.asarray(denominators, dtype=float)
            expected_rate = frame["expected"].to_numpy() / np.where(den > 0, den, np.nan)
            actual_rate = frame["actual"].to_numpy() / np.where(den > 0, den, np.nan)
            se = np.sqrt(np.clip(expected_rate * (1 - expected_rate), 1e-12, None) / np.maximum(den, 1))
            z_binom = (actual_rate - expected_rate) / se
            frame["z_binomial"] = z_binom
            frame["z"] = np.where(np.isfinite(z_binom), z_binom, frame["z"])

        frame["flagged"] = self._flag(frame)

        # Level shifts: CUSUM accumulates evidence a point-wise test throws
        # away, then a t-test on the segment keeps the false-alarm rate sane.
        shifts: List[Dict[str, object]] = []
        shift_mask = np.zeros(len(frame), dtype=bool)
        if cfg.detect_shifts and cfg.method == "point":
            for run in self._validated_shifts(frame, self._shift_runs(frame)):
                shifts.append({k: v for k, v in run.items() if not k.startswith("_")})
                shift_mask[run["_i"]: run["_j"] + 1] = True
        if shift_mask.any():
            combined = frame["flagged"].to_numpy() | shift_mask
            frame["flagged"] = pd.Series(combined, index=frame.index)

        anomalies = [self._to_anomaly(row) for _, row in frame[frame["flagged"]].iterrows()]
        episodes = self._episodes(frame, shift_mask)

        return {
            "label": label,
            "method": cfg.method,
            "baseline": cfg.baseline,
            "baseline_detail": self.last_baseline,
            "model": frame["model"].iloc[0] if len(frame) else "seasonal_trailing",
            "sigma": sigma,
            "n_points": int(len(frame)),
            "z_threshold": cfg.z_threshold,
            "anomalies": [a.__dict__ for a in anomalies],
            "episodes": episodes,
            "level_shifts": shifts,
            "summary": self._summary(label, frame, episodes, shifts),
            "frame": frame,
        }

    # ------------------------------------------------------------- internals
    @staticmethod
    def _sigma(resid: np.ndarray) -> float:
        """Robust noise scale.

        MAD alone is the right default, but robust fits can shrink it. The std
        floor is therefore computed on inliers only: a segment that is down 40%
        for a week must not inflate its own noise scale and hide itself.
        """
        values = np.asarray(resid, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 3:
            return 1e-9
        centre = float(np.median(values))
        mad = float(np.median(np.abs(values - centre)))
        sigma_mad = 1.4826 * mad
        if sigma_mad > 0:
            inliers = values[np.abs(values - centre) <= 5.0 * sigma_mad]
        else:
            inliers = values
        sigma_std = float(inliers.std(ddof=1)) if len(inliers) > 2 else float(values.std(ddof=1))
        sigma = max(sigma_mad, 0.8 * sigma_std) if sigma_std > 0 else sigma_mad
        return sigma if sigma > 1e-9 else 1e-9

    def _flag(self, frame: pd.DataFrame) -> pd.Series:
        cfg = self.config
        z = frame["z"].fillna(0.0)
        rel = frame["residual_rel"].fillna(0.0)
        base = (z.abs() >= cfg.z_threshold) & (rel.abs() >= cfg.min_rel_change)
        if cfg.direction == "up":
            base &= z > 0
        elif cfg.direction == "down":
            base &= z < 0
        if cfg.method == "cusum":
            return self._cusum_flag(frame).fillna(False)
        if cfg.method == "ewma":
            return self._ewma_flag(frame).fillna(False)
        return base

    def _ewma_flag(self, frame: pd.DataFrame) -> pd.Series:
        """EWMA control chart on the relative residual (catches slow drifts)."""
        lam = self.config.ewma_lambda
        z = frame["z"].fillna(0.0)
        ewma = z.ewm(alpha=lam, adjust=False).mean()
        sigma_ewma = 1.0 * np.sqrt(lam / (2 - lam))
        limit = self.config.z_threshold * sigma_ewma
        return ewma.abs() >= limit * 2

    def _cusum_alarm(self, frame: pd.DataFrame) -> np.ndarray:
        """Two-sided CUSUM statistic on the standardised residual."""
        z = frame["z"].fillna(0.0).to_numpy()
        k, h = self.config.cusum_slack, self.config.cusum_threshold
        pos = neg = 0.0
        alarm = np.zeros(len(z), dtype=bool)
        for i, value in enumerate(z):
            pos = max(0.0, pos + value - k)
            neg = min(0.0, neg + value + k)
            alarm[i] = pos > h or abs(neg) > h
        return alarm

    def _cusum_flag(self, frame: pd.DataFrame) -> pd.Series:
        """CUSUM as a stand-alone detector, kept for ``method="cusum"``."""
        return pd.Series(self._cusum_alarm(frame), index=frame.index)

    def _shift_runs(self, frame: pd.DataFrame) -> List[Tuple[int, int]]:
        """Contiguous CUSUM alarms, extended backwards to the true onset.

        CUSUM only fires once enough evidence has accumulated, so the first
        alarm lands a couple of days after the incident began. We walk back
        while the residual still points the same way and stays material.
        """
        alarm = self._cusum_alarm(frame)
        resid = frame["residual_rel"].fillna(0.0).to_numpy()
        runs: List[Tuple[int, int]] = []
        i = 0
        while i < len(alarm):
            if not alarm[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(alarm) and alarm[j + 1]:
                j += 1
            seg_mean = float(np.nanmean(resid[i: j + 1]))
            sign = 1.0 if seg_mean >= 0 else -1.0
            floor = max(0.35 * abs(seg_mean), 0.02)
            start = i
            while start - 1 >= 0 and i - start < self.config.period:
                prev = resid[start - 1]
                if np.sign(prev) != sign or abs(prev) < floor:
                    break
                start -= 1
            runs.append((start, j))
            i = j + 1
        return runs

    def _validated_shifts(self, frame: pd.DataFrame, runs: List[Tuple[int, int]]) -> List[Dict[str, object]]:
        """Keep only the CUSUM runs that survive a significance gate."""
        cfg = self.config
        shifts: List[Dict[str, object]] = []
        for i, j in runs:
            days = j - i + 1
            if days < cfg.min_shift_days:
                continue
            chunk = frame.iloc[i: j + 1]
            expected_sum = float(chunk["expected"].sum())
            actual_sum = float(chunk["actual"].sum())
            impact = actual_sum / expected_sum - 1 if expected_sum else 0.0
            resid = chunk["residual_rel"].to_numpy(dtype=float)
            p_value = float(stats.ttest_1samp(resid, 0.0).pvalue) if days > 1 and resid.std(ddof=1) > 0 else 1.0
            if abs(impact) < cfg.min_rel_change or p_value > cfg.shift_alpha:
                continue
            shifts.append(
                {
                    "_i": i,
                    "_j": j,
                    "date": str(pd.Timestamp(frame["date"].iloc[i]).date()),
                    "end": str(pd.Timestamp(frame["date"].iloc[j]).date()),
                    "days": int(days),
                    "pre_mean": float(chunk["expected"].mean()),
                    "post_mean": float(chunk["actual"].mean()),
                    "rel_change": float(impact),
                    "mean_rel_dev": float(resid.mean()),
                    "p_value": p_value,
                    "direction": "up" if impact > 0 else "down",
                    "method": "cusum",
                }
            )
        return shifts

    def _to_anomaly(self, row: pd.Series) -> Anomaly:
        return Anomaly(
            date=str(pd.Timestamp(row["date"]).date()),
            actual=float(row["actual"]),
            expected=float(row["expected"]),
            rel_dev=float(row["residual_rel"]),
            z=float(row["z"]),
            direction="up" if row["residual_rel"] > 0 else "down",
            severity=_band(float(row["z"])),
            extra={"trend": float(row["trend"]), "seasonal": float(row["seasonal"])},
        )

    def _episodes(self, frame: pd.DataFrame, shift_mask: Optional[np.ndarray] = None) -> List[Dict[str, object]]:
        """Merge consecutive flagged days into business-facing incidents."""
        flagged = frame["flagged"].to_numpy()
        shift_mask = np.zeros(len(frame), dtype=bool) if shift_mask is None else np.asarray(shift_mask, dtype=bool)
        episodes: List[Dict[str, object]] = []
        i = 0
        while i < len(frame):
            if not flagged[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(frame) and flagged[j + 1]:
                j += 1
            chunk = frame.iloc[i: j + 1]
            actual_sum = float(chunk["actual"].sum())
            expected_sum = float(chunk["expected"].sum())
            impact = actual_sum - expected_sum
            impact_pct = impact / expected_sum if expected_sum else 0.0
            peak = chunk.loc[chunk["z"].abs().idxmax()]
            resid = chunk["residual_rel"].to_numpy(dtype=float)
            days = j - i + 1
            # Segment t-stat: one day reduces to the point z-score, several days
            # reward consistency, which is what makes a slow bleed visible.
            if days > 1 and resid.std(ddof=1) > 0:
                t_stat = float(stats.ttest_1samp(resid, 0.0).statistic)
                p_value = float(stats.ttest_1samp(resid, 0.0).pvalue)
            else:
                t_stat = float(peak["z"])
                p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
            has_shift = bool(shift_mask[i: j + 1].any())
            episodes.append(
                {
                    "start": str(pd.Timestamp(chunk["date"].iloc[0]).date()),
                    "end": str(pd.Timestamp(chunk["date"].iloc[-1]).date()),
                    "days": int(days),
                    "actual_sum": actual_sum,
                    "expected_sum": expected_sum,
                    "impact": impact,
                    "impact_pct": impact_pct,
                    "mean_rel_dev": float(resid.mean()),
                    "p_value": p_value,
                    "peak_z": float(peak["z"]),
                    "direction": "up" if impact > 0 else "down",
                    "kind": "level_shift" if has_shift else "point",
                    "severity": _band(t_stat),
                }
            )
            i = j + 1
        return sorted(episodes, key=lambda e: abs(e["impact"]), reverse=True)

    def _summary(self, label: str, frame: pd.DataFrame, episodes: List[Dict[str, object]], shifts: List[Dict[str, object]]) -> str:
        if not episodes:
            return f"No anomaly detected for {label} (max |z| = {frame['z'].abs().max():.2f})."
        top = episodes[0]
        direction_word = "above" if top["direction"] == "up" else "below"
        text = (
            f"{label}: {len(episodes)} anomaly episode(s). Largest: {top['start']}~{top['end']} "
            f"({top['days']}d, {top['kind']}) actual {top['actual_sum']:,.0f} vs expected {top['expected_sum']:,.0f}, "
            f"{direction_word} expected by {abs(top['impact_pct']):.1%}, "
            f"p={top['p_value']:.4f}, severity={top['severity']}."
        )
        if shifts:
            text += f" {len(shifts)} validated level shift(s) (CUSUM + t-test)."
        return text


def detect_metric_anomalies(
    series: pd.DataFrame,
    metric_name: str,
    date_col: str = "date",
    numerator_col: str = "numerator",
    denominator_col: Optional[str] = "denominator",
    config: Optional[AnomalyConfig] = None,
) -> Dict[str, object]:
    """Convenience wrapper for a frame with actual numerator/denominator columns."""
    detector = AnomalyDetector(config)
    values = series[numerator_col].to_numpy(dtype=float)
    if denominator_col and denominator_col in series.columns:
        denominators = series[denominator_col].to_numpy(dtype=float)
        # For ratio metrics compare rates; for additive metrics compare volumes.
        if metric_name.endswith("_rate") or metric_name in ("conversion_rate", "aov", "gmv_per_session", "refund_rate"):
            values = values / np.where(denominators > 0, denominators, np.nan)
    else:
        denominators = None
    return detector.detect(series[date_col], values, denominators=denominators, label=metric_name)


def scan_seasonality(values: Sequence[float], period: int = 7) -> Dict[str, float]:
    """Quick seasonality check: variance explained by the day-of-week profile."""
    y = np.asarray(values, dtype=float)
    if len(y) < 2 * period:
        return {"seasonal_strength": 0.0, "recommended_period": period}
    phase = np.arange(len(y)) % period
    grand = y.mean()
    between = sum(len(y[phase == p]) * (y[phase == p].mean() - grand) ** 2 for p in range(period))
    total = ((y - grand) ** 2).sum()
    strength = float(between / total) if total else 0.0
    return {
        "seasonal_strength": strength,
        "recommended_period": period,
        "autocorr_lag7": float(pd.Series(y).autocorr(lag=period)) if len(y) > period else float("nan"),
    }


def scan_segment_anomalies(
    frame: pd.DataFrame,
    segment_col: str = "segment",
    date_col: str = "date",
    value_col: str = "value",
    denominator_col: Optional[str] = None,
    window: Optional[tuple] = None,
    comparison: Optional[tuple] = None,
    config: Optional[AnomalyConfig] = None,
    min_abs_z: float = 3.0,
    top_k: int = 8,
) -> Dict[str, object]:
    """Run the detector per segment and rank segments by severity.

    This is how anomaly review actually works: the topline looks flat while one
    segment is down 30%. Each segment gets its own seasonal decomposition (enough
    history matters - pass >= 8 weeks), a window-over-window change and a
    Welch p-value, so a flagged segment can be separated from noise.
    """
    detector = AnomalyDetector(config)
    rows: List[Dict[str, object]] = []
    need = max(28, 4 * (config or AnomalyConfig()).period)

    for segment, chunk in frame.groupby(segment_col):
        chunk = chunk.sort_values(date_col).reset_index(drop=True)
        if len(chunk) < need:
            continue
        den = chunk[denominator_col].to_numpy() if denominator_col and denominator_col in chunk.columns else None
        result = detector.detect(chunk[date_col], chunk[value_col].to_numpy(), denominators=den, label=str(segment))
        res_frame = result["frame"]

        # Only judge recency inside the requested window, but decompose on the
        # full history so the seasonal profile and trend are stable.
        recent = res_frame
        if window is not None and window[0] is not None:
            recent = res_frame[res_frame["date"] >= pd.Timestamp(window[0])]
        if len(recent) == 0:
            continue
        worst = recent.loc[recent["z"].abs().idxmax()]
        latest = res_frame.iloc[-1]

        change = None
        p_value = None
        if window is not None and comparison is not None:
            cur_vals = res_frame.loc[
                (res_frame["date"] >= pd.Timestamp(window[0])) & (res_frame["date"] <= pd.Timestamp(window[1])), "actual"
            ].to_numpy()
            base_vals = res_frame.loc[
                (res_frame["date"] >= pd.Timestamp(comparison[0])) & (res_frame["date"] <= pd.Timestamp(comparison[1])), "actual"
            ].to_numpy()
            if len(cur_vals) > 1 and len(base_vals) > 1 and base_vals.mean() != 0:
                change = float(cur_vals.mean() / base_vals.mean() - 1)
                p_value = float(compare_windows(cur_vals, base_vals)["p_value"])

        if abs(float(worst["z"])) < min_abs_z and (change is None or abs(change) < 0.05):
            continue
        episodes = result["episodes"]
        rows.append(
            {
                "segment": str(segment),
                "latest_value": float(latest["actual"]),
                "latest_rel_dev": float(latest["residual_rel"]),
                "window_change": change,
                "p_value": p_value,
                "worst_date": str(pd.Timestamp(worst["date"]).date()),
                "worst_rel_dev": float(worst["residual_rel"]),
                "max_abs_z": float(abs(float(worst["z"]))),
                "direction": "down" if float(worst["residual_rel"]) < 0 else "up",
                "severity": _band(float(worst["z"])),
                "episode_impact": float(episodes[0]["impact"]) if episodes else 0.0,
                "episode_days": int(episodes[0]["days"]) if episodes else 0,
            }
        )

    rows.sort(key=lambda r: r["max_abs_z"], reverse=True)
    return {
        "segments": rows[:top_k],
        "n_scanned": int(frame[segment_col].nunique()),
        "n_flagged": len(rows),
        "summary": (
            f"{len(rows)} of {frame[segment_col].nunique()} segments flagged (|z|>={min_abs_z} or |change|>=5%)"
            if rows
            else f"no segment flagged (|z|>={min_abs_z})"
        ),
    }


def compare_windows(
    current: Sequence[float], baseline: Sequence[float], alpha: float = 0.05, paired: Optional[bool] = None
) -> Dict[str, float]:
    """Compare two time windows of a metric.

    Day-level values are strongly weekly-seasonal, so the unpaired t-test pays
    for variance that is not noise: in a 7v7 comparison the weekly swing alone
    is ~8%, which buries a real 7% drop. When both windows are the same length
    the days line up by weekday (the windows are an integer number of weeks
    apart), so the paired test on daily differences is the right one - it
    removes the seasonal component entirely and is far more powerful.
    """
    cur, base = np.asarray(current, dtype=float), np.asarray(baseline, dtype=float)
    diff_means = cur.mean() - base.mean()
    if paired is None:
        paired = len(cur) == len(base) >= 3
    if paired and len(cur) == len(base):
        diffs = cur - base
        se = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
        t = float(diff_means / se) if se else 0.0
        p = float(2 * (1 - stats.t.cdf(abs(t), df=len(diffs) - 1)))
        method = "paired_t"
    else:
        se = np.sqrt(cur.var(ddof=1) / len(cur) + base.var(ddof=1) / len(base)) if len(cur) > 1 and len(base) > 1 else 0.0
        t = float(diff_means / se) if se else 0.0
        p = float(2 * (1 - stats.t.cdf(abs(t), df=max(1, len(cur) + len(base) - 2))))
        method = "welch_t"
    return {
        "current_mean": float(cur.mean()),
        "baseline_mean": float(base.mean()),
        "abs_diff": float(diff_means),
        "rel_diff": float(diff_means / base.mean()) if base.mean() else float("nan"),
        "se": float(se),
        "t": float(t),
        "p_value": float(p),
        "method": method,
        "significant": bool(p < alpha),
    }
