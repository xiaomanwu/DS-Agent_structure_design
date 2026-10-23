"""Anomaly detection: planted incidents must be found, clean series must stay quiet."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ds_agent.tools.anomaly import AnomalyConfig, AnomalyDetector, scan_segment_anomalies


def _series(n=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    base = 1000 + np.arange(n) * 1.5
    seasonal = np.array([1.0, 1.02, 0.98, 0.99, 1.05, 1.12, 1.06])[dates.dayofweek]
    values = base * seasonal * rng.normal(1.0, 0.01, n)
    return dates, values


def test_detects_planted_dip():
    dates, values = _series()
    values = values.copy()
    values[60:65] *= 0.75  # planted 5-day incident
    result = AnomalyDetector(AnomalyConfig(z_threshold=3.0)).detect(dates, values, label="gmv")

    assert result["episodes"], "the planted dip should produce at least one episode"
    top = result["episodes"][0]
    assert top["direction"] == "down"
    assert top["days"] >= 3
    assert 55 <= dates.get_loc(pd.Timestamp(top["start"])) <= 66


def test_quiet_on_clean_series():
    dates, values = _series(seed=5)
    result = AnomalyDetector(AnomalyConfig(z_threshold=4.0, min_rel_change=0.05)).detect(dates, values)
    assert len(result["episodes"]) == 0


def test_seasonality_is_not_flagged():
    """A clean weekly pattern must not be mistaken for anomalies."""
    dates = pd.date_range("2026-01-01", periods=112, freq="D")
    weekly = np.array([1.0, 0.95, 1.30, 1.10, 1.0, 0.9, 0.8])[dates.dayofweek] * 5000
    result = AnomalyDetector().detect(dates, weekly)
    assert len(result["episodes"]) == 0


def test_frames_are_aligned():
    """Regression: a sliced series used to misalign dates and produce NaT."""
    dates, values = _series(n=60)
    frame = pd.DataFrame({"d": dates, "v": values}).iloc[10:].reset_index(drop=True)
    result = AnomalyDetector().detect(frame["d"], frame["v"].to_numpy())
    assert result["frame"]["date"].notna().all()


def test_additive_series_still_works():
    """Zero-containing series fall back to the additive model."""
    dates, values = _series(n=60)
    values = values.copy()
    values[::9] = 0
    result = AnomalyDetector().detect(dates, values)
    assert result["model"] == "additive"
    assert len(result["frame"]) == 60


def test_segment_scan_ranks_the_bleeding_segment():
    dates, _ = _series(n=90)
    rng = np.random.default_rng(3)
    frames = []
    for seg, drift in (("a", 1.0), ("b", 1.0), ("c", 0.60)):
        level = (1000 + np.arange(90) * 2) * rng.normal(1.0, 0.01, 90)
        level[-7:] *= drift
        frames.append(pd.DataFrame({"segment": seg, "date": dates, "value": level}))
    result = scan_segment_anomalies(pd.concat(frames), min_abs_z=3.0)
    assert result["segments"], "segment c should be flagged"
    assert result["segments"][0]["segment"] == "c"
    assert result["segments"][0]["direction"] == "down"
