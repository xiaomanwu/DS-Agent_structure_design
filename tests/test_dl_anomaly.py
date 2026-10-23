"""Deep-learning anomaly detection: autoencoder + isolation forest + confidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ds_agent.tools.dl_anomaly import (
    autoencoder_scores,
    dl_anomaly_confidence,
    isolation_forest_scores,
    _flagged_days,
)


def _series(with_anomaly: bool, n: int = 120, seed: int = 11):
    rng = np.random.default_rng(seed)
    base = 1000 * (1 + 0.02 * np.sin(2 * np.pi * np.arange(n) / 7)) + rng.normal(0, 15, n)
    if with_anomaly:
        base = base.copy()
        base[60:66] -= 220  # planted 6-day level shift
        base[90] -= 300  # planted single-day spike
    return base


def test_autoencoder_detects_planted_shift_and_spike():
    flagged = autoencoder_scores(_series(True), window=7)
    days = _flagged_days(flagged, 120)
    assert days[60:66].all()  # the planted shift
    assert days[90]  # the planted spike


def test_autoencoder_is_quiet_on_a_clean_series():
    result = autoencoder_scores(_series(False), window=7)
    assert not result["flagged"].any()


def test_autoencoder_reports_reconstruction_in_original_units():
    result = autoencoder_scores(_series(True), window=7)
    rec = np.asarray(result["reconstruction"], dtype=float)
    actual = np.asarray(result["actual"], dtype=float)
    assert (rec > 0).all()  # GMV-scale numbers, not ratio-space values
    rel = np.median(np.abs(rec - actual) / actual)
    assert rel < 0.10  # reconstruction tracks the level well


def test_confidence_checks_are_day_level_and_pass_on_clean_detection():
    result = autoencoder_scores(_series(True), window=7)
    seasonal = np.zeros(120, dtype=bool)
    seasonal[60:66] = True
    seasonal[90] = True
    conf = dl_anomaly_confidence(result, seasonal)
    statuses = {c["name_en"]: c["status"] for c in conf["checks"]}
    assert statuses["alert volume"] == "PASS"  # ~6% of days, not window-inflated
    assert statuses["seasonal-baseline cross-check"] == "PASS"  # Jaccard 1.0
    assert conf["verdict"]["label_en"] == "high confidence"


def test_confidence_fails_when_threshold_collapses():
    result = autoencoder_scores(_series(False), window=7)
    result["flagged"] = np.ones(len(result["flagged"]), dtype=bool)  # everything flagged
    conf = dl_anomaly_confidence(result)
    statuses = {c["name_en"]: c["status"] for c in conf["checks"]}
    assert statuses["alert volume"] == "FAIL"
    assert conf["verdict"]["label_en"] == "low confidence"


def test_isolation_forest_flags_the_planted_days():
    result = isolation_forest_scores(_series(True))
    flagged = set(np.asarray(result["idx"])[result["flagged"]].tolist())
    assert 60 in flagged and 90 in flagged


def test_autoencoder_needs_enough_data():
    with pytest.raises(ValueError):
        autoencoder_scores(np.arange(20, dtype=float), window=7)
