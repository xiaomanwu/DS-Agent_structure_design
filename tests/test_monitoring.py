"""Model monitoring: PSI bands, KS drift, decay verdicts, alert wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ds_agent.tools.monitoring import (
    PSI_CRITICAL,
    PSI_WARN,
    feature_drift_table,
    ks_data_drift,
    monitor_report,
    performance_decay,
    psi,
)


def test_psi_identical_is_zero_and_shift_is_large():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < PSI_WARN
    assert psi(ref, rng.normal(3, 1, 5000)) > PSI_CRITICAL


def test_psi_quantile_edges_detect_absence_bin():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 4000)
    # 30% of the population moves far outside the reference support — the
    # overflow must land in the edge bin and blow up the PSI, not vanish.
    actual = np.concatenate([rng.normal(0, 1, 2800), rng.normal(6, 0.1, 1200)])
    assert psi(ref, actual) > PSI_CRITICAL


def test_feature_drift_table_bands_and_verdict():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 4000), "b": rng.normal(0, 1, 4000)})
    cur = pd.DataFrame({"a": rng.normal(0.9, 1, 4000), "b": rng.normal(0, 1, 4000)})
    out = feature_drift_table(ref, cur)
    by_name = {r["feature"]: r for r in out["columns"]}
    assert by_name["a"]["severity"] == "critical"
    assert by_name["b"]["severity"] == "ok"
    assert out["verdict"] == "critical" and out["n_critical"] == 1


def test_ks_drift_flags_shifted_distribution():
    rng = np.random.default_rng(3)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 3000), "cat": np.repeat(["p", "q"], 1500)})
    cur = pd.DataFrame({"x": rng.normal(1.2, 1, 3000), "cat": np.repeat(["p", "q"], 1500)})
    out = ks_data_drift(ref, cur)
    assert out["verdict"] == "drifted"
    assert out["tests"][0]["feature"] == "x"


def test_performance_decay_thresholds():
    hist = [{"slice": f"w{i}", "auc": 0.80} for i in range(4)] + [
        {"slice": f"w{i}", "auc": 0.72} for i in range(4, 8)
    ]
    out = performance_decay(hist)
    assert out["verdict"] == "critical"          # drop 0.08 > 0.05
    assert out["drop"] == pytest.approx(0.08, abs=0.01)
    flat = performance_decay([{"slice": "a", "auc": 0.8}, {"slice": "b", "auc": 0.8}])
    assert flat["verdict"] == "ok"


def test_monitor_report_wires_alerts_and_overall():
    rng = np.random.default_rng(4)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 4000), "b": rng.normal(0, 1, 4000)})
    cur = pd.DataFrame({"a": rng.normal(2.5, 1, 4000), "b": rng.normal(0, 1, 4000)})
    hist = [{"slice": "old", "auc": 0.80}] * 4 + [{"slice": "new", "auc": 0.70}] * 4
    rep = monitor_report(ref, cur, hist)
    assert rep["overall"] == "critical"
    checks = {a["check"] for a in rep["alerts"]}
    assert "feature_drift" in checks and "performance_decay" in checks
    assert rep["thresholds"]["psi_critical"] == PSI_CRITICAL


def test_agent_routes_monitoring_question(agent):
    reply = agent.ask("检查转化模型的特征漂移和性能衰减")
    assert "monitoring" in reply.data
    assert "模型监控" in reply.text or "Model monitoring" in reply.text
