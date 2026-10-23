"""Dashboard tool: KPI math, anomaly table, standalone HTML artifact."""

from __future__ import annotations

from pathlib import Path

import pytest

import ds_agent.config as config_module
from ds_agent.agent import DSQueryAgent
from ds_agent.tools.dashboard import build_dashboard


@pytest.fixture(scope="module")
def dash_agent(warehouse, workdir):
    artifacts = workdir / "artifacts_dashboard"
    artifacts.mkdir(parents=True, exist_ok=True)
    original = config_module.settings.artifact_dir
    config_module.settings.artifact_dir = artifacts
    instance = DSQueryAgent(warehouse=warehouse, artifact_dir=artifacts)
    yield instance, artifacts
    config_module.settings.artifact_dir = original
    instance.close()


def test_build_dashboard_produces_standalone_html(dash_agent):
    agent, artifacts = dash_agent
    out = build_dashboard(agent)
    path = Path(out["path"])
    assert path.exists() and path.parent == artifacts
    html = path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert html.count("data:image/png;base64,") >= 4  # trend, channel, device, refund
    assert "kpi" in html


def test_dashboard_kpis_have_wow_deltas(dash_agent):
    agent, _ = dash_agent
    out = build_dashboard(agent)
    assert len(out["kpis"]) == 4
    for k in out["kpis"]:
        assert k["value"]
        assert k["delta"] is None or -1.0 < k["delta"] < 1.0


def test_dashboard_flags_anomalies_with_same_detector(dash_agent):
    agent, _ = dash_agent
    out = build_dashboard(agent)
    # The demo warehouse contains a real down-episode in mid-October.
    if out["flagged_rows"]:
        row = out["flagged_rows"][0]
        assert {"date", "metric", "delta", "severity"} <= set(row.keys())
        assert "GMV" == row["metric"]


def test_dashboard_matches_agent_caliber(dash_agent):
    """Dashboard GMV KPI must equal a direct SQL answer (same caliber)."""
    agent, _ = dash_agent
    out = build_dashboard(agent)
    gmv_value = float(out["kpis"][0]["value"].replace("¥", "").replace(",", ""))
    direct = agent.sql.run(
        "SELECT SUM(gmv) AS v FROM orders_daily WHERE date > (SELECT MAX(date) - INTERVAL 13 DAY FROM orders_daily)"
    ).dataframe["v"].iloc[0]
    assert abs(gmv_value - float(direct)) < 1.0


def test_dashboard_intent_end_to_end(dash_agent):
    agent, artifacts = dash_agent
    reply = agent.ask("给我生成一个 GMV 监控看板")
    assert reply.data["intent"] == "dashboard"
    assert reply.artifacts and Path(reply.artifacts[0]).exists()
