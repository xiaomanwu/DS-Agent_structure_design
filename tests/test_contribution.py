"""Contribution decomposition must be exact, and attribution must find the driver."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ds_agent.semantic import Metric
from ds_agent.tools.contribution import attribute_change, contribution_table, decompose_change

GMV = Metric(name="gmv", numerator="gmv", table="t")
CVR = Metric(name="conversion_rate", numerator="orders", denominator="sessions", table="t")


def _frames():
    # channel 'feed' collapses, everything else grows: the answer is unambiguous.
    base = pd.DataFrame(
        {
            "channel": ["feed", "search", "social"],
            "device": ["iOS", "iOS", "Android"],
            "gmv": [400.0, 300.0, 300.0],
            "orders": [40.0, 30.0, 30.0],
            "sessions": [1000.0, 1000.0, 1000.0],
        }
    )
    cur = pd.DataFrame(
        {
            "channel": ["feed", "search", "social"],
            "device": ["iOS", "iOS", "Android"],
            "gmv": [200.0, 330.0, 315.0],
            "orders": [20.0, 33.0, 31.5],
            "sessions": [1000.0, 1000.0, 1000.0],
        }
    )
    return cur, base


def test_additive_decomposition_is_exact():
    cur, base = _frames()
    result = decompose_change(cur, base, GMV, "channel")
    assert result["total_delta"] == pytest.approx(-155.0)
    assert result["residual"] == pytest.approx(0.0, abs=1e-9)
    assert result["segments"]["contribution"].sum() == pytest.approx(result["total_delta"])
    top = result["segments"].iloc[0]
    assert top["segment"] == "feed"
    assert top["contribution"] == pytest.approx(-200.0)
    assert top["contribution_pct"] == pytest.approx(-200 / -155)


def test_ratio_decomposition_adds_up():
    cur, base = _frames()
    result = decompose_change(cur, base, CVR, "channel")
    total = result["rate_effect_total"] + result["mix_effect_total"] + result["interaction_total"]
    assert total == pytest.approx(result["total_delta"], abs=1e-12)
    assert result["residual"] == pytest.approx(0.0, abs=1e-12)
    assert result["kind"] == "ratio"


def test_ratio_mix_effect_detects_composition_shift():
    """Same rates everywhere, only the volume mix changes: pure mix effect."""
    base = pd.DataFrame({"channel": ["a", "b"], "orders": [10.0, 90.0], "sessions": [100.0, 100.0]})
    cur = pd.DataFrame({"channel": ["a", "b"], "orders": [90.0, 10.0], "sessions": [100.0, 100.0]})
    result = decompose_change(cur, base, CVR, "channel")
    # Rates are unchanged per segment (0.1 and 0.9), so the change is 100% mix.
    assert abs(result["rate_effect_total"]) < 1e-12
    assert abs(result["mix_effect_total"] - result["total_delta"]) < 1e-12


def test_attribute_picks_the_dominant_dimension_and_drills_down():
    cur, base = _frames()
    result = attribute_change(cur, base, GMV, ["channel", "device"])
    assert result["primary"]["dimension"] == "channel"
    assert result["primary_segment"] == "feed"
    assert result["drilldown"], "a second-level drill-down should be produced"
    # Gross movement can exceed the net change when segments offset each other.
    assert result["explained_share"] >= 0.9


def test_new_and_lost_segments_are_reported():
    base = pd.DataFrame({"channel": ["a", "b"], "gmv": [100.0, 100.0]})
    cur = pd.DataFrame({"channel": ["b", "c"], "gmv": [100.0, 50.0]})
    result = decompose_change(cur, base, GMV, "channel")
    assert "c" in result["new_segments"]
    assert "a" in result["lost_segments"]


def test_contribution_table_is_renderable():
    cur, base = _frames()
    result = decompose_change(cur, base, GMV, "channel")
    table = contribution_table(result, top_k=2)
    assert len(table) == 2
    assert "share_of_change" in table.columns
    assert table.iloc[0]["channel"] == "feed"
