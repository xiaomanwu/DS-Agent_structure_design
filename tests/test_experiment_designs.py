"""Experiment designs beyond a plain A/B: switchback, reversal, nested, two-sided, sensitivity.

Every test recovers a planted ground truth from a generator in
``ds_agent.data.synthetic``, plus the falsification check that tells a reader
whether to believe it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ds_agent.data.synthetic import (
    build_experiment_users,
    build_nested_users,
    build_reverse_log,
    build_switchback_log,
    build_two_sided_market,
)
from ds_agent.tools.experiment_designs import (
    analyze_nested_experiment,
    analyze_reverse,
    analyze_switchback,
    analyze_two_sided,
    sensitivity_analysis,
)


@pytest.fixture(scope="module")
def switchback():
    return analyze_switchback(build_switchback_log(), metric="gmv", washout=1)


@pytest.fixture(scope="module")
def reverse():
    return analyze_reverse(build_reverse_log(), metric="revenue", phase_col="phase")


@pytest.fixture(scope="module")
def nested():
    return analyze_nested_experiment(build_nested_users(), metric="converted")


@pytest.fixture(scope="module")
def two_sided():
    return analyze_two_sided(
        build_two_sided_market(),
        metrics={"buyer": "buyer_conversion", "seller": "listing_rate", "market": "gmv_per_buyer"},
        n_boot=60,
    )


# --------------------------------------------------------------------- switchback
def test_switchback_recovers_planted_effect(switchback):
    """Ground truth is +4.5%; the block-level estimate must land near it."""
    assert switchback["relative"] == pytest.approx(0.045, abs=0.015)
    assert switchback["p_value"] < 0.01
    assert switchback["n_blocks"] == 14


def test_switchback_uses_blocks_not_rows(switchback):
    """The collapsed series is the inference unit, not the 448 market-days."""
    assert switchback["n_obs"] < switchback["n_blocks"] * 5
    assert np.isfinite(switchback["residual_autocorr"])


def test_switchback_paired_and_carryover_are_reported(switchback):
    paired = switchback["paired_blocks"]
    assert paired["n_pairs"] >= 5
    # Paired differencing and the regression agree on direction.
    assert np.sign(paired["mean_diff"]) == np.sign(switchback["effect"])
    assert {"lag_coef", "lag_p", "ratio", "contaminated"}.issubset(switchback["carryover"])


def test_switchback_washout_changes_the_estimate():
    """Dropping the switch days must move the estimate: that is the point of them."""
    log = build_switchback_log()
    clean = analyze_switchback(log, metric="gmv", washout=1)
    dirty = analyze_switchback(log, metric="gmv", washout=0)
    assert clean["effect"] != pytest.approx(dirty["effect"], abs=1e-6)
    assert clean["washout_days"] == 1


# ----------------------------------------------------------------------- reversal
def test_reversal_detects_the_sign_flip(reverse):
    assert reverse["reversal_detected"]
    assert reverse["tau1"] > 0
    assert reverse["tau2"] < 0


def test_reversal_persistence_in_range(reverse):
    """The generator keeps 70% of the effect after the swap."""
    assert reverse["persistence"] == pytest.approx(0.7, abs=0.25)
    assert reverse["pooled"]["p_value"] < 0.01


def test_reversal_sum_test_exposes_drift(reverse):
    """tau1 + tau2 tests for instrumentation drift; both keys must be reported."""
    assert "p_value" in reverse["symmetry"]
    assert reverse["symmetric"] is bool(reverse["symmetry"]["p_value"] > 0.05)


# ------------------------------------------------------------------------- nested
def test_nested_layers_are_orthogonal(nested):
    assert nested["orthogonal"]
    assert nested["orthogonality"]["p_value"] > 0.01


def test_nested_recovers_child_effect(nested):
    """Truth is +6% on the child layer, on top of a +3% parent."""
    assert nested["adjusted"]["relative"] == pytest.approx(0.06, abs=0.02)
    assert nested["adjusted"]["p_value"] < 0.05


def test_nested_interaction_is_not_flagged(nested):
    """The generator builds no parent x child interaction."""
    assert not nested["interaction"]["detected"]


def test_nested_requires_enough_users():
    """Under 1,000 users the four cells cannot support an interaction test."""
    small = build_nested_users(n=400)
    with pytest.raises(ValueError, match="at least"):
        analyze_nested_experiment(small, metric="converted")


# ---------------------------------------------------------------------- two-sided
def test_two_sided_direct_effects_recover_truth(two_sided):
    """Generator: buyer +8% conversion, seller +6% listing rate."""
    assert two_sided["results"]["buyer"]["buyer_direct_relative"] == pytest.approx(0.08, abs=0.04)
    assert two_sided["results"]["seller"]["seller_direct_relative"] == pytest.approx(0.06, abs=0.04)
    assert two_sided["results"]["buyer"]["seller_direct_relative"] == pytest.approx(0.015, abs=0.03)


def test_two_sided_reports_spillover_ratios(two_sided):
    spill = two_sided["spillover"]
    assert spill["buyer"]["ratio"] > 0
    assert spill["buyer"]["cross_side"] < spill["buyer"]["direct"]


def test_two_sided_every_metric_has_four_cells(two_sided):
    for res in two_sided["results"].values():
        assert set(res["cells"]) == {"00", "10", "01", "11"}
        assert np.isfinite(res["interaction_p"])


# -------------------------------------------------------------------- sensitivity
def test_sensitivity_runs_every_specification():
    result = sensitivity_analysis(
        build_experiment_users(), metric="converted", segment_col="device", n_boot=60
    )
    assert result["n_specs"] >= 3
    assert set(["spec", "rel_lift", "p_value", "significant"]).issubset(result["specs"].columns)
    lo, hi = result["lift_range"]
    assert lo <= result["baseline"]["rel_lift"] <= hi


def test_sensitivity_detects_proportion_and_skips_trimming():
    """0/1 metrics must not be trimmed or logged: it would be meaningless."""
    result = sensitivity_analysis(build_experiment_users(), metric="converted", n_boot=40)
    specs = " ".join(result["specs"]["spec"].astype(str))
    assert "截尾" not in specs and "trim" not in specs.lower()
    assert result["sign_stable"]


def test_sensitivity_placebo_rate_is_reported():
    result = sensitivity_analysis(
        build_experiment_users(), metric="revenue", covariate="pre_revenue",
        segment_col="device", n_boot=40,
    )
    assert 0.0 <= result["placebo_false_positive_rate"] <= 0.3
    assert result["jackknife"], "jackknife over segments should be present when segment_col is given"
