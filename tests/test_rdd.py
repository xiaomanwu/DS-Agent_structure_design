"""Regression discontinuity: sharp, fuzzy, and the falsification tests that matter."""

from __future__ import annotations

import numpy as np
import pytest

from ds_agent.data.synthetic import build_rdd_scores
from ds_agent.tools.rdd import density_test, regression_discontinuity, select_bandwidth

TRUE_JUMP = 22.0


@pytest.fixture(scope="module")
def scores():
    return build_rdd_scores()


@pytest.fixture(scope="module")
def sharp(scores):
    # `pre_spend` is a pre-determined covariate: continuous in the score by
    # construction, so it must not jump at the cutoff. Passing it in is what
    # makes the falsification test meaningful.
    return regression_discontinuity(
        scores, outcome="outcome", running="score", cutoff=600.0, covariates=("pre_spend",)
    )


def test_sharp_rdd_recovers_the_jump(sharp):
    """The generator plants a jump of +22 at score 600."""
    assert sharp["effect"] == pytest.approx(TRUE_JUMP, abs=6.0)
    assert sharp["p_value"] < 0.01
    assert sharp["n_left"] > 50 and sharp["n_right"] > 50


def test_bandwidth_is_chosen_and_positive(sharp):
    assert sharp["bandwidth"] > 0
    assert sharp["bandwidth_selection"]


def test_bandwidth_sensitivity_is_stable(sharp):
    """A real discontinuity keeps its sign when the bandwidth moves."""
    table = sharp["bandwidth_sensitivity"]
    assert len(table) >= 3
    assert sharp["bandwidth_stable"]
    assert np.all(np.sign(table["effect"]) == np.sign(sharp["effect"]))


def test_no_manipulation_on_clean_scores(sharp):
    """Nobody games the score in the generator, so the density must be continuous."""
    assert not sharp["density"]["manipulation"]
    assert sharp["density"]["p_value"] > 0.05


def test_covariate_balance_and_placebos(sharp):
    assert sharp["covariate_balance"], "pre-period covariates should be checked"
    assert sharp["covariate_balance_ok"]
    # A placebo cutoff away from the real one must not produce a significant jump
    # at every single location.
    hits = [p for p in sharp["placebo_cutoffs"] if p.get("p_value", 1.0) < 0.05]
    assert len(hits) <= max(1, len(sharp["placebo_cutoffs"]) // 2)


def test_plot_payload_is_present(sharp):
    plot = sharp["plot"]
    assert {"bins", "left", "right"}.issubset(plot)
    assert len(plot["bins"]) > 0
    assert len(plot["left"]) and len(plot["right"])


def test_fuzzy_rdd_returns_a_late():
    fuzzy_data = build_rdd_scores(fuzzy=True)
    result = regression_discontinuity(
        fuzzy_data, outcome="outcome", running="score", cutoff=600.0, fuzzy="treated"
    )
    assert result["first_stage"]["jump"] > 0.1
    assert result["first_stage"]["compliance"]
    # The Wald ratio must land near the same 22 the sharp design plants.
    assert result["effect"] == pytest.approx(TRUE_JUMP, abs=8.0)


def test_bandwidth_selector_is_deterministic(scores):
    x = scores["score"].to_numpy(dtype=float)
    y = scores["outcome"].to_numpy(dtype=float)
    first = select_bandwidth(x, y, 600.0)
    second = select_bandwidth(x, y, 600.0)
    assert first["bandwidth"] == pytest.approx(second["bandwidth"])


def test_density_test_flags_a_pile_up_at_the_cutoff():
    """A manufactured pile-up just above the cutoff must be caught."""
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(500, 80, 3000), rng.normal(600, 40, 1500), rng.normal(700, 80, 2000)])
    result = density_test(x, cutoff=600.0, bandwidth=120.0)
    assert result["p_value"] < 0.05


def test_rdd_needs_enough_data():
    rng = np.random.default_rng(0)
    tiny = {"score": rng.normal(600, 50, 60), "outcome": rng.normal(10, 1, 60)}
    import pandas as pd

    with pytest.raises(ValueError):
        regression_discontinuity(pd.DataFrame(tiny), outcome="outcome", running="score", cutoff=600.0)
