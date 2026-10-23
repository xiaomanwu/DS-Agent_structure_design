"""Uplift modelling: causal tree, causal forest, meta-learners and the Qini checks."""

from __future__ import annotations

import numpy as np
import pytest

from ds_agent.data.synthetic import build_uplift_users
from ds_agent.tools.uplift import (
    CausalTree,
    binary_treatment,
    causal_forest,
    causal_tree,
    design_matrix,
    qini_coefficient,
    qini_curve,
    uplift_deciles,
    uplift_learners,
)

FEATURES = ["tenure_days", "spend_30d", "sessions_30d", "channel", "device"]
TRUE_ATE = 12.50


@pytest.fixture(scope="module")
def users():
    return build_uplift_users()


@pytest.fixture(scope="module")
def forest(users):
    return causal_forest(users, treatment="group", outcome="revenue", features=FEATURES, n_trees=20, seed=7)


# ------------------------------------------------------------------- primitives
def test_binary_treatment_accepts_arm_labels(users):
    t, positive = binary_treatment(users["group"])
    assert positive == "treatment"
    assert set(np.unique(t)) == {0.0, 1.0}
    assert t.sum() == (users["group"] != "control").sum()


def test_binary_treatment_rejects_three_arms():
    with pytest.raises(ValueError):
        binary_treatment(np.array(["control", "a", "b"]))


def test_design_matrix_is_numeric_and_complete(users):
    X, names = design_matrix(users, FEATURES)
    assert np.isfinite(X).all(), "missing values must be filled, not passed through"
    assert X.shape[1] == len(names)


def test_perfect_ranking_beats_random_and_reversal_flips_sign():
    """The Qini is a ranking score: perfect > random, and reversing the ranking mirrors it.

    The absolute ceiling depends on how heterogeneous the true effects are, so the
    assertions are comparative - which is also how it should be read in a readout.
    """
    rng = np.random.default_rng(0)
    n = 20_000
    t = (rng.random(n) < 0.5).astype(float)
    true_tau = rng.uniform(0, 10, n)
    y = 10 + true_tau * t + rng.normal(0, 1, n)
    perfect = qini_coefficient(y, t, true_tau)
    random_rank = qini_coefficient(y, t, rng.normal(0, 1, n))
    reversed_rank = qini_coefficient(y, t, -true_tau)
    assert perfect > 0.25
    assert perfect > random_rank + 0.2
    assert reversed_rank == pytest.approx(-perfect, abs=0.03)


def test_random_ranking_has_near_zero_qini():
    rng = np.random.default_rng(1)
    n = 4000
    t = rng.integers(0, 2, n).astype(float)
    tau = rng.normal(5, 2, n)
    y = 10 + tau * t + rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)  # unrelated ranking
    assert abs(qini_coefficient(y, t, noise)) < 0.15


def test_qini_curve_shape():
    rng = np.random.default_rng(2)
    n = 500
    t = rng.integers(0, 2, n).astype(float)
    y = rng.normal(0, 1, n) + 3 * t
    curve = qini_curve(y, t, np.linspace(-1, 1, n))
    assert curve["fraction"].iloc[-1] == pytest.approx(1.0)
    assert curve["fraction"].is_monotonic_increasing
    assert (curve["random"] >= 0).all()


def test_deciles_cover_the_population(users):
    t, _ = binary_treatment(users["group"])
    y = users["revenue"].to_numpy(dtype=float)
    deciles = uplift_deciles(y, t, np.linspace(-1, 1, len(y)))
    assert len(deciles) == 10
    assert deciles["n"].sum() == len(users)


# ------------------------------------------------------------------ causal tree
def test_causal_tree_recovers_ate(users):
    result = causal_tree(users, treatment="group", outcome="revenue", features=FEATURES, max_depth=3)
    assert result["ate"] == pytest.approx(TRUE_ATE, abs=3.0)
    assert result["n_leaves"] >= 2


def test_causal_tree_finds_a_heterogeneous_segment(users):
    result = causal_tree(users, treatment="group", outcome="revenue", features=FEATURES, max_depth=3)
    assert result["best_segment"]["tau"] > result["ate"] > result["worst_segment"]["tau"]
    assert result["best_segment"]["n"] > 100


def test_causal_tree_respects_leaf_constraints(users):
    tree = CausalTree(max_depth=2, min_samples_leaf=200, min_treat_per_leaf=40)
    X, names = design_matrix(users, FEATURES)
    t, _ = binary_treatment(users["group"])
    y = users["revenue"].to_numpy(dtype=float)
    tree.fit(X, t, y, names)
    for leaf in tree.leaves():
        assert leaf["n"] >= 200


# ---------------------------------------------------------------- causal forest
def test_forest_ate_matches_truth_and_aipw(forest):
    assert forest["ate"] == pytest.approx(TRUE_ATE, abs=2.5)
    # The forest and the doubly-robust check must agree, or the forest is fitting noise.
    assert abs(forest["ate"] - forest["ate_aipw"]["ate"]) < 2.5


def test_forest_ranking_beats_random(forest):
    assert forest["qini"] > 0.15
    deciles = forest["deciles"]
    assert deciles.iloc[0]["uplift"] > deciles.iloc[-1]["uplift"]


def test_forest_targeting_gain_is_monotone_in_population(forest):
    """Targeting more people must never capture less total uplift."""
    shares = [t["captured_share"] for t in forest["targeting"]]
    assert shares == sorted(shares)
    assert 0 < shares[0] < 1


def test_forest_is_deterministic(users):
    a = causal_forest(users, treatment="group", outcome="revenue", features=FEATURES, n_trees=8, seed=3)
    b = causal_forest(users, treatment="group", outcome="revenue", features=FEATURES, n_trees=8, seed=3)
    assert a["ate"] == pytest.approx(b["ate"], abs=1e-9)
    assert a["qini"] == pytest.approx(b["qini"], abs=1e-9)


# --------------------------------------------------------------- meta-learners
def test_learners_all_produce_a_ranking(users):
    result = uplift_learners(users, treatment="group", outcome="revenue", features=FEATURES)
    assert set(result["learners"]) == {"t", "s", "x"}
    for stats in result["learners"].values():
        assert np.isfinite(stats["qini"])
        assert stats["ate"] == pytest.approx(TRUE_ATE, abs=3.5)
    assert result["best"] in {"t", "s", "x"}
    assert len(result["best_deciles"]) == 10
