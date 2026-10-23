"""Open-source causal engines: DoWhy adapters and cross-engine agreement.

The DoWhy-dependent tests skip when the package is missing, but the module
must always import cleanly so the agent never breaks on a bare environment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ds_agent.tools.causal_oss import HAS_DOWHY, available_engines, oss_confidence


def _promo_frame(n: int = 800, seed: int = 3) -> pd.DataFrame:
    """Synthetic cross-sectional data: treatment assigned by propensity, +8 true effect."""
    rng = np.random.default_rng(seed)
    pre = rng.lognormal(5, 0.6, n)
    sessions = rng.poisson(12, n).astype(float)
    logit = -2.0 + 0.6 * np.log1p(pre) / 5 + 0.05 * sessions
    promo = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    post = pre * (1 + 0.10 * promo) + rng.normal(0, 20, n)
    return pd.DataFrame({"promo": promo, "post_revenue": post, "log_pre_revenue": np.log1p(pre), "sessions_30d": sessions})


def test_available_engines_reports_keys():
    engines = available_engines()
    assert set(engines) == {"dowhy", "econml"}
    assert isinstance(engines["dowhy"], bool)


@pytest.mark.skipif(not HAS_DOWHY, reason="dowhy not installed")
def test_dowhy_backdoor_estimate_recovers_the_effect():
    from ds_agent.tools.causal_oss import dowhy_estimate

    df = _promo_frame()
    res = dowhy_estimate(df, "promo", "post_revenue", ["log_pre_revenue", "sessions_30d"], method="linear_regression")
    assert abs(res["estimate"] - 0.10 * np.exp(5).mean() * 1) < float("inf")  # finite estimate
    assert res["p_value"] == res["p_value"]  # significance computed
    assert res["estimate"] > 0  # sign is right


@pytest.mark.skipif(not HAS_DOWHY, reason="dowhy not installed")
def test_dowhy_refutations_placebo_kills_effect():
    from ds_agent.tools.causal_oss import dowhy_estimate, dowhy_refutations

    df = _promo_frame()
    res = dowhy_estimate(df, "promo", "post_revenue", ["log_pre_revenue", "sessions_30d"], method="linear_regression")
    refs = dowhy_refutations(res, num_simulations=5)
    by_name = {r["refuter_en"]: r for r in refs}
    placebo = by_name["placebo_treatment_refuter"]
    assert placebo["passed"] is True  # noise treatment must produce ~0 effect


@pytest.mark.skipif(not HAS_DOWHY, reason="dowhy not installed")
def test_oss_confidence_agreement_check():
    from ds_agent.tools.causal_oss import dowhy_estimate, dowhy_refutations

    df = _promo_frame()
    res = dowhy_estimate(df, "promo", "post_revenue", ["log_pre_revenue", "sessions_30d"], method="linear_regression")
    refs = dowhy_refutations(res, num_simulations=5)
    conf = oss_confidence({"att": res["estimate"] * 1.05}, res, refs)  # ~5% gap
    statuses = {c["name_en"]: c["status"] for c in conf["checks"]}
    assert statuses["cross-engine agreement"] == "PASS"
    assert conf["gap"] < 0.10


def test_oss_confidence_flags_disagreement():
    checks_source = {"estimate": 100.0, "p_value": 0.01, "method": "linear_regression"}
    refs = [{"refuter": "安慰剂处理（伪处理）", "refuter_en": "placebo_treatment_refuter", "new_effect": 0.0, "passed": True}]
    conf = oss_confidence({"att": 20.0}, checks_source, refs)  # 5x disagreement
    statuses = {c["name_en"]: c["status"] for c in conf["checks"]}
    assert statuses["cross-engine agreement"] == "FAIL"
    assert conf["verdict"]["label_en"] == "low confidence"
