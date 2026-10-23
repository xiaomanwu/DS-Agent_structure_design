"""Synthetic demo warehouse with deliberately planted metric anomalies.

The data mimics a mid-size e-commerce business for 180 days:
  * organic weekly seasonality and a slow growth trend
  * two planted incidents used to demo anomaly detection and attribution
      1) a feed-channel tracking outage (GMV drop of ~35% for 5 days)
      2) an iOS payment-SDK regression (conversion drop for the last 6 days)
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Tuple

import duckdb
import numpy as np
import pandas as pd

CHANNELS = ["organic", "feed", "search", "social", "affiliate"]
CHANNEL_WEIGHTS = np.array([0.34, 0.26, 0.18, 0.14, 0.08])
DEVICES = ["iOS", "Android", "PC"]
DEVICE_WEIGHTS = np.array([0.30, 0.52, 0.18])
REGIONS = ["east", "north", "south", "southwest", "central"]
REGION_WEIGHTS = np.array([0.38, 0.24, 0.20, 0.10, 0.08])
CATEGORIES = ["3C", "apparel", "beauty", "grocery", "home"]
CATEGORY_WEIGHTS = np.array([0.26, 0.24, 0.18, 0.20, 0.12])

# AOV by category (CNY)
CATEGORY_AOV = {"3C": 620.0, "apparel": 180.0, "beauty": 210.0, "grocery": 85.0, "home": 260.0}
# Baseline conversion by device and channel
CVR_DEVICE = {"iOS": 0.031, "Android": 0.026, "PC": 0.019}
CVR_CHANNEL = {"organic": 1.25, "feed": 0.85, "search": 1.35, "social": 0.70, "affiliate": 0.95}


def _day_factors(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Trend + weekly seasonality + noise, shared by all traffic sources."""
    n = len(dates)
    trend = np.linspace(0.92, 1.10, n)
    dow = dates.dayofweek.to_numpy()
    weekly = np.array([1.02, 1.00, 0.98, 0.99, 1.06, 1.14, 1.08])[dow]
    noise = rng.normal(1.0, 0.035, n)
    return trend * weekly * noise


def build_orders_daily(days: int = 180, seed: int = 7) -> pd.DataFrame:
    """Build the daily aggregated fact table."""
    rng = np.random.default_rng(seed)
    end = date(2026, 10, 20)
    dates = pd.date_range(end=end, periods=days, freq="D")
    day_factor = _day_factors(dates, rng)

    rows = []
    base_sessions = 420_000.0
    for idx, ts in enumerate(dates):
        day_offset = days - 1 - idx  # 0 == most recent day
        for ci, channel in enumerate(CHANNELS):
            for di, device in enumerate(DEVICES):
                for ri, region in enumerate(REGIONS):
                    for ki, category in enumerate(CATEGORIES):
                        share = (
                            CHANNEL_WEIGHTS[ci]
                            * DEVICE_WEIGHTS[di]
                            * REGION_WEIGHTS[ri]
                            * CATEGORY_WEIGHTS[ki]
                        )
                        sessions = base_sessions * share * day_factor[idx] * rng.normal(1.0, 0.06)

                        # Incident 2: iOS payment SDK regression on the last 6 days.
                        ios_penalty = 1.0
                        if device == "iOS" and day_offset <= 5:
                            ios_penalty = 0.62

                        cvr = (
                            CVR_DEVICE[device]
                            * CVR_CHANNEL[channel]
                            * ios_penalty
                            * rng.normal(1.0, 0.07)
                        )
                        cvr = float(np.clip(cvr, 0.002, 0.15))
                        orders = rng.poisson(max(sessions * cvr, 0.1))

                        aov = CATEGORY_AOV[category] * rng.normal(1.0, 0.05)
                        gmv = orders * aov

                        # Incident 1: feed-channel tracking outage 6-10 days ago.
                        if channel == "feed" and 6 <= day_offset <= 10:
                            orders = int(orders * 0.52)
                            gmv = gmv * 0.52

                        refund_rate = 0.035 + (0.02 if category == "apparel" else 0.0)
                        refund_orders = rng.binomial(orders, refund_rate) if orders > 0 else 0
                        refund_amount = refund_orders * aov * rng.normal(0.95, 0.05)

                        buyers = int(orders * rng.uniform(0.86, 0.95)) if orders > 0 else 0
                        rows.append(
                            (
                                ts.date(),
                                channel,
                                device,
                                region,
                                category,
                                "new" if rng.random() < 0.38 else "returning",
                                int(orders),
                                round(float(gmv), 2),
                                max(buyers, 0),
                                int(max(sessions, 0)),
                                int(refund_orders),
                                round(float(refund_amount), 2),
                            )
                        )

    df = pd.DataFrame(
        rows,
        columns=[
            "date", "channel", "device", "region", "category", "user_type",
            "orders", "gmv", "buyers", "sessions", "refund_orders", "refund_amount",
        ],
    )
    df["is_new_user"] = np.where(df["user_type"] == "new", 1, 0)
    return df


def build_experiment_users(n_per_arm: int = 50_000, true_lift: float = 0.06, seed: int = 11) -> pd.DataFrame:
    """Build a user-level AB test dataset with a known true relative lift.

    Conversion is generated from a logistic model that depends on device and on
    pre-period revenue, so CUPED / covariate adjustment has something to work
    with. Treatment raises conversion by ``true_lift`` multiplicatively.
    """
    rng = np.random.default_rng(seed)
    n = n_per_arm * 2
    group = np.array(["control"] * n_per_arm + ["treatment"] * n_per_arm)
    device = rng.choice(DEVICES, size=n, p=DEVICE_WEIGHTS)
    pre_revenue = rng.lognormal(mean=3.4, sigma=1.0, size=n)  # skewed pre-period spend

    device_effect = np.where(device == "iOS", 0.35, np.where(device == "Android", 0.0, -0.30))
    logit = -3.15 + device_effect + 0.16 * np.log1p(pre_revenue)
    if true_lift:
        logit = logit + np.where(group == "treatment", np.log1p(true_lift), 0.0)
    prob = 1.0 / (1.0 + np.exp(-logit))
    converted = rng.binomial(1, prob)

    aov = np.where(device == "PC", 320.0, 210.0)
    revenue = converted * rng.lognormal(mean=np.log(aov), sigma=0.45, size=n)
    visits = rng.poisson(3.1, size=n)

    return pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "dt": pd.Timestamp("2026-10-20").date(),
            "group": group,
            "device": device,
            "pre_revenue": np.round(pre_revenue, 2),
            "converted": converted,
            "revenue": np.round(revenue, 2),
            "visits": visits,
        }
    )


def build_city_panel(n_cities: int = 20, n_weeks: int = 40, seed: int = 23) -> pd.DataFrame:
    """City x week panel with a staggered policy rollout, for DiD / synthetic control.

    Six cities receive a subsidy policy from week 26, lifting GMV by ~9%. The
    pre-period is generated from a shared national trend plus city-level noise,
    so parallel trends roughly hold and the estimators have something real to find.
    """
    rng = np.random.default_rng(seed)
    cities = [f"city_{i:02d}" for i in range(n_cities)]
    treat_start = 26
    treated_cities = set(cities[:6])

    base = rng.uniform(1.2e6, 3.4e6, n_cities)
    trend = np.linspace(0.95, 1.18, n_weeks)
    national = np.array([1.0, 1.01, 0.99, 1.02, 1.06, 1.12, 1.05])  # weekly shape on 7-week cycle

    rows = []
    for ci, city in enumerate(cities):
        city_noise = rng.normal(0.0, 0.02, n_weeks).cumsum() * 0.0015 + 1.0
        for w in range(1, n_weeks + 1):
            seasonal = national[(w - 1) % 7]
            shock = rng.normal(1.0, 0.012)
            level = base[ci] * trend[w - 1] * seasonal * city_noise[w - 1] * shock
            if city in treated_cities and w >= treat_start:
                level *= 1.09
            rows.append(
                {
                    "city": city,
                    "week": w,
                    "date": (date(2025, 10, 6) + timedelta(weeks=w - 1)),
                    "gmv": round(level, 2),
                    "orders": int(level / rng.uniform(180, 240)),
                    "treated": 1 if city in treated_cities else 0,
                    "post": 1 if w >= treat_start else 0,
                    "is_policy_city": 1 if city in treated_cities else 0,
                }
            )
    return pd.DataFrame(rows)


def build_promo_users(n: int = 40_000, true_effect: float = 12.0, seed: int = 31) -> pd.DataFrame:
    """Observational promo dataset with deliberate confounding.

    Promo exposure depends on pre-period revenue and tenure, so the naive
    difference in post-period revenue overstates the effect. The true effect on
    post revenue is a constant ``true_effect`` CNY, which makes PSM / IPW / AIPW
    verifiable: they should land near 12 while the naive comparison does not.
    """
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 37, n)
    sessions_30d = rng.poisson(8, n)
    pre_revenue = rng.lognormal(mean=3.6, sigma=0.9, size=n)

    # Exposure mechanism: heavy users and long-tenure users are much more likely
    # to receive the promo - this is the confounding we need to remove.
    logit = -1.6 + 0.55 * (np.log1p(pre_revenue) - 3.6) + 0.035 * tenure + 0.03 * sessions_30d
    ps = 1 / (1 + np.exp(-logit))
    promo = rng.binomial(1, ps)

    baseline_post = (
        18.0
        + 0.85 * pre_revenue
        + 1.8 * tenure
        + 6.0 * sessions_30d
        + rng.lognormal(mean=2.0, sigma=0.8, size=n)
    )
    post_revenue = baseline_post + true_effect * promo

    return pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "promo": promo,
            "tenure_months": tenure,
            "sessions_30d": sessions_30d,
            "pre_revenue": np.round(pre_revenue, 2),
            "post_revenue": np.round(post_revenue, 2),
        }
    )


def build_switchback_log(
    days: int = 56, markets: int = 8, block_days: int = 4, true_effect: float = 0.045, seed: int = 41
) -> pd.DataFrame:
    """Market x day panel from a time-sliced (switchback) experiment.

    The arm alternates every ``block_days`` days across all markets at once, so
    day-of-week and slow trends are shared by both arms and cancel out. Each
    market carries its own AR(1) noise, and the first day of a block keeps 25%
    of the previous arm's effect - the carryover a washout has to absorb.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=date(2026, 10, 20), periods=days, freq="D")
    market_ids = [f"mkt_{i:02d}" for i in range(markets)]
    base = rng.uniform(1.4e5, 3.2e5, markets)
    trend = np.linspace(0.95, 1.08, days)
    weekly = np.array([1.02, 1.00, 0.98, 0.99, 1.05, 1.13, 1.07])
    common = np.zeros(days)
    for i in range(1, days):
        common[i] = 0.55 * common[i - 1] + rng.normal(0, 0.012)

    arm_schedule = ["treatment" if (d // block_days) % 2 == 0 else "control" for d in range(days)]
    rows = []
    for mi, market in enumerate(market_ids):
        ar = np.zeros(days)
        for i in range(1, days):
            ar[i] = 0.4 * ar[i - 1] + rng.normal(0, 0.02)
        for d, ts in enumerate(dates):
            arm = arm_schedule[d]
            level = base[mi] * trend[d] * weekly[ts.dayofweek] * (1 + common[d] + ar[d])
            effect = true_effect if arm == "treatment" else 0.0
            # Carryover: the day after a switch still feels the old arm.
            if d > 0 and arm_schedule[d] != arm_schedule[d - 1]:
                effect = 0.75 * (true_effect if arm_schedule[d - 1] == "treatment" else 0.0)
            gmv = level * (1 + effect)
            rows.append(
                {
                    "date": ts.date(),
                    "market": market,
                    "arm": arm,
                    "gmv": round(float(gmv), 2),
                    "orders": int(gmv / rng.uniform(190, 240)),
                    "sessions": int(gmv / rng.uniform(26, 34)),
                }
            )
    return pd.DataFrame(rows)


def build_reverse_log(
    days: int = 48,
    switch_day: int = 24,
    users_per_arm_day: int = 2500,
    effect_phase1: float = 0.06,
    effect_phase2: float = 0.042,
    seed: int = 43,
) -> pd.DataFrame:
    """Reversal experiment: the variants are swapped halfway through.

    Convention: the arm *label* stays fixed while the *experience* swaps, so in
    phase 2 the group labelled ``treatment`` actually receives the old control
    variant. A real effect therefore flips sign (``tau2 < 0``), which is the
    whole point of the design -- a broken tracker does not flip.

    Phase 2 keeps a slightly smaller effect (``effect_phase2``), so the
    persistence ratio is informative rather than trivially 1: a real but decaying
    effect, which is what most shipped UI changes actually look like.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=date(2026, 10, 20), periods=days, freq="D")
    weekly = np.array([1.02, 1.00, 0.98, 0.99, 1.05, 1.13, 1.07])
    rows = []
    uid = 1
    for d, ts in enumerate(dates):
        phase = 1 if d < switch_day else 2
        # Phase 2 swaps the experience behind each label, so the sign flips.
        effect = effect_phase1 if phase == 1 else -effect_phase2
        day_level = 1 + 0.06 * d / days
        for arm in ("control", "treatment"):
            n = int(rng.normal(users_per_arm_day, 40))
            pre = rng.lognormal(mean=3.3, sigma=0.9, size=n)
            base = rng.lognormal(mean=4.1, sigma=0.7, size=n) * day_level * weekly[ts.dayofweek]
            lift = effect if arm == "treatment" else 0.0
            revenue = base * (1 + lift) * (1 + 0.15 * (pre / pre.mean() - 1))
            rows.append(
                pd.DataFrame(
                    {
                        "user_id": np.arange(uid, uid + n),
                        "phase": phase,
                        "date": ts.date(),
                        "arm": arm,
                        "revenue": np.round(revenue, 2),
                        "orders": rng.poisson(np.clip(revenue / 210, 0.05, 6)),
                        "converted": (revenue > 0).astype(int),
                        "pre_revenue": np.round(pre, 2),
                    }
                )
            )
            uid += n
    return pd.concat(rows, ignore_index=True)


def build_nested_users(
    n: int = 60_000,
    parent_effect: float = 0.03,
    child_effect: float = 0.06,
    orthogonal: bool = True,
    seed: int = 47,
) -> pd.DataFrame:
    """Two-layer experiment: a ranking (child) test inside a UI (parent) test.

    With ``orthogonal=True`` the child bucket is hashed independently, so the
    layer assignment is independent of the parent. With ``orthogonal=False`` the
    child test only runs inside the parent's treatment arm, which is the mistake
    this analysis is designed to catch (complete confounding, no within-parent
    comparison available for the control parent).
    """
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1)
    parent = np.where(rng.random(n) < 0.5, "control", "treatment")
    if orthogonal:
        child = np.where(rng.random(n) < 0.5, "control", "ranking_v2")
    else:
        # Child treatment only reachable inside the parent treatment arm.
        child = np.where((parent == "treatment") & (rng.random(n) < 0.6), "ranking_v2", "control")

    base = rng.lognormal(mean=4.2, sigma=0.75, size=n)
    lift = 1.0 + parent_effect * (parent == "treatment") + child_effect * (child == "ranking_v2")
    revenue = base * lift * (1 + rng.normal(0, 0.08, n))

    return pd.DataFrame(
        {
            "user_id": ids,
            "parent_arm": parent,
            "child_arm": child,
            "revenue": np.round(np.clip(revenue, 0.01, None), 2),
            # Both outcomes carry the *same* relative lift on purpose: a 6% child
            # effect should read as +6% on revenue and +6% on conversion, so the
            # demo cannot accidentally compare two different numbers.
            "converted": rng.binomial(1, np.clip(0.18 * lift, 0.01, 0.6)),
            "device": rng.choice(DEVICES, size=n, p=DEVICE_WEIGHTS),
        }
    )


def build_two_sided_market(n_markets: int = 240, seed: int = 53) -> pd.DataFrame:
    """Marketplace 2x2 design: treat buyers, sellers, both, or neither.

    Buyer treatment lifts buyer conversion by 8% and spills 1.5% onto seller
    listing rates; seller treatment lifts listing rates by 6% and spills 2% onto
    buyer conversion; the interaction is slightly negative (cannibalisation),
    which is exactly the effect a one-sided readout misses.

    Metrics are expressed as *rates* (per buyer / per seller), not totals: with
    total GMV the between-market size variation (CV ~25%) swamps a 2% spillover,
    and the analysis would be measuring market size rather than the treatment.
    """
    rng = np.random.default_rng(seed)
    cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
    rows = []
    for i in range(n_markets):
        buyer_t, seller_t = cells[i % 4]
        size = rng.lognormal(mean=11.0, sigma=0.25)
        buyer_cvr = 0.045 * (1 + 0.08 * buyer_t + 0.02 * seller_t) * rng.normal(1, 0.04)
        listing_rate = 0.32 * (1 + 0.06 * seller_t + 0.015 * buyer_t) * rng.normal(1, 0.04)
        interaction = -0.015 if (buyer_t and seller_t) else 0.0
        gmv_per_buyer = buyer_cvr * 260 * (1 + interaction) * rng.normal(1, 0.03)
        sellers = size * 0.09
        rows.append(
            {
                "market": f"geo_{i:03d}",
                "buyer_treat": buyer_t,
                "seller_treat": seller_t,
                "buyers": int(size),
                "sellers": int(sellers),
                "buyer_conversion": round(float(buyer_cvr), 5),
                "seller_listings": round(float(sellers * listing_rate), 1),
                "listing_rate": round(float(listing_rate), 5),
                "gmv_per_buyer": round(float(gmv_per_buyer), 2),
                "gmv": round(float(size * gmv_per_buyer), 2),
            }
        )
    return pd.DataFrame(rows)


def build_rdd_scores(
    n: int = 6000, cutoff: float = 600.0, jump: float = 22.0, fuzzy: bool = False, seed: int = 59
) -> pd.DataFrame:
    """Credit-score style running variable with a treatment threshold.

    ``score`` is the running variable, ``treated`` switches on at ``cutoff``.
    The outcome has a linear slope in the score plus a level jump at the cutoff,
    so a local linear fit on each side must recover ``jump``. With
    ``fuzzy=True`` compliance is imperfect (85% above, 15% below), and the Wald
    ratio should still recover the same jump.
    """
    rng = np.random.default_rng(seed)
    score = rng.normal(cutoff, 110, n)
    above = (score >= cutoff).astype(float)
    if fuzzy:
        p_treat = np.where(above == 1, 0.85, 0.15)
        treated = rng.binomial(1, p_treat)
    else:
        treated = above
    base = 60.0 + 0.10 * (score - cutoff)
    # Real effect flows through actual treatment, not through the threshold, so
    # the fuzzy design needs the Wald ratio to recover `jump`.
    outcome = base + jump * (treated if fuzzy else above) + rng.normal(0, 14, n)
    covariate = 30.0 + 0.02 * (score - cutoff) + rng.normal(0, 8, n)
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "score": np.round(score, 2),
            "above_cutoff": above.astype(int),
            "treated": treated.astype(int),
            "outcome": np.round(outcome, 2),
            "pre_spend": np.round(np.clip(covariate, 0.1, None), 2),
        }
    )


def build_uplift_users(n: int = 24_000, seed: int = 61) -> pd.DataFrame:
    """User-level data with a known heterogeneous treatment effect.

    The true per-user effect is stored in ``true_tau`` so uplift models can be
    scored against the truth, not just against a Qini curve. The effect is large
    for new users and feed-channel users, small for long-tenure organic users -
    a pattern a two-model learner can find but an average treatment effect hides.
    """
    rng = np.random.default_rng(seed)
    tenure = rng.exponential(scale=260, size=n).clip(3, 2000)
    spend = rng.lognormal(mean=3.2, sigma=0.9, size=n)
    sessions = rng.poisson(6, n) + 1
    channel = rng.choice(CHANNELS, size=n, p=CHANNEL_WEIGHTS)
    device = rng.choice(DEVICES, size=n, p=DEVICE_WEIGHTS)
    group = np.where(rng.random(n) < 0.5, "control", "treatment")

    tau = (
        6.0
        + 14.0 * (tenure < 90)
        + 7.0 * (channel == "feed")
        + 4.0 * (device == "iOS")
        - 3.0 * (spend > np.quantile(spend, 0.8))
    )
    base = (
        12.0
        + 0.55 * spend
        + 1.4 * sessions
        + 0.02 * tenure
        + 5.0 * (channel == "search")
        + rng.normal(0, 12, n)
    )
    revenue = np.clip(base + tau * (group == "treatment"), 0.0, None)
    converted = (revenue > 40).astype(int)

    return pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "group": group,
            "tenure_days": np.round(tenure, 0),
            "spend_30d": np.round(spend, 2),
            "sessions_30d": sessions,
            "channel": channel,
            "device": device,
            "revenue": np.round(revenue, 2),
            "converted": converted,
            "true_tau": np.round(tau, 2),
        }
    )


def build_its_series(
    days: int = 140, intervention_day: int = 95, effect: float = 0.12, seed: int = 67
) -> pd.DataFrame:
    """Interrupted time series with a level shift, plus two control markets.

    The controls share the weekly shape and trend but not the intervention, so a
    structural time-series model can use them to sharpen the counterfactual.
    Noise is AR(1): ignoring that autocorrelation is the classic way to make an
    interrupted-time-series p-value look far better than it is.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=date(2026, 10, 20), periods=days, freq="D")
    weekly = np.array([1.02, 1.00, 0.98, 0.99, 1.05, 1.13, 1.07])
    trend = np.linspace(0.92, 1.10, days)
    common = np.zeros(days)
    for i in range(1, days):
        common[i] = 0.6 * common[i - 1] + rng.normal(0, 0.018)
    shift = np.where(np.arange(days) >= intervention_day, 1 + effect, 1.0)

    rows = []
    base = 3.1e6
    for i, ts in enumerate(dates):
        level = base * trend[i] * weekly[ts.dayofweek] * (1 + common[i]) * shift[i]
        c1 = base * 0.62 * trend[i] * weekly[ts.dayofweek] * (1 + 0.9 * common[i]) * rng.normal(1, 0.01)
        c2 = base * 0.48 * trend[i] * weekly[ts.dayofweek] * (1 + 0.8 * common[i]) * rng.normal(1, 0.01)
        rows.append(
            {
                "date": ts.date(),
                "gmv": round(float(level * rng.normal(1, 0.012)), 2),
                "control_gmv_1": round(float(c1), 2),
                "control_gmv_2": round(float(c2), 2),
                "orders": int(level / 210),
            }
        )
    frame = pd.DataFrame(rows)
    frame["intervention"] = (np.arange(days) >= intervention_day).astype(int)
    return frame


def build_demo_warehouse(warehouse: Path, force: bool = False) -> Tuple[Path, pd.DataFrame]:
    """Create the DuckDB warehouse (idempotent unless ``force``)."""
    warehouse = Path(warehouse)
    if warehouse.exists() and not force:
        con = duckdb.connect(str(warehouse), read_only=True)
        try:
            orders = con.execute("SELECT * FROM orders_daily").df()
        finally:
            con.close()
        return warehouse, orders

    warehouse.parent.mkdir(parents=True, exist_ok=True)
    orders = build_orders_daily()
    users = build_experiment_users()
    panel = build_city_panel()
    promo = build_promo_users()
    switchback = build_switchback_log()
    reverse = build_reverse_log()
    nested = build_nested_users()
    two_sided = build_two_sided_market()
    rdd = build_rdd_scores()
    uplift = build_uplift_users()
    its = build_its_series()

    if warehouse.exists():
        warehouse.unlink()
    con = duckdb.connect(str(warehouse))
    try:
        con.register("orders_df", orders)
        con.register("users_df", users)
        con.register("panel_df", panel)
        con.register("promo_df", promo)
        con.register("switchback_df", switchback)
        con.register("reverse_df", reverse)
        con.register("nested_df", nested)
        con.register("two_sided_df", two_sided)
        con.register("rdd_df", rdd)
        con.register("uplift_df", uplift)
        con.register("its_df", its)
        con.execute("CREATE TABLE orders_daily AS SELECT * FROM orders_df")
        con.execute("CREATE TABLE experiment_users AS SELECT * FROM users_df")
        con.execute("CREATE TABLE city_panel AS SELECT * FROM panel_df")
        con.execute("CREATE TABLE promo_users AS SELECT * FROM promo_df")
        con.execute("CREATE TABLE switchback_log AS SELECT * FROM switchback_df")
        con.execute("CREATE TABLE reverse_log AS SELECT * FROM reverse_df")
        con.execute("CREATE TABLE nested_users AS SELECT * FROM nested_df")
        con.execute("CREATE TABLE two_sided_market AS SELECT * FROM two_sided_df")
        con.execute("CREATE TABLE rdd_scores AS SELECT * FROM rdd_df")
        con.execute("CREATE TABLE uplift_users AS SELECT * FROM uplift_df")
        con.execute("CREATE TABLE its_series AS SELECT * FROM its_df")
    finally:
        con.close()
    return warehouse, orders
