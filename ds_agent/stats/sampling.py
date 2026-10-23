"""Sampling theory made executable: sample size, draw, representativeness.

Three questions a data scientist actually asks about sampling, each with a
function the agent can call and cite:

1. **How large a sample do I need?** — ``sample_size_mean`` /
   ``sample_size_proportion`` invert the two-sided z-test power equation,
   exactly the formula an experiment-design readout uses.
2. **How do I draw the sample?** — ``draw_sample`` implements simple random,
   systematic and stratified sampling with a fixed seed, so the draw is
   reproducible and auditable.
3. **Is my sample representative?** — ``representativeness`` compares the
   sample against the population with the proper test (chi-square GOF for a
   categorical distribution, Welch t + KS for a numeric measure) instead of
   eyeballing means.

Theory anchor (see docs/methodology.md §7): by the central limit theorem the
sample mean is approximately normal with standard error sigma/sqrt(n), so a
two-sided test with size alpha and power 1-beta needs

    n = (z_{1-alpha/2} + z_{1-beta})^2 * sigma^2 / delta^2        (means)
    n = (z_{1-alpha/2} + z_{1-beta})^2 * p0(1-p0) / delta^2       (proportions)

and any estimate should ship with its standard error and confidence interval.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from scipy import stats as sps


def _z(level: float) -> float:
    """Two-sided normal quantile: _z(0.95) = 1.96."""
    return float(sps.norm.ppf(0.5 + level / 2.0))


def sample_size_mean(
    sd: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> Dict[str, Any]:
    """Sample size per group for detecting a mean shift of ``mde`` (2-sided)."""
    if sd <= 0 or mde <= 0:
        raise ValueError("sd and mde must be positive")
    z_a, z_b = _z(1 - alpha), _z(power)
    n = ((z_a + z_b) ** 2) * (sd**2) / (mde**2)
    return {
        "kind": "mean",
        "sd": sd,
        "mde": mde,
        "alpha": alpha,
        "power": power,
        "z_alpha": round(z_a, 4),
        "z_power": round(z_b, 4),
        "n_per_group": int(np.ceil(n)),
        "n_total": int(np.ceil(n)) * 2,
        "formula": "n = (z_{1-α/2}+z_{1-β})² · σ² / δ²",
    }


def sample_size_proportion(
    p0: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> Dict[str, Any]:
    """Sample size per group for detecting a proportion shift of ``mde``.

    Uses the classical normal-approximation formula with the baseline
    variance p0(1-p0); for tiny p0 (<0.01) exact binomial sizing is safer —
    the result carries a note saying so.
    """
    if not 0 < p0 < 1 or mde <= 0:
        raise ValueError("need 0<p0<1 and a positive absolute mde")
    z_a, z_b = _z(1 - alpha), _z(power)
    n = ((z_a + z_b) ** 2) * p0 * (1 - p0) / (mde**2)
    note = ""
    if p0 < 0.01:
        note = "p0 < 1%: the normal approximation is optimistic; verify with exact binomial sizing"
    return {
        "kind": "proportion",
        "p0": p0,
        "mde": mde,
        "alpha": alpha,
        "power": power,
        "z_alpha": round(z_a, 4),
        "z_power": round(z_b, 4),
        "n_per_group": int(np.ceil(n)),
        "n_total": int(np.ceil(n)) * 2,
        "formula": "n = (z_{1-α/2}+z_{1-β})² · p₀(1-p₀) / δ²",
        "note": note,
    }


def draw_sample(
    df: pd.DataFrame,
    n: Optional[int] = None,
    *,
    method: str = "srs",
    by: Optional[str] = None,
    seed: int = 7,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Draw a reproducible sample: ``srs``, ``systematic`` or ``stratified``.

    * srs — uniform without replacement (numpy default_rng permutation);
    * systematic — every k-th row after a random start (k = N/n);
    * stratified — proportional allocation within ``by`` groups, SRS inside
      each stratum, which guarantees small groups keep representation.
    """
    if method not in ("srs", "systematic", "stratified"):
        raise ValueError("method must be srs | systematic | stratified")
    total = len(df)
    if not n or n >= total:
        n = total if n is None else min(n, total)
    rng = np.random.default_rng(seed)
    log: Dict[str, Any] = {"method": method, "n_requested": int(n), "population": int(total), "seed": seed}

    if method == "srs" or (method == "stratified" and not by):
        idx = rng.choice(total, size=n, replace=False)
        log["fraction"] = round(n / total, 4)
        return df.iloc[np.sort(idx)].copy(), log

    if method == "systematic":
        k = max(1, total // n)
        start = int(rng.integers(0, k))
        idx = np.arange(start, total, k)[:n]
        log["k"] = k
        log["start"] = start
        log["fraction"] = round(len(idx) / total, 4)
        return df.iloc[idx].copy(), log

    # stratified: proportional allocation, then SRS per stratum
    strata = df[by]
    counts = strata.value_counts()
    picked: list[int] = []
    allocation: Dict[str, Dict[str, int]] = {}
    for stratum, size in counts.items():
        take = int(round(n * size / total))
        take = min(max(take, 1 if size > 0 else 0), int(size))
        members = np.flatnonzero((strata == stratum).to_numpy())
        chosen = rng.choice(members, size=take, replace=False)
        picked.extend(int(i) for i in chosen)
        allocation[str(stratum)] = {"population": int(size), "sampled": int(take)}
    log["stratum_column"] = str(by)
    log["allocation"] = allocation
    log["fraction"] = round(len(picked) / total, 4)
    picked.sort()
    return df.iloc[picked].copy(), log


def standard_error(series: Sequence[float]) -> Dict[str, Any]:
    """Mean with its standard error and 95% CI — the CLT in one table row."""
    x = np.asarray([v for v in series if np.isfinite(v)], dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 observations")
    se = float(x.std(ddof=1) / np.sqrt(n))
    mean = float(x.mean())
    crit = float(sps.t.ppf(0.975, n - 1))
    fpc = None
    return {
        "n": n,
        "mean": round(mean, 4),
        "se": round(se, 6),
        "ci95": [round(mean - crit * se, 4), round(mean + crit * se, 4)],
        "cv": round(float(x.std(ddof=1) / abs(mean)), 4) if mean else None,
        "formula": "SE = s/√n · √((N-n)/(N-1)) (finite-population factor omitted, n≪N)",
    }


def representativeness(
    sample: pd.Series,
    population: pd.Series,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Test whether a sample matches its population distribution.

    Categorical column → chi-square goodness-of-fit with the population
    proportions as expected shares (small expected counts fall back to the
    exact multinomial note). Numeric column → Welch t on the means plus a
    two-sample KS on the whole distribution, so "same average" and "same
    shape" are checked separately.
    """
    s = sample.dropna()
    p = population.dropna()
    if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(p):
        from ds_agent.stats.tests import ks_two_sample, welch_t

        t_res = welch_t(s.to_numpy(dtype=float), p.to_numpy(dtype=float), alpha=alpha)
        ks = ks_two_sample(s.to_numpy(dtype=float), p.to_numpy(dtype=float), alpha=alpha)
        return {
            "column_type": "numeric",
            "tests": [t_res, ks],
            "primary": [t_res, ks],
        }
    from ds_agent.stats.tests import chi_square_gof

    # Expected counts under the population proportions.
    props = p.value_counts(normalize=True)
    categories = list(props.index)
    s_counts = s.value_counts()
    observed = np.array([int(s_counts.get(c, 0)) for c in categories], dtype=float)
    gof = chi_square_gof(observed, expected_ratio=props.to_numpy(dtype=float), alpha=alpha)
    # Table the viewer actually reads: share vs share.
    rows = [
        {
            "category": str(c),
            "sample_share": round(observed[i] / max(observed.sum(), 1), 4),
            "population_share": round(float(props.iloc[i]), 4),
            "diff": round(observed[i] / max(observed.sum(), 1) - float(props.iloc[i]), 4),
        }
        for i, c in enumerate(categories)
    ]
    rows.sort(key=lambda r: abs(r["diff"]), reverse=True)
    return {
        "column_type": "categorical",
        "tests": [gof],
        "primary": [gof],
        "shares": rows[:8],
    }
