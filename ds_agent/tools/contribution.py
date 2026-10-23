"""Contribution decomposition and root-cause attribution for metric moves.

Two exact decompositions are used:

Additive metrics (GMV, orders, sessions)
    dTotal = sum_i (v_i,current - v_i,baseline)
    contribution_i = dTotal_i / dTotal
    Plus segmentation of new / lost segments, which additive maths handles for free.

Ratio metrics (AOV, conversion rate, refund rate)
    R = sum_i w_i * r_i  with w_i = d_i / D (volume share)
    dR = sum_i w_i,base * dr_i            (rate effect: behaviour changed)
       + sum_i r_i,base * dw_i            (mix effect: traffic composition changed)
       + sum_i dw_i * dr_i                (interaction)
    The three terms sum exactly to dR; the residual is asserted to be ~0.

Attribution across several dimensions uses "peeling": rank dimensions by the
amount of change they explain, then drill into the top segment along the next
dimension. Every step reports how much of the total move is explained, so a
reader can see when the answer is only partial.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..semantic import Metric


def _segment_sums(df: pd.DataFrame, metric: Metric, dim: str) -> pd.DataFrame:
    """Aggregate numerator/denominator per segment for one dimension."""
    if dim not in df.columns:
        raise KeyError(f"dimension '{dim}' not present in frame")
    num_col = metric.numerator
    den_col = metric.denominator
    agg = {num_col: "sum"}
    if den_col:
        agg[den_col] = "sum"
    grouped = df.groupby(dim, dropna=False).agg(agg).reset_index()
    grouped = grouped.rename(columns={dim: "segment"})
    grouped["numerator"] = grouped[num_col].astype(float)
    grouped["denominator"] = grouped[den_col].astype(float) if den_col else np.nan
    return grouped[["segment", "numerator", "denominator"]]


def decompose_change(
    current: pd.DataFrame, baseline: pd.DataFrame, metric: Metric, dim: str
) -> Dict[str, object]:
    """Exact decomposition of a metric change along one dimension."""
    cur = _segment_sums(current, metric, dim)
    base = _segment_sums(baseline, metric, dim)
    merged = cur.merge(base, on="segment", how="outer", suffixes=("_cur", "_base")).fillna(0.0)

    if not metric.is_ratio:
        merged["value_cur"] = merged["numerator_cur"]
        merged["value_base"] = merged["numerator_base"]
        merged["delta"] = merged["value_cur"] - merged["value_base"]
        total_cur, total_base = merged["value_cur"].sum(), merged["value_base"].sum()
        total_delta = total_cur - total_base
        merged["contribution"] = merged["delta"]
        merged["contribution_pct"] = np.where(total_delta != 0, merged["delta"] / total_delta, np.nan)
        merged["growth"] = np.where(
            merged["value_base"] > 0, merged["delta"] / merged["value_base"], np.where(merged["value_cur"] > 0, np.inf, 0.0)
        )
        merged["rate_effect"] = merged["delta"]
        merged["mix_effect"] = 0.0
        merged["interaction"] = 0.0
        residual = float(total_delta - merged["delta"].sum())
        result = {
            "metric": metric.name,
            "kind": "additive",
            "dimension": dim,
            "total_current": float(total_cur),
            "total_baseline": float(total_base),
            "total_delta": float(total_delta),
            "total_rel_change": float(total_delta / total_base) if total_base else float(np.nan),
            "residual": residual,
            "segments": merged.sort_values("contribution", key=lambda s: s.abs(), ascending=False),
        }
    else:
        merged["value_cur"] = merged["numerator_cur"] / merged["denominator_cur"].replace(0, np.nan)
        merged["value_base"] = merged["numerator_base"] / merged["denominator_base"].replace(0, np.nan)
        total_num_cur, total_den_cur = merged["numerator_cur"].sum(), merged["denominator_cur"].sum()
        total_num_base, total_den_base = merged["numerator_base"].sum(), merged["denominator_base"].sum()
        r_cur = total_num_cur / total_den_cur if total_den_cur else np.nan
        r_base = total_num_base / total_den_base if total_den_base else np.nan
        merged["w_cur"] = merged["denominator_cur"] / total_den_cur if total_den_cur else np.nan
        merged["w_base"] = merged["denominator_base"] / total_den_base if total_den_base else np.nan
        dr = merged["value_cur"].fillna(0) - merged["value_base"].fillna(0)
        dw = merged["w_cur"].fillna(0) - merged["w_base"].fillna(0)
        merged["rate_effect"] = merged["w_base"].fillna(0) * dr
        merged["mix_effect"] = merged["value_base"].fillna(0) * dw
        merged["interaction"] = dw * dr
        merged["contribution"] = merged["rate_effect"] + merged["mix_effect"] + merged["interaction"]
        total_delta = r_cur - r_base
        merged["contribution_pct"] = np.where(total_delta != 0, merged["contribution"] / total_delta, np.nan)
        merged["growth"] = np.where(
            merged["value_base"].abs() > 0,
            merged["value_cur"] / merged["value_base"] - 1,
            np.nan,
        )
        residual = float(total_delta - merged["contribution"].sum()) if np.isfinite(total_delta) else float("nan")
        result = {
            "metric": metric.name,
            "kind": "ratio",
            "dimension": dim,
            "total_current": float(r_cur),
            "total_baseline": float(r_base),
            "total_delta": float(total_delta),
            "total_rel_change": float(total_delta / r_base) if r_base else float(np.nan),
            "residual": residual,
            "rate_effect_total": float(merged["rate_effect"].sum()),
            "mix_effect_total": float(merged["mix_effect"].sum()),
            "interaction_total": float(merged["interaction"].sum()),
            "segments": merged.sort_values("contribution", key=lambda s: s.abs(), ascending=False),
        }

    # Segments that did not exist in the baseline (or disappeared) are called out
    # separately: they are a frequent root cause of step changes.
    result["new_segments"] = [
        s for s, vb in zip(result["segments"]["segment"], result["segments"]["value_base"]) if (pd.isna(vb) or vb == 0)
    ]
    result["lost_segments"] = [
        s for s, vc in zip(result["segments"]["segment"], result["segments"]["value_cur"]) if (pd.isna(vc) or vc == 0)
    ]
    result["concentration"] = _concentration(result["segments"])
    result["explained_abs"] = float(result["segments"]["contribution"].abs().sum())
    result["narrative"] = _decompose_narrative(result)
    return result


def _concentration(segments: pd.DataFrame) -> Dict[str, float]:
    """How concentrated is the move? Pareto-style diagnostics."""
    contrib = segments["contribution"].abs().sort_values(ascending=False).to_numpy()
    total = contrib.sum()
    if total <= 0:
        return {"top1_share": 0.0, "top3_share": 0.0, "segments": len(contrib)}
    return {
        "top1_share": float(contrib[0] / total),
        "top3_share": float(contrib[:3].sum() / total),
        "segments": int(len(contrib)),
        "segments_needed_for_80pct": int(np.searchsorted(np.cumsum(contrib) / total, 0.8) + 1),
    }


def _decompose_narrative(result: Dict[str, object]) -> str:
    seg: pd.DataFrame = result["segments"]
    if seg.empty:
        return "No segments to decompose."
    top = seg.iloc[0]
    metric = result["metric"]
    direction = "decreased" if result["total_delta"] < 0 else "increased"
    unit = "" if result["kind"] == "ratio" else ""
    text = (
        f"{metric} {direction} from {result['total_baseline']:,.4g} to {result['total_current']:,.4g} "
        f"({result['total_delta']:+,.4g}, {result['total_rel_change']:+.2%}) along '{result['dimension']}'. "
        f"Largest contributor: {top['segment']} ({top['contribution']:+,.4g}, "
        f"{top['contribution_pct']:+.1%} of the move, own change {top['growth']:+.2%})."
    )
    if result["kind"] == "ratio":
        text += (
            f" Overall: rate effect {result['rate_effect_total']:+.4g}, mix effect "
            f"{result['mix_effect_total']:+.4g}, interaction {result['interaction_total']:+.4g}."
        )
    conc = result["concentration"]
    text += (
        f" Concentration: top-1 explains {conc['top1_share']:.0%} of absolute movement, "
        f"{conc['segments_needed_for_80pct']} segment(s) cover 80%."
    )
    return text


def attribute_change(
    current: pd.DataFrame,
    baseline: pd.DataFrame,
    metric: Metric,
    dimensions: Sequence[str],
    max_depth: int = 2,
    top_k: int = 5,
    focus: Optional[str] = None,
) -> Dict[str, object]:
    """Rank dimensions by explained movement, then drill down along the best next dimension.

    This is the "peeling" strategy analysts use in practice: it is greedy, but
    every step is exact and reports the share of the total move it accounts for.

    ``focus`` pins the primary dimension. When the user explicitly asks for
    "GMV by category", answering with device because it scored higher is
    technically better and practically useless.
    """
    if not dimensions:
        raise ValueError("at least one dimension is required")

    per_dimension: List[Dict[str, object]] = []
    for dim in dimensions:
        try:
            dec = decompose_change(current, baseline, metric, dim)
        except KeyError:
            continue
        per_dimension.append(dec)

    if not per_dimension:
        raise ValueError("none of the dimensions exist in the provided frames")

    # Rank dimensions by the absolute size of their largest single contributor,
    # tempered by concentration: a dimension where one segment carries most of the
    # move is a better explanation than one where the move is spread thin.
    def score(dec: Dict[str, object]) -> float:
        conc = dec["concentration"]
        top1_abs = float(dec["segments"]["contribution"].abs().iloc[0])
        return top1_abs * (0.5 + 0.5 * conc["top1_share"])

    ranked = sorted(per_dimension, key=score, reverse=True)
    if focus:
        pinned = [dec for dec in ranked if dec["dimension"] == focus]
        if pinned:
            ranked = pinned + [dec for dec in ranked if dec["dimension"] != focus]
    primary = ranked[0]
    total_delta = primary["total_delta"]
    explained = float(primary["segments"]["contribution"].abs().sum())
    explained_share = explained / abs(total_delta) if total_delta else 0.0

    drilldown: List[Dict[str, object]] = []
    if max_depth > 1 and len(dimensions) > 1:
        top_segment = primary["segments"].iloc[0]["segment"]
        remaining = [d for d in dimensions if d != primary["dimension"]]
        # Restrict both windows to the top segment, then decompose again.
        cur_sub = current[current[primary["dimension"]] == top_segment]
        base_sub = baseline[baseline[primary["dimension"]] == top_segment]
        for dim in remaining:
            try:
                sub = decompose_change(cur_sub, base_sub, metric, dim)
            except (KeyError, ValueError):
                continue
            drilldown.append(sub)
        drilldown.sort(key=lambda d: d["concentration"]["top1_share"] * abs(d["total_delta"]), reverse=True)

    return {
        "metric": metric.name,
        "kind": primary["kind"],
        "total_delta": total_delta,
        "total_rel_change": primary["total_rel_change"],
        "explained_share": explained_share,
        "primary": primary,
        "primary_segment": primary["segments"].iloc[0]["segment"],
        "drilldown": drilldown,
        "dimension_scores": [
            {
                "dimension": d["dimension"],
                "concentration_top1": d["concentration"]["top1_share"],
                "segments_for_80pct": d["concentration"]["segments_needed_for_80pct"],
                "top_segment": d["segments"].iloc[0]["segment"],
                "top_contribution": float(d["segments"].iloc[0]["contribution"]),
            }
            for d in ranked
        ],
        "narrative": _attribute_narrative(primary, drilldown, explained_share),
        "top_segments": primary["segments"].head(top_k)[
            ["segment", "value_base", "value_cur", "contribution", "contribution_pct", "growth"]
        ].to_dict(orient="records"),
        "all_dimensions": {d["dimension"]: d for d in per_dimension},
    }


def _attribute_narrative(primary: Dict[str, object], drilldown: List[Dict[str, object]], explained_share: float) -> str:
    seg = primary["segments"].iloc[0]
    direction = "down" if primary["total_delta"] < 0 else "up"
    text = (
        f"{primary['metric']} moved {direction} {primary['total_rel_change']:+.2%} "
        f"({primary['total_delta']:+,.4g}); the dominant driver is "
        f"{primary['dimension']} = '{seg['segment']}' with {seg['contribution']:+,.4g} "
        f"({seg['contribution_pct']:+.0%} of the move)."
    )
    if drilldown:
        sub = drilldown[0]
        sub_seg = sub["segments"].iloc[0]
        text += (
            f" Within '{seg['segment']}', {sub['dimension']} = '{sub_seg['segment']}' explains "
            f"{sub_seg['contribution']:+,.4g} of its contribution."
        )
    if explained_share and explained_share < 0.9:
        text += f" Warning: the primary dimension only accounts for {explained_share:.0%} of gross movement; residual is diffuse."
    return text


def contribution_table(result: Dict[str, object], top_k: int = 10) -> pd.DataFrame:
    """Renderable contribution table for reports and UI cards."""
    seg: pd.DataFrame = result["segments"].copy()
    cols = ["segment", "value_base", "value_cur", "contribution", "contribution_pct", "growth"]
    if result["kind"] == "ratio":
        cols = ["segment", "w_base", "w_cur", "value_base", "value_cur", "rate_effect", "mix_effect", "contribution", "contribution_pct"]
    out = seg.head(top_k)[[c for c in cols if c in seg.columns]].copy()
    return out.rename(
        columns={
            "segment": result["dimension"],
            "value_base": "baseline",
            "value_cur": "current",
            "contribution_pct": "share_of_change",
        }
    )
