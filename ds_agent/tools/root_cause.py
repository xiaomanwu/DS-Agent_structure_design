"""Multi-level drill-down decision tree for metric-move root causes.

Contribution decomposition answers "how much did each segment add along one
dimension". Real root-cause work goes deeper: after pinning the primary
dimension you re-ask the question *inside* the worst segment, along the next
most informative dimension, and keep going until the path explains the move.
That is exactly what this module automates as a decision tree:

- each node holds a filtered data slice (e.g. ``device=iOS → is_new_user=returning``),
  the metric change inside it, and the share of the *root* move it carries;
- at every node the remaining dimensions are scored (top-segment contribution
  weighted by concentration) and the winner branches into its top segments;
- branching stops at ``max_depth``, when no remaining dimension explains
  ``min_share`` of the node's move, or when the node budget is exhausted;
- leaves are ranked by |share of root| — that ranked list *is* the root-cause
  shortlist, each entry a readable path like ``device=iOS > is_new_user=returning``.

The greedy choice per node is deliberate: it mirrors how analysts drill and
keeps every node's numbers exact (the same additive/ratio decompositions as
``contribution.py``), which a fitted surrogate tree could not promise.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..semantic import Metric
from .contribution import decompose_change


def _node_delta(cur: pd.DataFrame, base: pd.DataFrame, metric: Metric) -> float:
    """Metric change inside the node's slice (level for additive, rate for ratio)."""
    num, den = metric.numerator, metric.denominator
    if not metric.is_ratio:
        return float(cur[num].sum() - base[num].sum())
    den_cur = float(cur[den].sum()) if den else float("nan")
    den_base = float(base[den].sum()) if den else float("nan")
    r_cur = float(cur[num].sum()) / den_cur if den_cur else float("nan")
    r_base = float(base[num].sum()) / den_base if den_base else float("nan")
    return r_cur - r_base


def root_cause_tree(
    current: pd.DataFrame,
    baseline: pd.DataFrame,
    metric: Metric,
    dimensions: Sequence[str],
    max_depth: int = 3,
    top_segments: int = 2,
    min_share: float = 0.08,
    max_nodes: int = 30,
    value_labels: Optional[Dict[str, Dict[object, str]]] = None,
) -> Dict[str, object]:
    """Recursive drill-down decision tree over the provided dimensions.

    Parameters mirror the analysis conventions: ``max_depth`` bounds how many
    levels the analyst would realistically read; ``top_segments`` branches only
    into the few segments that carry the move; ``min_share`` ignores dimensions
    whose best segment explains less than that share of the node's change.
    ``value_labels`` maps dimension codes to readable labels (from the semantic
    layer) so paths read ``is_new_user=新客`` instead of ``is_new_user=0.0``.
    """
    if not dimensions:
        raise ValueError("at least one dimension is required")
    labels = value_labels or {}

    budget = {"nodes": 0, "cap": max_nodes}
    root_delta = _node_delta(current, baseline, metric)

    def label_for(dim: str, value: object) -> str:
        mapping = labels.get(dim)
        if mapping:
            for key, text in mapping.items():
                try:
                    # Float-tolerant: raw frames carry 0.0/1.0 while the semantic
                    # layer labels keys as 0/1 or "0"/"1".
                    if float(key) == float(value):
                        return text
                    continue
                except (TypeError, ValueError):
                    if str(key) == str(value):
                        return text
        return str(value)

    def expand(cur: pd.DataFrame, base: pd.DataFrame, dims: List[str], depth: int,
               path: List[Dict[str, str]]) -> Dict[str, object]:
        budget["nodes"] += 1
        delta = _node_delta(cur, base, metric)
        node: Dict[str, object] = {
            "depth": depth,
            "path": list(path),
            "path_label": " > ".join(f"{s['dimension']}={s['value']}" for s in path) or "ALL",
            "delta": delta,
            "share_of_root": (delta / root_delta) if root_delta else float("nan"),
            "dimension": None,
            "children": [],
            "n_cur": int(len(cur)),
        }

        if depth >= max_depth or not dims or budget["nodes"] >= budget["cap"]:
            return node

        # Score every remaining dimension inside this slice.
        best = None
        for dim in dims:
            try:
                dec = decompose_change(cur, base, metric, dim)
            except (KeyError, ValueError):
                continue
            segs = dec["segments"]
            if segs.empty:
                continue
            top = segs.iloc[0]
            score = abs(float(top["contribution"])) * (0.5 + 0.5 * dec["concentration"]["top1_share"])
            if best is None or score > best["score"]:
                best = {"dim": dim, "dec": dec, "score": score}

        if best is None:
            return node

        segs: pd.DataFrame = best["dec"]["segments"]
        node_delta_abs = abs(delta) or 1.0
        branches = [
            row for _, row in segs.iterrows()
            if abs(float(row["contribution"])) >= min_share * node_delta_abs
        ][:top_segments]
        if not branches:
            return node

        node["dimension"] = best["dim"]
        remaining = [d for d in dims if d != best["dim"]]
        for row in branches:
            value = row["segment"]
            cur_sub = cur[cur[best["dim"]] == value]
            base_sub = base[base[best["dim"]] == value]
            if len(cur_sub) == 0 and len(base_sub) == 0:
                continue
            child = expand(
                cur_sub, base_sub, remaining, depth + 1,
                path + [{"dimension": best["dim"], "value": label_for(best["dim"], value)}],
            )
            child["contribution"] = float(row["contribution"])
            node["children"].append(child)
        return node

    root = expand(current, baseline, [d for d in dimensions if d], 0, [])

    # Ranked leaves: paths that actually carry the move.
    leaves: List[Dict[str, object]] = []

    def collect(node: Dict[str, object]) -> None:
        if not node["children"]:
            leaves.append(node)
            return
        for child in node["children"]:
            collect(child)

    collect(root)
    # A node with no children because depth/budget ran out still counts; rank by |delta|.
    leaves.sort(key=lambda n: abs(float(n["delta"])), reverse=True)

    flat_nodes: List[Dict[str, object]] = []

    def flatten(node: Dict[str, object]) -> None:
        flat_nodes.append(node)
        for child in node["children"]:
            flatten(child)

    flatten(root)

    return {
        "metric": metric.name,
        "root_delta": root_delta,
        "root": root,
        "leaves": leaves[:5],
        "nodes": flat_nodes,
        "n_nodes": len(flat_nodes),
        "depth_reached": max((n["depth"] for n in flat_nodes), default=0),
        "narrative": _tree_narrative(leaves, root_delta, metric.name),
    }


def _tree_narrative(leaves: List[Dict[str, object]], root_delta: float, metric: str) -> str:
    """One-line summary of the top root-cause paths."""
    if not leaves or not root_delta:
        return "No drill-down path explains a material share of the move."
    top = leaves[0]
    path = top["path_label"]
    share = abs(float(top["delta"])) / abs(root_delta)
    more = ""
    if len(leaves) > 1 and abs(float(leaves[1]["delta"])) > 0:
        second_share = abs(float(leaves[1]["delta"])) / abs(root_delta)
        more = f" Next: {leaves[1]['path_label']} ({second_share:.0%})."
    direction = "decline" if root_delta < 0 else "increase"
    return (
        f"Root-cause tree: the {direction} in {metric} concentrates at **{path}** "
        f"({float(top['delta']):+,.4g}, ~{share:.0%} of the root move).{more}"
    )


def tree_table(tree: Dict[str, object], lang: str = "zh") -> pd.DataFrame:
    """Depth-indented table of the tree for blocks / reports."""
    zh = lang != "en"
    rows = []
    for node in tree["nodes"]:
        indent = "　" * int(node["depth"])  # full-width space keeps alignment in md/html
        label = indent + (node["path_label"] or "ALL")
        rows.append(
            [
                int(node["depth"]),
                label,
                f"{float(node['delta']):+,.4g}",
                f"{float(node['share_of_root']):+.0%}" if np.isfinite(node["share_of_root"]) else "-",
                node["dimension"] or ("-"),
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            zh and "层级" or "depth",
            zh and "下钻路径" or "drill-down path",
            zh and "指标变化" or "metric change",
            zh and "占总变动" or "share of root",
            zh and "下钻维度" or "drill dimension",
        ],
    )
