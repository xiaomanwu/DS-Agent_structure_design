"""Root-cause drill-down decision tree: structure, ranking and chart."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from ds_agent import charts
from ds_agent.semantic import SemanticLayer
from ds_agent.tools.root_cause import root_cause_tree, tree_table


DB = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "demo_data" / "warehouse.duckdb")
DIMS = ["channel", "device", "region", "category", "is_new_user"]


@pytest.fixture(scope="module")
def windows():
    table = SemanticLayer().default_table()
    metric = table.metrics["gmv"]
    # Join whatever mode the warehouse is already open in suite-wide.
    con = duckdb.connect(DB)
    current = con.execute("SELECT * FROM orders_daily WHERE date >= DATE '2026-10-16'").df()
    baseline = con.execute(
        "SELECT * FROM orders_daily WHERE date < DATE '2026-10-16' AND date >= DATE '2026-09-16'"
    ).df()
    con.close()
    return current, baseline, metric, table


def test_tree_finds_the_planted_root_cause(windows):
    current, baseline, metric, table = windows
    tree = root_cause_tree(current, baseline, metric, DIMS, max_depth=3, value_labels=table.value_labels)
    assert tree["n_nodes"] > 1
    assert tree["depth_reached"] >= 2
    # The demo warehouse plants an iOS payment-SDK regression: some ranked
    # root-cause path must go through device=iOS.
    assert any(
        any(s["dimension"] == "device" and s["value"] == "iOS" for s in leaf["path"])
        for leaf in tree["leaves"]
    )
    # Value labels must replace 0/1 codes in every path label.
    for leaf in tree["leaves"]:
        assert "0.0" not in leaf["path_label"] and "1.0" not in leaf["path_label"]


def test_tree_structure_is_parent_child_consistent(windows):
    current, baseline, metric, table = windows
    tree = root_cause_tree(current, baseline, metric, DIMS, max_depth=3)
    nodes = tree["nodes"]
    by_path = {n["path_label"]: n for n in nodes}
    for node in nodes:
        for child in node["children"]:
            assert child["depth"] == node["depth"] + 1
            assert child["path_label"] in by_path
            # A child's slice cannot move more than its parent's slice.
            assert abs(float(child["delta"])) <= abs(float(node["delta"])) + 1e-6


def test_tree_respects_max_depth_and_single_dimension(windows):
    current, baseline, metric, table = windows
    shallow = root_cause_tree(current, baseline, metric, DIMS, max_depth=1)
    assert shallow["depth_reached"] <= 1
    single = root_cause_tree(current, baseline, metric, ["device"], max_depth=3)
    dims_used = {n["dimension"] for n in single["nodes"] if n["dimension"]}
    assert dims_used <= {"device"}


def test_tree_table_and_chart_render(windows):
    current, baseline, metric, table = windows
    tree = root_cause_tree(current, baseline, metric, DIMS, max_depth=2, value_labels=table.value_labels)
    df = tree_table(tree, "zh")
    assert "下钻路径" in df.columns
    assert len(df) == tree["n_nodes"]
    uri = charts.root_cause_tree_chart(tree["nodes"])
    assert uri.startswith("data:image/png;base64,")
    assert tree["narrative"]
