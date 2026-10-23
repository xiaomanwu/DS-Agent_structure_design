"""Use the tools as a library, without the agent loop.

    python examples/02_library_usage.py

Everything the agent does is a plain function call, which is what makes the
answers reproducible and testable.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ds_agent.config import settings
from ds_agent.data import build_demo_warehouse
from ds_agent.data.synthetic import build_its_series
from ds_agent.semantic import semantic
from ds_agent.stats import power as pw
from ds_agent.tools import SQLTool, Warehouse
from ds_agent.tools.anomaly import AnomalyConfig, AnomalyDetector, scan_segment_anomalies
from ds_agent.tools.causal import difference_in_differences, synthetic_control
from ds_agent.tools.contribution import attribute_change
from ds_agent.tools.experiment import ExperimentDesign, analyze_experiment, design_experiment
from ds_agent.tools.experiment_designs import (
    analyze_nested_experiment,
    analyze_reverse,
    analyze_switchback,
    analyze_two_sided,
    sensitivity_analysis,
)
from ds_agent.tools.forecast import compare_methods
from ds_agent.tools.rdd import regression_discontinuity
from ds_agent.tools.ts_causal import arima_intervention, causal_impact
from ds_agent.tools.uplift import causal_forest, uplift_learners


def main() -> None:
    build_demo_warehouse(settings.warehouse)
    wh = Warehouse(settings.warehouse)
    sql = SQLTool(wh)
    table = semantic.default_table()

    # ---------------------------------------------------------------- 1. SQL
    print("\n== 1. metric query ==")
    print(sql.ask("近7天转化率按设备拆分")["sql"])

    # ------------------------------------------------------ 2. anomaly scan
    print("\n== 2. anomaly detection ==")
    series = sql.metric_series("gmv", grain="day")
    detection = AnomalyDetector().detect(series["date"], series["metric_value"].to_numpy(), label="gmv")
    print("model:", detection["model"], "| sigma:", round(detection["sigma"], 4))
    for episode in detection["episodes"][:3]:
        print(
            f"  {episode['start']}~{episode['end']}  {episode['days']}d  {episode['kind']:<11}"
            f" {episode['impact_pct']:+.1%}  p={episode['p_value']:.4f}  severity={episode['severity']}"
        )

    # --------------------------------------------- 2b. time-series baselines
    print("\n== 2b. which baseline? ==")
    values = series["metric_value"].to_numpy(dtype=float)
    backtest = compare_methods(values)
    print(backtest[["label", "mase", "rmse"]].sort_values("mase").to_string(index=False))
    hw = AnomalyDetector(AnomalyConfig(baseline="holt_winters")).detect(
        series["date"], values, label="gmv"
    )
    detail = hw["baseline_detail"]
    print(
        f"  holt-winters: {detail['label']}  MASE={detail['accuracy']['mase']:.4f}"
        f"  episodes={len(hw['episodes'])}"
    )

    # ----------------------------------------------- 3. contribution/attrib
    print("\n== 3. attribution ==")
    import datetime as dt

    from ds_agent.tools.sql import TimeWindow

    dims = list(table.dimensions)
    anchor = wh.max_date(table.name, table.time_column)
    cur_w = TimeWindow(anchor - dt.timedelta(days=6), anchor, "last 7d", 7)
    base_w = TimeWindow(anchor - dt.timedelta(days=13), anchor - dt.timedelta(days=7), "prior 7d", 7)
    metric = table.metrics["gmv"]
    # dimension_expr() applies the semantic layer's code -> label mapping, so the
    # narrative says "returning" instead of leaking the raw 0/1 code.
    select = ", ".join(table.dimension_expr(d) for d in dims) + ", SUM(gmv) AS gmv"
    group_by = ", ".join(str(i + 1) for i in range(len(dims)))
    frame_cur = wh.run(f"SELECT {select} FROM {table.name} WHERE {cur_w.clause('date')} GROUP BY {group_by}").dataframe
    frame_base = wh.run(f"SELECT {select} FROM {table.name} WHERE {base_w.clause('date')} GROUP BY {group_by}").dataframe
    result = attribute_change(frame_cur, frame_base, metric, dims)
    print(" ", result["narrative"])
    print("  top segments:")
    top = result["primary"]["segments"].head(3)
    for _, row in top.iterrows():
        print(f"    {row['segment']:<10} {row['contribution']:>14,.0f}  {row['contribution_pct']:+.0%} of the move")

    # ------------------------------------------------------- 4. experiment
    print("\n== 4. experiment sizing ==")
    design = design_experiment(
        ExperimentDesign(goal="checkout CVR", metric_kind="proportion", baseline=0.03, mde_rel=0.05, daily_traffic=80_000)
    )
    print(" ", design["sizing"]["n_total"], "users,", design["duration"]["recommended_days"], "days")
    print("  power curve:", pw.power_curve(baseline=0.03, mde_grid=(0.02, 0.05, 0.10))[:3])

    print("\n== 5. experiment readout ==")
    users = wh.run("SELECT * FROM experiment_users", max_rows=200_000).dataframe
    readout = analyze_experiment(
        users,
        [
            {"name": "conversion", "kind": "proportion", "numerator": "converted"},
            {"name": "revenue_per_user", "kind": "mean", "numerator": "revenue"},
        ],
        covariate_col="pre_revenue",
    )
    for m in readout["metrics"]:
        print(f"  {m['metric']:<18} lift {m['rel_lift']:+.2%}  p={m['p_value']:.4f}  adjusted p={m['p_adjusted']:.4f}")

    # ------------------------------------------------------------ 6. causal
    print("\n== 6. causal inference ==")
    panel = wh.run("SELECT * FROM city_panel").dataframe
    did = difference_in_differences(panel, "gmv", "city", "week", "treated", "post", log_outcome=True)
    print(f"  DiD (log): {did['relative_effect']:+.2%}  p={did['p_value']:.4f}")
    donors = sorted(panel.loc[panel.treated == 0, "city"].unique().tolist())
    sc = synthetic_control(panel, "city", "week", "gmv", "city_00", 26, donors=donors)
    print(f"  synthetic control: {sc['relative_effect']:+.2%}  placebo p={sc['placebo_p_value']:.3f}")

    # ------------------------------------ 7. designs beyond a standard A/B
    print("\n== 7. switchback / reversal / nested / two-sided ==")
    switch = analyze_switchback(
        wh.run("SELECT * FROM switchback_log", max_rows=5_000).dataframe,
        metric="gmv", date_col="date", arm_col="arm",
        treatment="treatment", control="control", washout=1,
    )
    print(
        f"  switchback: {switch['relative']:+.2%}  p={switch['p_value']:.4f}"
        f"  blocks={switch['n_blocks']}  washout={switch['washout_days']}d"
        f"  carryover ratio={switch['carryover']['ratio']:+.0%}"
    )

    reverse = analyze_reverse(
        wh.run("SELECT * FROM reverse_log", max_rows=300_000).dataframe,
        metric="revenue", date_col="date", arm_col="arm", phase_col="phase",
        treatment="treatment", control="control",
    )
    print(
        f"  reversal: tau1={reverse['tau1']:+.2f}  tau2={reverse['tau2']:+.2f}"
        f"  persistence={reverse['persistence']:.2f}  pipeline_ok={reverse['infrastructure_valid']}"
    )

    nested = analyze_nested_experiment(
        wh.run("SELECT * FROM nested_users", max_rows=100_000).dataframe,
        metric="converted", parent_col="parent_arm", child_col="child_arm",
    )
    print(
        f"  nested: orthogonal={nested['orthogonal']}  child={nested['adjusted']['relative']:+.2%}"
        f"  interaction p={nested['interaction']['p_value']:.3f}"
    )

    # The two-sided readout is 2x2 at the market level: one column per side's
    # treatment flag, one metric per side, read off the marginals.
    two = analyze_two_sided(
        wh.run("SELECT * FROM two_sided_market", max_rows=5_000).dataframe,
        metrics={"buyer": "buyer_conversion", "seller": "listing_rate", "market": "gmv_per_buyer"},
        market_col="market", buyer_treat_col="buyer_treat", seller_treat_col="seller_treat", n_boot=120,
    )
    buyer, seller, market = two["results"]["buyer"], two["results"]["seller"], two["results"]["market"]
    print(
        f"  two-sided: buyer {buyer['buyer_direct_relative']:+.2%}  seller {seller['seller_direct_relative']:+.2%}"
        f"  market interaction {market['interaction_relative']:+.2%} (p={market['interaction_p']:.3f})"
    )

    # ------------------------------------------------ 8. sensitivity analysis
    print("\n== 8. sensitivity analysis ==")
    users = wh.run("SELECT * FROM experiment_users", max_rows=200_000).dataframe
    sensitivity = sensitivity_analysis(
        users, metric="revenue", group_col="group", control="control", treatment="treatment",
        covariate="pre_revenue", segment_col="device", n_boot=200,
    )
    print(
        f"  {sensitivity['n_specs']} specs  sign_stable={sensitivity['sign_stable']}"
        f"  A/A false positive rate={sensitivity['placebo_false_positive_rate']:.1%}"
        f"  lift range={sensitivity['lift_range'][0]:+.2%}..{sensitivity['lift_range'][1]:+.2%}"
    )

    # ------------------------------------------------------------- 9. uplift
    print("\n== 9. uplift ==")
    uplift_df = wh.run("SELECT * FROM uplift_users", max_rows=50_000).dataframe
    features = ["tenure_days", "spend_30d", "sessions_30d"]
    forest = causal_forest(uplift_df, treatment="group", outcome="revenue", features=features, n_trees=25)
    learners = uplift_learners(uplift_df, treatment="group", outcome="revenue", features=features)
    top_row, bottom_row = forest["deciles"].iloc[0], forest["deciles"].iloc[-1]
    print(
        f"  causal forest ATE={forest['ate']:.2f}  AIPW={forest['ate_aipw']['ate']:.2f}"
        f"  Qini={forest['qini']:.3f}"
    )
    print(
        f"  top decile uplift={top_row['uplift']:.2f}  bottom decile={bottom_row['uplift']:.2f}"
        f"  top 20% captures {forest['targeting'][1]['captured_share']:.0%} of the gain"
    )
    for name, scored in learners["learners"].items():
        print(f"  {name}-learner: ATE={scored['ate']:.2f}  Qini={scored['qini']:.3f}")
    print(f"  best learner: {learners['best']}")

    # ------------------------------------------------------------------ 10. RDD
    print("\n== 10. regression discontinuity ==")
    rdd = regression_discontinuity(
        wh.run("SELECT * FROM rdd_scores", max_rows=20_000).dataframe,
        outcome="outcome", running="score", cutoff=600.0, covariates=("pre_spend",),
    )
    print(
        f"  jump at 600: {rdd['effect']:+.2f}  p={rdd['p_value']:.4f}  bandwidth={rdd['bandwidth']:.0f}"
        f"  manipulation p={rdd['density']['p_value']:.3f}  balance_ok={rdd['covariate_balance_ok']}"
    )

    # -------------------------------------------------- 11. time-series causal
    print("\n== 11. time-series causal ==")
    its_df = build_its_series()
    its = arima_intervention(
        its_df, time_col="date", outcome="gmv", intervention="2026-09-06",
        covariates=("control_gmv_1", "control_gmv_2"), placebo_n=4,
    )
    impact = causal_impact(
        its_df, time_col="date", outcome="gmv", intervention="2026-09-06",
        controls=("control_gmv_1", "control_gmv_2"), placebo_n=4,
    )
    pre = its_df.loc[its_df["date"] < __import__("datetime").date(2026, 9, 6), "gmv"].mean()
    post = its_df.loc[its_df["date"] >= __import__("datetime").date(2026, 9, 6), "gmv"].mean()
    print(f"  naive pre/post:      {post / pre - 1:+.2%}  (bills the trend to the launch)")
    print(f"  ARIMA ITS:           {its['relative']:+.2%}  p={its['p_value']:.2e}  Ljung-Box={its['ljung_box_p']:.3f}")
    print(f"  structural impact:   {impact['relative']:+.2%}  p={impact['p_value']:.2e}  pre-fit R2={impact['pre_fit_r2']:.3f}")

    wh.close()


if __name__ == "__main__":
    main()
