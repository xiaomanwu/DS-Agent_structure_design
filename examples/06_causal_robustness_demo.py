"""Causal demo, end to end: PSM with every layer a serious estimate needs.

This script walks the full chain a causal answer should ship with — not just a
point estimate:

  1. problem definition + identifying assumptions (stated, not implied)
  2. feasibility checks  — covariate balance (SMD) and propensity overlap
  3. the estimate itself — PSM ATT with bootstrap CI, next to IPW / AIPW
  4. robustness checks   — placebo refutation + estimator agreement
  5. sensitivity analysis — Rosenbaum bounds for hidden confounding
  6. method selection    — what to run instead when an assumption fails
  7. conclusion          — estimate, uncertainty, business interpretation

    python examples/06_causal_robustness_demo.py

Same data the agent uses for "用倾向得分匹配评估促销对转化的效果" — run the
agent question to see the same chain rendered as a chat answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ds_agent.tools.causal import (
    compare_estimators,
    method_selection_advice,
    placebo_refutation,
    propensity_score_matching,
    rosenbaum_bounds,
)


def rule(char: str = "-") -> str:
    return char * 78


def main() -> None:
    from ds_agent.data import build_demo_warehouse
    from ds_agent.config import settings

    wh = settings.warehouse
    if not Path(wh).exists():
        build_demo_warehouse(Path(wh), force=True)

    import duckdb

    con = duckdb.connect(str(wh), read_only=True)
    df = con.execute("SELECT * FROM promo_users").df()
    con.close()

    df["log_pre_revenue"] = np.log1p(df["pre_revenue"])
    covariates = ["log_pre_revenue", "tenure_months", "sessions_30d"]

    # 1. ------------------------------------------------- problem & assumptions
    print(rule("="))
    print("1. Problem definition & identifying assumptions / PROBLEM & ASSUMPTIONS")
    print(rule("="))
    print(f"""
Problem:  estimate the causal effect (ATT) of promo exposure (promo) on next-month revenue (post_revenue).
Difficulty: exposure is not random — high-value users are more likely to be targeted, so a naive
          mean difference carries selection bias.
Assumptions:
  A1 Conditional ignorability   given covariates {covariates}, exposure is approximately random
                                (untestable; argued through design)
  A2 Overlap                    every treated unit has comparable controls (diagnosable from data)
  A3 SUTVA                      no spillover between users (untestable)""")

    # 2. ------------------------------------------------------ feasibility
    print(rule("="))
    print("2. Feasibility checks / FEASIBILITY")
    print(rule("="))
    psm = propensity_score_matching(df, "promo", "post_revenue", covariates, n_bootstrap=200)
    print(f"\nCovariate balance (SMD, before → after matching; |SMD|<0.1 treated as balanced):")
    for b in psm["balance"]:
        flag = "OK " if b["balanced"] else "FAIL"
        print(f"  {b['covariate']:>18}  {b['smd_before']:+.3f} → {b['smd_after']:+.3f}   [{flag}]")
    s = psm["common_support"]
    print(f"\nOverlap: common support [{s['low']:.3f}, {s['high']:.3f}]; "
          f"{s['n_treated_outside']} treated units outside the interval ({s['share_outside']:.1%})")

    # 3. ----------------------------------------------------- the estimate
    print()
    print(rule("="))
    print("3. Effect estimation / THE ESTIMATE")
    print(rule("="))
    naive = float(df.loc[df["promo"] == 1, "post_revenue"].mean()
                  - df.loc[df["promo"] == 0, "post_revenue"].mean())
    print(f"\n  Naive mean difference (biased):   {naive:+.4g}")
    print(f"  PSM  ATT:            {psm['att']:+.4g}   95% CI {psm['ci'][0]:+.4g}..{psm['ci'][1]:+.4g}   p={psm['p_value']:.4f}")
    print(f"  Matched pairs: {psm['n_matched_pairs']:,} (caliper {psm['caliper']:.3f})")

    # 4. ----------------------------------------------------- robustness
    print()
    print(rule("="))
    print("4. Robustness checks / ROBUSTNESS")
    print(rule("="))
    print("\n[4a] Estimator agreement (PSM / IPW / AIPW):")
    est = compare_estimators(df, "promo", "post_revenue", covariates, n_bootstrap=60)
    for name in ("psm", "ipw", "aipw"):
        r = est.get(name, {})
        if "error" in r:
            print(f"  {name.upper():5s}  error: {r['error']}")
        elif "estimate" in r:
            print(f"  {name.upper():5s}  {r['estimate']:+.4g}  CI [{r['ci'][0]:+.4g}, {r['ci'][1]:+.4g}]")
        else:
            print(f"  {name.upper():5s}  {r['att']:+.4g}")
    print(f"  → spread across estimators {est['summary']['spread_pct']:.1%} "
          f"({'robust' if est['summary']['robust'] else 'caution warranted'})")

    print("\n[4b] Placebo refutation (shuffle treatment labels; the effect must vanish):")
    pl = placebo_refutation(df, "promo", "post_revenue", covariates, n_placebo=8)
    print(f"  {pl['n_placebo']} placebo runs: ATT {pl['placebo_mean']:+.4g}  "
          f"90% range [{pl['placebo_ci'][0]:+.4g}, {pl['placebo_ci'][1]:+.4g}]")
    print(f"  real ATT {pl['real_att']:+.4g} → {'outside the placebo band; refutation passed' if pl['robust'] else 'inside the placebo band - not credible'}")

    # 5. ---------------------------------------------------- sensitivity
    print()
    print(rule("="))
    print("5. Sensitivity analysis / SENSITIVITY (Rosenbaum bounds)")
    print(rule("="))
    print(f"\nWhen hidden confounding multiplies the odds of treatment assignment by at most Γ, does significance survive (p_upper < 0.05):")
    rb = rosenbaum_bounds(np.asarray(psm["pair_diffs"]))
    for r in rb:
        print(f"  Γ={r['gamma']:<4}  p in [{r['p_lower']:.2e}, {r['p_upper']:.2e}]   "
              f"{'survives' if r['survives'] else 'fails'}")
    ok = [r["gamma"] for r in rb if r["survives"]]
    print(f"  → conclusion is robust up to Γ≤{max(ok):g}" if ok else "  → conclusion is highly fragile to hidden confounding")

    # 6. ------------------------------------------------- method selection
    print()
    print(rule("="))
    print("6. Method selection when assumptions fail / METHOD SELECTION")
    print(rule("="))
    for a in method_selection_advice(psm):
        print(f"\n  Situation: {a['situation']}")
        print(f"  → {a['zh']}")

    # 7. ------------------------------------------------------ conclusion
    print()
    print(rule("="))
    print("7. Conclusion & uncertainty / CONCLUSION & UNCERTAINTY")
    print(rule("="))
    monthly = psm["att"] * psm["n_treated"]
    print(f"""
  Estimate:    ATT = {psm['att']:+.4g} per user-month  (bootstrap 95% CI [{psm['ci'][0]:+.4g}, {psm['ci'][1]:+.4g}], p={psm['p_value']:.4f})
  Uncertainty: holds only for the comparable population inside the common support; Rosenbaum bounds
               cap the tolerated confounding; the {est['summary']['spread_pct']:.1%} spread across the three
               estimators is a direct measure of method uncertainty.
  Business:    scaled to the {psm['n_treated']:,} treated users, roughly ¥{round(monthly):,} in incremental monthly revenue.
               The naive difference {naive:+.4g} overstates the effect by {abs(naive - psm['att']) / abs(psm['att']):.0%} — it books
               "high-value users would have spent more anyway" as promo impact.
  Decision:    if the increment exceeds the per-user subsidy cost, expand delivery to a population
               similar to the treated group and verify with an A/B test.
""")


if __name__ == "__main__":
    main()
