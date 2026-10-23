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
    print("1. 问题定义与识别假设 / PROBLEM & ASSUMPTIONS")
    print(rule("="))
    print(f"""
问题: 估计促销曝光 (promo) 对用户次月收入 (post_revenue) 的因果效应 (ATT)。
难点: 曝光非随机 —— 高价值用户更容易被选中，朴素均值差混入选择偏差。
假设:
  A1 条件可忽略性   给定协变量 {covariates}，曝光近似随机（不可检验，靠设计论证）
  A2 重叠性         每个处理组个体在对照组中有可比对象（可用数据诊断）
  A3 SUTVA          用户之间无溢出（不可检验）""")

    # 2. ------------------------------------------------------ feasibility
    print(rule("="))
    print("2. 可行性检验 / FEASIBILITY")
    print(rule("="))
    psm = propensity_score_matching(df, "promo", "post_revenue", covariates, n_bootstrap=200)
    print(f"\n协变量平衡性 (SMD，匹配前 → 匹配后，|SMD|<0.1 视为平衡):")
    for b in psm["balance"]:
        flag = "OK " if b["balanced"] else "FAIL"
        print(f"  {b['covariate']:>18}  {b['smd_before']:+.3f} → {b['smd_after']:+.3f}   [{flag}]")
    s = psm["common_support"]
    print(f"\n重叠性: 共同支撑 [{s['low']:.3f}, {s['high']:.3f}]，"
          f"{s['n_treated_outside']} 个处理组个体在区间外 ({s['share_outside']:.1%})")

    # 3. ----------------------------------------------------- the estimate
    print()
    print(rule("="))
    print("3. 效应估计 / THE ESTIMATE")
    print(rule("="))
    naive = float(df.loc[df["promo"] == 1, "post_revenue"].mean()
                  - df.loc[df["promo"] == 0, "post_revenue"].mean())
    print(f"\n  朴素均值差 (有偏):   {naive:+.4g}")
    print(f"  PSM  ATT:            {psm['att']:+.4g}   95% CI {psm['ci'][0]:+.4g}..{psm['ci'][1]:+.4g}   p={psm['p_value']:.4f}")
    print(f"  匹配对数: {psm['n_matched_pairs']:,} (卡尺 {psm['caliper']:.3f})")

    # 4. ----------------------------------------------------- robustness
    print()
    print(rule("="))
    print("4. 稳健性检验 / ROBUSTNESS")
    print(rule("="))
    print("\n[4a] 多方法一致性 (PSM / IPW / AIPW):")
    est = compare_estimators(df, "promo", "post_revenue", covariates, n_bootstrap=60)
    for name in ("psm", "ipw", "aipw"):
        r = est.get(name, {})
        if "error" in r:
            print(f"  {name.upper():5s}  error: {r['error']}")
        elif "estimate" in r:
            print(f"  {name.upper():5s}  {r['estimate']:+.4g}  CI [{r['ci'][0]:+.4g}, {r['ci'][1]:+.4g}]")
        else:
            print(f"  {name.upper():5s}  {r['att']:+.4g}")
    print(f"  → 估计量间差异 {est['summary']['spread_pct']:.1%} "
          f"({'稳健' if est['summary']['robust'] else '需警惕'})")

    print("\n[4b] 安慰剂反驳 (打乱处理标签，效应必须消失):")
    pl = placebo_refutation(df, "promo", "post_revenue", covariates, n_placebo=8)
    print(f"  {pl['n_placebo']} 次安慰剂: ATT {pl['placebo_mean']:+.4g}  "
          f"90% 范围 [{pl['placebo_ci'][0]:+.4g}, {pl['placebo_ci'][1]:+.4g}]")
    print(f"  真实 ATT {pl['real_att']:+.4g} → {'在安慰剂带之外，反驳检验通过' if pl['robust'] else '落在安慰剂带内 — 不可信'}")

    # 5. ---------------------------------------------------- sensitivity
    print()
    print(rule("="))
    print("5. 敏感性分析 / SENSITIVITY (Rosenbaum bounds)")
    print(rule("="))
    print(f"\n不可观测混杂使处理分配优势比最多扩大 Γ 倍时，显著性是否存续 (p_upper < 0.05):")
    rb = rosenbaum_bounds(np.asarray(psm["pair_diffs"]))
    for r in rb:
        print(f"  Γ={r['gamma']:<4}  p in [{r['p_lower']:.2e}, {r['p_upper']:.2e}]   "
              f"{'survives' if r['survives'] else 'fails'}")
    ok = [r["gamma"] for r in rb if r["survives"]]
    print(f"  → 结论在 Γ≤{max(ok):g} 内稳健" if ok else "  → 结论对隐藏混杂极脆弱")

    # 6. ------------------------------------------------- method selection
    print()
    print(rule("="))
    print("6. 假设不满足时的方法选择 / METHOD SELECTION")
    print(rule("="))
    for a in method_selection_advice(psm):
        print(f"\n  情形: {a['situation']}")
        print(f"  → {a['zh']}")

    # 7. ------------------------------------------------------ conclusion
    print()
    print(rule("="))
    print("7. 结论与不确定性 / CONCLUSION & UNCERTAINTY")
    print(rule("="))
    monthly = psm["att"] * psm["n_treated"]
    print(f"""
  估计值:   ATT = {psm['att']:+.4g} 元/人/月  (bootstrap 95% CI [{psm['ci'][0]:+.4g}, {psm['ci'][1]:+.4g}], p={psm['p_value']:.4f})
  不确定性: 仅对共同支撑区内的可比人群成立；Rosenbaum 边界给出混杂容忍上限；
            三估计量差异 {est['summary']['spread_pct']:.1%} 是方法间不确定性的直接度量。
  业务解释: 按处理组 {psm['n_treated']:,} 人折算，月增量收入约 ¥{round(monthly):,}。
            朴素差异 {naive:+.4g} 高估了 {abs(naive - psm['att']) / abs(psm['att']):.0%} —— 那是把
            "高价值用户本来就会花更多" 算成了促销效果。
  决策建议: 若增量 > 单人补贴成本，可在与处理组相似的人群中扩大投放并用 A/B 验证。
""")


if __name__ == "__main__":
    main()
