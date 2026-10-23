"""Advanced readouts: the experiment designs and causal estimators beyond a plain A/B.

``agent.py`` owns routing plus the query / diagnosis / design handlers. The eight
handlers here share one shape - load a table, call exactly one tool from
``ds_agent.tools``, compose KPI + chart + table + narrative - so they live in
their own module and are mixed into ``DSQueryAgent``.

Each one records a ``Section``, which is what makes a follow-up "write this up"
produce a report without re-running anything.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import charts
from .confidence import confidence_table, ts_causal_confidence, verdict_line
from .reply import AgentReply, fmt_num, fmt_pct
from .stats import panels as test_panels
from .stats import tests as stat_tests
from .tools.experiment_designs import (
    analyze_nested_experiment,
    analyze_reverse,
    analyze_switchback,
    analyze_two_sided,
    sensitivity_analysis,
)
from .tools.rdd import regression_discontinuity
from .tools.report import Section
from .tools.ts_causal import arima_intervention, causal_impact, compare_ts_methods
from .tools.uplift import causal_forest, causal_tree, uplift_learners


class ReadoutHandlers:
    """Mixin: expects ``self.sql``, ``self.lang``, ``self._t``, ``self._table_block``,
    ``self._record`` and ``self.session`` to come from ``DSQueryAgent``."""

    # ------------------------------------------------------------------ switchback
    def _handle_switchback(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """Time-sliced rotation: the inference unit is the time block, not the row."""
        df = self.sql.run("SELECT * FROM switchback_log", max_rows=100000).dataframe
        # One washout day is the default: the switch day itself mixes both arms,
        # and keeping it is exactly the mistake that shortens the measured effect.
        washout = 0 if re.search(r"不清洗|不设清洗|no washout", question, re.I) else 1
        result = analyze_switchback(
            df, metric="gmv", date_col="date", arm_col="arm",
            treatment="treatment", control="control", washout=washout,
        )
        blocks = [
            {
                "type": "kpi",
                "items": [
                    {
                        "label": self._t( "relative effect", "relative effect"),
                        "value": fmt_pct(result["relative"]),
                        "sub": f"p={result['p_value']:.2e}",
                        "good": result["relative"] > 0,
                    },
                    {
                        "label": self._t( "time blocks", "time blocks"),
                        "value": f"{result['n_blocks']}",
                        "sub": self._t( f"washout {result['washout_days']}d", f"washout {result['washout_days']}d"),
                    },
                    {
                        "label": self._t( "residual autocorr", "residual autocorr"),
                        "value": f"{result['residual_autocorr']:+.2f}",
                        "sub": self._t( f"HAC {result['hac_lags']} lags", f"HAC {result['hac_lags']} lags"),
                    },
                ],
            }
        ]
        # Daily series coloured by arm: the schedule is the design, so it belongs
        # on the chart rather than in a table.
        daily = (
            df.assign(_date=pd.to_datetime(df["date"]))
            .groupby(["_date", "arm"], as_index=False)["gmv"]
            .sum()
            .rename(columns={"_date": "time", "gmv": "value"})
        )
        schedule_chart = ""
        try:
            schedule_chart = charts.switchback_chart(daily)
            blocks.append(
                {
                    "type": "chart",
                    "title": self._t( "Switchback schedule and daily metric", "Switchback schedule and daily metric"),
                    "image": schedule_chart,
                }
            )
        except Exception:  # noqa: BLE001 - a chart must never fail the answer
            pass

        paired = result["paired_blocks"]
        rows = [
            [
                self._t( "block mean difference (HAC)", "block mean difference (HAC)"),
                fmt_num(result["effect"]),
                fmt_num(result["se"]),
                f"{result['p_value']:.2e}",
            ],
            [
                self._t( "paired block difference", "paired block difference"),
                fmt_num(paired["mean_diff"]),
                fmt_num(paired["se"]),
                f"{paired['p_value']:.3f}",
            ],
        ]
        blocks.append(
            self._table_block(
                pd.DataFrame(rows, columns=[self._t( "estimator", "estimator"), self._t( "effect", "effect"), "se", "p"]),
                self._t( "Two estimators", "Two estimators"),
            )
        )

        carry = result["carryover"]
        carry_line = ""
        if carry and np.isfinite(carry.get("ratio", float("nan"))):
            carry_line = self._t(
                f"**Carryover test**: the lagged schedule coefficient is {carry['lag_coef']:+,.0f}, i.e. "
                f"{carry['ratio']:+.0%} of the current effect (p={carry['lag_p']:.3f}) - "
                + (
                    "significant, so the washout is too short and yesterday's arm is bleeding into today."
                    if carry["contaminated"]
                    else "not significant, so a one-day washout is enough."
                ),
                f"**Carryover test**: the lagged schedule coefficient is {carry['lag_coef']:+,.0f}, i.e. "
                f"{carry['ratio']:+.0%} of the current effect (p={carry['lag_p']:.3f}) - "
                + (
                    "significant, so the washout is too short and yesterday's arm is bleeding into today."
                    if carry["contaminated"]
                    else "not significant, so a one-day washout is enough."
                ),
            )
        lines = [
            self._t(
                f"**Switchback**: treatment vs control **{fmt_pct(result['relative'])}** "
                f"({fmt_num(result['effect'])}, 95% CI [{fmt_num(result['ci'][0])}, {fmt_num(result['ci'][1])}], "
                f"p={result['p_value']:.2e}), {'significant' if result['significant'] else 'not significant'}.",
                f"**Switchback**: treatment vs control **{fmt_pct(result['relative'])}** "
                f"({fmt_num(result['effect'])}, 95% CI [{fmt_num(result['ci'][0])}, {fmt_num(result['ci'][1])}], "
                f"p={result['p_value']:.2e}), {'significant' if result['significant'] else 'not significant'}.",
            ),
            "",
            self._t(
                f"The inference unit is the **time block** ({result['n_blocks']} blocks, {result['n_obs']} daily "
                f"observations after washout). Treating market-days as independent inflates the effective sample size by an "
                f"order of magnitude; residual autocorrelation is {result['residual_autocorr']:+.2f}, so SEs are HAC with "
                f"day-of-week and trend terms.",
                f"The inference unit is the **time block** ({result['n_blocks']} blocks, {result['n_obs']} daily "
                f"observations after washout). Treating market-days as independent inflates the effective sample size by an "
                f"order of magnitude; residual autocorrelation is {result['residual_autocorr']:+.2f}, so SEs are HAC with "
                f"day-of-week and trend terms.",
            ),
        ]
        if carry_line:
            lines += ["", carry_line]
        if result["blocks_per_arm_needed"]:
            lines += [
                "",
                self._t(
                    f"**Next run**: to halve the MDE you would need about {result['blocks_per_arm_needed']:.0f} blocks per arm "
                    f"(this run had {result['n_blocks']}).",
                    f"**Next run**: to halve the MDE you would need about {result['blocks_per_arm_needed']:.0f} blocks per arm "
                    f"(this run had {result['n_blocks']}).",
                ),
            ]
        lines += ["", self._t( "**Assumption**", "**Assumption**"), result["assumption"]]

        text = "\n".join(lines)
        self._record(
            self._t(
                f"Switchback: {fmt_pct(result['relative'])} (p={result['p_value']:.2e}) over {result['n_blocks']} blocks.",
                f"Switchback: {fmt_pct(result['relative'])} (p={result['p_value']:.2e}) over {result['n_blocks']} blocks.",
            ),
            Section(
                title=self._t( "Switchback experiment", "Switchback experiment"),
                prose=self._plain_line(result["narrative"]),
                bullets=[carry_line] if carry_line else [],
                table=pd.DataFrame(rows, columns=[self._t( "estimator", "estimator"), self._t( "effect", "effect"), "se", "p"]),
                chart=schedule_chart or None,
            ),
        )
        return AgentReply(
            text=text,
            blocks=blocks,
            tool_calls=[{"tool": "analyze_switchback", "args": {"washout": washout}, "result": {"relative": result["relative"], "p": result["p_value"]}}],
            data={"switchback": {k: v for k, v in result.items() if k != "paired_blocks"} | {"paired_blocks": paired}},
        )

    # ------------------------------------------------------------------- reversal
    def _handle_reversal(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """Swap-back experiment: two phases, a sum test for drift, a persistence ratio."""
        df = self.sql.run("SELECT * FROM reverse_log", max_rows=300000).dataframe
        result = analyze_reverse(
            df, metric="revenue", date_col="date", arm_col="arm", phase_col="phase",
            treatment="treatment", control="control",
        )
        # phases is keyed by phase number; relative effects are not stored because
        # the two phases have different baselines, so derive them here.
        p1, p2 = result["phases"][1], result["phases"][2]
        rel1 = p1["effect"] / p1["control_mean"] if p1["control_mean"] else float("nan")
        rel2 = p2["effect"] / p2["control_mean"] if p2["control_mean"] else float("nan")
        phases = [dict(p1, relative=rel1), dict(p2, relative=rel2)]
        blocks = [
            {
                "type": "kpi",
                "items": [
                    {"label": self._t( "phase 1 tau1", "phase 1 tau1"), "value": fmt_pct(rel1), "sub": f"p={p1['p_value']:.2e}"},
                    {"label": self._t( "phase 2 tau2", "phase 2 tau2"), "value": fmt_pct(rel2), "sub": f"p={p2['p_value']:.2e}"},
                    {
                        "label": self._t( "drift test", "drift test"),
                        "value": "PASS" if result["symmetric"] else "FAIL",
                        "sub": f"τ₁+τ₂ p={result['symmetry']['p_value']:.3f}",
                        "good": result["symmetric"],
                    },
                ],
            }
        ]
        forest = [
            {"metric": self._t( "phase 1", "phase 1"), "abs_diff": p1["effect"], "ci_abs": (p1["effect"] - 1.96 * p1["se"], p1["effect"] + 1.96 * p1["se"])},
            {"metric": self._t( "phase 2", "phase 2"), "abs_diff": p2["effect"], "ci_abs": (p2["effect"] - 1.96 * p2["se"], p2["effect"] + 1.96 * p2["se"])},
        ]
        forest_chart = ""
        try:
            forest_chart = charts.effect_forest(forest)
            blocks.append({"type": "chart", "title": self._t( "Effect by phase", "Effect by phase"), "image": forest_chart})
        except Exception:  # noqa: BLE001
            pass

        rows = [
            [self._t( "phase 1", "phase 1"), fmt_num(p1["control_mean"]), fmt_num(p1["treatment_mean"]), fmt_pct(rel1), f"{p1['p_value']:.2e}"],
            [self._t( "phase 2 (swapped)", "phase 2 (swapped)"), fmt_num(p2["control_mean"]), fmt_num(p2["treatment_mean"]), fmt_pct(rel2), f"{p2['p_value']:.2e}"],
        ]
        blocks.append(
            self._table_block(
                pd.DataFrame(rows, columns=[self._t( "phase", "phase"), "control", "treatment", "effect", "p"]),
                self._t( "Phase-level effects", "Phase-level effects"),
                pct_cols=("effect",),
            )
        )

        sum_p = result["symmetry"]["p_value"]
        persistence = result["persistence"]
        lines = [
            self._t(
                f"**Reversal experiment**: phase 1 tau1 = **{fmt_pct(rel1)}** ({fmt_num(p1['effect'])}), "
                f"after the swap tau2 = **{fmt_pct(rel2)}** ({fmt_num(p2['effect'])}), "
                f"{p1['n'] + p2['n']:,} user-period rows.",
                f"**Reversal experiment**: phase 1 tau1 = **{fmt_pct(rel1)}** ({fmt_num(p1['effect'])}), "
                f"after the swap tau2 = **{fmt_pct(rel2)}** ({fmt_num(p2['effect'])}), "
                f"{p1['n'] + p2['n']:,} user-period rows.",
            ),
            "",
            self._t(
                f"**Symmetry test**: tau1+tau2 = {fmt_num(result['symmetry']['sum'])} (p={sum_p:.3f}) - "
                + (
                    "not significant, so the phase difference is fully explained by swapping the experience: no instrumentation drift."
                    if result["symmetric"]
                    else "significant: the phases are asymmetric, so suspect an instrumentation change, audience shift or time trend."
                ),
                f"**Symmetry test**: tau1+tau2 = {fmt_num(result['symmetry']['sum'])} (p={sum_p:.3f}) - "
                + (
                    "not significant, so the phase difference is fully explained by swapping the experience: no instrumentation drift."
                    if result["symmetric"]
                    else "significant: the phases are asymmetric, so suspect an instrumentation change, audience shift or time trend."
                ),
            ),
            "",
            self._t(
                f"**Persistence**: |tau2|/|tau1| = {persistence:.2f}, "
                + ("the effect persists almost fully, so no novelty decay." if persistence > 0.8 else "the effect decays after the swap, so phase 1 included a novelty component.")
                + f" The symmetric pooled estimate is {fmt_pct(result['pooled']['relative'])} (p={result['pooled']['p_value']:.2e}).",
                f"**Persistence**: |tau2|/|tau1| = {persistence:.2f}, "
                + ("the effect persists almost fully, so no novelty decay." if persistence > 0.8 else "the effect decays after the swap, so phase 1 included a novelty component.")
                + f" The symmetric pooled estimate is {fmt_pct(result['pooled']['relative'])} (p={result['pooled']['p_value']:.2e}).",
            ),
        ]
        if not result["reversal_detected"]:
            lines += ["", self._t( "Note: no reverse effect in phase 2 - the premise of the design (the swap really changed the experience) may not hold.", "Note: no reverse effect in phase 2 - the premise of the design (the swap really changed the experience) may not hold.")]
        lines += ["", self._t( "**Assumption**", "**Assumption**"), result["assumption"]]

        verdict = self._t(
            f"Reversal: tau1={fmt_pct(rel1)}, tau2={fmt_pct(rel2)}, "
            f"symmetry p={sum_p:.3f}, persistence {persistence:.2f}.",
            f"Reversal: tau1={fmt_pct(rel1)}, tau2={fmt_pct(rel2)}, "
            f"symmetry p={sum_p:.3f}, persistence {persistence:.2f}.",
        )
        self._record(
            verdict,
            Section(
                title=self._t( "Reversal experiment", "Reversal experiment"),
                prose=self._plain_line(result["narrative"]),
                bullets=[
                    self._t( f"tau1+tau2 p={sum_p:.3f}: {'no drift' if result['symmetric'] else 'drift detected'}.", f"tau1+tau2 p={sum_p:.3f}: {'no drift' if result['symmetric'] else 'drift detected'}."),
                    self._t( f"Persistence {persistence:.2f}.", f"Persistence {persistence:.2f}."),
                ],
                table=pd.DataFrame(rows, columns=[self._t( "phase", "phase"), "control", "treatment", "effect", "p"]),
                chart=forest_chart or None,
            ),
        )
        return AgentReply(
            text="\n".join(lines),
            blocks=blocks,
            tool_calls=[{"tool": "analyze_reverse", "args": {}, "result": {"tau1": result["tau1"], "tau2": result["tau2"]}}],
            data={"reversal": {k: v for k, v in result.items() if k != "phases"} | {"phases": phases}},
        )

    # --------------------------------------------------------------- parent-child
    def _handle_nested(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """Parent-child layers: orthogonality first, then the child effect net of the parent."""
        df = self.sql.run("SELECT * FROM nested_users", max_rows=100000).dataframe
        result = analyze_nested_experiment(
            df, metric="converted", parent_col="parent_arm", child_col="child_arm", lang=self.lang
        )
        naive = result["naive"]
        adjusted = result["adjusted"] or naive
        inter = result["interaction"]
        orth = result["orthogonality"]
        blocks = [
            {
                "type": "kpi",
                "items": [
                    {
                        "label": self._t( "orthogonality", "orthogonality"),
                        "value": "PASS" if result["orthogonal"] else "FAIL",
                        "sub": f"chi2 p={orth['p_value']:.3f}",
                        "good": result["orthogonal"],
                    },
                    {
                        "label": self._t( "child effect (adjusted)", "child effect (adjusted)"),
                        "value": fmt_pct(adjusted["relative"]),
                        "sub": f"p={adjusted['p_value']:.4f}",
                        "good": adjusted["relative"] > 0,
                    },
                    {
                        "label": self._t( "interaction", "interaction"),
                        "value": fmt_pct(inter["relative"]) if np.isfinite(inter.get("relative", np.nan)) else "-",
                        "sub": f"p={inter['p_value']:.3f}" if np.isfinite(inter["p_value"]) else self._t( "too few", "too few"),
                    },
                ],
            }
        ]
        rows = [
            [self._t( "naive", "naive"), fmt_pct(naive["relative"]), f"{naive['p_value']:.2e}", f"{result['n']:,}"],
            [
                self._t( "parent FE adjusted", "parent FE adjusted"),
                fmt_pct(adjusted["relative"]),
                f"{adjusted['p_value']:.4f}",
                f"{result['n']:,}",
            ],
        ]
        blocks.append(
            self._table_block(
                pd.DataFrame(rows, columns=[self._t( "estimator", "estimator"), "effect", "p", "n"]),
                self._t( "Child effect", "Child effect"),
                pct_cols=("effect",),
            )
        )
        cells = result["cells"]
        # At tens of thousands of users a chi-square test flags deviations that are
        # commercially irrelevant, so the max relative cell deviation is the number
        # to read; the p-value only rules out gross imbalance.
        max_dev = float("nan")
        if isinstance(cells, pd.DataFrame) and len(cells):
            share = cells.to_numpy(dtype=float)
            totals = share.sum(axis=0, keepdims=True)
            expected = share.sum(axis=1, keepdims=True) * totals / max(share.sum(), 1e-9)
            with np.errstate(divide="ignore", invalid="ignore"):
                dev = np.abs(share - expected) / np.where(expected > 0, expected, np.nan)
            max_dev = float(np.nanmax(dev)) if np.isfinite(dev).any() else float("nan")
            blocks.append(self._table_block(cells.reset_index(), self._t( "Parent x child cells", "Parent x child cells")))

        gap = adjusted["relative"] - naive["relative"]
        lines = [
            self._t(
                f"**Parent-child**: the layers "
                + ("are orthogonal" if result["orthogonal"] else "**are not orthogonal**")
                + f" (chi2 p={orth['p_value']:.3f} against a 0.01 threshold; worst cell deviation {max_dev:.1%}), "
                + (
                    f"so the child effect reads cleanly as **{fmt_pct(adjusted['relative'])}**."
                    if result["orthogonal"]
                    else "so the child effect is contaminated by the parent and only the adjusted estimate is usable."
                ),
                f"**Parent-child**: the layers "
                + ("are orthogonal" if result["orthogonal"] else "**are not orthogonal**")
                + f" (chi2 p={orth['p_value']:.3f} against a 0.01 threshold; worst cell deviation {max_dev:.1%}), "
                + (
                    f"so the child effect reads cleanly as **{fmt_pct(adjusted['relative'])}**."
                    if result["orthogonal"]
                    else "so the child effect is contaminated by the parent and only the adjusted estimate is usable."
                ),
            ),
            "",
            self._t(
                f"Naive {fmt_pct(naive['relative'])} vs parent-FE adjusted {fmt_pct(adjusted['relative'])} "
                f"(gap {fmt_pct(gap)}). "
                + ("The gap is small: the parent layer was not biasing the child." if abs(gap) < 0.005 else "The gap matters: the naive read credits the parent effect to the child."),
                f"Naive {fmt_pct(naive['relative'])} vs parent-FE adjusted {fmt_pct(adjusted['relative'])} "
                f"(gap {fmt_pct(gap)}). "
                + ("The gap is small: the parent layer was not biasing the child." if abs(gap) < 0.005 else "The gap matters: the naive read credits the parent effect to the child."),
            ),
        ]
        if np.isfinite(inter["p_value"]):
            lines += [
                "",
                self._t(
                    f"**Interaction**: {fmt_pct(inter['relative'])} (p={inter['p_value']:.3f}) - "
                    + ("not significant, so the child works the same in both parent levels and can be reported pooled." if inter["p_value"] > 0.05 else "significant, so the child effect depends on the parent level and must be split."),
                    f"**Interaction**: {fmt_pct(inter['relative'])} (p={inter['p_value']:.3f}) - "
                    + ("not significant, so the child works the same in both parent levels and can be reported pooled." if inter["p_value"] > 0.05 else "significant, so the child effect depends on the parent level and must be split."),
                ),
            ]
        lines += [
            "",
            self._t( f"**Smallest cell**: {result['min_cell']:,} users; below ~1,000 the interaction test has almost no power.", f"**Smallest cell**: {result['min_cell']:,} users; below ~1,000 the interaction test has almost no power."),
            "",
            self._t( "**Assumption**", "**Assumption**"),
            result["assumption"],
        ]
        verdict = self._t(
            f"Parent-child: child effect {fmt_pct(adjusted['relative'])} (p={adjusted['p_value']:.4f}), "
            f"orthogonality {'pass' if result['orthogonal'] else 'fail'}.",
            f"Parent-child: child effect {fmt_pct(adjusted['relative'])} (p={adjusted['p_value']:.4f}), "
            f"orthogonality {'pass' if result['orthogonal'] else 'fail'}.",
        )
        self._record(
            verdict,
            Section(
                title=self._t( "Parent-child experiment", "Parent-child experiment"),
                prose=self._plain_line(result["narrative"]),
                bullets=[verdict],
                table=pd.DataFrame(rows, columns=[self._t( "estimator", "estimator"), "effect", "p", "n"]),
            ),
        )
        return AgentReply(
            text="\n".join(lines),
            blocks=blocks,
            tool_calls=[{"tool": "analyze_nested_experiment", "args": {}, "result": {"orthogonal": result["orthogonal"], "adjusted": adjusted["relative"]}}],
            data={"nested": result},
        )

    # ------------------------------------------------------------------ two-sided
    def _handle_two_sided(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """2x2 buyer x seller cells: direct effects, cross-side spillover, interaction."""
        df = self.sql.run("SELECT * FROM two_sided_market", max_rows=100000).dataframe
        result = analyze_two_sided(
            df,
            metrics={"buyer": "buyer_conversion", "seller": "listing_rate", "market": "gmv_per_buyer"},
            market_col="market",
            buyer_treat_col="buyer_treat",
            seller_treat_col="seller_treat",
            n_boot=150,
            lang=self.lang,
        )
        res = result["results"]
        buyer, seller, market = res["buyer"], res["seller"], res["market"]
        blocks = [
            {
                "type": "kpi",
                "items": [
                    {
                        "label": self._t( "buyer-side direct", "buyer-side direct"),
                        "value": fmt_pct(buyer["buyer_direct_relative"]),
                        "sub": buyer["metric"],
                        "good": buyer["buyer_direct"] > 0,
                    },
                    {
                        "label": self._t( "seller-side direct", "seller-side direct"),
                        "value": fmt_pct(seller["seller_direct_relative"]),
                        "sub": seller["metric"],
                        "good": seller["seller_direct"] > 0,
                    },
                    {
                        "label": self._t( "interaction", "interaction"),
                        "value": fmt_pct(market["interaction_relative"]),
                        "sub": f"p={market['interaction_p']:.3f}",
                    },
                ],
            }
        ]
        rows = []
        for item in (buyer, seller, market):
            cells = item["cells"]
            rows.append(
                [
                    item["metric"],
                    fmt_num(cells["00"], 4),
                    fmt_num(cells["10"], 4),
                    fmt_num(cells["01"], 4),
                    fmt_num(cells["11"], 4),
                    fmt_num(item["buyer_direct"], 4),
                    fmt_num(item["seller_direct"], 4),
                    fmt_num(item["interaction"], 4),
                    f"{item['interaction_p']:.3f}",
                ]
            )
        effect_table = pd.DataFrame(
            rows,
            columns=[
                self._t( "metric", "metric"),
                self._t( "00 neither", "00 neither"),
                self._t( "10 buyer", "10 buyer"),
                self._t( "01 seller", "01 seller"),
                self._t( "11 both", "11 both"),
                self._t( "buyer direct", "buyer direct"),
                self._t( "seller direct", "seller direct"),
                self._t( "interaction", "interaction"),
                "p",
            ],
        )
        blocks.append(self._table_block(effect_table, self._t( "2x2 cells and effects", "2x2 cells and effects"), max_rows=5, pct_cols=()))

        spill = result["spillover"]
        spill_rows = [
            [
                self._t( f"{role} side", f"{role} side"),
                fmt_num(v["direct"], 4),
                fmt_num(v["cross_side"], 4),
                fmt_pct(v["ratio"]) if np.isfinite(v["ratio"]) else "-",
            ]
            for role, v in spill.items()
        ]
        blocks.append(
            self._table_block(
                pd.DataFrame(
                    spill_rows,
                    columns=[self._t( "side", "side"), self._t( "own direct", "own direct"), self._t( "cross-side", "cross-side"), self._t( "ratio", "ratio")],
                ),
                self._t( "Cross-side spillover", "Cross-side spillover"),
                pct_cols=(),
            )
        )

        lines = [
            self._t(
                f"**Two-sided 2x2**: treating buyers moves buyer conversion {fmt_pct(buyer['buyer_direct_relative'])}; "
                f"treating sellers moves listing rate {fmt_pct(seller['seller_direct_relative'])}; "
                f"{result['n_markets']} markets.",
                f"**Two-sided 2x2**: treating buyers moves buyer conversion {fmt_pct(buyer['buyer_direct_relative'])}; "
                f"treating sellers moves listing rate {fmt_pct(seller['seller_direct_relative'])}; "
                f"{result['n_markets']} markets.",
            ),
            "",
            self._t(
                "Direct effects come from the 2x2 marginals; cross-side spillover is what happens to the side you did *not* "
                "touch - exactly what user-level randomisation cannot see, and the easiest way to double-count the gain.",
                "Direct effects come from the 2x2 marginals; cross-side spillover is what happens to the side you did *not* "
                "touch - exactly what user-level randomisation cannot see, and the easiest way to double-count the gain.",
            ),
            "",
            self._t(
                f"**Interaction**: treating both sides at once moves market GMV per buyer by an extra "
                f"{fmt_pct(market['interaction_relative'])} (p={market['interaction_p']:.3f}) - "
                + (
                    "not significant, so the two side effects add up."
                    if market["interaction_p"] > 0.05
                    else "significant and negative: the two sides cannibalise each other, so single-side readouts overstate the joint rollout."
                    if market["interaction"] < 0
                    else "significant and positive: the two sides reinforce each other and a joint rollout beats doing them separately."
                ),
                f"**Interaction**: treating both sides at once moves market GMV per buyer by an extra "
                f"{fmt_pct(market['interaction_relative'])} (p={market['interaction_p']:.3f}) - "
                + (
                    "not significant, so the two side effects add up."
                    if market["interaction_p"] > 0.05
                    else "significant and negative: the two sides cannibalise each other, so single-side readouts overstate the joint rollout."
                    if market["interaction"] < 0
                    else "significant and positive: the two sides reinforce each other and a joint rollout beats doing them separately."
                ),
            ),
            "",
            self._t(
                f"**Cross-side spillover**: treating only buyers moves seller listing rate by "
                f"{spill['seller']['ratio']:.0%} of the direct seller effect; treating only sellers moves buyer conversion by "
                f"{spill['buyer']['ratio']:.0%} of the direct buyer effect. The smaller these ratios, the more separable the "
                f"two sides - and the safer it is to extrapolate a single-side result.",
                f"**Cross-side spillover**: treating only buyers moves seller listing rate by "
                f"{spill['seller']['ratio']:.0%} of the direct seller effect; treating only sellers moves buyer conversion by "
                f"{spill['buyer']['ratio']:.0%} of the direct buyer effect. The smaller these ratios, the more separable the "
                f"two sides - and the safer it is to extrapolate a single-side result.",
            ),
            "",
            self._t( "**Assumption**", "**Assumption**"),
            result["assumption"],
        ]

        headline = self._t(
            f"Two-sided: buyer {fmt_pct(buyer['buyer_direct_relative'])}, seller {fmt_pct(seller['seller_direct_relative'])}, "
            f"interaction {fmt_num(market['interaction'])} (p={market['interaction_p']:.3f}).",
            f"Two-sided: buyer {fmt_pct(buyer['buyer_direct_relative'])}, seller {fmt_pct(seller['seller_direct_relative'])}, "
            f"interaction {fmt_num(market['interaction'])} (p={market['interaction_p']:.3f}).",
        )
        self._record(
            headline,
            Section(
                title=self._t( "Two-sided experiment", "Two-sided experiment"),
                prose=self._plain_line(result["narrative"]),
                bullets=[headline],
                table=effect_table,
            ),
        )
        return AgentReply(
            text="\n".join(lines),
            blocks=blocks,
            tool_calls=[
                {
                    "tool": "analyze_two_sided",
                    "args": {"metrics": [buyer["metric"], seller["metric"], market["metric"]]},
                    "result": {"buyer": buyer["buyer_direct"], "seller": seller["seller_direct"], "interaction_p": market["interaction_p"]},
                }
            ],
            data={"two_sided": result},
        )

    # ---------------------------------------------------------------- sensitivity
    def _handle_sensitivity(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """Re-run one treatment effect under every defensible specification."""
        df = self.sql.run("SELECT * FROM experiment_users", max_rows=200000).dataframe
        metric = "revenue" if re.search(r"金额|收入|gmv|revenue|arpu", question, re.I) else "converted"
        result = sensitivity_analysis(
            df,
            metric=metric,
            group_col="group",
            control="control",
            treatment="treatment",
            covariate="pre_revenue" if metric != "converted" else None,
            segment_col="device",
            n_boot=200,
            lang=self.lang,
        )
        specs = result["specs"]
        blocks = [
            {
                "type": "kpi",
                "items": [
                    {"label": self._t( "baseline lift", "baseline lift"), "value": fmt_pct(result["baseline"]["rel_lift"]), "sub": f"p={result['baseline']['p_value']:.4f}"},
                    {"label": self._t( "specifications", "specifications"), "value": f"{result['n_specs']}", "sub": self._t( f"{result['significant_count']} significant", f"{result['significant_count']} significant")},
                    {
                        "label": self._t( "sign stability", "sign stability"),
                        "value": "PASS" if result["sign_stable"] else "FAIL",
                        "sub": self._t( "all same sign", "all same sign") if result["sign_stable"] else self._t( "some flip sign", "some flip sign"),
                        "good": result["sign_stable"],
                    },
                ],
            }
        ]
        spec_rows = [
            [str(r["spec"]), fmt_pct(float(r["rel_lift"])), f"{float(r['p_value']):.4f}", str(r.get("note", ""))]
            for _, r in specs.iterrows()
        ]
        spec_table = pd.DataFrame(
            spec_rows, columns=[self._t( "specification", "specification"), "lift", "p", self._t( "note", "note")]
        )
        blocks.append(
            self._table_block(spec_table, self._t( "Specification curve", "Specification curve"), max_rows=20)
        )

        lo, hi = result["lift_range"]
        lines = [
            self._t(
                f"**Sensitivity** (`{metric}`): across {result['n_specs']} specifications the lift ranges "
                f"**{fmt_pct(lo)} ~ {fmt_pct(hi)}**, baseline {fmt_pct(result['baseline']['rel_lift'])}, "
                f"{result['significant_count']}/{result['n_specs']} significant.",
                f"**Sensitivity** (`{metric}`): across {result['n_specs']} specifications the lift ranges "
                f"**{fmt_pct(lo)} ~ {fmt_pct(hi)}**, baseline {fmt_pct(result['baseline']['rel_lift'])}, "
                f"{result['significant_count']}/{result['n_specs']} significant.",
            ),
            "",
            result["verdict"],
        ]
        if result["placebo_false_positive_rate"] is not None and np.isfinite(result["placebo_false_positive_rate"]):
            fp = result["placebo_false_positive_rate"]
            lines += [
                "",
                self._t(
                    f"**A/A placebo**: shuffling the treatment label 50 times gives a false-positive rate of {fp:.1%}"
                    + (", close to nominal alpha - the test itself is fine." if fp <= 0.1 else ", too high - the test is not robust to some structure (clustering, heavy tails)."),
                    f"**A/A placebo**: shuffling the treatment label 50 times gives a false-positive rate of {fp:.1%}"
                    + (", close to nominal alpha - the test itself is fine." if fp <= 0.1 else ", too high - the test is not robust to some structure (clustering, heavy tails)."),
                ),
            ]
        jack = result.get("jackknife") or []
        if jack:
            worst = max(jack, key=lambda j: abs(j["rel_lift"] - result["baseline"]["rel_lift"]))
            lines += [
                "",
                self._t(
                    f"**Jackknife**: dropping one segment at a time moves the lift between "
                    f"{fmt_pct(min(j['rel_lift'] for j in jack))} and {fmt_pct(max(j['rel_lift'] for j in jack))}; the most "
                    f"influential segment is {worst['dropped']} (lift becomes {fmt_pct(worst['rel_lift'])}, p={worst['p_value']:.3f}).",
                    f"**Jackknife**: dropping one segment at a time moves the lift between "
                    f"{fmt_pct(min(j['rel_lift'] for j in jack))} and {fmt_pct(max(j['rel_lift'] for j in jack))}; the most "
                    f"influential segment is {worst['dropped']} (lift becomes {fmt_pct(worst['rel_lift'])}, p={worst['p_value']:.3f}).",
                ),
            ]
        verdict = self._t(
            f"Sensitivity: {result['n_specs']} specifications, lift {fmt_pct(lo)}~{fmt_pct(hi)}, "
            f"{'sign stable' if result['sign_stable'] else 'sign unstable'}.",
            f"Sensitivity: {result['n_specs']} specifications, lift {fmt_pct(lo)}~{fmt_pct(hi)}, "
            f"{'sign stable' if result['sign_stable'] else 'sign unstable'}.",
        )
        self._record(
            verdict,
            Section(
                title=self._t( "Sensitivity analysis", "Sensitivity analysis"),
                prose=self._plain_line(result["narrative"]),
                bullets=[result["verdict"]],
                table=spec_table,
            ),
        )
        return AgentReply(
            text="\n".join(lines),
            blocks=blocks,
            tool_calls=[{"tool": "sensitivity_analysis", "args": {"metric": metric}, "result": {"sign_stable": result["sign_stable"], "n_specs": result["n_specs"]}}],
            data={"sensitivity": {k: v for k, v in result.items() if k != "specs"}},
        )

    # --------------------------------------------------------------------- uplift
    def _handle_uplift(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """Causal tree / forest readout: who responds, and how much targeting is worth."""
        df = self.sql.run("SELECT * FROM uplift_users", max_rows=100000).dataframe
        features = ["tenure_days", "spend_30d", "sessions_30d", "channel", "device"]
        want_tree = bool(re.search(r"因果树|causal tree|树模型|分群|segment", question, re.I))
        qini_chart_uri = ""  # only the forest branch produces a Qini chart
        blocks: List[Dict[str, Any]] = []
        lines: List[str] = []

        if want_tree:
            result = causal_tree(df, treatment="group", outcome="revenue", features=features, max_depth=3, min_samples_leaf=120)
            best, worst = result["best_segment"], result["worst_segment"]
            blocks.append(
                {
                    "type": "kpi",
                    "items": [
                        {"label": "ATE", "value": fmt_num(result["ate"], 3), "sub": self._t( "average effect", "average effect")},
                        {"label": self._t( "best segment", "best segment"), "value": fmt_num(best["tau"], 3), "sub": f"n={best['n']:,}", "good": best["tau"] > 0},
                        {"label": self._t( "leaves", "leaves"), "value": f"{result['n_leaves']}", "sub": self._t( "max depth 3", "max depth 3")},
                    ],
                }
            )
            leaf_rows = [[l["rule"], fmt_num(l["tau"], 2), f"{l['n']:,}"] for l in result["leaves"]]
            leaf_table = pd.DataFrame(leaf_rows, columns=[self._t( "rule", "rule"), self._t( "tau", "tau"), "n"])
            blocks.append(self._table_block(leaf_table, self._t( "Leaf-level effects", "Leaf-level effects"), max_rows=12))
            lines += [
                self._t(
                    f"**Causal tree** (depth 3, {result['n_leaves']} leaves): ATE {fmt_num(result['ate'], 3)}; the strongest "
                    f"segment `{best['rule']}` has effect **{fmt_num(best['tau'], 3)}** (n={best['n']:,}), the weakest "
                    f"`{worst['rule']}` only {fmt_num(worst['tau'], 3)}.",
                    f"**Causal tree** (depth 3, {result['n_leaves']} leaves): ATE {fmt_num(result['ate'], 3)}; the strongest "
                    f"segment `{best['rule']}` has effect **{fmt_num(best['tau'], 3)}** (n={best['n']:,}), the weakest "
                    f"`{worst['rule']}` only {fmt_num(worst['tau'], 3)}.",
                ),
                "",
                self._t(
                    "The split criterion is the *effect difference*, not predictive accuracy: each split maximises "
                    "(n_L n_R/n)(tau_L-tau_R)^2, so the tree finds the population with the largest response gap, not the one "
                    "most likely to convert. Those two are often opposites.",
                    "The split criterion is the *effect difference*, not predictive accuracy: each split maximises "
                    "(n_L n_R/n)(tau_L-tau_R)^2, so the tree finds the population with the largest response gap, not the one "
                    "most likely to convert. Those two are often opposites.",
                ),
            ]
            self.session["last_uplift"] = result
            section_prose = result["narrative"]
            table = leaf_table
            data = {"causal_tree": {k: v for k, v in result.items() if k != "cate"}}
        else:
            result = causal_forest(df, treatment="group", outcome="revenue", features=features, n_trees=30)
            learners = uplift_learners(df, treatment="group", outcome="revenue", features=features)
            deciles = result["deciles"]
            top = deciles.iloc[0]
            bottom = deciles.iloc[-1]
            top20 = next((t for t in result["targeting"] if abs(t["fraction"] - 0.2) < 1e-6), result["targeting"][0])
            blocks.append(
                {
                    "type": "kpi",
                    "items": [
                        {"label": "ATE", "value": fmt_num(result["ate"], 3), "sub": f"AIPW {fmt_num(result['ate_aipw']['ate'], 3)}"},
                        {"label": "Qini", "value": f"{result['qini']:.3f}", "sub": self._t( "honest split, 0=random", "honest split, 0=random")},
                        {"label": self._t("Top10% uplift", "top-10% uplift"), "value": fmt_num(top["uplift"], 3), "sub": self._t( f"bottom {fmt_num(bottom['uplift'], 3)}", f"bottom {fmt_num(bottom['uplift'], 3)}")},
                    ],
                }
            )
            try:
                qini_chart_uri = charts.qini_chart(result["qini_curve"], deciles)
                blocks.append(
                    {
                        "type": "chart",
                        "title": self._t( "Qini curve and decile uplift", "Qini curve and decile uplift"),
                        "image": qini_chart_uri,
                    }
                )
            except Exception:  # noqa: BLE001
                pass
            dec_rows = [
                [int(r["decile"]), fmt_num(r["pred_uplift"], 2), fmt_num(r["uplift"], 2), f"{int(r['n']):,}", f"{float(r['uplift_share']):.0%}" if np.isfinite(r["uplift_share"]) else "-"]
                for _, r in deciles.iterrows()
            ]
            blocks.append(
                self._table_block(
                    pd.DataFrame(dec_rows, columns=[self._t( "decile", "decile"), self._t( "predicted", "predicted"), self._t( "actual", "actual"), "n", self._t( "share of gain", "share of gain")]),
                    self._t( "Uplift by decile", "Uplift by decile"),
                    max_rows=10,
                )
            )
            learner_row = [[k.upper(), f"{v['qini']:.3f}", fmt_num(v["ate"], 3), fmt_num(v["top_decile_uplift"], 3)] for k, v in learners["learners"].items()]
            blocks.append(
                self._table_block(
                    pd.DataFrame(learner_row, columns=["learner", "qini", "ate", self._t("Top10% uplift", "top-10% uplift")]),
                    self._t( "Meta-learners compared", "Meta-learners compared"),
                )
            )
            gap = result["ate_gap"]
            lines += [
                self._t(
                    f"**Causal forest** ({result['n_trees']} trees, {result['n_fit']:,} fit / {result['n_eval']:,} held out): "
                    f"ATE **{fmt_num(result['ate'], 3)}**, cross-fitted AIPW check {fmt_num(result['ate_aipw']['ate'], 3)} "
                    f"(gap {fmt_num(gap, 3)}); honest Qini **{result['qini']:.3f}**.",
                    f"**Causal forest** ({result['n_trees']} trees, {result['n_fit']:,} fit / {result['n_eval']:,} held out): "
                    f"ATE **{fmt_num(result['ate'], 3)}**, cross-fitted AIPW check {fmt_num(result['ate_aipw']['ate'], 3)} "
                    f"(gap {fmt_num(gap, 3)}); honest Qini **{result['qini']:.3f}**.",
                ),
                "",
                self._t(
                    f"**Decile monotonicity**: the top 10% by predicted uplift actually lifts {fmt_num(top['uplift'], 3)} vs "
                    f"{fmt_num(bottom['uplift'], 3)} in the bottom 10% ({top['uplift'] / bottom['uplift']:.1f}x). "
                    f"Targeting only the top 20% captures {top20['captured_share']:.0%} of the total incremental outcome.",
                    f"**Decile monotonicity**: the top 10% by predicted uplift actually lifts {fmt_num(top['uplift'], 3)} vs "
                    f"{fmt_num(bottom['uplift'], 3)} in the bottom 10% ({top['uplift'] / bottom['uplift']:.1f}x). "
                    f"Targeting only the top 20% captures {top20['captured_share']:.0%} of the total incremental outcome.",
                ),
                "",
                self._t(
                    "**Meta-learners**: " + ", ".join(f"{k.upper()} Qini {v['qini']:.3f}" for k, v in learners["learners"].items())
                    + f"; {learners['best'].upper()} ranks best out of sample. Their ATEs agree but their rankings differ - "
                    "use Qini to choose whom to target, ATE to size the overall effect.",
                    "**Meta-learners**: " + ", ".join(f"{k.upper()} Qini {v['qini']:.3f}" for k, v in learners["learners"].items())
                    + f"; {learners['best'].upper()} ranks best out of sample. Their ATEs agree but their rankings differ - "
                    "use Qini to choose whom to target, ATE to size the overall effect.",
                ),
            ]
            self.session["last_uplift"] = result
            section_prose = result["narrative"]
            table = pd.DataFrame(dec_rows, columns=[self._t( "decile", "decile"), self._t( "predicted", "predicted"), self._t( "actual", "actual"), "n", self._t( "share of gain", "share of gain")])
            data = {"causal_forest": {k: v for k, v in result.items() if k not in ("cate", "qini_curve", "deciles")}}

        lines += ["", self._t( "**Assumption**", "**Assumption**"), result["assumption"]]
        verdict = self._t(
            f"Uplift: ATE {fmt_num(result['ate'], 3)}, top-decile effect {fmt_num(result.get('best_decile_uplift', result.get('best_segment', {}).get('tau', float('nan'))), 3)}.",
            f"Uplift: ATE {fmt_num(result['ate'], 3)}, top-decile effect {fmt_num(result.get('best_decile_uplift', result.get('best_segment', {}).get('tau', float('nan'))), 3)}.",
        )
        self._record(
            verdict,
            Section(
                title=self._t( "Uplift: causal tree", "Uplift: causal tree")
                if want_tree
                else self._t( "Uplift: causal forest", "Uplift: causal forest"),
                prose=self._plain_line(section_prose),
                bullets=[verdict],
                table=table,
                chart=qini_chart_uri or None,
            ),
        )
        return AgentReply(
            text="\n".join(lines),
            blocks=blocks,
            tool_calls=[{"tool": "causal_tree" if want_tree else "causal_forest", "args": {"features": features}, "result": {"ate": result["ate"]}}],
            data=data,
        )

    # ------------------------------------------------------------------------ RDD
    # "门槛 620" and "620 分以上" are both natural ways to name a cutoff, so the
    # number is allowed on either side of the keyword.
    _CUTOFF_AFTER = re.compile(r"(?:阈值|门槛|断点|cutoff|分数|score)\D{0,8}(\d{2,5})", re.I)
    _CUTOFF_BEFORE = re.compile(r"(\d{2,5})\s*(?:分|分以上|分以下|分及以上|分及以下|points?)", re.I)

    def _handle_rdd(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """Regression discontinuity on the membership-score cutoff."""
        df = self.sql.run("SELECT * FROM rdd_scores").dataframe
        cutoff = 600.0
        match = self._CUTOFF_AFTER.search(question) or self._CUTOFF_BEFORE.search(question)
        if match:
            cutoff = float(match.group(1))
        fuzzy = bool(re.search(r"模糊|fuzzy|不完全合规|依从", question, re.I))
        result = regression_discontinuity(
            df,
            outcome="outcome",
            running="score",
            cutoff=cutoff,
            fuzzy="treated" if fuzzy else None,
            covariates=("pre_spend",),
            lang=self.lang,
        )
        density = result["density"]
        blocks = [
            {
                "type": "kpi",
                "items": [
                    {
                        "label": self._t( "RD effect", "RD effect") + (" (LATE)" if fuzzy else ""),
                        "value": fmt_num(result["effect"], 3),
                        "sub": f"p={result['p_value']:.2e}",
                        "good": result["effect"] > 0,
                    },
                    {"label": self._t( "bandwidth", "bandwidth"), "value": f"{result['bandwidth']:.1f}", "sub": result["bandwidth_selection"] if isinstance(result["bandwidth_selection"], str) else self._t( "rule of thumb", "rule of thumb")},
                    {
                        "label": self._t( "density test", "density test"),
                        "value": "PASS" if not density.get("manipulation") else "FAIL",
                        "sub": f"p={density['p_value']:.3f}",
                        "good": not density.get("manipulation"),
                    },
                ],
            }
        ]
        rdd_chart = ""
        try:
            plot = result["plot"]
            rdd_chart = charts.rdd_plot(plot["bins"], plot["left"], plot["right"], result["cutoff"])
            blocks.append(
                {
                    "type": "chart",
                    "title": self._t( "Discontinuity at the cutoff", "Discontinuity at the cutoff"),
                    "image": rdd_chart,
                }
            )
        except Exception:  # noqa: BLE001
            pass

        sens = result["bandwidth_sensitivity"]
        if isinstance(sens, pd.DataFrame) and len(sens):
            blocks.append(
                self._table_block(
                    pd.DataFrame(
                        [[f"{r['bandwidth']:.1f}", fmt_num(r["effect"], 3), f"{int(r['n']):,}"] for _, r in sens.iterrows()],
                        columns=[self._t( "bandwidth", "bandwidth"), self._t( "effect", "effect"), "n"],
                    ),
                    self._t( "Bandwidth sensitivity", "Bandwidth sensitivity"),
                )
            )
        balance = result["covariate_balance"]
        if balance:
            blocks.append(
                self._table_block(
                    pd.DataFrame(
                        [[b["covariate"], fmt_num(b["jump"], 3), f"{b['p_value']:.3f}"] for b in balance],
                        columns=[self._t( "covariate", "covariate"), self._t( "jump", "jump"), "p"],
                    ),
                    self._t( "Covariate balance (should be flat)", "Covariate balance (should be flat)"),
                )
            )

        lines = [
            self._t(
                f"**Regression discontinuity** ({'fuzzy' if fuzzy else 'sharp'} RDD, cutoff {result['cutoff']:.0f}): "
                f"jump of **{fmt_num(result['effect'], 3)}** ({fmt_pct(result['relative'])}, "
                f"95% CI [{fmt_num(result['ci'][0], 3)}, {fmt_num(result['ci'][1], 3)}], p={result['p_value']:.2e}).",
                f"**Regression discontinuity** ({'fuzzy' if fuzzy else 'sharp'} RDD, cutoff {result['cutoff']:.0f}): "
                f"jump of **{fmt_num(result['effect'], 3)}** ({fmt_pct(result['relative'])}, "
                f"95% CI [{fmt_num(result['ci'][0], 3)}, {fmt_num(result['ci'][1], 3)}], p={result['p_value']:.2e}).",
            ),
            "",
            self._t(
                f"Bandwidth {result['bandwidth']:.1f} with {result['n_left']:,} / {result['n_right']:,} observations on each side; "
                f"local-linear fits use a triangular kernel with HC1 robust SEs.",
                f"Bandwidth {result['bandwidth']:.1f} with {result['n_left']:,} / {result['n_right']:,} observations on each side; "
                f"local-linear fits use a triangular kernel with HC1 robust SEs.",
            ),
        ]
        if fuzzy and result["first_stage"]:
            fs = result["first_stage"]
            lines += [
                "",
                self._t(
                    f"**First stage**: the probability of treatment jumps {fs['jump']:.3f} at the cutoff "
                    f"(p={fs['p_value']:.2e}), so this is a LATE for compliers, not an ATE.",
                    f"**First stage**: the probability of treatment jumps {fs['jump']:.3f} at the cutoff "
                    f"(p={fs['p_value']:.2e}), so this is a LATE for compliers, not an ATE.",
                ),
            ]
        lines += [
            "",
            self._t(
                f"**No manipulation**: density test p={density['p_value']:.3f}"
                + (", the running variable is continuous at the cutoff - nobody is gaming the score." if not density.get("manipulation") else " - the density jumps at the cutoff, which points to manipulation and invalidates the design."),
                f"**No manipulation**: density test p={density['p_value']:.3f}"
                + (", the running variable is continuous at the cutoff - nobody is gaming the score." if not density.get("manipulation") else " - the density jumps at the cutoff, which points to manipulation and invalidates the design."),
            ),
            self._t(
                f"**Covariate balance**: {'pass' if result['covariate_balance_ok'] else 'fail'}; "
                f"**bandwidth stability**: {'same sign across bandwidths' if result['bandwidth_stable'] else 'the sign flips with the bandwidth - unstable'}.",
                f"**Covariate balance**: {'pass' if result['covariate_balance_ok'] else 'fail'}; "
                f"**bandwidth stability**: {'same sign across bandwidths' if result['bandwidth_stable'] else 'the sign flips with the bandwidth - unstable'}.",
            ),
        ]
        if result["placebo_cutoffs"]:
            placebos = result["placebo_cutoffs"]
            hits = [p for p in placebos if p.get("p_value", 1) < 0.05]
            lines += [
                "",
                self._t(
                    f"**Placebo cutoffs**: {len(hits)} of {len(placebos)} fake cutoffs show a significant jump"
                    + (" - the real one may be noise." if len(hits) >= max(1, len(placebos) // 3) else ", so the real cutoff stands out."),
                    f"**Placebo cutoffs**: {len(hits)} of {len(placebos)} fake cutoffs show a significant jump"
                    + (" - the real one may be noise." if len(hits) >= max(1, len(placebos) // 3) else ", so the real cutoff stands out."),
                ),
            ]
        lines += ["", self._t( "**Assumption**", "**Assumption**"), result["assumption"]]

        verdict = self._t(
            f"RDD: effect {fmt_num(result['effect'], 3)} at cutoff {result['cutoff']:.0f} (p={result['p_value']:.2e}), "
            f"density test {'pass' if not density.get('manipulation') else 'fail'}.",
            f"RDD: effect {fmt_num(result['effect'], 3)} at cutoff {result['cutoff']:.0f} (p={result['p_value']:.2e}), "
            f"density test {'pass' if not density.get('manipulation') else 'fail'}.",
        )
        self._record(
            verdict,
            Section(
                title=self._t( "Regression discontinuity", "Regression discontinuity"),
                prose=self._plain_line(result["narrative"]),
                bullets=[verdict],
                table=pd.DataFrame(
                    [["effect", fmt_num(result["effect"], 3)], ["p", f"{result['p_value']:.2e}"], ["bandwidth", f"{result['bandwidth']:.1f}"]],
                    columns=[self._t( "item", "item"), self._t( "value", "value")],
                ),
                chart=rdd_chart or None,
            ),
        )
        return AgentReply(
            text="\n".join(lines),
            blocks=blocks,
            tool_calls=[{"tool": "regression_discontinuity", "args": {"cutoff": cutoff, "fuzzy": fuzzy}, "result": {"effect": result["effect"], "p": result["p_value"]}}],
            data={"rdd": {k: v for k, v in result.items() if k != "plot"}},
        )

    # ------------------------------------------------------------------ timeseries
    def _handle_ts_causal(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """Interrupted time series plus the structural counterfactual, side by side."""
        df = self.sql.run("SELECT * FROM its_series").dataframe
        cutoff = pd.Timestamp(df.loc[df["intervention"] == 1, "date"].min())
        match = re.search(r"(\d{4}-\d{2}-\d{2})", question)
        if match and pd.Timestamp(match.group(1)) in set(pd.to_datetime(df["date"])):
            cutoff = pd.Timestamp(match.group(1))
        controls = ("control_gmv_1", "control_gmv_2")
        comparison = compare_ts_methods(df, "date", "gmv", cutoff, controls=controls, lang=self.lang)
        its = comparison["arima_its"]
        impact = comparison["causal_impact"]

        blocks = [
            {
                "type": "kpi",
                "items": [
                    {"label": self._t("ARIMA ITS", "ARIMA ITS"), "value": fmt_pct(its["relative"]), "sub": f"p={its['p_value']:.2e}"},
                    {"label": self._t( "structural", "structural"), "value": fmt_pct(impact["relative"]), "sub": f"p={impact['p_value']:.2e}"},
                    {
                        "label": self._t( "naive", "naive"),
                        "value": fmt_pct(comparison["naive"]["relative"]),
                        "sub": self._t( "includes trend", "includes trend"),
                    },
                ],
            }
        ]
        ts_chart = ""
        try:
            ts_chart = charts.ts_causal_chart(impact["history"], impact["frame"], cutoff)
            blocks.append(
                {
                    "type": "chart",
                    "title": self._t( "Actual vs counterfactual", "Actual vs counterfactual"),
                    "image": ts_chart,
                }
            )
        except Exception:  # noqa: BLE001
            pass
        blocks.append(
            self._table_block(
                comparison["table"].assign(
                    relative=lambda d: d["relative"].map(lambda v: fmt_pct(float(v)) if pd.notna(v) else "-")
                ),
                self._t( "Three readings", "Three readings"),
                pct_cols=(),
            )
        )
        spec = its["spec_table"]
        # Confidence card: residual whiteness, placebo dates, model agreement,
        # counterfactual fit and specification-curve stability.
        conf = ts_causal_confidence(its, impact, comparison["agree"])
        blocks.append(
            self._table_block(
                confidence_table(conf["checks"], self.lang, conf["verdict"]),
                self._t( "Confidence validation", "Confidence validation"),
            )
        )
        if isinstance(spec, pd.DataFrame) and len(spec):
            blocks.append(
                self._table_block(
                    pd.DataFrame(
                        [[str(r["spec"]), fmt_pct(float(r["relative"])), f"{float(r['p_value']):.1e}"] for _, r in spec.iterrows()],
                        columns=[self._t( "specification", "specification"), "effect", "p"],
                    ),
                    self._t( "ITS specification curve", "ITS specification curve"),
                    pct_cols=("effect",),
                )
            )

        # Test panel: stationarity of the series (ADF + KPSS), whiteness and
        # autocorrelation of the residuals. Without these the reported p-value
        # of the intervention dummy is not interpretable.
        panel = None
        try:
            panel = test_panels.its_panel(
                its["residuals"].to_numpy(dtype=float),
                df["gmv"].to_numpy(dtype=float),
            )
            blocks.append(
                self._table_block(
                    stat_tests.to_frame(panel["tests"], self.lang),
                    self._t( "Statistical tests (stationarity + residual whiteness)", "Statistical tests (stationarity + residual whiteness)"),
                    max_rows=10,
                )
            )
        except Exception:  # noqa: BLE001
            panel = None

        lines = [
            self._t(
                f"**Time-series causal** (intervention {cutoff.date()}): ARIMA ITS gives {fmt_pct(its['relative'])} "
                f"({fmt_num(its['effect'])}, p={its['p_value']:.2e}); the structural model gives {fmt_pct(impact['relative'])} "
                f"({fmt_num(impact['effect'])}, p={impact['p_value']:.2e}) - the two {'agree' if comparison['agree'] else 'disagree'}.",
                f"**Time-series causal** (intervention {cutoff.date()}): ARIMA ITS gives {fmt_pct(its['relative'])} "
                f"({fmt_num(its['effect'])}, p={its['p_value']:.2e}); the structural model gives {fmt_pct(impact['relative'])} "
                f"({fmt_num(impact['effect'])}, p={impact['p_value']:.2e}) - the two {'agree' if comparison['agree'] else 'disagree'}.",
            ),
            "",
            self._t(
                f"**Why the naive number is wrong**: the uncorrected pre/post difference is {fmt_pct(comparison['naive']['relative'])}, "
                f"{fmt_pct(comparison['naive']['relative'] - impact['relative'])} above the model estimates - that gap is the trend that was happening anyway.",
                f"**Why the naive number is wrong**: the uncorrected pre/post difference is {fmt_pct(comparison['naive']['relative'])}, "
                f"{fmt_pct(comparison['naive']['relative'] - impact['relative'])} above the model estimates - that gap is the trend that was happening anyway.",
            ),
            "",
            self._t(
                f"**Diagnostics**: ITS residual Ljung-Box p={its['ljung_box_p']:.3f}, placebo p={its['placebo_p']:.3f}; "
                f"structural pre-period one-step R2={impact['pre_fit_r2']:.3f}, placebo p={impact['placebo_p']:.3f}.",
                f"**Diagnostics**: ITS residual Ljung-Box p={its['ljung_box_p']:.3f}, placebo p={its['placebo_p']:.3f}; "
                f"structural pre-period one-step R2={impact['pre_fit_r2']:.3f}, placebo p={impact['placebo_p']:.3f}.",
            ),
            verdict_line(conf["verdict"], self.lang),
        ]
        if panel is not None:
            lines.append(
                self._t(
                    f"**Statistical tests** (method / statistic / p / conclusion): {panel['headline_en']} {panel['summary_en']}",
                    f"**Statistical tests** (method / statistic / p / conclusion): {panel['headline_en']} {panel['summary_en']}",
                )
            )
        if abs(its["relative"] - impact["relative"]) > 0.05:
            lines += [
                "",
                self._t(
                    "The two models differ by more than 5pp: check for a concurrent campaign on the intervention date, "
                    "contamination of the control series, and whether the weekly pattern is absorbed.",
                    "The two models differ by more than 5pp: check for a concurrent campaign on the intervention date, "
                    "contamination of the control series, and whether the weekly pattern is absorbed.",
                ),
            ]
        verdict = self._t(
            f"Time-series causal: ITS {fmt_pct(its['relative'])}, structural {fmt_pct(impact['relative'])}, "
            f"naive {fmt_pct(comparison['naive']['relative'])}.",
            f"Time-series causal: ITS {fmt_pct(its['relative'])}, structural {fmt_pct(impact['relative'])}, "
            f"naive {fmt_pct(comparison['naive']['relative'])}.",
        )
        self._record(
            verdict,
            Section(
                title=self._t( "Time-series causal inference", "Time-series causal inference"),
                prose=self._plain_line(impact["narrative"]),
                bullets=[verdict],
                table=comparison["table"]
                .assign(relative=lambda d: d["relative"].map(lambda v: fmt_pct(float(v)) if pd.notna(v) else "-")),
                chart=ts_chart or None,
            ),
        )
        self._record(
            verdict_line(conf["verdict"], self.lang),
            Section(
                title=self._t( "Time-series causal confidence validation", "Time-series causal confidence validation"),
                prose=verdict_line(conf["verdict"], self.lang)
                + " " + self._t(
                    "Checks cover residual whiteness, placebo intervention dates, two-model agreement, counterfactual fit and the specification curve.",
                    "Checks cover residual whiteness, placebo intervention dates, two-model agreement, counterfactual fit and the specification curve.",
                ),
                table=confidence_table(conf["checks"], self.lang, conf["verdict"]),
            ),
        )
        return AgentReply(
            text="\n".join(lines),
            blocks=blocks,
            tool_calls=[
                {"tool": "arima_intervention", "args": {"intervention": str(cutoff)}, "result": {"relative": its["relative"]}},
                {"tool": "causal_impact", "args": {"controls": list(controls)}, "result": {"relative": impact["relative"]}},
            ],
            data={"its": {k: v for k, v in its.items() if k != "frame"}, "impact": {k: v for k, v in impact.items() if k not in ("frame", "history")}},
        )

    # ------------------------------------------------------------ method select
    def _handle_method_select(self, question: str, slots: Dict[str, Any]) -> AgentReply:
        """Recommend the right estimator for a described scenario."""
        from ds_agent.tools.method_advisor import recommend_methods

        result = recommend_methods(question, lang=self.lang)
        ranked = result["recommendations"]
        blocks: List[Dict[str, Any]] = []
        if ranked:
            blocks.append(
                {
                    "type": "kpi",
                    "items": [
                        {
                            "label": self._t( "first choice", "first choice"),
                            "value": ranked[0]["name"],
                            "sub": ranked[0]["tool"],
                        },
                        {
                            "label": self._t( "alternatives", "alternatives"),
                            "value": ", ".join(r["name"] for r in ranked[1:3]) or "-",
                            "sub": self._t( "for cross-checks", "for cross-checks"),
                        },
                        {
                            "label": self._t( "methods considered", "methods considered"),
                            "value": str(len(ranked)),
                            "sub": self._t( "ranked for this scenario", "ranked for this scenario"),
                        },
                    ],
                }
            )
            rows = [
                [r["name"], r["reason"], r["assumption"], r["pitfall"]]
                for r in ranked[:6]
            ]
            blocks.append(
                self._table_block(
                    pd.DataFrame(rows, columns=[self._t( "method", "method"), self._t( "why", "why"), self._t( "assumption", "assumption"), self._t( "pitfall", "pitfall")]),
                    self._t( "Ranked methods", "Ranked methods"),
                )
            )
            rows_ck = [[c["item"] if self.lang == "zh" else c["item_en"]] for c in result["checklist"]]
            blocks.append(
                self._table_block(
                    pd.DataFrame(rows_ck, columns=[self._t( "Robustness checklist", "Robustness checklist")]),
                    self._t( "Before you conclude", "Before you conclude"),
                )
            )
        lines = [result["narrative"], ""]
        if ranked:
            lines.append(
                self._t(
                    "Run one or two of the runners-up as cross-checks: agreement between estimators beats any single p-value.",
                    "Run one or two of the runners-up as cross-checks: agreement between estimators beats any single p-value.",
                )
            )
        bullets = (
            [self._t( f"First choice: {ranked[0]['name']}.", f"First choice: {ranked[0]['name']}.")] if ranked else []
        )
        self._record(
            self._plain_line(result["narrative"]),
            Section(
                title=self._t( "Recommended methodology", "Recommended methodology"),
                prose=self._plain_line(result["narrative"]),
                bullets=bullets,
                table=pd.DataFrame(
                    [[r["name"], r["reason"]] for r in ranked[:5]],
                    columns=[self._t( "method", "method"), self._t( "reason", "reason")],
                ),
            ),
        )
        return AgentReply(
            text="\n".join(lines),
            blocks=blocks,
            tool_calls=[{"tool": "recommend_methods", "args": {"question": question[:120]}, "result": {"primary": ranked[0]["method"] if ranked else None}}],
            data={"method_select": {k: v for k, v in result.items()}},
        )

    # ------------------------------------------------------------------- helpers
    def _plain_line(self, text: str) -> str:
        """Strip markdown emphasis so a narrative can sit inside report prose."""
        return re.sub(r"\*\*|`", "", text or "")
