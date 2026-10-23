"""Chart rendering for reports and the web UI.

Charts are rendered with matplotlib and returned as base64 PNG data URLs so a
single HTML file is fully self-contained (no external asset requests). Axis
labels default to English because that is guaranteed to render; localised text
that does reach a chart (metric and segment names, which come from the semantic
layer and are Chinese in ``zh`` mode) renders through the CJK fallback installed
below when the host has such a font, and is the only case that needs it.
"""

from __future__ import annotations

import base64
import io
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# CJK families in rough order of how good they look next to DejaVu Sans.
_CJK_FONTS = (
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "SimHei",
    "STHeiti",
)


def _install_fonts() -> None:
    """Put DejaVu first and a CJK family behind it as a per-glyph fallback.

    matplotlib resolves glyphs family by family, so Latin text keeps DejaVu's
    metrics while Chinese characters fall through to the CJK font. Without this
    a Chinese segment name renders as a row of empty boxes, which is worse than
    an untranslated label because it looks like a rendering bug.
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [name for name in _CJK_FONTS if name in available][:1]
    stack = ["DejaVu Sans"] + cjk
    # `font.family` has to hold the explicit list: matplotlib only falls back
    # glyph by glyph when the family list itself names the candidate fonts. One
    # CJK family is enough - listing several only makes matplotlib probe them in
    # turn and log a "font weight" notice for each miss.
    matplotlib.rcParams["font.family"] = stack
    matplotlib.rcParams["font.sans-serif"] = stack
    # A CJK font's minus sign sits outside DejaVu's box and renders as tofu.
    matplotlib.rcParams["axes.unicode_minus"] = False


_install_fonts()

PALETTE = {
    "primary": "#2563eb",
    "danger": "#dc2626",
    "success": "#16a34a",
    "muted": "#94a3b8",
    "grid": "#e2e8f0",
    "text": "#0f172a",
}


def _style(ax, title: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=12, color=PALETTE["text"], loc="left", pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color="#475569")
    ax.tick_params(colors="#475569", labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PALETTE["grid"])
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _to_data_url(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def anomaly_chart(frame: pd.DataFrame, title: str = "Metric vs expected") -> str:
    """Actual vs seasonally-adjusted expectation with anomaly markers."""
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    ax.plot(frame["date"], frame["actual"], color=PALETTE["primary"], linewidth=1.8, label="actual")
    ax.plot(frame["date"], frame["expected"], color=PALETTE["muted"], linewidth=1.2, linestyle="--", label="expected")
    flagged = frame[frame.get("flagged", False) == True]  # noqa: E712
    if len(flagged):
        ax.scatter(flagged["date"], flagged["actual"], color=PALETTE["danger"], s=38, zorder=5, label="anomaly")
    ax.fill_between(frame["date"], frame["expected"], frame["actual"], color=PALETTE["primary"], alpha=0.08)
    _style(ax, title)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.autofmt_xdate(rotation=0)
    return _to_data_url(fig)


def ps_overlap_chart(payload: Dict[str, object], title: str = "Propensity score overlap") -> str:
    """Histogram of propensity scores by arm — the visual common-support check."""
    edges = np.asarray(payload["edges"], dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    ax.bar(centers - width * 0.2, payload["treated"], width=width * 0.4, color=PALETTE["primary"], label="treated", alpha=0.85)
    ax.bar(centers + width * 0.2, payload["control"], width=width * 0.4, color=PALETTE["muted"], label="control", alpha=0.85)
    _style(ax, title, "units")
    ax.set_xlabel("propensity score", fontsize=9, color="#475569")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return _to_data_url(fig)


def contribution_bar(segments: pd.DataFrame, dimension: str, title: str = "Change contribution") -> str:
    """Horizontal bar chart of contribution by segment, sorted by magnitude."""
    data = segments.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.4, max(2.6, 0.34 * len(data) + 1.2)))
    colors = [PALETTE["success"] if v >= 0 else PALETTE["danger"] for v in data["contribution"]]
    ax.barh(data["segment"].astype(str), data["contribution"], color=colors, height=0.62)
    ax.axvline(0, color="#475569", linewidth=0.9)
    _style(ax, title, f"contribution to change ({dimension})")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
    return _to_data_url(fig)


def effect_forest(metrics: List[Dict[str, object]], title: str = "Treatment effect (95% CI)") -> str:
    """Forest plot of relative lift per metric."""
    rows = [m for m in metrics if m.get("ci_abs") and m.get("abs_diff") is not None]
    if not rows:
        return ""
    fig, ax = plt.subplots(figsize=(8.4, max(2.2, 0.5 * len(rows) + 1.2)))
    ys = np.arange(len(rows))
    for i, m in enumerate(rows):
        lo, hi = m["ci_abs"]
        color = PALETTE["success"] if lo > 0 else (PALETTE["danger"] if hi < 0 else PALETTE["muted"])
        ax.plot([lo, hi], [i, i], color=color, linewidth=2.2, solid_capstyle="round")
        ax.scatter([m["abs_diff"]], [i], color=color, s=46, zorder=5)
    ax.axvline(0, color="#475569", linewidth=0.9, linestyle="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([m["metric"] for m in rows], fontsize=9)
    _style(ax, title, "absolute difference")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
    return _to_data_url(fig)


def power_chart(curve: Sequence[Dict[str, float]], baseline: float) -> str:
    """Sample size needed as a function of the target relative MDE."""
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    x = [c["mde_rel"] * 100 for c in curve]
    y = [c["n_per_arm"] for c in curve]
    ax.plot(x, y, color=PALETTE["primary"], marker="o", linewidth=1.8)
    _style(ax, f"Sample size per arm vs MDE (baseline {baseline:.2%})", "users per arm")
    ax.set_xlabel("relative MDE (%)", fontsize=9, color="#475569")
    ax.set_yscale("log")
    return _to_data_url(fig)


def gap_chart(gap_frame: pd.DataFrame, treat_time, title: str = "Synthetic control gap") -> str:
    """Actual vs synthetic series for the synthetic-control estimator."""
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    ax.plot(gap_frame["time"], gap_frame["actual"], color=PALETTE["primary"], linewidth=1.8, label="treated unit")
    ax.plot(gap_frame["time"], gap_frame["synthetic"], color=PALETTE["muted"], linewidth=1.4, linestyle="--", label="synthetic")
    ax.axvline(treat_time, color=PALETTE["danger"], linewidth=1.0, linestyle=":")
    _style(ax, title)
    ax.legend(frameon=False, fontsize=8)
    return _to_data_url(fig)


def event_study_chart(coefficients: Sequence[Dict[str, float]], title: str = "Event study") -> str:
    """Event-study coefficients with confidence intervals."""
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    periods = [c["rel_period"] for c in coefficients]
    coefs = [c["coef"] for c in coefficients]
    lows = [c["ci"][0] for c in coefficients]
    highs = [c["ci"][1] for c in coefficients]
    ax.errorbar(periods, coefs, yerr=[np.array(coefs) - np.array(lows), np.array(highs) - np.array(coefs)],
                fmt="o", color=PALETTE["primary"], ecolor=PALETTE["muted"], capsize=3, markersize=5)
    ax.axhline(0, color="#475569", linewidth=0.9, linestyle="--")
    ax.axvline(-0.5, color=PALETTE["danger"], linewidth=0.9, linestyle=":")
    _style(ax, title, "coefficient")
    ax.set_xlabel("periods relative to treatment", fontsize=9, color="#475569")
    return _to_data_url(fig)


def qini_chart(curve: pd.DataFrame, deciles: pd.DataFrame, title: str = "Uplift model quality") -> str:
    """Qini curve next to the decile uplift profile.

    The Qini curve answers "if I target the top x% by predicted uplift, how much
    incremental outcome do I capture"; the decile bars answer "is the ranking
    monotone", which is the part stakeholders actually argue about.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.plot(curve["fraction"] * 100, curve["model"], color=PALETTE["primary"], linewidth=2.0, label="model")
    ax.plot(curve["fraction"] * 100, curve["random"], color=PALETTE["muted"], linewidth=1.2, linestyle="--", label="random")
    ax.fill_between(curve["fraction"] * 100, curve["random"], curve["model"], color=PALETTE["primary"], alpha=0.10)
    _style(ax, f"{title}: Qini", "cumulative incremental outcome")
    ax.set_xlabel("% of population targeted", fontsize=9, color="#475569")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax2 = axes[1]
    colors = [PALETTE["primary"] if v >= 0 else PALETTE["danger"] for v in deciles["uplift"]]
    ax2.bar(deciles["decile"].astype(str), deciles["uplift"], color=colors, width=0.68)
    ax2.axhline(0, color="#475569", linewidth=0.8)
    _style(ax2, "Uplift by predicted decile", "average uplift")
    ax2.set_xlabel("decile of predicted uplift (1 = highest)", fontsize=9, color="#475569")
    ax2.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    return _to_data_url(fig)


def rdd_plot(bins: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, cutoff: float,
             title: str = "Regression discontinuity") -> str:
    """Binned outcome against the running variable with local-linear fits per side."""
    fig, ax = plt.subplots(figsize=(8.4, 3.3))
    ax.scatter(bins["x"], bins["y"], s=22, color=PALETTE["muted"], alpha=0.85, label="bin mean", zorder=3)
    if len(left):
        ax.plot(left["x"], left["fit"], color=PALETTE["primary"], linewidth=2.0, label="fit (control side)")
    if len(right):
        ax.plot(right["x"], right["fit"], color=PALETTE["danger"], linewidth=2.0, label="fit (treated side)")
    ax.axvline(cutoff, color="#475569", linewidth=1.0, linestyle=":")
    _style(ax, title, "outcome")
    ax.set_xlabel("running variable", fontsize=9, color="#475569")
    ax.legend(frameon=False, fontsize=8, loc="best")
    return _to_data_url(fig)


def ts_causal_chart(history: pd.DataFrame, frame: pd.DataFrame, intervention, title: str = "Counterfactual") -> str:
    """Observed series, pre-period fit and post-period counterfactual with a band."""
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.plot(history["time"], history["actual"], color=PALETTE["primary"], linewidth=1.6, label="actual")
    ax.plot(history["time"], history["counterfactual"], color=PALETTE["muted"], linewidth=1.0, linestyle="--",
            label="model fit")
    ax.plot(frame["time"], frame["actual"], color=PALETTE["primary"], linewidth=1.8)
    ax.plot(frame["time"], frame["counterfactual"], color=PALETTE["danger"], linewidth=1.6, label="counterfactual")
    ax.fill_between(frame["time"], frame["lower"], frame["upper"], color=PALETTE["danger"], alpha=0.12)
    ax.axvline(pd.Timestamp(intervention), color="#475569", linewidth=1.0, linestyle=":")
    _style(ax, title, "")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.autofmt_xdate(rotation=0)
    return _to_data_url(fig)


def dl_reconstruction_chart(result: Dict[str, Any], dates: Sequence[Any], title: str = "Autoencoder anomaly detection") -> str:
    """Actual vs reconstruction with the anomaly score and threshold underneath."""
    idx = np.asarray(result["idx"])
    dates = np.asarray(dates)
    day_axis = dates[idx - 1] if result.get("window") else dates[idx]
    actual = np.asarray(result["actual"], dtype=float)
    rec = np.asarray(result.get("reconstruction", actual), dtype=float)
    err = np.asarray(result["error"], dtype=float)
    flagged = np.asarray(result["flagged"], dtype=bool)
    threshold = result.get("threshold")

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.6, 4.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    ax.plot(day_axis, actual, color=PALETTE["primary"], linewidth=1.4, label="actual")
    ax.plot(day_axis, rec, color=PALETTE["muted"], linewidth=1.2, linestyle="--", label="reconstruction")
    ax.scatter(day_axis[flagged], actual[flagged], s=42, color=PALETTE["danger"], zorder=4, label="anomaly")
    _style(ax, title, "value")
    ax.legend(frameon=False, fontsize=8, loc="best")

    ax2.plot(day_axis, err, color=PALETTE["muted"], linewidth=1.1)
    if threshold is not None:
        ax2.axhline(threshold, color=PALETTE["danger"], linestyle=":", linewidth=1.0)
    ax2.fill_between(day_axis, err, 0, where=flagged, color=PALETTE["danger"], alpha=0.25)
    _style(ax2, None, "score")
    fig.autofmt_xdate(rotation=0)
    return _to_data_url(fig)


def engine_comparison_chart(rows: List[Dict[str, Any]], title: str = "Cross-engine estimates") -> str:
    """Horizontal bars comparing effect estimates from independent engines."""
    labels = [str(r["engine"]) for r in rows]
    values = [float(r["estimate"]) for r in rows]
    colors = [PALETTE["primary"] if r.get("engine") != "dowhy" else PALETTE["success"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 0.62 * max(2, len(rows)) + 1.2))
    ypos = np.arange(len(rows))[::-1]
    ax.barh(ypos, values, color=colors, height=0.55)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    for y, v in zip(ypos, values):
        ax.text(v, y, f" {v:,.4g}", va="center", fontsize=8, color=PALETTE["text"])
    ax.axvline(0, color=PALETTE["grid"], linewidth=0.8)
    ax.set_title(title, fontsize=11, color=PALETTE["text"], loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("effect estimate", fontsize=9, color=PALETTE["muted"])
    fig.tight_layout()
    return _to_data_url(fig)


def switchback_chart(frame: pd.DataFrame, title: str = "Switchback schedule") -> str:
    """Daily series coloured by the arm that was live, so blocks are visible."""
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    work = frame.copy()
    work["time"] = pd.to_datetime(work["time"])
    treated = work[work["arm"] == "treatment"]
    control = work[work["arm"] == "control"]
    ax.plot(work["time"], work["value"], color=PALETTE["grid"], linewidth=1.0, zorder=1)
    ax.scatter(control["time"], control["value"], s=26, color=PALETTE["muted"], label="control block", zorder=3)
    ax.scatter(treated["time"], treated["value"], s=26, color=PALETTE["primary"], label="treatment block", zorder=3)
    _style(ax, title, "daily metric")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.autofmt_xdate(rotation=0)
    return _to_data_url(fig)


# --------------------------------------------------------------------------
# Confidence-validation charts: the visual half of the confidence card.
# --------------------------------------------------------------------------
def confidence_band_chart(frame: pd.DataFrame, title: str = "Actual vs expected ± 2.5σ") -> str:
    """Actual series against its expectation with a noise band.

    A deviation is credible as an anomaly only when it leaves the band the
    model's own residual scale implies — this chart is that judgement, drawn.
    """
    fig, ax = plt.subplots(figsize=(8.6, 3.1))
    ax.fill_between(frame["date"], frame["band_low"], frame["band_high"], color=PALETTE["primary"], alpha=0.12, label="±2.5σ band")
    ax.plot(frame["date"], frame["actual"], color=PALETTE["primary"], linewidth=1.7, label="actual")
    ax.plot(frame["date"], frame["expected"], color=PALETTE["muted"], linewidth=1.1, linestyle="--", label="expected")
    flagged = frame[frame["flagged"] == True]  # noqa: E712
    if len(flagged):
        outside = flagged[(flagged["actual"] > flagged["band_high"]) | (flagged["actual"] < flagged["band_low"])]
        ax.scatter(flagged["date"], flagged["actual"], s=30, color=PALETTE["danger"], alpha=0.55, zorder=4, label="flagged")
        if len(outside):
            ax.scatter(outside["date"], outside["actual"], s=58, facecolors="none", edgecolors=PALETTE["danger"], linewidths=1.6, zorder=5, label="outside band")
    _style(ax, title)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    fig.autofmt_xdate(rotation=0)
    return _to_data_url(fig)


def permutation_chart(null_diffs: Sequence[float], observed: float, title: str = "Permutation null distribution") -> str:
    """Histogram of permuted arm differences with the observed effect marked.

    Shows how far the real effect sits in the randomisation null — the visual
    cross-check on the analytic p-value.
    """
    arr = np.asarray(null_diffs, dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    ax.hist(arr, bins=36, color=PALETTE["muted"], alpha=0.85)
    ax.axvline(observed, color=PALETTE["danger"], linewidth=1.8, label=f"observed {observed:+.4g}")
    ax.axvline(0, color="#475569", linewidth=0.9, linestyle="--")
    _style(ax, title, "frequency")
    ax.set_xlabel("difference in means under permuted labels", fontsize=9, color="#475569")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return _to_data_url(fig)


def placebo_chart(placebo_effects: Sequence[float], observed: float, title: str = "In-space placebo distribution") -> str:
    """Histogram of fake-treatment effects with the real estimate marked.

    If the real ATT sits inside this distribution, the design cannot separate
    the policy from ordinary unit-level noise — that failure should be visible,
    not footnoted.
    """
    arr = np.asarray(placebo_effects, dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    ax.hist(arr, bins=24, color=PALETTE["muted"], alpha=0.85)
    ax.axvline(observed, color=PALETTE["danger"], linewidth=1.8, label=f"real ATT {observed:+.4g}")
    ax.axvline(0, color="#475569", linewidth=0.9, linestyle="--")
    _style(ax, title, "frequency")
    ax.set_xlabel("placebo ATT (fake treated units)", fontsize=9, color="#475569")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return _to_data_url(fig)


def root_cause_tree_chart(nodes: List[Dict[str, object]], title: str = "Root-cause drill-down tree") -> str:
    """Icicle-style drill-down tree: one bar per node, indented by depth.

    Bar length is the node's |share of root move|, colour the sign. Reading
    order is depth-first, so a child always sits directly under its parent and
    the deepest indent is the deepest drill the tree reached.
    """
    # Width scale: the largest |share| on any node sets the full bar length.
    shares = [abs(float(n["share_of_root"])) for n in nodes if np.isfinite(n.get("share_of_root", np.nan))]
    scale = max(shares) or 1.0
    n = len(nodes)
    fig, ax = plt.subplots(figsize=(8.8, max(2.4, 0.42 * n + 1.0)))
    ys = np.arange(n)[::-1]  # root on top, depth-first order below
    for y, node in zip(ys, nodes):
        share = float(node.get("share_of_root", 0.0))
        length = abs(share) / scale
        depth = int(node["depth"])
        color = PALETTE["danger"] if share < 0 else (PALETTE["success"] if share > 0 else PALETTE["muted"])
        ax.barh(y, length, left=0, height=0.62, color=color, alpha=0.85)
        share_txt = f"{share:+.0%}" if np.isfinite(share) else "-"
        label = node.get("path_label") or "ALL"
        ax.text(0.004, y, label, va="center", ha="left", fontsize=8.2, color="white" if length > 0.28 else PALETTE["text"],
                transform=ax.get_yaxis_transform())
        ax.text(1.005, y, f"{share_txt}  ({float(node['delta']):+,.3g})", va="center", ha="left",
                fontsize=7.6, color="#475569", transform=ax.get_yaxis_transform())
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, n - 0.4)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["grid"])
    ax.set_title(title, fontsize=12, color=PALETTE["text"], loc="left", pad=10)
    ax.set_xlabel("share of root metric move (bar length, indentation = drill depth)", fontsize=8.4, color="#475569")
    return _to_data_url(fig)



def confusion_matrix_chart(matrix: Sequence[Sequence[int]], labels: Sequence[str], title: str = "Confusion matrix") -> str:
    """Heatmap of a 2x2 (or nxn) confusion matrix with counts annotated."""
    m = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlabel("predicted", fontsize=9, color="#475569")
    ax.set_ylabel("actual", fontsize=9, color="#475569")
    thresh = m.max() / 2 if m.size else 0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=11,
                    color="white" if m[i, j] > thresh else PALETTE["text"])
    ax.set_title(title, fontsize=12, color=PALETTE["text"], loc="left", pad=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _to_data_url(fig)


def roc_chart(fpr: Sequence[float], tpr: Sequence[float], auc: float, title: str = "ROC curve") -> str:
    """ROC curve with the AUC annotated; diagonal = random baseline."""
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(fpr, tpr, color=PALETTE["primary"], linewidth=1.8, label=f"model (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color=PALETTE["muted"], linewidth=1.1, linestyle="--", label="random")
    ax.fill_between(fpr, tpr, color=PALETTE["primary"], alpha=0.08)
    ax.set_xlabel("false positive rate", fontsize=9, color="#475569")
    ax.set_ylabel("true positive rate", fontsize=9, color="#475569")
    ax.legend(fontsize=8, frameon=False)
    _style(ax, title)
    return _to_data_url(fig)


def importance_bar(rows: List[Dict[str, Any]], title: str = "Feature importance") -> str:
    """Horizontal importance bars (permutation importance or mean |SHAP|)."""
    rows = list(rows)[:12][::-1]
    names = [r["feature"] for r in rows]
    vals = [float(r["value"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.0, max(2.2, 0.38 * len(rows) + 0.9)))
    ax.barh(names, vals, color=PALETTE["primary"], alpha=0.85, height=0.6)
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=8, color="#475569")
    _style(ax, title, "")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8, alpha=0.9)
    ax.grid(axis="y", visible=False)
    return _to_data_url(fig)


def pred_vs_actual_chart(y_true: Sequence[float], y_pred: Sequence[float], title: str = "Predicted vs actual") -> str:
    """Scatter of predictions against truth with the 45-degree reference line."""
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(y_true, y_pred, s=14, color=PALETTE["primary"], alpha=0.45, edgecolors="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color=PALETTE["muted"], linewidth=1.1, linestyle="--")
    ax.set_xlabel("actual", fontsize=9, color="#475569")
    ax.set_ylabel("predicted", fontsize=9, color="#475569")
    _style(ax, title)
    return _to_data_url(fig)
