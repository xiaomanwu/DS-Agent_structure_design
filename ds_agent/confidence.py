"""Confidence validation for the agent's first-class analyses.

Every headline number (an anomaly, an experiment lift, a causal estimate)
ends with the same stakeholder question: *how much should this be trusted?*
This module makes the answer standard and visible instead of scattered
across prose:

- a list of named checks (``PASS`` / ``WARN`` / ``FAIL``) tailored to the
  analysis kind — significance, SRM, parallel trends, placebo permutation,
  specification agreement, outlier dependence, model agreement, ...
- an overall verdict (高置信 / 中等置信 / 低置信) derived from the checks;
- the data for a dedicated validation chart, so the confidence story ships
  with the answer rather than living in a log.

The builders below only assemble checks and chart inputs; the charts
themselves live in ``ds_agent.charts`` and the handlers decide what to show.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_THRESHOLD = 0.05


def check(name_zh: str, name_en: str, status: str, detail_zh: str, detail_en: str) -> Dict[str, str]:
    """One validation row: bilingual name, status, bilingual explanation."""
    return {
        "name_zh": name_zh,
        "name_en": name_en,
        "status": status,
        "detail_zh": detail_zh,
        "detail_en": detail_en,
    }


def verdict(checks: Sequence[Dict[str, str]]) -> Dict[str, object]:
    """Overall verdict from a check list.

    High confidence: no FAIL and ≥ 80% PASS. Low confidence: any FAIL.
    Medium: everything else. FAIL weights more than a missing WARN — a single
    broken randomisation check invalidates the whole readout.
    """
    n = len(checks) or 1
    n_pass = sum(1 for c in checks if c["status"] == PASS)
    n_fail = sum(1 for c in checks if c["status"] == FAIL)
    score = n_pass / n
    if n_fail:
        label_zh, label_en = "低置信", "low confidence"
    elif score >= 0.8:
        label_zh, label_en = "高置信", "high confidence"
    else:
        label_zh, label_en = "中等置信", "medium confidence"
    return {"score": round(score, 3), "n_fail": n_fail, "label_zh": label_zh, "label_en": label_en}


def confidence_table(checks: Sequence[Dict[str, str]], lang: str, verdict_info: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    """Renderable check table for blocks / reports."""
    zh = lang != "en"
    rows = []
    for c in checks:
        rows.append(
            [
                c["name_zh"] if zh else c["name_en"],
                c["status"],
                c["detail_zh"] if zh else c["detail_en"],
            ]
        )
    df = pd.DataFrame(rows, columns=["检查项" if zh else "check", "结果" if zh else "result", "说明" if zh else "detail"])
    if verdict_info:
        df.loc[len(df)] = [
            "总体结论" if zh else "overall",
            verdict_info["label_zh"] if zh else verdict_info["label_en"],
            f"通过率 {verdict_info['score']:.0%}" if zh else f"{verdict_info['score']:.0%} checks passed",
        ]
    return df


def verdict_line(verdict_info: Dict[str, object], lang: str) -> str:
    zh = lang != "en"
    if zh:
        return f"**置信度验证**：通过率 {verdict_info['score']:.0%} → **{verdict_info['label_zh']}**。"
    return f"**Confidence validation**: {verdict_info['score']:.0%} of checks passed -> **{verdict_info['label_en']}**."


# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------
def anomaly_band(frame: pd.DataFrame, sigma_rel: float, z: float = 2.5) -> pd.DataFrame:
    """Build the ±z·σ noise band around the expectation for display.

    ``sigma_rel`` is the detector's robust residual scale in *relative* terms;
    scaling it by the expected level keeps the band proportional across the
    series instead of a flat corridor that means nothing at trend peaks.
    """
    band = frame[["date", "actual", "expected", "flagged"]].copy()
    half = band["expected"].abs() * sigma_rel * z
    band["band_low"] = band["expected"] - half
    band["band_high"] = band["expected"] + half
    return band


def anomaly_confidence(
    sig: Optional[Dict[str, object]],
    window_values: np.ndarray,
    window_expected: np.ndarray,
    window_z: Optional[np.ndarray],
    chosen_method: str,
    mase: Optional[float],
    episodes: Sequence[Dict[str, object]],
    scan_top: Optional[Dict[str, object]],
    total_delta: float,
    sigma_rel: float,
) -> Dict[str, object]:
    """Checks for an anomaly diagnosis.

    Includes a leave-one-out recheck: drop the single most extreme in-window
    day and re-test. A shift that survives without its worst day is not an
    artefact of one outlier — this is the cheap version of an influence check.
    """
    checks: List[Dict[str, str]] = []

    # 1. Window significance vs the seasonal baseline.
    if sig is not None:
        p = float(sig["p_value"])
        ok = p < _THRESHOLD
        checks.append(
            check(
                "窗口显著性", "window significance",
                PASS if ok else FAIL,
                f"相对季节性基线 p={p:.4f}（{'显著' if ok else '不显著'}）",
                f"p={p:.4f} vs the seasonal baseline ({'significant' if ok else 'not significant'})",
            )
        )
    else:
        checks.append(check("窗口显著性", "window significance", WARN, "基线期望缺失，无法检验", "no model expectation to test against"))

    # 2. Baseline quality: a bad MASE produces confident nonsense.
    if chosen_method and mase is not None:
        ok = mase < 1.0
        checks.append(
            check(
                "基线质量", "baseline quality",
                PASS if ok else WARN,
                f"{chosen_method} 滚动回测 MASE={mase:.3f}（{'优于季节朴素' if ok else '劣于季节朴素，预期不可全信'}）",
                f"{chosen_method} rolling MASE={mase:.3f} ({'better than seasonal naive' if ok else 'worse than seasonal naive'})",
            )
        )
    else:
        checks.append(check("基线质量", "baseline quality", WARN, "使用 trailing 窗口基线，未做模型回测", "trailing baseline, no backtest"))

    # 3. Persistence: a level shift that lasts is not a one-day blip.
    if episodes:
        top = episodes[0]
        days = int(top.get("days", 1))
        ok = days >= 3
        checks.append(
            check(
                "持续性", "persistence",
                PASS if ok else WARN,
                f"最大异常区间持续 {days} 天（{'持续位移' if ok else '疑似单日噪声'}）",
                f"largest episode lasts {days} day(s) ({'sustained shift' if ok else 'possibly one-day noise'})",
            )
        )
    else:
        checks.append(check("持续性", "persistence", WARN, "未检出异常区间", "no episodes detected"))

    # 4. Leave-one-out: does the shift survive without its worst day?
    loo_status = WARN
    loo_detail_zh = "样本不足，跳过"
    loo_detail_en = "skipped: not enough points"
    if len(window_values) >= 5 and np.isfinite(window_expected).all():
        resid = np.abs(window_values - window_expected)
        keep = resid < resid.max()  # drop the single worst day
        try:
            # Paired: the expectation is one value per day, so day-pairs are the unit.
            before = sps.ttest_rel(window_values, window_expected).pvalue
            after = sps.ttest_rel(window_values[keep], window_expected[keep]).pvalue
            loo_status = PASS if after < _THRESHOLD else FAIL
            loo_detail_zh = f"去掉最极端一天后 p={after:.4f}（全窗口 p={before:.4f}）"
            loo_detail_en = f"p={after:.4f} after dropping the most extreme day (full window p={before:.4f})"
        except ValueError:
            pass
    checks.append(check("极端值依赖（留一）", "outlier dependence (leave-one-out)", loo_status, loo_detail_zh, loo_detail_en))

    # 5. Segment agreement: the segment scan should point the same way.
    if scan_top and scan_top.get("window_change") is not None:
        same = (scan_top["window_change"] > 0) == (total_delta > 0)
        checks.append(
            check(
                "分段一致性", "segment agreement",
                PASS if same else WARN,
                f"最异常分段方向与总体{'一致' if same else '相反'}（分段 {scan_top['window_change']:+.1%} vs 总体）",
                f"worst segment direction {'agrees' if same else 'disagrees'} with the topline ({scan_top['window_change']:+.1%} vs overall)",
            )
        )
    else:
        checks.append(check("分段一致性", "segment agreement", WARN, "分段扫描无结果", "no segment-scan result"))

    v = verdict(checks)
    return {"checks": checks, "verdict": v}


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def permutation_test(values: np.ndarray, groups: np.ndarray, n_iter: int = 1000, seed: int = 42) -> Dict[str, object]:
    """Label-permutation null distribution for the difference in means.

    Answers "is the analytic p-value believable" without distributional
    assumptions: shuffle arm labels, rebuild the diff, and see where the
    observed diff lands. Deterministic via a fixed seed.
    """
    obs = float(values[groups == 1].mean() - values[groups == 0].mean())
    rng = np.random.default_rng(seed)
    n = len(values)
    null = np.empty(n_iter)
    for i in range(n_iter):
        # Permute the labels only: values stay fixed, so the null distribution
        # is over reassignments, not over reshuffled copies of the data.
        g = rng.permutation(groups)
        null[i] = values[g == 1].mean() - values[g == 0].mean()
    p_perm = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_iter + 1))
    return {"observed": obs, "null": null, "p_perm": p_perm}


def experiment_confidence(
    srm: Dict[str, object],
    primary: Dict[str, object],
    cuped: Optional[Dict[str, object]],
    cuped_metric: Optional[Dict[str, object]],
    segments: Sequence[Dict[str, object]],
    perm: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Checks for a standard AB readout.

    ``perm`` is the label-permutation result for the primary metric; it is
    ``None`` for ratio metrics, where a permuted difference of per-row values
    would test a different estimand — the check then degrades to a WARN note
    instead of pretending.
    """
    checks: List[Dict[str, str]] = []

    srm_ok = bool(srm["passed"])
    checks.append(
        check(
            "SRM 分流比检查", "SRM (sample-ratio mismatch)",
            PASS if srm_ok else FAIL,
            f"{srm['message']}（p={srm['p_value']:.3f}）",
            f"{srm['message']} (p={srm['p_value']:.3f})",
        )
    )

    prim_ok = bool(primary["significant_adjusted"])
    checks.append(
        check(
            "主指标显著性（BH 校正后）", "primary significance (BH-adjusted)",
            PASS if prim_ok else WARN,
            f"p={primary['p_value']:.4f}，校正后 p={primary['p_adjusted']:.4f}，CI 不含 0: {primary['ci_abs'][0] > 0 or primary['ci_abs'][1] < 0}",
            f"p={primary['p_value']:.4f}, adjusted p={primary['p_adjusted']:.4f}, CI excludes 0: "
            f"{primary['ci_abs'][0] > 0 or primary['ci_abs'][1] < 0}",
        )
    )

    # Permutation agreement: the analytic p and the randomisation p should tell
    # the same story. This is the "don't trust one formula" check.
    if perm is not None:
        agree = (perm["p_perm"] < _THRESHOLD) == prim_ok
        gap = abs(perm["p_perm"] - float(primary["p_value"]))
        checks.append(
            check(
                "置换检验交叉验证", "permutation cross-check",
                PASS if agree and gap < 0.03 else WARN,
                f"重排 p={perm['p_perm']:.4f} vs 解析 p={primary['p_value']:.4f}（{'结论一致' if agree else '结论不一致，需警惕'}）",
                f"permutation p={perm['p_perm']:.4f} vs analytic p={primary['p_value']:.4f} ({'agrees' if agree else 'disagrees'})",
            )
        )
    else:
        checks.append(
            check(
                "置换检验交叉验证", "permutation cross-check",
                WARN,
                "比率类指标未做置换检验（重排会改变估计对象）",
                "skipped for ratio metrics (permutation would change the estimand)",
            )
        )

    if cuped and cuped_metric:
        same_sign = (cuped["rel_lift"] > 0) == (primary["rel_lift"] > 0)
        checks.append(
            check(
                "CUPED 方向一致", "CUPED direction agreement",
                PASS if same_sign else WARN,
                f"校正后 {cuped['rel_lift']:+.2%} vs 校正前 {primary['rel_lift']:+.2%}（方差缩减 {cuped['variance_reduction']:.0%}）",
                f"adjusted {cuped['rel_lift']:+.2%} vs raw {primary['rel_lift']:+.2%} (variance cut {cuped['variance_reduction']:.0%})",
            )
        )

    risky = [s for s in segments if s["rel_lift"] * primary["rel_lift"] < 0 and s["p_value"] < 0.05]
    checks.append(
        check(
            "分群无反向显著", "no adverse significant segment",
            PASS if not risky else WARN,
            "无反向且显著的分群" if not risky else "存在反向显著分群: " + ", ".join(str(s["segment"]) for s in risky[:3]),
            "no significant adverse segment" if not risky
            else "significant adverse segments: " + ", ".join(str(s["segment"]) for s in risky[:3]),
        )
    )

    v = verdict(checks)
    return {"checks": checks, "verdict": v}


# --------------------------------------------------------------------------
# Difference-in-differences
# --------------------------------------------------------------------------
def did_placebo(panel: pd.DataFrame, outcome: str, unit: str, time: str,
                n_iter: int = 60, seed: int = 42) -> Dict[str, object]:
    """In-space placebo: re-run TWFE DiD with fake treated cities.

    The treated label is permuted across cities (same count as real), the DiD
    re-estimated each time. The real ATT should sit in the far tail of the
    placebo distribution; if it sits in the middle, the design cannot separate
    the policy from city-level noise.
    """
    from .tools.causal import difference_in_differences

    n_treated = int(panel.loc[panel["treated"] == 1, unit].nunique())
    cities = panel[unit].unique()
    rng = np.random.default_rng(seed)
    effects = []
    for _ in range(n_iter):
        fake = set(rng.choice(cities, size=n_treated, replace=False).tolist())
        fake_panel = panel.copy()
        fake_panel["treated"] = fake_panel[unit].isin(fake).astype(int)
        try:
            res = difference_in_differences(fake_panel, outcome, unit, time, "treated", "post")
            effects.append(float(res["att"]))
        except Exception:  # noqa: BLE001 - a single failed placebo run is not fatal
            continue
    return {"effects": effects, "n": len(effects)}


def did_confidence(
    did: Dict[str, object],
    did_log: Dict[str, object],
    es: Dict[str, object],
    placebo: Dict[str, object],
    n_treated_units: int,
) -> Dict[str, object]:
    """Checks for the DiD readout."""
    checks: List[Dict[str, str]] = []

    pre_ok = bool(es["parallel_trends_ok"])
    checks.append(
        check(
            "平行趋势（前置期联合检验）", "parallel trends (joint pre-trend)",
            PASS if pre_ok else FAIL,
            f"前置期系数联合检验 p={es['pre_trend_joint_p']:.3f}",
            f"joint test on pre-period leads p={es['pre_trend_joint_p']:.3f}",
        )
    )

    pre_coefs = [c for c in es["coefficients"] if c["rel_period"] < 0]
    if pre_coefs:
        max_abs_t = max(abs(c["coef"]) / c["se"] for c in pre_coefs if c["se"] > 0)
        checks.append(
            check(
                "前置期系数均不显著", "no significant pre-period lead",
                PASS if max_abs_t < 2.0 else WARN,
                f"最大 |t|={max_abs_t:.2f}（阈值 2.0）",
                f"max |t|={max_abs_t:.2f} (threshold 2.0)",
            )
        )

    if placebo["n"] >= 20:
        eff = np.asarray(placebo["effects"])
        tail = float(np.mean(np.abs(eff) >= abs(did["att"])))
        checks.append(
            check(
                "安慰剂检验（假处理城市）", "in-space placebo (fake treated cities)",
                PASS if tail <= 0.05 else (WARN if tail <= 0.15 else FAIL),
                f"实际 ATT 超过 {tail:.0%} 的安慰剂效应（n={placebo['n']}）",
                f"real ATT exceeds {tail:.0%} of placebo effects (n={placebo['n']})",
            )
        )

    same_sign = (did["att"] > 0) == (did_log["relative_effect"] > 0)
    checks.append(
        check(
            "口径稳健（level vs log）", "specification robustness (level vs log)",
            PASS if same_sign else WARN,
            f"level ATT {did['att']:+.4g} 与 log 相对效应 {did_log['relative_effect']:+.2%} {'同号' if same_sign else '异号'}",
            f"level ATT {did['att']:+.4g} and log relative {did_log['relative_effect']:+.2%} "
            f"{'agree' if same_sign else 'disagree'} in sign",
        )
    )

    checks.append(
        check(
            "处理组覆盖", "treated-cluster coverage",
            PASS if n_treated_units >= 3 else WARN,
            f"{n_treated_units} 个处理城市（{'集群推断有效' if n_treated_units >= 3 else '集群过少，推断偏乐观'}）",
            f"{n_treated_units} treated cities ({'cluster inference sound' if n_treated_units >= 3 else 'few clusters, CIs optimistic'})",
        )
    )

    v = verdict(checks)
    return {"checks": checks, "verdict": v}


# --------------------------------------------------------------------------
# Matching / weighting (PSM/IPW/AIPW)
# --------------------------------------------------------------------------
def matching_confidence(
    psm: Dict[str, object],
    ipw: Dict[str, object],
    aipw: Dict[str, object],
) -> Dict[str, object]:
    """Checks for the matching readout — assembled from existing diagnostics."""
    checks: List[Dict[str, str]] = []

    balance = psm.get("balance") or []
    if balance:
        worst = max(b["smd_after"] for b in balance)
        checks.append(
            check(
                "匹配后协变量平衡", "post-match covariate balance",
                PASS if worst < 0.1 else (WARN if worst < 0.2 else FAIL),
                f"匹配后最大 SMD={worst:.3f}（阈值 0.1）",
                f"max post-match SMD={worst:.3f} (threshold 0.1)",
            )
        )

    support = psm.get("common_support") or {}
    if support:
        share = float(support.get("share_outside", 0.0))
        checks.append(
            check(
                "共同支撑 / 重叠性", "common support / overlap",
                PASS if share < 0.05 else (WARN if share < 0.15 else FAIL),
                f"{share:.1%} 的处理组落在共同支撑外",
                f"{share:.1%} of treated units fall outside common support",
            )
        )

    psm_att = psm.get("att")
    aipw_est = aipw.get("estimate") if isinstance(aipw, dict) else None
    if psm_att is not None and aipw_est:
        gap = abs(psm_att - aipw_est) / max(abs(aipw_est), 1e-9)
        checks.append(
            check(
                "方法一致性（PSM vs AIPW）", "estimator agreement (PSM vs AIPW)",
                PASS if gap < 0.3 else WARN,
                f"PSM ATT {psm_att:+.4g} vs AIPW ATE {aipw_est:+.4g}（差 {gap:.0%}）",
                f"PSM ATT {psm_att:+.4g} vs AIPW ATE {aipw_est:+.4g} ({gap:.0%} apart)",
            )
        )

    if psm_att is not None and psm.get("p_value") is not None:
        p = float(psm["p_value"])
        checks.append(
            check(
                "PSM 显著性", "PSM significance",
                PASS if p < _THRESHOLD else WARN,
                f"p={p:.4f}",
                f"p={p:.4f}",
            )
        )

    checks.append(
        check(
            "隐性混淆提示", "hidden-confounding caveat",
            WARN,
            "匹配只能平衡已观测协变量，需配合敏感性分析",
            "matching balances observed covariates only; pair with a sensitivity pass",
        )
    )

    v = verdict(checks)
    return {"checks": checks, "verdict": v}


# --------------------------------------------------------------------------
# Interrupted time series
# --------------------------------------------------------------------------
def ts_causal_confidence(its: Dict[str, object], impact: Dict[str, object], agree: bool) -> Dict[str, object]:
    """Checks for the time-series causal readout."""
    checks: List[Dict[str, str]] = []

    lb = float(its["ljung_box_p"])
    checks.append(
        check(
            "ITS 残差白噪声", "ITS residual whiteness",
            PASS if lb > 0.05 else WARN,
            f"Ljung-Box p={lb:.3f}（残差{'无' if lb > 0.05 else '有'}自相关）",
            f"Ljung-Box p={lb:.3f} (residuals {'white' if lb > 0.05 else 'autocorrelated'})",
        )
    )

    p1 = float(its["placebo_p"])
    p2 = float(impact["placebo_p"])
    ok = p1 > 0.10 and p2 > 0.10
    checks.append(
        check(
            "安慰剂检验（假干预日）", "placebo intervention dates",
            PASS if ok else (WARN if min(p1, p2) > 0.05 else FAIL),
            f"ITS p={p1:.3f}，结构模型 p={p2:.3f}（均应 >0.10）",
            f"ITS p={p1:.3f}, structural p={p2:.3f} (both should exceed 0.10)",
        )
    )

    checks.append(
        check(
            "双模型一致性", "two-model agreement",
            PASS if agree else WARN,
            f"ITS {its['relative']:+.2%} vs 结构模型 {impact['relative']:+.2%}",
            f"ITS {its['relative']:+.2%} vs structural {impact['relative']:+.2%}",
        )
    )

    r2 = float(impact.get("pre_fit_r2", 0.0))
    checks.append(
        check(
            "反事实拟合质量", "counterfactual fit quality",
            PASS if r2 >= 0.7 else WARN,
            f"干预前一步拟合 R²={r2:.3f}",
            f"pre-period one-step fit R2={r2:.3f}",
        )
    )

    spec = its.get("spec_table")
    if isinstance(spec, pd.DataFrame) and len(spec) and {"relative", "p_value"} <= set(spec.columns):
        same = bool((spec["relative"] > 0).all() or (spec["relative"] < 0).all())
        sig_all = bool((spec["p_value"] < _THRESHOLD).all())
        checks.append(
            check(
                "口径曲线稳健", "specification-curve robustness",
                PASS if same and sig_all else WARN,
                f"{'全部同号且显著' if same and sig_all else '存在异号或不显著的口径'}（{len(spec)} 个设定）",
                f"{'all same sign and significant' if same and sig_all else 'mixed signs or insignificant specs'} ({len(spec)} specs)",
            )
        )

    v = verdict(checks)
    return {"checks": checks, "verdict": v}
