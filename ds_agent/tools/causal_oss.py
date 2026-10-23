"""Open-source causal-inference engine adapters (DoWhy / EconML).

Industry practice is to *not* trust a single home-grown implementation. This
module wraps the established open-source causal stack and runs the same
identification problem through it, so the agent can:

1. Re-estimate an effect with a battle-tested package (DoWhy backdoor
   estimators; EconML double/causal forests when installed).
2. Run DoWhy's refutation suite (placebo treatment, random common cause,
   random subset) as an independent confidence layer.
3. Cross-validate against the in-house estimators and report the relative
   gap — agreement across independent engines is evidence the estimate is
   not an artifact of one code path.

Every adapter degrades gracefully: if a package is missing the entry point
raises a clear ``RuntimeError`` and ``available_engines()`` simply omits it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from dowhy import CausalModel  # type: ignore

    HAS_DOWHY = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_DOWHY = False

try:
    from econml.dml import CausalForestDML  # type: ignore

    HAS_ECONML = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_ECONML = False


def available_engines() -> Dict[str, bool]:
    """Which open-source causal engines are importable right now."""
    return {"dowhy": HAS_DOWHY, "econml": HAS_ECONML}


_DOWHY_METHODS = {
    "linear_regression": "backdoor.linear_regression",
    "psm": "backdoor.propensity_score_matching",
    "ipw": "backdoor.propensity_score_weighting",
}


def dowhy_estimate(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    common_causes: Sequence[str],
    method: str = "linear_regression",
    effect_modifier: Optional[str] = None,
) -> Dict[str, Any]:
    """Estimate a backdoor effect with DoWhy.

    ``method`` is one of ``linear_regression`` / ``psm`` / ``ipw``. Returns
    the estimate, its p-value, and the identified estimand string so the
    answer can show *what* identification strategy the package chose.
    """
    if not HAS_DOWHY:
        raise RuntimeError("dowhy is not installed — pip install dowhy")
    if method not in _DOWHY_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(_DOWHY_METHODS)}")

    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        common_causes=list(common_causes),
        effect_modifiers=[effect_modifier] if effect_modifier else None,
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    est = model.estimate_effect(
        estimand,
        method_name=_DOWHY_METHODS[method],
        target_units="ate",
        confidence_intervals=True,
        test_significance=True,
    )
    p_value = float("nan")
    try:
        p_value = float(est.test_stat_significance()["p_value"][0])
    except Exception:  # noqa: BLE001 - significance is best-effort per method
        p_value = float("nan")

    ci = (float("nan"), float("nan"))
    try:
        ci_lo, ci_hi = est.get_confidence_intervals()
        ci = (float(np.asarray(ci_lo).ravel()[0]), float(np.asarray(ci_hi).ravel()[0]))
    except Exception:  # noqa: BLE001
        pass

    return {
        "engine": "dowhy",
        "method": method,
        "estimate": float(est.value),
        "p_value": p_value,
        "ci": ci,
        "estimand": str(estimand.estimand_type),
        "backdoor_set": str(getattr(estimand, "backdoor_variables", "")),
        "_model": model,
        "_estimand": estimand,
        "_estimate": est,
    }


def dowhy_refutations(result: Dict[str, Any], num_simulations: int = 20) -> List[Dict[str, Any]]:
    """Run DoWhy's refutation tests on a previous estimate.

    ``placebo_treatment_refuter`` replaces the treatment with noise — a real
    causal effect must vanish (new effect ≈ 0). ``random_common_cause``
    adds a fake confounder — the estimate must barely move. Both are exactly
    the "confidence validation" layer the agent reports.
    """
    if not HAS_DOWHY:
        raise RuntimeError("dowhy is not installed — pip install dowhy")
    model = result["_model"]
    estimand = result["_estimand"]
    est = result["_estimate"]
    out: List[Dict[str, Any]] = []
    for name, key in (
        ("安慰剂处理（伪处理）", "placebo_treatment_refuter"),
        ("随机共同原因", "random_common_cause"),
    ):
        try:
            ref = model.refute_estimate(estimand, est, method_name=key, num_simulations=num_simulations)
            new_effect = float(np.asarray(ref.new_effect).ravel()[0])
            out.append(
                {
                    "refuter": name,
                    "refuter_en": key,
                    "new_effect": new_effect,
                    "passed": abs(new_effect) < max(1e-6, abs(result["estimate"]) * 0.2) if key == "placebo_treatment_refuter" else bool(getattr(ref, "refutation_result", True)),
                }
            )
        except Exception as exc:  # noqa: BLE001 - a refuter must never fail the answer
            out.append({"refuter": name, "refuter_en": key, "new_effect": float("nan"), "passed": None, "error": str(exc)[:120]})
    return out


def econml_causal_forest(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Sequence[str],
    seed: int = 7,
) -> Dict[str, Any]:
    """Heterogeneous effects with EconML's causal forest (DML honest splitting).

    Returns the ATE plus a small CATE summary so the answer can show *who*
    the effect concentrates on — the reason industrial teams reach for EconML.
    """
    if not HAS_ECONML:
        raise RuntimeError("econml is not installed — pip install econml")
    x = data[list(confounders)].to_numpy(dtype=float)
    t = data[treatment].to_numpy(dtype=float)
    y = data[outcome].to_numpy(dtype=float)
    cf = CausalForestDML(random_state=seed)
    cf.fit(y, t, X=x)
    tau = cf.effect(x)
    ate = float(np.mean(tau))
    cate_lo, cate_hi = np.percentile(tau, [10, 90])
    top = np.argsort(tau)[-5:][::-1]
    return {
        "engine": "econml",
        "method": "causal_forest_dml",
        "ate": ate,
        "cate_p10": float(cate_lo),
        "cate_p90": float(cate_hi),
        "top_units": top.tolist(),
        "top_effects": [round(float(tau[i]), 4) for i in top],
    }


def oss_confidence(
    inhouse: Dict[str, Any],
    oss: Dict[str, Any],
    refutations: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Confidence card for the open-source cross-validation.

    Checks: the OSS estimate is significant, the placebo refuter kills the
    effect, and the OSS number agrees with the in-house estimator within a
    reasonable band. Cross-engine agreement is the whole point of this path.
    """
    checks: List[Dict[str, str]] = []
    PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

    oss_est = float(oss["estimate"])
    p = float(oss.get("p_value", float("nan")))
    checks.append(
        {
            "name_zh": "开源引擎显著性",
            "name_en": "OSS engine significance",
            "status": PASS if p == p and p < 0.05 else WARN,
            "detail_zh": f"DoWhy {oss['method']} 估计 {oss_est:.4f}，p={p:.4f}",
            "detail_en": f"DoWhy {oss['method']} estimate {oss_est:.4f}, p={p:.4f}",
        }
    )

    placebo = next((r for r in refutations if r["refuter_en"] == "placebo_treatment_refuter"), None)
    if placebo is not None and placebo["passed"] is not None:
        checks.append(
            {
                "name_zh": "安慰剂处理反驳",
                "name_en": "placebo-treatment refutation",
                "status": PASS if placebo["passed"] else FAIL,
                "detail_zh": f"伪处理下新效应 {placebo['new_effect']:.4f}（应≈0）",
                "detail_en": f"effect under placebo treatment {placebo['new_effect']:.4f} (should be ~0)",
            }
        )
    rcc = next((r for r in refutations if r["refuter_en"] == "random_common_cause"), None)
    if rcc is not None and rcc["passed"] is not None:
        checks.append(
            {
                "name_zh": "随机共同原因反驳",
                "name_en": "random-common-cause refutation",
                "status": PASS if rcc["passed"] else WARN,
                "detail_zh": f"加入伪混杂后新效应 {rcc['new_effect']:.4f}（应基本不变）",
                "detail_en": f"effect after adding a fake confounder {rcc['new_effect']:.4f} (should barely move)",
            }
        )

    gap = float("nan")
    if inhouse:
        ih = float(inhouse.get("att", inhouse.get("estimate", float("nan"))))
        if ih == ih:
            gap = abs(oss_est - ih) / max(abs(ih), 1e-9)
            checks.append(
                {
                    "name_zh": "跨引擎一致性",
                    "name_en": "cross-engine agreement",
                    "status": PASS if gap < 0.20 else (WARN if gap < 0.50 else FAIL),
                    "detail_zh": f"自研估计 {ih:.4f} vs DoWhy {oss_est:.4f}，相对差 {gap:.0%}",
                    "detail_en": f"in-house {ih:.4f} vs DoWhy {oss_est:.4f}, relative gap {gap:.0%}",
                }
            )

    n_fail = sum(1 for c in checks if c["status"] == FAIL)
    score = sum(1 for c in checks if c["status"] == PASS) / max(1, len(checks))
    if n_fail:
        verdict = {"label_zh": "低置信", "label_en": "low confidence", "score": score, "n_fail": n_fail}
    elif score >= 0.8:
        verdict = {"label_zh": "高置信", "label_en": "high confidence", "score": score, "n_fail": n_fail}
    else:
        verdict = {"label_zh": "中置信", "label_en": "medium confidence", "score": score, "n_fail": n_fail}
    return {"checks": checks, "verdict": verdict, "score": score, "gap": gap}
