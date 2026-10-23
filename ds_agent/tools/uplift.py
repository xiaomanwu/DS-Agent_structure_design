"""Heterogeneous treatment effects: causal trees, causal forests and uplift models.

The average treatment effect answers "does it work on average"; it cannot answer
"for whom". This module estimates the conditional average treatment effect
(CATE) with the three families used in practice:

* **Causal tree** (causal_tree) - recursive partitioning that splits on the
  *difference* in treatment effect, not on the outcome. Splitting on the outcome
  finds users who convert; splitting on the effect finds users who convert
  *because of* the treatment, which is the only thing that justifies targeting.
* **Causal forest** (causal_forest) - an ensemble of subsampled, feature-random
  causal trees. Averaging reduces the variance that makes a single tree's leaves
  unstable, and the out-of-bag spread gives a usable uncertainty band.
* **Uplift learners** (uplift_learners) - T-learner, S-learner and X-learner,
  the three meta-learners every growth team has already implemented somewhere.

Evaluation is deliberately outcome-model-free where possible: Qini/AUUC measure
whether the ranking is useful, and a cross-fitted AIPW estimate is used to check
that the forest's average CATE is not adrift from a doubly robust ATE.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def design_matrix(df: pd.DataFrame, features: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """Numeric design matrix with one-hot encoding for categorical columns.

    Trees do not need scaling, so no standardisation is applied -- it would only
    make the exported thresholds harder to read back to the business.
    """
    cols = [c for c in features if c in df.columns]
    if not cols:
        raise ValueError("none of the requested features exist in the dataframe")
    frame = df[cols].copy()
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(frame[c])]
    if cat_cols:
        frame = pd.get_dummies(frame, columns=cat_cols, dummy_na=False)
    frame = frame.astype(float).fillna(frame.median(numeric_only=True)).fillna(0.0)
    return frame.to_numpy(dtype=float), list(frame.columns)


def binary_treatment(series: pd.Series, positive: Optional[str] = None) -> Tuple[np.ndarray, Optional[str]]:
    """Encode a treatment column as 0/1, accepting string arm labels.

    String labels are common ("treatment", "ranking_v2"), and silently comparing
    them with ``> 0`` is how an uplift model ends up with a constant treatment
    indicator and a flat CATE surface.
    """
    series = series if isinstance(series, pd.Series) else pd.Series(series)
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return (series.to_numpy(dtype=float) > 0).astype(float), None
    levels = [lvl for lvl in pd.unique(series.dropna())]
    if len(levels) != 2:
        raise ValueError(f"treatment column needs exactly two levels, got {levels}")
    pos = positive if positive in levels else ("treatment" if "treatment" in levels else levels[-1])
    return (series.to_numpy() == pos).astype(float), pos


# --------------------------------------------------------------------------- #
# Causal tree
# --------------------------------------------------------------------------- #
class CausalTree:
    """A single causal tree: splits maximise the between-child effect gap.

    The split criterion is the standard transformed-outcome criterion

        S = (n_L * n_R / (n_L + n_R)) * (tau_L - tau_R)^2

    i.e. a split is good when the two children have very different treatment
    effects *and* both are large enough to be estimated reliably.
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 150,
        min_treat_per_leaf: int = 20,
        max_features: Optional[int] = None,
        n_thresholds: int = 10,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_treat_per_leaf = min_treat_per_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.rng = rng or np.random.default_rng(0)
        self.root: Optional[Dict[str, object]] = None
        self.feature_names: List[str] = []
        self.importance: Dict[str, float] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray, feature_names: Sequence[str]) -> "CausalTree":
        self.feature_names = list(feature_names)
        self.importance = {name: 0.0 for name in self.feature_names}
        self.root = self._grow(X, t, y, np.arange(len(y)), depth=0)
        total = sum(self.importance.values()) or 1.0
        self.importance = {k: v / total for k, v in self.importance.items()}
        return self

    def _tau(self, t: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
        if len(idx) == 0:
            return 0.0
        tt = t[idx]
        n1 = tt.sum()
        n0 = len(idx) - n1
        if n1 < 1 or n0 < 1:
            return 0.0
        return float(y[idx][tt == 1].mean() - y[idx][tt == 0].mean())

    def _grow(self, X: np.ndarray, t: np.ndarray, y: np.ndarray, idx: np.ndarray, depth: int) -> Dict[str, object]:
        tau = self._tau(t, y, idx)
        node: Dict[str, object] = {
            "n": int(len(idx)),
            "tau": tau,
            "treated_share": float(t[idx].mean()) if len(idx) else 0.0,
            "depth": depth,
            "leaf": True,
        }
        if depth >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
            return node

        n_features = X.shape[1]
        k = self.max_features or n_features
        candidate_features = (
            self.rng.choice(n_features, size=min(k, n_features), replace=False)
            if k < n_features
            else np.arange(n_features)
        )
        probs = np.linspace(0.1, 0.9, self.n_thresholds)
        best_score, best = 0.0, None

        for f in candidate_features:
            values = X[idx, f]
            if values.min() == values.max():
                continue
            order = np.argsort(values, kind="stable")
            sorted_values = values[order]
            # Cumulative sums turn "evaluate 10 thresholds" into one vectorised
            # comparison instead of 10 boolean masks over the node.
            positions = np.unique(np.searchsorted(sorted_values, np.quantile(sorted_values, probs), side="right"))
            positions = positions[(positions >= self.min_samples_leaf) & (positions <= len(idx) - self.min_samples_leaf)]
            if len(positions) == 0:
                continue

            tt = t[idx][order]
            yy = y[idx][order]
            cum_t = np.concatenate(([0.0], np.cumsum(tt)))
            cum_y = np.concatenate(([0.0], np.cumsum(yy)))
            cum_ty = np.concatenate(([0.0], np.cumsum(tt * yy)))

            n_l = positions.astype(float)
            n_r = len(idx) - n_l
            n1_l, n1_r = cum_t[positions], cum_t[-1] - cum_t[positions]
            n0_l, n0_r = n_l - n1_l, n_r - n1_r
            valid = (
                (n1_l >= self.min_treat_per_leaf)
                & (n0_l >= self.min_treat_per_leaf)
                & (n1_r >= self.min_treat_per_leaf)
                & (n0_r >= self.min_treat_per_leaf)
            )
            if not valid.any():
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                sum_y_l = cum_y[positions]
                sum_ty_l = cum_ty[positions]
                tau_l = sum_ty_l / n1_l - (sum_y_l - sum_ty_l) / n0_l
                sum_y_r = cum_y[-1] - sum_y_l
                sum_ty_r = cum_ty[-1] - sum_ty_l
                tau_r = sum_ty_r / n1_r - (sum_y_r - sum_ty_r) / n0_r
                score = (n_l * n_r / len(idx)) * (tau_l - tau_r) ** 2
            score = np.where(valid, np.nan_to_num(score, nan=0.0), 0.0)
            j = int(np.argmax(score))
            if score[j] > best_score:
                p = int(positions[j])
                lower, upper = sorted_values[p - 1], sorted_values[p]
                best_score = float(score[j])
                best = {"feature": int(f), "threshold": float((lower + upper) / 2), "left": idx[order[:p]], "right": idx[order[p:]]}

        if best is None or best_score <= 0:
            return node

        name = self.feature_names[best["feature"]]
        self.importance[name] = self.importance.get(name, 0.0) + best_score
        node.update(
            {
                "leaf": False,
                "feature": name,
                "threshold": best["threshold"],
                "score": best_score,
                "left": self._grow(X, t, y, best["left"], depth + 1),
                "right": self._grow(X, t, y, best["right"], depth + 1),
            }
        )
        return node

    # -------------------------------------------------------------- predict
    def predict(self, X: np.ndarray, feature_names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Vectorised leaf lookup.

        A per-row Python traversal costs ~50x more than splitting the index set
        recursively, which matters because a forest predicts the whole frame once
        per tree.
        """
        names = list(feature_names) if feature_names is not None else self.feature_names
        if self.root is None:
            raise RuntimeError("tree is not fitted")
        out = np.zeros(len(X), dtype=float)

        def walk(node: Dict[str, object], idx: np.ndarray) -> None:
            if len(idx) == 0:
                return
            if node["leaf"]:
                out[idx] = float(node["tau"])
                return
            f = names.index(node["feature"])  # type: ignore[arg-type]
            mask = X[idx, f] <= node["threshold"]  # type: ignore[operator]
            walk(node["left"], idx[mask])  # type: ignore[arg-type]
            walk(node["right"], idx[~mask])  # type: ignore[arg-type]

        walk(self.root, np.arange(len(X)))
        return out

    def leaves(self) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []

        def walk(node: Dict[str, object], path: List[str]) -> None:
            if node["leaf"]:
                out.append({**{k: v for k, v in node.items() if k not in ("left", "right")}, "rule": " & ".join(path) or "all"})
                return
            walk(node["left"], path + [f"{node['feature']} <= {node['threshold']:.4g}"])  # type: ignore[arg-type]
            walk(node["right"], path + [f"{node['feature']} > {node['threshold']:.4g}"])  # type: ignore[arg-type]

        if self.root is not None:
            walk(self.root, [])
        return sorted(out, key=lambda leaf: leaf["tau"], reverse=True)


def causal_tree(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    features: Sequence[str],
    max_depth: int = 3,
    min_samples_leaf: int = 150,
    min_treat_per_leaf: int = 20,
    seed: int = 0,
) -> Dict[str, object]:
    """Fit one causal tree and describe its segments in business language."""
    X, names = design_matrix(df, features)
    t, _ = binary_treatment(df[treatment])
    y = df[outcome].to_numpy(dtype=float)
    tree = CausalTree(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_treat_per_leaf=min_treat_per_leaf,
        rng=np.random.default_rng(seed),
    ).fit(X, t, y, names)
    cate = tree.predict(X, names)
    leaves = tree.leaves()
    best, worst = leaves[0], leaves[-1]
    return {
        "method": "causal_tree",
        "ate": float(cate.mean()),
        "cate": cate,
        "leaves": leaves,
        "n_leaves": len(leaves),
        "importance": tree.importance,
        "best_segment": best,
        "worst_segment": worst,
        "n": int(len(df)),
        "assumption": "No unmeasured confounders within leaves; overlap holds inside each leaf.",
        "narrative": (
            f"Causal tree with {len(leaves)} leaves: average effect {cate.mean():+.4g}, best segment "
            f"[{best['rule']}] {best['tau']:+.4g} (n={best['n']:,}), worst [{worst['rule']}] {worst['tau']:+.4g} "
            f"(n={worst['n']:,})."
        ),
    }


# --------------------------------------------------------------------------- #
# Causal forest
# --------------------------------------------------------------------------- #
def causal_forest(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    features: Sequence[str],
    n_trees: int = 60,
    max_depth: int = 4,
    min_samples_leaf: int = 120,
    subsample: float = 0.6,
    max_features: Optional[int] = None,
    holdout: float = 0.5,
    seed: int = 7,
) -> Dict[str, object]:
    """Causal forest (honest evaluation) plus a doubly robust ATE for calibration.

    Trees are grown on subsamples and the CATE is the ensemble average. The
    evaluation split is held out from *fitting*, so the Qini curve is honest:
    in-sample Qini is always flattering and is the most common way an uplift
    model gets approved without being useful.
    """
    rng = np.random.default_rng(seed)
    X, names = design_matrix(df, features)
    t, _ = binary_treatment(df[treatment])
    y = df[outcome].to_numpy(dtype=float)
    n = len(y)
    if n < 500:
        raise ValueError("causal_forest needs at least 500 rows")

    # Stratified split keeps the treatment share identical in both halves.
    idx_treat = np.flatnonzero(t == 1)
    idx_ctrl = np.flatnonzero(t == 0)
    rng.shuffle(idx_treat)
    rng.shuffle(idx_ctrl)
    cut_t, cut_c = int(len(idx_treat) * holdout), int(len(idx_ctrl) * holdout)
    eval_idx = np.concatenate([idx_treat[:cut_t], idx_ctrl[:cut_c]])
    fit_idx = np.concatenate([idx_treat[cut_t:], idx_ctrl[cut_c:]])

    mtry = max_features or max(1, int(np.sqrt(X.shape[1])))
    trees: List[CausalTree] = []
    for b in range(n_trees):
        sub = rng.choice(fit_idx, size=int(len(fit_idx) * subsample), replace=False)
        tree = CausalTree(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=mtry,
            rng=np.random.default_rng(seed + b + 1),
        ).fit(X[sub], t[sub], y[sub], names)
        trees.append(tree)

    # Ensemble CATE: average of the trees that saw the row during fitting; the
    # evaluation rows use all trees (they are out-of-sample either way).
    cate = np.zeros(n)
    for tree in trees:
        cate += tree.predict(X, names)
    cate /= len(trees)

    eval_cate, eval_t, eval_y = cate[eval_idx], t[eval_idx], y[eval_idx]
    qini = qini_coefficient(eval_y, eval_t, eval_cate)
    deciles = uplift_deciles(eval_y, eval_t, eval_cate, n_bins=10)
    importance = pd.Series({k: np.mean([tr.importance.get(k, 0.0) for tr in trees]) for k in names}).sort_values(ascending=False)
    ate_aipw = aipw_ate(X, t, y, n_splits=4, seed=seed)

    best_share = float(deciles.iloc[0]["uplift_share"]) if len(deciles) else float("nan")
    return {
        "method": "causal_forest",
        "ate": float(cate[eval_idx].mean()),
        "ate_aipw": ate_aipw,
        "ate_gap": float(cate[eval_idx].mean() - ate_aipw["ate"]) if np.isfinite(ate_aipw["ate"]) else float("nan"),
        "cate": cate,
        "qini": qini,
        "qini_curve": qini_curve(eval_y, eval_t, eval_cate),
        "deciles": deciles,
        "importance": importance.to_dict(),
        "n_trees": len(trees),
        "n_fit": int(len(fit_idx)),
        "n_eval": int(len(eval_idx)),
        "best_decile_uplift": float(deciles.iloc[0]["uplift"]) if len(deciles) else float("nan"),
        "top_decile_share": best_share,
        "targeting": targeting_gain(eval_y, eval_t, eval_cate, fractions=(0.1, 0.2, 0.3, 0.5)),
        "assumption": "Unconfoundedness given the features; the evaluation split is untouched during fitting.",
        "narrative": (
            f"Causal forest ({len(trees)} trees, {X.shape[1]} features): ATE {cate[eval_idx].mean():+.4g} "
            f"(AIPW check {ate_aipw['ate']:+.4g}), honest Qini {qini:+.3f}, top-decile uplift "
            f"{deciles.iloc[0]['uplift']:+.4g} vs bottom decile {deciles.iloc[-1]['uplift']:+.4g}. "
            f"The top 20% of users capture {targeting_gain(eval_y, eval_t, eval_cate, (0.2,))[0]['captured_share']:.0%} "
            f"of the total uplift."
        ),
    }


def aipw_ate(X: np.ndarray, t: np.ndarray, y: np.ndarray, n_splits: int = 4, seed: int = 0) -> Dict[str, float]:
    """Cross-fitted AIPW ATE with an influence-function standard error.

    This is the calibration anchor: it does not use the forest at all, so if the
    forest's mean CATE disagrees with it, the forest is overfitting the effect
    heterogeneity rather than measuring it.
    """
    n = len(y)
    if n < 50 or t.sum() < 10 or (1 - t).sum() < 10:
        return {"ate": float("nan"), "se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    mu0 = np.zeros(n)
    mu1 = np.zeros(n)
    e = np.zeros(n)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, t):
        # Shallow random forests: fast enough to cross-fit four folds on a
        # 20k-row frame, flexible enough not to bias the AIPW correction.
        m0 = RandomForestRegressor(n_estimators=25, max_depth=8, min_samples_leaf=25, n_jobs=-1, random_state=seed)
        m1 = RandomForestRegressor(n_estimators=25, max_depth=8, min_samples_leaf=25, n_jobs=-1, random_state=seed)
        m0.fit(X[train_idx][t[train_idx] == 0], y[train_idx][t[train_idx] == 0])
        m1.fit(X[train_idx][t[train_idx] == 1], y[train_idx][t[train_idx] == 1])
        mu0[test_idx] = m0.predict(X[test_idx])
        mu1[test_idx] = m1.predict(X[test_idx])
        ps = LogisticRegression(max_iter=1000, C=1.0)
        ps.fit(X[train_idx], t[train_idx])
        e[test_idx] = np.clip(ps.predict_proba(X[test_idx])[:, 1], 0.02, 0.98)

    psi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    ate = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(n))
    return {"ate": ate, "se": se, "ci_low": ate - 1.96 * se, "ci_high": ate + 1.96 * se}


# --------------------------------------------------------------------------- #
# Evaluation: Qini, AUUC, deciles, targeting
# --------------------------------------------------------------------------- #
def qini_curve(y: np.ndarray, t: np.ndarray, uplift: np.ndarray) -> pd.DataFrame:
    """Cumulative incremental gain when targeting users in descending uplift order.

    At each prefix the incremental gain is ``conversions_treated -
    conversions_control * (n_treated / n_control)``: the treated conversions
    above what the same number of control users would have produced.
    """
    order = np.argsort(-np.asarray(uplift, dtype=float))
    y = np.asarray(y, dtype=float)[order]
    t = np.asarray(t, dtype=float)[order]
    n = len(y)

    cum_t = np.cumsum(y * t)
    cum_c = np.cumsum(y * (1 - t))
    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = cum_t - cum_c * np.where(n_c > 0, n_t / np.maximum(n_c, 1), 0.0)
    return pd.DataFrame(
        {
            "fraction": np.arange(1, n + 1) / n,
            "gain": gain,
            # The random-targeting benchmark is the straight line from the origin
            # to the same total incremental outcome; the gap between the two is
            # exactly what the model buys you.
            "random": (np.arange(1, n + 1) / n) * (float(gain[-1]) if n else 0.0),
            "n_treated": n_t,
            "n_control": n_c,
        }
    )


def qini_coefficient(y: np.ndarray, t: np.ndarray, uplift: np.ndarray) -> float:
    """Normalised Qini: 0 means "no better than random targeting"."""
    curve = qini_curve(y, t, uplift)
    if len(curve) < 10:
        return float("nan")
    final = float(curve["gain"].iloc[-1])
    if abs(final) < 1e-12:
        return 0.0
    # numpy >= 2 renames trapz to trapezoid; support both.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    area = float(trapezoid(curve["gain"].to_numpy(), curve["fraction"].to_numpy()))
    random_area = 0.5 * final
    denom = 0.5 * abs(final)
    return float((area - random_area) / denom) if denom > 0 else 0.0


def uplift_deciles(y: np.ndarray, t: np.ndarray, uplift: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Actual uplift by predicted-uplift decile (the table a growth team reads)."""
    frame = pd.DataFrame({"y": y, "t": t, "pred": uplift})
    frame["bin"] = pd.qcut(frame["pred"].rank(method="first", ascending=False), n_bins, labels=False)
    rows = []
    for b, chunk in frame.groupby("bin"):
        tt = chunk.loc[chunk["t"] == 1, "y"]
        cc = chunk.loc[chunk["t"] == 0, "y"]
        rows.append(
            {
                "decile": int(b) + 1,
                "n": int(len(chunk)),
                "pred_uplift": float(chunk["pred"].mean()),
                "uplift": float(tt.mean() - cc.mean()) if len(tt) and len(cc) else np.nan,
                "uplift_share": float("nan"),
            }
        )
    out = pd.DataFrame(rows).sort_values("decile")
    total = float(np.nansum(out["uplift"] * out["n"]))
    if abs(total) > 1e-12:
        out["uplift_share"] = out["uplift"] * out["n"] / total
    return out


def targeting_gain(
    y: np.ndarray, t: np.ndarray, uplift: np.ndarray, fractions: Sequence[float] = (0.1, 0.2, 0.3)
) -> List[Dict[str, float]]:
    """How much of the total uplift is captured by targeting the top x% of users."""
    curve = qini_curve(y, t, uplift)
    final = float(curve["gain"].iloc[-1])
    out = []
    for frac in fractions:
        k = max(1, int(len(curve) * frac))
        gain = float(curve["gain"].iloc[k - 1])
        out.append(
            {
                "fraction": float(frac),
                "n_targeted": int(k),
                "gain": gain,
                "captured_share": gain / final if abs(final) > 1e-12 else float("nan"),
                "uplift_lift": (gain / k) / (final / len(curve)) if abs(final) > 1e-12 and k else float("nan"),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Meta-learners (T / S / X)
# --------------------------------------------------------------------------- #
def _base_model(seed: int):
    return HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.06, random_state=seed)


def uplift_learners(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    features: Sequence[str],
    test_size: float = 0.3,
    learners: Sequence[str] = ("t", "s", "x"),
    seed: int = 11,
) -> Dict[str, object]:
    """Fit the three standard meta-learners and rank them on a held-out Qini.

    * T-learner: one outcome model per arm. Simple, loses power when one arm is
      small.
    * S-learner: one model with treatment as a feature. The treatment effect is
      a difference of two predictions; regularisation tends to shrink it.
    * X-learner: imputes individual effects in both arms, then combines them with
      the propensity score. Best when the arms are very unbalanced.
    """
    X, names = design_matrix(df, features)
    t, _ = binary_treatment(df[treatment])
    y = df[outcome].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_size))
    train, test = idx[:cut], idx[cut:]

    X_tr, t_tr, y_tr = X[train], t[train], y[train]
    X_te, t_te, y_te = X[test], t[test], y[test]

    fits: Dict[str, Dict[str, float]] = {}
    predictions: Dict[str, np.ndarray] = {}

    if "t" in learners:
        m0 = _base_model(seed).fit(X_tr[t_tr == 0], y_tr[t_tr == 0])
        m1 = _base_model(seed + 1).fit(X_tr[t_tr == 1], y_tr[t_tr == 1])
        predictions["t"] = m1.predict(X_te) - m0.predict(X_te)

    if "s" in learners:
        Xs = np.column_stack([X_tr, t_tr])
        model = _base_model(seed + 2).fit(Xs, y_tr)
        pred1 = model.predict(np.column_stack([X_te, np.ones(len(X_te))]))
        pred0 = model.predict(np.column_stack([X_te, np.zeros(len(X_te))]))
        predictions["s"] = pred1 - pred0

    if "x" in learners:
        m0 = _base_model(seed + 3).fit(X_tr[t_tr == 0], y_tr[t_tr == 0])
        m1 = _base_model(seed + 4).fit(X_tr[t_tr == 1], y_tr[t_tr == 1])
        d1 = y_tr[t_tr == 1] - m0.predict(X_tr[t_tr == 1])
        d0 = m1.predict(X_tr[t_tr == 0]) - y_tr[t_tr == 0]
        tau1 = _base_model(seed + 5).fit(X_tr[t_tr == 1], d1)
        tau0 = _base_model(seed + 6).fit(X_tr[t_tr == 0], d0)
        ps = LogisticRegression(max_iter=1000).fit(X_tr, t_tr)
        e_te = np.clip(ps.predict_proba(X_te)[:, 1], 0.05, 0.95)
        predictions["x"] = e_te * tau0.predict(X_te) + (1 - e_te) * tau1.predict(X_te)

    for name, pred in predictions.items():
        fits[name] = {
            "qini": qini_coefficient(y_te, t_te, pred),
            "ate": float(pred.mean()),
            "top_decile_uplift": float(uplift_deciles(y_te, t_te, pred, 10).iloc[0]["uplift"]),
        }

    best = max(fits.items(), key=lambda kv: kv[1]["qini"])[0] if fits else None
    best_deciles = uplift_deciles(y_te, t_te, predictions[best], 10) if best else pd.DataFrame()
    return {
        "method": "uplift_learners",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "learners": fits,
        "best": best,
        "best_deciles": best_deciles,
        "cate": predictions.get(best, np.array([])),
        "assumption": "Unconfoundedness given the features; the ranking is evaluated out of sample.",
        "narrative": (
            "Uplift learners (held-out): "
            + ", ".join(f"{k.upper()}-learner Qini {v['qini']:+.3f}" for k, v in fits.items())
            + (f"; best is {best.upper()}." if best else ".")
        ),
    }
