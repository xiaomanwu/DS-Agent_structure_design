"""Deep-learning style anomaly detection.

Industry teams increasingly detect anomalies with learned representations
rather than hand-built seasonal baselines. This module ships the two most
common recipes that run *without* a GPU cluster:

- ``autoencoder_scores``: a bottleneck autoencoder (sklearn ``MLPRegressor``
  with an ``n -> h -> d -> h -> n`` layout) trained to reconstruct the recent
  window of a metric. Reconstruction error is the anomaly score — the model
  learns "what normal looks like" and flags anything it cannot rebuild.
- ``isolation_forest_scores``: sklearn ``IsolationForest`` over engineered
  per-day features (level, day-over-day, week-over-week, z vs trailing
  median). The standard multivariate detector for tabular monitoring data.

If PyTorch is available an LSTM autoencoder path is used instead (sequence
aware); otherwise the MLP autoencoder is the deterministic fallback. Both
expose the same API, so callers never branch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:  # optional deep-learning backend
    import torch  # type: ignore

    HAS_TORCH = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_TORCH = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def _window_features(values: np.ndarray, window: int = 7) -> tuple:
    """Sliding-window feature matrix: each row is one window of ``window`` days.

    The autoencoder reconstructs whole windows, so a change in *shape*
    (level shift, seasonal break) is as visible as a single-day spike.
    """
    n = len(values)
    if n <= window:
        return np.empty((0, window)), np.arange(0)
    rows = [values[i - window : i] for i in range(window, n + 1)]
    return np.asarray(rows, dtype=float), np.arange(window, n + 1)


def autoencoder_scores(
    values: Sequence[float],
    window: int = 7,
    hidden: int = 8,
    bottleneck: int = 2,
    seed: int = 7,
    quiet_band: float = 0.05,
    epochs: int = 250,
) -> Dict[str, Any]:
    """Reconstruction-error anomaly scores from an autoencoder.

    Industry recipe: the AE is trained only on *quiet* windows (within
    ``quiet_band`` of the trailing median), so it learns what normal looks
    like instead of memorising incidents. Every window is then scored and
    the cut is the 99th percentile of the quiet-window error distribution
    (with a robust MAD floor).

    Returns a frame-aligned dict: ``idx`` (last index of each window),
    ``actual``/``reconstruction``, ``error`` (per-window MSE),
    ``threshold`` and the ``flagged`` mask.
    """
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required for the autoencoder detector")
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < max(10, 4 * window) + window:
        raise ValueError(f"need at least {max(10, 4 * window) + window} points, got {n}")

    # Stationary input: ratios to the trailing 28-day median. Raw levels make
    # the reconstruction error dominated by trend/weekday composition, which
    # is exactly the noise a monitoring AE should NOT have to model. In ratio
    # space a normal day is ~1.0 and the AE only has to learn the weekly
    # shape of deviations.
    trailing = pd.Series(x).rolling(28, min_periods=7).median().to_numpy()
    denom = np.where(trailing > 0, trailing, np.nan)
    ratio = x / denom
    feats, idx = _window_features(ratio, window)
    valid = ~np.isnan(feats).any(axis=1)
    feats = feats[valid]
    idx = idx[valid]
    if feats.shape[0] < max(10, 4 * window):
        raise ValueError(f"not enough complete windows after the trailing-median warmup: {feats.shape[0]}")

    # Quiet-window prefilter: a window is quiet when every day sits within
    # ``quiet_band`` of the trailing median. Training on quiet windows only is
    # what keeps the AE from learning the anomaly itself.
    day_quiet = np.abs(ratio - 1.0) < quiet_band
    win_quiet = np.array([bool(day_quiet[end - window : end].all()) for end in idx])
    if win_quiet.sum() < max(20, 2 * window):
        win_quiet = np.ones(len(idx), dtype=bool)  # not enough quiet data: fall back to all

    train = feats[win_quiet]
    model: Any = None
    if HAS_TORCH:
        enc, mid, up, dec = _lstm_autoencoder(
            train, hidden=hidden, bottleneck=bottleneck, seed=seed, epochs=epochs
        )
        backend = "lstm_autoencoder(torch)"
        # Same layout as training: each window is a length-d sequence of
        # scalars, so the LSTM sees intra-window order.
        seq = torch.tensor(feats, dtype=torch.float32).unsqueeze(-1)  # (n, d, 1)
        with torch.no_grad():
            out, _ = enc(seq)
            rec_seq, _ = dec(torch.relu(up(mid(out))))
            rec = rec_seq.squeeze(-1).numpy()  # (n, d) reconstructed windows
    else:
        model = _mlp_autoencoder(train, hidden=hidden, bottleneck=bottleneck, seed=seed)
        backend = "mlp_autoencoder(sklearn)"
        scaler = model._scaler
        rec = scaler.inverse_transform(model.predict(scaler.transform(feats)))

    error = np.mean((feats - rec) ** 2, axis=1)
    quiet_err = error[win_quiet]
    med = float(np.median(quiet_err))
    mad = float(np.median(np.abs(quiet_err - med))) * 1.4826
    mad_cut = med + 4.0 * max(mad, 1e-9)
    q_cut = float(np.quantile(quiet_err, 0.99)) if len(quiet_err) >= 20 else mad_cut
    threshold = max(mad_cut, q_cut)
    flagged = error > threshold
    # Report in original units: reconstruction scaled back by the window medians.
    win_median = np.array([trailing[end - 1] if np.isfinite(trailing[end - 1]) else np.nan for end in idx])
    return {
        "backend": backend,
        "window": window,
        "idx": idx,
        "actual": x[idx - 1],
        "reconstruction": rec.mean(axis=1) * win_median,
        "error": error,
        "threshold": float(threshold),
        "flagged": flagged,
        "n_quiet_windows": int(win_quiet.sum()),
        "day_dev": np.abs(ratio - 1.0),  # per-day deviation, used for day-level attribution
        "model": model,
    }


def _mlp_autoencoder(feats: np.ndarray, hidden: int, bottleneck: int, seed: int):
    """Sklearn MLP autoencoder: symmetric bottleneck trained to reconstruct.

    Returns the fitted network only; scoring is the caller's job (train and
    score sets usually differ — train on quiet windows, score everything).
    """
    scaler = StandardScaler().fit(feats)
    z = scaler.transform(feats)
    net = MLPRegressor(
        hidden_layer_sizes=(hidden, max(2, bottleneck), hidden),
        activation="relu",
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=seed,
    )
    net.fit(z, z)  # targets are the inputs: this *is* the autoencoder objective
    net._scaler = scaler  # kept so callers can score unseen windows
    return net


def _lstm_autoencoder(feats: np.ndarray, hidden: int, bottleneck: int, seed: int, epochs: int = 60):
    """Torch LSTM autoencoder (used only when PyTorch is installed).

    Each ``d``-feature window is treated as a length-``d`` sequence of scalars,
    so the LSTM learns the intra-window shape. Returns the trained modules;
    scoring (train on quiet windows, score everything) is the caller's job.
    """
    torch.manual_seed(seed)
    enc = torch.nn.LSTM(1, hidden, batch_first=True)
    mid = torch.nn.Linear(hidden, bottleneck)
    up = torch.nn.Linear(bottleneck, hidden)
    dec = torch.nn.LSTM(hidden, 1, batch_first=True)
    params = list(enc.parameters()) + list(dec.parameters()) + list(mid.parameters()) + list(up.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)
    seq = torch.tensor(feats, dtype=torch.float32).unsqueeze(-1)  # (n, d, 1)
    for _ in range(epochs):
        opt.zero_grad()
        out, _ = enc(seq)  # (n, d, hidden)
        lat = mid(out)  # (n, d, bottleneck)
        hid = torch.relu(up(lat))  # (n, d, hidden)
        rec, _ = dec(hid)  # (n, d, 1)
        loss = torch.nn.functional.mse_loss(rec, seq)
        loss.backward()
        opt.step()
    return enc, mid, up, dec


def isolation_forest_scores(
    values: Sequence[float],
    denominators: Optional[Sequence[float]] = None,
    contamination: float = 0.03,
    seed: int = 7,
) -> Dict[str, Any]:
    """Isolation Forest over per-day engineered features.

    Features: raw level, day-over-day change, week-over-week change, and the
    ratio to the trailing 28-day median — the monitoring feature set most
    alerting platforms use. Returns per-day scores (higher = more anomalous).
    """
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required for the isolation forest detector")
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 21:
        raise ValueError(f"need at least 21 points, got {n}")

    def _shift(arr: np.ndarray, k: int) -> np.ndarray:
        out = np.full_like(arr, np.nan)
        out[k:] = arr[:-k]
        return out

    dod = x - _shift(x, 1)
    wow = x - _shift(x, 7)
    trailing = pd.Series(x).rolling(28, min_periods=7).median().to_numpy()
    ratio = x / np.where(trailing == 0, np.nan, trailing) - 1.0
    feats = np.column_stack([x, dod, wow, ratio])
    valid = ~np.isnan(feats).any(axis=1)

    iso = IsolationForest(contamination=contamination, random_state=seed)
    iso.fit(feats[valid])
    scores = np.full(n, np.nan)
    scores[valid] = -iso.score_samples(feats[valid])  # higher = more anomalous
    flagged = np.zeros(n, dtype=bool)
    flagged[valid] = iso.predict(feats[valid]) == -1
    return {
        "backend": "isolation_forest(sklearn)",
        "idx": np.arange(n),
        "actual": x,
        "error": scores,
        "flagged": flagged,
        "contamination": contamination,
        "model": iso,
    }


def _flagged_days(result: Dict[str, Any], total_len: int) -> np.ndarray:
    """Map window-level flags onto days (windows overlap, so one incident
    flags many windows — the confidence checks must count days, not windows).

    Within a flagged window only the days that actually deviate (``day_dev``
    above the quiet band) are marked; if none does (a pure shape anomaly),
    the whole window falls back to being marked.
    """
    flagged = np.asarray(result["flagged"], dtype=bool)
    idx = np.asarray(result.get("idx", np.arange(len(flagged))))
    win = result.get("window")
    if win is None:  # day-level detector (isolation forest)
        return flagged
    dev = np.asarray(result.get("day_dev", np.full(total_len, 1.0)), dtype=float)
    quiet_band = 0.05
    days = np.zeros(total_len, dtype=bool)
    for j, is_flag in enumerate(flagged):
        if not is_flag:
            continue
        end = int(idx[j])
        start = max(0, end - int(win))
        hot = dev[start:end] >= quiet_band
        if hot.any():
            days[start:end] |= hot
        else:
            days[start:end] = True
    return days


def dl_anomaly_confidence(result: Dict[str, Any], seasonal_flagged: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Standardized confidence checks for the deep-learning detector output."""
    checks: List[Dict[str, str]] = []
    err = np.asarray(result["error"], dtype=float)
    flagged = np.asarray(result["flagged"], dtype=bool)
    valid = np.isfinite(err)
    n_days = len(result.get("actual", np.array([]))) or len(flagged)
    day_flags = _flagged_days(result, max(n_days, int(np.max(result["idx"])) if len(result.get("idx", [])) else n_days))
    n_flag = int(day_flags.sum())
    n_total = max(1, len(day_flags))
    day_rate = n_flag / n_total

    # 1. The detector must actually fire somewhere, else it learned nothing.
    checks.append(
        {
            "name_zh": "detection power",
            "name_en": "detection power",
            "status": PASS if n_flag > 0 else WARN,
            "detail_zh": f"{n_flag} anomalous days" + ("" if n_flag else "; zero hits can mean underfit or a loose threshold"),
            "detail_en": f"{n_flag} anomalous days" + ("" if n_flag else "; zero hits can mean underfit or a loose threshold"),
        }
    )

    # 2. Alert volume sanity: >10% of days flagged means the threshold collapsed.
    checks.append(
        {
            "name_zh": "alert volume",
            "name_en": "alert volume",
            "status": PASS if day_rate <= 0.10 else FAIL,
            "detail_zh": f"anomalous-day share {day_rate:.1%} (>10% means the threshold collapsed)",
            "detail_en": f"anomalous-day share {day_rate:.1%} (>10% means the threshold collapsed)",
        }
    )

    # 3. Reconstruction quality: median relative error should be small.
    actual = np.asarray(result.get("actual", np.array([])), dtype=float)
    rec = np.asarray(result.get("reconstruction", np.array([])), dtype=float)
    if len(actual) == len(rec) and len(actual):
        denom = np.where(np.abs(actual) > 1e-9, np.abs(actual), 1.0)
        rel = float(np.median(np.abs(rec - actual) / denom))
        checks.append(
            {
                "name_zh": "reconstruction quality",
                "name_en": "reconstruction quality",
                "status": PASS if rel < 0.05 else (WARN if rel < 0.15 else FAIL),
                "detail_zh": f"median relative reconstruction error {rel:.1%} (too large means capacity/window mismatch)",
                "detail_en": f"median relative reconstruction error {rel:.1%} (too large means capacity/window mismatch)",
            }
        )

    # 4. Score separation: flagged errors should sit far above the quiet bulk.
    if valid.sum() >= 10 and n_flag > 0:
        quiet = err[valid & ~flagged]
        loud = err[valid & flagged]
        sep = float(np.median(loud) / max(np.median(quiet), 1e-12))
        checks.append(
            {
                "name_zh": "score separation",
                "name_en": "score separation",
                "status": PASS if sep >= 2.0 else WARN,
                "detail_zh": f"flagged/quiet median score ratio {sep:.1f}x (>=2x is a clean separation)",
                "detail_en": f"flagged/quiet median score ratio {sep:.1f}x (>=2x is a clean separation)",
            }
        )

    # 5. Cross-method agreement with the seasonal baseline, when available.
    # Day-level flags on both sides (AE window flags are mapped to days).
    if seasonal_flagged is not None:
        seasonal_flagged = np.asarray(seasonal_flagged, dtype=bool)
        day_flags = day_flags[: len(seasonal_flagged)]
        agree = int((seasonal_flagged & day_flags).sum())
        union = int((seasonal_flagged | day_flags).sum())
        jac = agree / union if union else 1.0
        checks.append(
            {
                "name_zh": "seasonal-baseline cross-check",
                "name_en": "seasonal-baseline cross-check",
                "status": PASS if jac > 0.3 or n_flag == 0 else WARN,
                "detail_zh": f"Jaccard={jac:.2f} ({agree}/{union} anomaly days shared with the seasonal baseline)",
                "detail_en": f"Jaccard={jac:.2f} ({agree}/{union} anomaly days shared with the seasonal baseline)",
            }
        )

    n_fail = sum(1 for c in checks if c["status"] == FAIL)
    score = sum(1 for c in checks if c["status"] == PASS) / max(1, len(checks))
    if n_fail:
        verdict = {"label_zh": "low confidence", "label_en": "low confidence", "score": score, "n_fail": n_fail}
    elif score >= 0.8:
        verdict = {"label_zh": "high confidence", "label_en": "high confidence", "score": score, "n_fail": n_fail}
    else:
        verdict = {"label_zh": "medium confidence", "label_en": "medium confidence", "score": score, "n_fail": n_fail}
    return {"checks": checks, "verdict": verdict, "score": score}


def dl_anomaly_table(dates: Sequence[Any], result: Dict[str, Any], lang: str = "zh") -> pd.DataFrame:
    """Frame for the chart/table block: date, actual, reconstruction, score, flag."""
    zh = lang != "en"
    idx = np.asarray(result["idx"])
    dates = np.asarray(dates)
    return pd.DataFrame(
        {
            zh and "date" or "date": dates[idx - 1] if result.get("backend", "").startswith("mlp") or result.get("backend", "").startswith("lstm") else dates[idx],
            zh and "actual" or "actual": np.asarray(result["actual"], dtype=float),
            zh and "reconstruction" or "reconstruction": np.asarray(result.get("reconstruction", result["actual"]), dtype=float),
            zh and "score" or "score": np.asarray(result["error"], dtype=float),
            zh and "threshold" or "threshold": result.get("threshold", float("nan")),
            zh and "anomaly" or "anomaly": np.asarray(result["flagged"], dtype=bool),
        }
    )
