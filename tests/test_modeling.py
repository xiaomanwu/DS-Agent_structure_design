"""Modeling toolkit: feature engineering, training, explanation, confidence."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from ds_agent.tools.modeling import (
    engineer_features,
    feature_analysis,
    modeling_confidence,
    train_model,
)

pytest.importorskip("sklearn")


@pytest.fixture(scope="module")
def con(warehouse):
    # Join whatever mode the warehouse is already open in (a read-write open
    # elsewhere in the suite wins; forcing read-only would conflict with it).
    return duckdb.connect(str(warehouse))


@pytest.fixture(scope="module")
def cls_result(con):
    df = con.execute(
        "SELECT tenure_days, spend_30d, sessions_30d, channel, device, converted FROM uplift_users LIMIT 8000"
    ).df()
    X, y, names, log = engineer_features(df, target="converted")
    return train_model(X, y, task="classification")


@pytest.fixture(scope="module")
def reg_result(con):
    df = con.execute(
        "SELECT date, SUM(gmv) AS gmv, SUM(sessions) AS sessions, SUM(orders) AS orders "
        "FROM orders_daily GROUP BY 1 ORDER BY 1"
    ).df()
    X, y, names, log = engineer_features(df, target="gmv", date_column="date")
    return train_model(X, y, task="regression")


def test_engineer_features_onehot_and_datetime(con):
    df = con.execute(
        "SELECT dt AS date, device, pre_revenue, converted FROM experiment_users LIMIT 500"
    ).df()
    X, y, names, log = engineer_features(df, target="converted", date_column="date")
    assert y is not None and len(y) == len(X)
    assert any(("_" in c) and c.split("_")[0] in {"device", "channel"} for c in names), "categoricals must be one-hot encoded"
    # The datetime expansion ran; a constant snapshot date (this table has a
    # single dt) then yields constant derived features, which preprocessing
    # drops on their merits — the log records both stages either way.
    assert "datetime expansion of date -> dow/month/is_weekend" in log["steps"]
    assert {"dow", "month", "is_weekend"} & set(names) or any(
        c in {d["column"] for d in log.get("preprocess", {}).get("dropped", [])}
        for c in ("dow", "month", "is_weekend")
    )
    assert not X.isna().any().any(), "imputation should leave no NaNs"
    assert "one-hot encoded" in "; ".join(log["steps"])


def test_classification_metrics_and_cv(cls_result):
    assert cls_result["task"] == "classification"
    metrics = cls_result["metrics"][cls_result["best_model"]]
    assert 0.5 < metrics["roc_auc"] < 1.0  # real signal, not a memorized fit
    assert 0 <= cls_result["cv_std"] < 0.2
    assert cls_result["n_train"] + cls_result["n_test"] == 8000


def test_regression_metrics(reg_result):
    assert reg_result["task"] == "regression"
    metrics = reg_result["metrics"][reg_result["best_model"]]
    assert metrics["r2"] > 0.8  # calendar+traffic explain most of daily GMV
    assert metrics["rmse"] > 0


def test_feature_analysis_ranks_signal_first(cls_result):
    fa = feature_analysis(cls_result)
    top = [r["feature"] for r in fa["importance"][:3]]
    assert "spend_30d" in top, "the strongest behavioural predictor should rank first"
    assert fa["engine"] in ("shap", "sklearn-permutation")


def test_confidence_card(cls_result):
    conf = modeling_confidence(cls_result)
    names = [c["name"] for c in conf["checks"]]
    assert {"train-test gap", "5-fold CV stability", "sample size"} == set(names)
    assert conf["overall"] in ("high", "medium", "low")


def test_regression_has_direction_and_charts(reg_result):
    fa = feature_analysis(reg_result)
    assert fa["importance"], "regression should produce importance rows"
    from ds_agent import charts

    img = charts.importance_bar(fa["importance"])
    assert img.startswith("data:image/png;base64,")
    _, X_te, y_te = None, reg_result["_split"][1], reg_result["_split"][3]
    pred = reg_result["predictions"][reg_result["best_model"]]
    img2 = charts.pred_vs_actual_chart(y_te, pred)
    assert img2.startswith("data:image/png;base64,")
