import numpy as np
import pandas as pd
import pytest
from src.atlasml.routing.router import (
    build_routing_dataset,
    train_router,
    evaluate_routing_strategies,
)


def test_build_routing_dataset():
    df = pd.DataFrame({"hour": [1, 2, 3], "lag_1": [10, 20, 30]})
    y_true = np.array([12, 18, 35])
    ridge_preds = np.array([10, 20, 30])  # errors: 2, 2, 5
    gbm_preds = np.array([12, 18, 30])    # errors: 0, 0, 5

    routing_df = build_routing_dataset(df, y_true, ridge_preds, gbm_preds)
    assert len(routing_df) == 3
    assert "best_model" in routing_df.columns
    assert list(routing_df["best_model"]) == ["gbm", "gbm", "ridge"]  # tie goes to ridge due to lower cost proxy


def test_train_router_and_evaluate():
    df = pd.DataFrame({
        "hour": [1, 2, 3, 4, 5, 6],
        "rolling_std_4": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    })
    y_true = np.array([10, 20, 30, 40, 50, 60])
    ridge_preds = np.array([11, 21, 31, 35, 45, 55])  # errors: 1, 1, 1, 5, 5, 5
    gbm_preds = np.array([15, 25, 35, 40, 50, 60])    # errors: 5, 5, 5, 0, 0, 0

    routing_df = build_routing_dataset(df, y_true, ridge_preds, gbm_preds)
    feature_cols = ["hour", "rolling_std_4"]

    router, train_acc = train_router(routing_df, feature_cols, max_depth=2)
    assert train_acc > 0.8

    metrics = evaluate_routing_strategies(routing_df, router, feature_cols)
    assert "always_ridge_mae" in metrics
    assert "always_gbm_mae" in metrics
    assert "router_mae" in metrics
    assert "oracle_mae" in metrics
    assert metrics["router_mae"] <= metrics["always_gbm_mae"]
    assert metrics["router_mae"] >= metrics["oracle_mae"]
