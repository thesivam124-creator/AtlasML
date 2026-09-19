import numpy as np
import pandas as pd
import pytest
from src.atlasml.models.baselines import (
    time_based_split,
    seasonal_naive_predict,
    evaluate,
    train_ridge,
    train_gbm,
)


def test_time_based_split():
    df = pd.DataFrame({
        "time_bucket": pd.date_range("2026-01-01", periods=10, freq="15min"),
        "pickup_count": range(10),
    })
    train, test = time_based_split(df, time_col="time_bucket", test_frac=0.2)
    assert len(train) == 8
    assert len(test) == 2
    assert train["time_bucket"].max() < test["time_bucket"].min()


def test_seasonal_naive_predict():
    df = pd.DataFrame({"lag_1": [10, 20, 30]})
    preds = seasonal_naive_predict(df, lag_col="lag_1")
    assert list(preds) == [10, 20, 30]


def test_evaluate():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])
    res = evaluate(y_true, y_pred, "test_model")
    assert res["model"] == "test_model"
    assert pytest.approx(res["mae"], 0.01) == 2.333
    assert "rmse" in res


def test_train_ridge_and_gbm():
    X_train = pd.DataFrame({
        "hour_sin": [0.1, 0.2, 0.3, 0.4, 0.5],
        "lag_1": [5, 10, 15, 20, 25],
    })
    y_train = pd.Series([6, 11, 16, 21, 26])
    X_test = pd.DataFrame({
        "hour_sin": [0.6],
        "lag_1": [30],
    })
    
    ridge_model, ridge_preds = train_ridge(X_train, y_train, X_test)
    assert len(ridge_preds) == 1
    assert ridge_preds[0] > 25

    gbm_model, gbm_preds = train_gbm(X_train, y_train, X_test, n_estimators=10)
    assert len(gbm_preds) == 1
