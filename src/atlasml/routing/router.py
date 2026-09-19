import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
from typing import Tuple, Dict, Any


def build_routing_dataset(
    df: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    ridge_preds: np.ndarray,
    gbm_preds: np.ndarray,
    ridge_cost: float = 0.001,
    gbm_cost: float = 0.01,
) -> pd.DataFrame:
    """
    Construct a dataset for training and evaluating the model router.

    Parameters
    ----------
    df : pd.DataFrame
        Input features dataframe.
    y_true : pd.Series | np.ndarray
        True target values.
    ridge_preds : np.ndarray
        Predictions from Ridge regression.
    gbm_preds : np.ndarray
        Predictions from LightGBM regressor.
    ridge_cost : float
        Latency/compute cost proxy for Ridge.
    gbm_cost : float
        Latency/compute cost proxy for LightGBM.

    Returns
    -------
    pd.DataFrame
        Routing feature matrix with best_model target labels.
    """
    y_true = np.asarray(y_true, dtype=float)
    ridge_preds = np.asarray(ridge_preds, dtype=float)
    gbm_preds = np.asarray(gbm_preds, dtype=float)

    ridge_error = np.abs(ridge_preds - y_true)
    gbm_error = np.abs(gbm_preds - y_true)

    # Cost-adjusted error decision: pick model with lower total loss (error + cost)
    ridge_loss = ridge_error + ridge_cost
    gbm_loss = gbm_error + gbm_cost

    best_model = np.where(gbm_loss < ridge_loss, "gbm", "ridge")

    routing_df = df.copy()
    routing_df["true_value"] = y_true
    routing_df["ridge_pred"] = ridge_preds
    routing_df["gbm_pred"] = gbm_preds
    routing_df["ridge_error"] = ridge_error
    routing_df["gbm_error"] = gbm_error
    routing_df["pred_diff"] = np.abs(ridge_preds - gbm_preds)
    routing_df["best_model"] = best_model

    return routing_df


def train_router(
    routing_df: pd.DataFrame,
    router_feature_cols: list[str],
    max_depth: int = 4,
    random_state: int = 42,
) -> Tuple[DecisionTreeClassifier, float]:
    """
    Train a decision tree model router to predict which model will perform best per request.

    Returns
    -------
    Tuple[DecisionTreeClassifier, float]
        (trained_router_model, train_accuracy)
    """
    X_router = routing_df[router_feature_cols]
    y_router = routing_df["best_model"]

    router = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    router.fit(X_router, y_router)

    train_acc = accuracy_score(y_router, router.predict(X_router))
    return router, float(train_acc)


def evaluate_routing_strategies(
    routing_df: pd.DataFrame,
    router: DecisionTreeClassifier,
    router_feature_cols: list[str],
) -> Dict[str, float]:
    """
    Compare Always Ridge, Always GBM, Router, and Oracle strategies.

    Returns
    -------
    Dict[str, float]
        MAEs for each strategy.
    """
    always_ridge_mae = float(routing_df["ridge_error"].mean())
    always_gbm_mae = float(routing_df["gbm_error"].mean())
    oracle_mae = float(np.minimum(routing_df["ridge_error"], routing_df["gbm_error"]).mean())

    X_router = routing_df[router_feature_cols]
    router_decisions = router.predict(X_router)

    selected_errors = np.where(router_decisions == "gbm", routing_df["gbm_error"], routing_df["ridge_error"])
    router_mae = float(np.mean(selected_errors))

    return {
        "always_ridge_mae": always_ridge_mae,
        "always_gbm_mae": always_gbm_mae,
        "router_mae": router_mae,
        "oracle_mae": oracle_mae,
    }
