import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import mlflow

# Add src directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from atlasml.models.baselines import time_based_split, evaluate, seasonal_naive_predict
from atlasml.uncertainty.estimation import residual_based_interval, check_calibration
from atlasml.routing.router import build_routing_dataset, train_router, evaluate_routing_strategies

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "lag_1",
    "lag_4",
    "rolling_mean_4",
    "rolling_std_4",
]
ROUTER_FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "lag_1",
    "lag_4",
    "rolling_mean_4",
    "rolling_std_4",
    "ridge_pred",
    "gbm_pred",
    "pred_diff",
]
TARGET_COL = "pickup_count"


def run_routing_pipeline():
    logger.info("Starting AtlasML Routing & Uncertainty Pipeline...")

    features_path = PROJECT_ROOT / "data" / "gold" / "features_2026_01.parquet"
    models_dir = PROJECT_ROOT / "experiments"

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found at {features_path}. Run run_baselines.py first.")

    df_clean = pd.read_parquet(features_path)
    logger.info(f"Loaded features dataset shape: {df_clean.shape}")

    # Split train and test
    train_df, test_df = time_based_split(df_clean, time_col="time_bucket", test_frac=0.2)
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    # Load baseline models
    ridge_model = joblib.load(models_dir / "ridge_model.joblib")
    gbm_model = joblib.load(models_dir / "gbm_model.joblib")

    logger.info("Generating predictions on train and test sets...")
    train_ridge_preds = ridge_model.predict(X_train)
    train_gbm_preds = gbm_model.predict(X_train)
    
    test_ridge_preds = ridge_model.predict(X_test)
    test_gbm_preds = gbm_model.predict(X_test)

    # 1. Uncertainty Estimation & Calibration Check (Step 9)
    logger.info("Estimating prediction uncertainty & evaluating calibration...")
    recent_residuals = y_train.values - train_gbm_preds
    lower_bounds, upper_bounds = residual_based_interval(test_gbm_preds, recent_residuals=recent_residuals, z=1.28)
    calib_res = check_calibration(y_test.values, lower_bounds, upper_bounds)
    logger.info(f"Uncertainty Coverage Rate (Target 80%): {calib_res['empirical_coverage']:.2%}")

    # 2. Router Dataset & Model Training (Step 10)
    logger.info("Building Model Router dataset...")
    routing_df = build_routing_dataset(
        test_df,
        y_test.values,
        test_ridge_preds,
        test_gbm_preds,
        ridge_cost=0.001,
        gbm_cost=0.01,
    )

    logger.info("Training Decision Tree Model Router...")
    router, train_acc = train_router(routing_df, ROUTER_FEATURE_COLS, max_depth=4)
    logger.info(f"Router Training Accuracy: {train_acc:.2%}")

    # Save Router model artifact
    joblib.dump(router, models_dir / "router_model.joblib")

    # 3. Evaluate Routing Strategies
    routing_metrics = evaluate_routing_strategies(routing_df, router, ROUTER_FEATURE_COLS)

    logger.info("\n=== Model Router Evaluation vs Baselines & Oracle ===")
    logger.info(f"Always Ridge MAE: {routing_metrics['always_ridge_mae']:.3f}")
    logger.info(f"Always GBM MAE:   {routing_metrics['always_gbm_mae']:.3f}")
    logger.info(f"Router MAE:       {routing_metrics['router_mae']:.3f}")
    logger.info(f"Oracle MAE:       {routing_metrics['oracle_mae']:.3f} (Unreachable Upper Bound)")

    # 4. MLflow Logging
    mlflow.set_experiment("atlasml_routing")
    with mlflow.start_run(run_name="router_v1"):
        mlflow.log_param("router_type", "DecisionTreeClassifier")
        mlflow.log_param("max_depth", 4)
        mlflow.log_metric("empirical_coverage", calib_res["empirical_coverage"])
        mlflow.log_metric("always_ridge_mae", routing_metrics["always_ridge_mae"])
        mlflow.log_metric("always_gbm_mae", routing_metrics["always_gbm_mae"])
        mlflow.log_metric("router_mae", routing_metrics["router_mae"])
        mlflow.log_metric("oracle_mae", routing_metrics["oracle_mae"])
        mlflow.log_artifact(str(models_dir / "router_model.joblib"))

    return routing_metrics


if __name__ == "__main__":
    run_routing_pipeline()
