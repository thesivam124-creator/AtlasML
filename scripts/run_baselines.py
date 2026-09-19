import logging
import sys
from pathlib import Path
import pandas as pd
import joblib
import mlflow

# Add src directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from atlasml.ingestion.load_raw import load_raw_to_bronze
from atlasml.validation.validators import validate_trips
from atlasml.features.aggregate import aggregate_demand
from atlasml.features.build_features import add_time_features, add_lag_and_rolling_features
from atlasml.models.baselines import (
    time_based_split,
    seasonal_naive_predict,
    evaluate,
    train_ridge,
    train_gbm,
)

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
TARGET_COL = "pickup_count"


def run_pipeline():
    logger.info("Starting AtlasML Baseline & Feature Pipeline...")

    # Data paths
    raw_path = PROJECT_ROOT / "data" / "raw" / "yellow_tripdata_2026-01.parquet"
    bronze_path = PROJECT_ROOT / "data" / "bronze" / "yellow_tripdata_2026-01.parquet"
    silver_path = PROJECT_ROOT / "data" / "silver" / "trips_2026_01.parquet"
    rejected_path = PROJECT_ROOT / "data" / "rejected" / "trips_2026_01.parquet"
    gold_path = PROJECT_ROOT / "data" / "gold" / "demand_2026_01.parquet"
    models_dir = PROJECT_ROOT / "experiments"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion (Raw -> Bronze)
    if not bronze_path.exists():
        logger.info("Bronze data missing. Ingesting from raw...")
        df_bronze = load_raw_to_bronze(str(raw_path), str(bronze_path))
    else:
        logger.info(f"Loading bronze data from {bronze_path}")
        df_bronze = pd.read_parquet(bronze_path)

    # 2. Validation (Bronze -> Silver & Rejected)
    logger.info("Validating trip records...")
    valid_df, rejected_df = validate_trips(df_bronze)
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    valid_df.to_parquet(silver_path, index=False)
    rejected_df.to_parquet(rejected_path, index=False)
    logger.info(f"Validation complete. Valid: {len(valid_df):,} rows, Rejected: {len(rejected_df):,} rows.")

    # 3. Aggregation (Silver -> Gold)
    logger.info("Aggregating into per-zone 15-minute time series buckets...")
    demand_df = aggregate_demand(valid_df, freq="15min")
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    demand_df.to_parquet(gold_path, index=False)
    logger.info(f"Gold aggregated dataset shape: {demand_df.shape}")

    # 4. Feature Engineering
    logger.info("Engineering time and lag/rolling features...")
    df_features = add_time_features(demand_df)
    df_features = add_lag_and_rolling_features(df_features, group_col="PULocationID")
    df_clean = df_features.dropna().copy()
    logger.info(f"Clean features dataset shape (after dropping NaNs): {df_clean.shape}")

    # Save feature matrix for downstream steps (routing/drift)
    df_clean.to_parquet(PROJECT_ROOT / "data" / "gold" / "features_2026_01.parquet", index=False)

    # 5. Time-based Split
    logger.info("Splitting dataset into train (80%) and test (20%) chronologically...")
    train_df, test_df = time_based_split(df_clean, time_col="time_bucket", test_frac=0.2)
    logger.info(f"Train size: {len(train_df):,}, Test size: {len(test_df):,}")

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    # 6. Model Training & MLflow Tracking (Step 8)
    mlflow.set_experiment("atlasml_baselines")
    results = []

    # A. Seasonal Naive Baseline
    logger.info("Evaluating Seasonal Naive Baseline...")
    with mlflow.start_run(run_name="seasonal_naive"):
        mlflow.log_param("model_type", "seasonal_naive")
        naive_preds = seasonal_naive_predict(test_df, lag_col="lag_1")
        naive_metrics = evaluate(y_test, naive_preds, "seasonal_naive")
        mlflow.log_metrics({"mae": naive_metrics["mae"], "rmse": naive_metrics["rmse"]})
        results.append(naive_metrics)

    # B. Ridge Regression
    logger.info("Training Ridge Regression...")
    with mlflow.start_run(run_name="ridge_v1"):
        alpha = 1.0
        mlflow.log_param("model_type", "ridge")
        mlflow.log_param("alpha", alpha)
        ridge_model, ridge_preds = train_ridge(X_train, y_train, X_test, alpha=alpha)
        ridge_metrics = evaluate(y_test, ridge_preds, "ridge")
        mlflow.log_metrics({"mae": ridge_metrics["mae"], "rmse": ridge_metrics["rmse"]})
        joblib.dump(ridge_model, models_dir / "ridge_model.joblib")
        mlflow.log_artifact(str(models_dir / "ridge_model.joblib"))
        results.append(ridge_metrics)

    # C. LightGBM Gradient Boosting
    logger.info("Training LightGBM Regressor...")
    with mlflow.start_run(run_name="gbm_v1"):
        n_est = 200
        lr = 0.05
        mlflow.log_param("model_type", "lightgbm")
        mlflow.log_param("n_estimators", n_est)
        mlflow.log_param("learning_rate", lr)
        gbm_model, gbm_preds = train_gbm(X_train, y_train, X_test, n_estimators=n_est, learning_rate=lr)
        gbm_metrics = evaluate(y_test, gbm_preds, "gradient_boosting")
        mlflow.log_metrics({"mae": gbm_metrics["mae"], "rmse": gbm_metrics["rmse"]})
        joblib.dump(gbm_model, models_dir / "gbm_model.joblib")
        mlflow.log_artifact(str(models_dir / "gbm_model.joblib"))
        results.append(gbm_metrics)

    # Summary Table
    logger.info("\n=== Baseline Results Summary ===")
    summary_df = pd.DataFrame(results)
    logger.info(f"\n{summary_df.to_string(index=False)}")

    return summary_df


if __name__ == "__main__":
    run_pipeline()
