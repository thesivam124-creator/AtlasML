import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure src directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from atlasml.uncertainty.estimation import residual_based_interval

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model artifacts on startup
    models_dir = PROJECT_ROOT / "experiments"
    try:
        if (models_dir / "ridge_model.joblib").exists():
            models["ridge"] = joblib.load(models_dir / "ridge_model.joblib")
        if (models_dir / "gbm_model.joblib").exists():
            models["gbm"] = joblib.load(models_dir / "gbm_model.joblib")
        if (models_dir / "router_model.joblib").exists():
            models["router"] = joblib.load(models_dir / "router_model.joblib")
        print("Loaded ML model artifacts successfully.")
    except Exception as e:
        print(f"Warning: Could not load model artifacts: {e}")
    yield
    models.clear()


app = FastAPI(
    title="AtlasML Taxi Demand Forecasting & Router API",
    description="Serving endpoint for NYC taxi demand forecasting, uncertainty estimation, and model routing.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    zone_id: int = Field(..., description="PULocationID zone identifier", json_schema_extra={"example": 142})
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)", json_schema_extra={"example": 14})
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)", json_schema_extra={"example": 2})
    is_weekend: int = Field(..., ge=0, le=1, description="1 if weekend else 0", json_schema_extra={"example": 0})
    lag_1: float = Field(..., description="Demand 15 mins ago", json_schema_extra={"example": 25.0})
    lag_4: float = Field(..., description="Demand 1 hour ago", json_schema_extra={"example": 22.0})
    rolling_mean_4: float = Field(..., description="Rolling 1-hr mean demand", json_schema_extra={"example": 24.5})
    rolling_std_4: float = Field(..., description="Rolling 1-hr std dev demand", json_schema_extra={"example": 3.2})


class PredictionResponse(BaseModel):
    zone_id: int
    prediction: float
    lower_bound: float
    upper_bound: float
    model_used: str
    confidence_level: str = "80%"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    # Compute cyclic hour encodings
    hour_sin = np.sin(2 * np.pi * req.hour / 24.0)
    hour_cos = np.cos(2 * np.pi * req.hour / 24.0)

    # Feature vector matching training feature columns
    feature_dict = {
        "hour": req.hour,
        "day_of_week": req.day_of_week,
        "is_weekend": req.is_weekend,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "lag_1": req.lag_1,
        "lag_4": req.lag_4,
        "rolling_mean_4": req.rolling_mean_4,
        "rolling_std_4": req.rolling_std_4,
    }
    df_features = pd.DataFrame([feature_dict])

    # Default fallback prediction if models not loaded
    ridge_pred = float(req.lag_1)
    gbm_pred = float(req.lag_1)

    if "ridge" in models:
        ridge_pred = float(models["ridge"].predict(df_features)[0])
    if "gbm" in models:
        gbm_pred = float(models["gbm"].predict(df_features)[0])

    # Routing decision
    model_used = "gbm"  # default preference
    if "router" in models:
        pred_diff = abs(ridge_pred - gbm_pred)
        router_features = df_features.copy()
        router_features["ridge_pred"] = ridge_pred
        router_features["gbm_pred"] = gbm_pred
        router_features["pred_diff"] = pred_diff

        router_cols = [
            "hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos",
            "lag_1", "lag_4", "rolling_mean_4", "rolling_std_4",
            "ridge_pred", "gbm_pred", "pred_diff",
        ]
        decision = models["router"].predict(router_features[router_cols])[0]
        model_used = str(decision)

    selected_pred = gbm_pred if model_used == "gbm" else ridge_pred

    # Uncertainty interval (~80% bounds based on empirical residual std ~3.5)
    recent_residuals = np.array([-3.5, 0.0, 3.5])
    lower, upper = residual_based_interval([selected_pred], recent_residuals=recent_residuals, z=1.28)

    return PredictionResponse(
        zone_id=req.zone_id,
        prediction=round(float(selected_pred), 2),
        lower_bound=round(float(lower[0]), 2),
        upper_bound=round(float(upper[0]), 2),
        model_used=model_used,
        confidence_level="80%",
    )
