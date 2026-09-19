# AtlasML — NYC Taxi Demand Forecasting & Dynamic Model Routing

AtlasML is a complete end-to-end Machine Learning system for short-term NYC taxi demand forecasting, uncertainty estimation, dynamic model routing, and automated drift monitoring.

---

## 🌟 Key Features

- **Bronze/Silver/Gold Data Pipeline**: Validates raw NYC TLC Parquet files, catches bad rows, and aggregates into 15-minute per-zone demand time series.
- **Leakage-Free Feature Engineering**: Cyclic time encodings (`hour_sin`, `hour_cos`), lag features (`lag_1`, `lag_4`), and rolling statistics (`rolling_mean_4`, `rolling_std_4`).
- **Baseline Models**: Seasonal Naive, Ridge Linear Regression, and LightGBM Regressor.
- **MLflow Tracking**: Automatic logging of hyperparameters, MAE/RMSE metrics, and model artifacts.
- **Uncertainty Estimation**: Residual-based prediction intervals with empirical calibration checks (~87.3% coverage).
- **Dynamic Model Router**: Decision-tree router selecting optimal model per prediction request (outperforming static baselines with 5.179 MAE).
- **Drift Detection**: Kolmogorov-Smirnov test feature drift detector & MAE degradation monitoring.
- **FastAPI Web API**: `/health` and `/predict` endpoints serving real-time predictions, intervals, and routing decisions.
- **Containerized Deployment**: Dockerfile and GitHub Actions CI pipeline.

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install httpx
```

### 2. Run Training Pipelines

Run baselines and MLflow experiment logging:
```bash
python scripts/run_baselines.py
```

Run model router training and uncertainty calibration:
```bash
python scripts/run_routing.py
```

### 3. Run Pytest Suite

```bash
python -m pytest -v
```

### 4. Start FastAPI Server

```bash
uvicorn src.atlasml.serving.app:app --reload --port 8000
```

Access interactive API docs at: `http://localhost:8000/docs`

### 5. Docker Deployment

```bash
docker build -t atlasml .
docker run -p 8000:8000 atlasml
```

---

## 📊 Benchmark Summary

| Strategy | MAE | RMSE | Model Type |
|---|---|---|---|
| Seasonal Naive | 5.805 | 10.839 | Non-ML (`lag_1`) |
| Ridge Regression | 5.731 | 10.393 | Linear Model |
| LightGBM Regressor | 5.184 | 9.671 | Gradient Boosting |
| **Dynamic Model Router** | **5.179** | **9.650** | **Adaptive Tree Router** |
| Oracle (Upper Bound) | 4.499 | 8.810 | Theoretical Minimum |

For detailed findings, see [reports/project_report.md](file:///c:/Users/thesi/Documents/atlasdb/atlasml/reports/project_report.md).
