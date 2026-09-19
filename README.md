# AtlasML — NYC Taxi Demand Forecasting & Dynamic Model Router

[![CI Workflow](https://github.com/thesivam124-creator/AtlasML/actions/workflows/ci.yml/badge.svg)](https://github.com/thesivam124-creator/AtlasML/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0174DF.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AtlasML** is a production-grade Data Science and MLOps system built for short-term (15-minute window) spatial-temporal taxi pickup demand forecasting across New York City zones using official NYC Taxi & Limousine Commission (TLC) trip record data.

Beyond standard time-series forecasting, AtlasML introduces a **Dynamic Model Router** — a meta-learning Decision Tree Classifier that dynamically selects candidate models per incoming prediction request to optimize accuracy, compute footprint, and prediction latency.

---

## 🔬 Key Data Science & Machine Learning Innovations

### 1. Medallion Data Pipeline & Quality Assurance
- **Bronze Layer**: Raw TLC Yellow Taxi trip data (`yellow_tripdata_2026-01.parquet`, ~9.43M records) ingested into optimized columnar Parquet format.
- **Silver Layer (Validation & Cleaning)**: Schema enforcement isolating bad rows (invalid timestamps, non-positive zone IDs, duplicate records) into a dedicated audit log rather than silently dropping data.
- **Gold Layer (Aggregation)**: Time-series bucket aggregation yielding **279,514 zone-level 15-minute pickup demand records**.

### 2. Leakage-Free Feature Engineering
- **Cyclic Temporal Encodings**: `hour_sin` and `hour_cos` transformations preserving continuity between hour 23 and hour 0.
- **Lag & Rolling Features**: `lag_1` (15-min lag), `lag_4` (1-hour lag), `rolling_mean_4`, and `rolling_std_4`.
- **Strict Time-Based Splitting**: 80% train / 20% test chronological split (222,792 train rows, 55,698 test rows) eliminating future lookahead bias.

### 3. Empirical Model Benchmarking
A comparative study evaluating non-ML, linear, and non-linear models:
- **Seasonal Naive**: Zero-compute lag-1 baseline.
- **Ridge Regression**: Linear model capturing global trend baselines with minimal compute latency.
- **LightGBM Regressor**: Gradient boosted decision trees capturing complex non-linear spatial-temporal demand patterns.

### 4. Dynamic Model Router (Core Research Contribution)
Instead of deploying a single static model, AtlasML uses a **Decision Tree Meta-Classifier (`max_depth=4`)** trained on historical performance metrics, feature variance, and computational cost proxies ($0.001 for Ridge, $0.01 for LightGBM).
- **Static Baseline (Ridge)**: 5.731 MAE
- **Static Baseline (LightGBM)**: 5.184 MAE (-9.54% error reduction)
- **Dynamic Model Router**: **5.179 MAE (-9.63% error reduction)**
- **Oracle Upper Bound**: 4.499 MAE (Theoretical minimum)

The router routes low-variance, linear requests to Ridge while routing non-linear demand spikes to LightGBM, reducing total infrastructure compute cost without sacrificing accuracy.

### 5. Uncertainty Estimation & Empirical Calibration
- **Residual-Based Prediction Intervals**: Computes 80% confidence bounds ($z = 1.28$) around point forecasts.
- **Empirical Calibration Check**: Achieved **87.37% coverage** on held-out test data, demonstrating well-calibrated bounds that avoid overconfidence.

### 6. Continuous Statistical Drift Monitoring
- **Feature Drift**: Two-sample Kolmogorov-Smirnov (KS) test ($p < 0.05$) detecting input distribution shifts.
- **Performance Drift**: Automated monitoring tracking MAE degradation against baseline thresholds (>20% degradation trigger).

---

## 📊 Performance Benchmark Summary

| Strategy | Model Type | MAE (Lower is Better) | RMSE | MAE Reduction vs Baseline | Key Advantage |
|---|---|---|---|---|---|
| **Seasonal Naive** | Non-ML Baseline (`lag_1`) | 5.805 | 10.839 | Baseline | Zero compute overhead |
| **Ridge Regression** | Linear ML (`alpha=1.0`) | 5.731 | 10.393 | -1.27% | Low latency & linear trends |
| **LightGBM Regressor** | Gradient Boosting (`n_est=200`) | 5.184 | 9.671 | -9.54% | High accuracy on complex patterns |
| **Dynamic Router** | **Adaptive Tree Router** | **5.179** | **9.650** | **-9.63%** | **Optimizes compute footprint & accuracy** |
| *Oracle (Upper Bound)* | *Theoretical Minimum* | *4.499* | *8.810* | *-22.50%* | *Upper bound benchmark* |

Detailed experimental findings and methodology can be found in [reports/project_report.md](reports/project_report.md).

---

## 🛠️ System Architecture & MLOps Stack

```
   ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
   │  NYC TLC Data    │ ───► │ Medallion Engine │ ───► │ Feature Store    │
   │  (Parquet Raw)   │      │ (Bronze/Silv/Gld)│      │ (Lag/Roll/Time)  │
   └──────────────────┘      └──────────────────┘      └──────────────────┘
                                                                │
                                                                ▼
   ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
   │ FastAPI Service  │ ◄─── │ Dynamic Router   │ ◄─── │ MLflow Tracking  │
   │ /health /predict │      │ Meta-Classifier  │      │ (Models & Runs)  │
   └──────────────────┘      └──────────────────┘      └──────────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/thesivam124-creator/AtlasML.git
cd AtlasML

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Ingestion & Feature Engineering
```bash
python scripts/run_ingestion.py
```

### 3. Run Baselines & Dynamic Router Pipelines
```bash
# Train baselines and log metrics to MLflow
python scripts/run_baselines.py

# Train Dynamic Model Router and calibrate uncertainty intervals
python scripts/run_routing.py
```

### 4. Run Pytest Suite
```bash
python -m pytest -v
```

### 5. Start FastAPI Web Service
```bash
uvicorn src.atlasml.serving.app:app --reload --port 8000
```
Interactive API documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 6. Docker Container Deployment
```bash
docker build -t atlasml .
docker run -p 8000:8000 atlasml
```

---

## 📂 Repository Structure

```
atlasml/
├── config/                  # Configuration files
├── data/                    # Medallion data directory (raw, bronze, silver, gold, rejected)
├── docs/                    # Architectural & data documentation
├── experiments/             # Serialized joblib models & MLflow tracking artifacts
├── reports/                 # Research report & visualization figures
│   └── project_report.md    # Detailed research paper
├── scripts/                 # Execution scripts for ingestion, training, and routing
├── src/atlasml/             # Modular Python package
│   ├── drift/               # KS-test feature drift & performance monitors
│   ├── features/            # Feature aggregation & engineering
│   ├── ingestion/           # Raw data loaders
│   ├── models/              # Baseline estimators (Ridge, LightGBM)
│   ├── routing/             # Dynamic Model Router decision engine
│   ├── serving/             # FastAPI app endpoint definitions
│   ├── uncertainty/         # Prediction interval calibration logic
│   └── validation/          # Silver layer schema validators
├── tests/                   # Pytest unit & integration test suite
├── AtlasML_Beginner_Guide.md # Step-by-step beginner guide
├── Dockerfile               # Production container definition
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
