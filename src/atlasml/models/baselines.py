import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from typing import Tuple, Dict, Any


def time_based_split(
    df: pd.DataFrame,
    time_col: str = "time_bucket",
    test_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-series dataframe into train and test sets chronologically.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series features and targets.
    time_col : str
        Column name for timestamp.
    test_frac : float
        Fraction of data to allocate to the test set (chronologically last).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    cutoff = int(len(df_sorted) * (1 - test_frac))
    train_df = df_sorted.iloc[:cutoff].copy()
    test_df = df_sorted.iloc[cutoff:].copy()
    return train_df, test_df


def seasonal_naive_predict(df: pd.DataFrame, lag_col: str = "lag_1") -> pd.Series:
    """
    Seasonal naive baseline prediction using historical lag values.
    """
    if lag_col not in df.columns:
        raise ValueError(f"Lag column '{lag_col}' not found in dataframe columns: {list(df.columns)}")
    return df[lag_col]


def evaluate(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, label: str) -> Dict[str, Any]:
    """
    Evaluate prediction performance using MAE and RMSE metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    print(f"{label:20s}  MAE={mae:.3f}  RMSE={rmse:.3f}")
    return {"model": label, "mae": mae, "rmse": rmse}


def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray = None,
    alpha: float = 1.0,
) -> Tuple[Ridge, np.ndarray]:
    """
    Train a Ridge regression model and generate test predictions.
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds


def train_gbm(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray = None,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Tuple[lgb.LGBMRegressor, np.ndarray]:
    """
    Train a LightGBM regressor and generate test predictions.
    """
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds
