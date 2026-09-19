import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def residual_based_interval(
    y_pred: np.ndarray | pd.Series,
    recent_residuals: np.ndarray | pd.Series,
    z: float = 1.28,  # ~80% coverage for standard normal
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate residual-based prediction intervals.

    Parameters
    ----------
    y_pred : np.ndarray | pd.Series
        Point predictions.
    recent_residuals : np.ndarray | pd.Series
        Recent prediction errors (y_true - y_pred).
    z : float
        Z-score for prediction interval (1.28 approx 80% coverage).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (lower_bound, upper_bound)
    """
    y_pred = np.asarray(y_pred, dtype=float)
    recent_residuals = np.asarray(recent_residuals, dtype=float)

    if len(recent_residuals) == 0:
        sigma = 1.0
    else:
        sigma = float(np.std(recent_residuals))
        if sigma == 0:
            sigma = 1e-5

    lower = y_pred - z * sigma
    upper = y_pred + z * sigma

    # Demand cannot be negative
    lower = np.maximum(0.0, lower)

    return lower, upper


def check_calibration(
    y_true: np.ndarray | pd.Series,
    lower: np.ndarray | pd.Series,
    upper: np.ndarray | pd.Series,
) -> Dict[str, Any]:
    """
    Compute empirical coverage rate to assess uncertainty calibration.

    Returns
    -------
    Dict[str, Any]
        Empirical coverage percentage and calibration stats.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    covered = (y_true >= lower) & (y_true <= upper)
    empirical_coverage = float(np.mean(covered))

    return {
        "empirical_coverage": empirical_coverage,
        "total_samples": len(y_true),
        "covered_samples": int(np.sum(covered)),
    }
