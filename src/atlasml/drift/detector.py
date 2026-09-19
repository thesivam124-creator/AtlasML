import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, Any


def detect_feature_drift(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Detect statistical distribution drift between reference and current feature data
    using the two-sample Kolmogorov-Smirnov (KS) test.

    Parameters
    ----------
    reference : pd.Series | np.ndarray
        Baseline/reference feature distribution (e.g., training data).
    current : pd.Series | np.ndarray
        Current feature distribution (e.g., recent production data).
    alpha : float
        Statistical significance threshold (default 0.05).

    Returns
    -------
    Dict[str, Any]
        Test statistic, p-value, drift flag, and threshold alpha.
    """
    ref_vals = np.asarray(reference, dtype=float)
    curr_vals = np.asarray(current, dtype=float)

    stat, p_value = ks_2samp(ref_vals, curr_vals)
    drifted = bool(p_value < alpha)

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "drifted": drifted,
        "alpha": alpha,
    }


def detect_performance_drift(
    reference_mae: float,
    current_mae: float,
    threshold_pct: float = 20.0,
) -> Dict[str, Any]:
    """
    Detect heuristic performance degradation by comparing current MAE to baseline reference MAE.

    Parameters
    ----------
    reference_mae : float
        Baseline model MAE on validation/historical data.
    current_mae : float
        Observed MAE on current operational batch.
    threshold_pct : float
        Percentage threshold increase allowed before flagging drift (default 20.0%).

    Returns
    -------
    Dict[str, Any]
        Reference MAE, current MAE, percentage change, drift flag.
    """
    if reference_mae <= 0:
        raise ValueError("reference_mae must be strictly positive.")

    pct_increase = float(((current_mae - reference_mae) / reference_mae) * 100.0)
    drifted = bool(pct_increase >= threshold_pct)

    return {
        "reference_mae": float(reference_mae),
        "current_mae": float(current_mae),
        "pct_increase": pct_increase,
        "drifted": drifted,
        "threshold_pct": threshold_pct,
    }
