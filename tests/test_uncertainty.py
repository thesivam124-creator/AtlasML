import numpy as np
import pytest
from src.atlasml.uncertainty.estimation import residual_based_interval, check_calibration


def test_residual_based_interval():
    y_pred = np.array([10.0, 20.0, 30.0])
    residuals = np.array([-2.0, 0.0, 2.0])
    
    lower, upper = residual_based_interval(y_pred, residuals, z=1.0)
    
    assert len(lower) == 3
    assert len(upper) == 3
    assert np.all(lower <= y_pred)
    assert np.all(upper >= y_pred)
    # Check lower is clipped at 0
    assert np.all(lower >= 0.0)


def test_check_calibration():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    lower = np.array([8.0, 18.0, 25.0, 50.0])  # row 4 is outside interval
    upper = np.array([12.0, 22.0, 35.0, 60.0])
    
    res = check_calibration(y_true, lower, upper)
    assert res["empirical_coverage"] == 0.75
    assert res["covered_samples"] == 3
    assert res["total_samples"] == 4
