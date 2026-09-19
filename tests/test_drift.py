import numpy as np
import pandas as pd
import pytest
from src.atlasml.drift.detector import detect_feature_drift, detect_performance_drift


def test_detect_feature_drift():
    np.random.seed(42)
    reference = pd.Series(np.random.normal(loc=0.0, scale=1.0, size=1000))
    current_same = pd.Series(np.random.normal(loc=0.0, scale=1.0, size=1000))
    current_shifted = pd.Series(np.random.normal(loc=2.0, scale=1.0, size=1000))

    # Same distribution -> no drift
    res_same = detect_feature_drift(reference, current_same, alpha=0.05)
    assert not res_same["drifted"]
    assert res_same["p_value"] > 0.05

    # Shifted distribution -> drift detected
    res_shifted = detect_feature_drift(reference, current_shifted, alpha=0.05)
    assert res_shifted["drifted"]
    assert res_shifted["p_value"] < 0.05


def test_detect_performance_drift():
    # Normal variation -> no drift
    res_normal = detect_performance_drift(reference_mae=5.0, current_mae=5.5, threshold_pct=20.0)
    assert not res_normal["drifted"]
    assert pytest.approx(res_normal["pct_increase"], 0.1) == 10.0

    # Significant degradation -> drift detected
    res_degraded = detect_performance_drift(reference_mae=5.0, current_mae=7.0, threshold_pct=20.0)
    assert res_degraded["drifted"]
    assert pytest.approx(res_degraded["pct_increase"], 0.1) == 40.0
