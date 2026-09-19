import pandas as pd
from src.atlasml.features.build_features import add_lag_and_rolling_features

def test_lag_feature_does_not_see_current_row():
    df = pd.DataFrame({
        "PULocationID": [1, 1, 1],
        "time_bucket": pd.date_range("2024-01-01", periods=3, freq="15min"),
        "pickup_count": [10, 20, 30],
    })
    out = add_lag_and_rolling_features(df)
    # the lag_1 value for row 3 must equal row 2's count, NOT row 3's own count
    assert out["lag_1"].iloc[2] == 20