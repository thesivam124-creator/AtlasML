import numpy as np
import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["time_bucket"].dt.hour
    df["day_of_week"] = df["time_bucket"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # cyclic encoding so the model knows hour 23 and hour 0 are close
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df

def add_lag_and_rolling_features(df: pd.DataFrame, group_col: str = "PULocationID") -> pd.DataFrame:
    df = df.sort_values([group_col, "time_bucket"]).copy()
    grp = df.groupby(group_col)["pickup_count"]

    # shift(1) ensures we only use PAST values — never the current row
    df["lag_1"] = grp.shift(1)
    df["lag_4"] = grp.shift(4)     # e.g., 1 hour ago if freq=15min
    df["rolling_mean_4"] = grp.shift(1).rolling(4).mean()
    df["rolling_std_4"] = grp.shift(1).rolling(4).std()
    return df