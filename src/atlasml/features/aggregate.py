import pandas as pd


def aggregate_demand(
    df: pd.DataFrame,
    freq: str = "15min",
) -> pd.DataFrame:
    """
    Aggregate trip-level data into per-zone,
    per-time-bucket pickup counts.

    Parameters
    ----------
    df : pd.DataFrame
        Valid trip-level records.

    freq : str
        Pandas time frequency. Default is 15 minutes.

    Returns
    -------
    pd.DataFrame
        One row per (PULocationID, time_bucket).
    """

    df = df.copy()

    # Make sure pickup timestamp is datetime.
    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"],
        errors="coerce",
    )

    # Create 15-minute time buckets.
    df["time_bucket"] = df["tpep_pickup_datetime"].dt.floor(freq)

    # Count pickups for each zone and time bucket.
    demand = (
        df.groupby(
            ["PULocationID", "time_bucket"]
        )
        .size()
        .reset_index(name="pickup_count")
        .sort_values(
            ["PULocationID", "time_bucket"]
        )
        .reset_index(drop=True)
    )

    return demand