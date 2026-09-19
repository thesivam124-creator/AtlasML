import pandas as pd


# Required columns for the first validation layer.
REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "PULocationID",
    "passenger_count",
]


def validate_trips(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate trip records and split them into:

    valid_rows
    rejected_rows

    Rejected rows contain a reject_reason column.
    """

    df = df.copy()

    # Store validation failure reasons for every row.
    reasons = pd.Series("", index=df.index, dtype="object")

    # ---------------------------------------------------------
    # 1. Check required columns
    # ---------------------------------------------------------
    missing_cols = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    # ---------------------------------------------------------
    # 2. Validate pickup timestamp
    # ---------------------------------------------------------
    bad_timestamp = df["tpep_pickup_datetime"].isna()

    reasons.loc[bad_timestamp] += "invalid_timestamp;"

    # ---------------------------------------------------------
    # 3. Validate pickup zone
    # ---------------------------------------------------------
    bad_zone = (
        df["PULocationID"].isna()
        | (df["PULocationID"] <= 0)
    )

    reasons.loc[bad_zone] += "invalid_zone;"

    # ---------------------------------------------------------
    # 4. Detect exact duplicate rows
    # ---------------------------------------------------------
    duplicates = df.duplicated()

    reasons.loc[duplicates] += "duplicate;"

    # ---------------------------------------------------------
    # 5. Split valid and rejected rows
    # ---------------------------------------------------------
    is_rejected = reasons != ""

    valid = df.loc[~is_rejected].copy()

    rejected = df.loc[is_rejected].copy()

    rejected["reject_reason"] = reasons.loc[is_rejected]

    return valid, rejected