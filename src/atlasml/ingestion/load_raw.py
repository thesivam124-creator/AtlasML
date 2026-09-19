from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_raw_to_bronze(raw_path: str, bronze_path: str) -> pd.DataFrame:
    """
    Load a raw Parquet trip file and save an unmodified
    copy to the bronze layer.
    """

    raw_file = Path(raw_path)

    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {raw_file.resolve()}"
        )

    logger.info(f"Reading raw data from {raw_file}")

    df = pd.read_parquet(raw_file)

    logger.info(f"Loaded {len(df):,} rows")
    logger.info(f"Columns: {list(df.columns)}")

    bronze_file = Path(bronze_path)
    bronze_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(bronze_file, index=False)

    logger.info(f"Wrote bronze layer to {bronze_file.resolve()}")

    return df