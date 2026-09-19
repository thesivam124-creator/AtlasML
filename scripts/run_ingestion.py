import logging
from pathlib import Path

from atlasml.ingestion.load_raw import load_raw_to_bronze


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

raw_path = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "yellow_tripdata_2026-01.parquet"
)

bronze_path = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "yellow_tripdata_2026-01.parquet"
)


if __name__ == "__main__":
    load_raw_to_bronze(
        raw_path=str(raw_path),
        bronze_path=str(bronze_path),
    )