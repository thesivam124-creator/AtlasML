import pandas as pd

from src.atlasml.validation.validators import validate_trips


def test_rejects_invalid_zone():

    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                [
                    "2024-01-01 10:00",
                    "2024-01-01 11:00",
                ]
            ),
            "PULocationID": [10, -1],
            "passenger_count": [1, 1],
        }
    )

    valid, rejected = validate_trips(df)

    assert len(valid) == 1
    assert len(rejected) == 1

    assert "invalid_zone" in rejected["reject_reason"].iloc[0]