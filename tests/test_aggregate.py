import pandas as pd

from src.atlasml.features.aggregate import aggregate_demand


def test_aggregate_demand():

    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                [
                    "2026-01-01 10:01",
                    "2026-01-01 10:05",
                    "2026-01-01 10:14",
                    "2026-01-01 10:16",
                    "2026-01-01 10:20",
                ]
            ),
            "PULocationID": [142, 142, 142, 142, 143],
        }
    )

    demand = aggregate_demand(df)

    # Zone 142:
    # 10:00 bucket -> 3 pickups
    # 10:15 bucket -> 1 pickup
    # Zone 143:
    # 10:15 bucket -> 1 pickup

    assert len(demand) == 3

    zone_142 = demand[demand["PULocationID"] == 142]

    assert zone_142.iloc[0]["pickup_count"] == 3
    assert zone_142.iloc[1]["pickup_count"] == 1