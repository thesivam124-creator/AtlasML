import pandas as pd
import matplotlib.pyplot as plt

from src.atlasml.features.aggregate import aggregate_demand


df = pd.read_parquet(
    "data/bronze/yellow_tripdata_2026-01.parquet"
)

demand = aggregate_demand(df)

zone_142 = demand[
    demand["PULocationID"] == 142
]

plt.figure(figsize=(14, 5))

plt.plot(
    zone_142["time_bucket"],
    zone_142["pickup_count"],
)

plt.title("Taxi Demand — Zone 142")
plt.xlabel("Time")
plt.ylabel("Pickup Count")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()