from pathlib import Path
import pandas as pd

# Project root = atlasml/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

path = PROJECT_ROOT / "data" / "raw" / "yellow_tripdata_2026-01.parquet"

if not path.exists():
    raise FileNotFoundError(f"File not found: {path}")

df = pd.read_parquet(path)

print("=" * 80)
print("NYC TLC RAW DATA INSPECTION")
print("=" * 80)

print("\nFile:")
print(path)

print("\nShape:")
print(df.shape)

print("\nColumns:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i:2}. {col}")

print("\nDtypes:")
print(df.dtypes)

print("\nSample rows:")
print(df.head(3).to_string())

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nMemory usage:")
print(f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

datetime_candidates = [
    col for col in df.columns
    if "datetime" in col.lower() or "date" in col.lower()
]

print("\nPotential datetime columns:")
print(datetime_candidates)

for col in datetime_candidates:
    converted = pd.to_datetime(df[col], errors="coerce")

    print(f"\n{col}:")
    print("  Min:", converted.min())
    print("  Max:", converted.max())
    print("  Invalid:", converted.isna().sum())

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)