
from pathlib import Path
import pandas as pd

path = Path("environmental_data/environmental_cleaned.csv")
df = pd.read_csv(path, parse_dates=["date"])

print("========== TEMPERATURE CONSISTENCY ==========")

temp_checks = {
    "Average below minimum": df["air_temp"] < df["air_temp_min"],
    "Average above maximum": df["air_temp"] > df["air_temp_max"],
    "Minimum above maximum": df["air_temp_min"] > df["air_temp_max"],
}

for name, condition in temp_checks.items():
    print(f"{name}: {int(condition.sum())} rows")

print("\n========== HUMIDITY CONSISTENCY ==========")

humidity_checks = {
    "Average below minimum": df["air_hum"] < df["air_hum_min"],
    "Average above maximum": df["air_hum"] > df["air_hum_max"],
    "Minimum above maximum": df["air_hum_min"] > df["air_hum_max"],
}

for name, condition in humidity_checks.items():
    print(f"{name}: {int(condition.sum())} rows")

print("\n========== PHYSICAL RANGE CHECKS ==========")

for col in ["air_hum", "air_hum_min", "air_hum_max"]:
    invalid = (df[col] < 0) | (df[col] > 100)
    print(f"{col} outside 0-100%: {int(invalid.sum())}")

for col in ["air_temp", "air_temp_min", "air_temp_max", "pluvm"]:
    print(f"{col}: minimum={df[col].min()}, maximum={df[col].max()}")

print("\n========== HIGHEST RAINFALL VALUES ==========")
print(
    df.nlargest(10, "pluvm")[
        ["date", "pluvm", "air_temp", "air_hum"]
    ].to_string(index=False)
)

print("\n========== POSSIBLE UNIT CHECK ==========")
print("Rainfall column name: pluvm")
print("Rainfall units must be confirmed from the original dataset documentation.")
print("Do not assume pluvm is measured in millimetres without verification.")

print("\nWeather quality checks complete.")
