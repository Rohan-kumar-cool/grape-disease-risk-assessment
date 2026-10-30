import pandas as pd

df = pd.read_csv("environmental_data/environmental_cleaned.csv")
df["date"] = pd.to_datetime(df["date"])

weather_cols = [
    "air_hum", "air_hum_min", "air_hum_max",
    "air_temp", "air_temp_min", "air_temp_max", "pluvm"
]
disease_cols = ["down_mildew", "powdery_mildew"]

print("WEATHER COMPARISON BY DISEASE INDICATOR")
for disease in disease_cols:
    print(f"\n{'=' * 55}")
    print(f"Disease indicator: {disease}")

    summary = df.groupby(disease)[weather_cols].mean().T
    summary.columns = ["Indicator = 0", "Indicator = 1"]
    summary["Difference (1 - 0)"] = (
        summary["Indicator = 1"] - summary["Indicator = 0"]
    )
    print(summary.round(3))

print("\nDISEASE INDICATOR COUNTS BY YEAR")
df["year"] = df["date"].dt.year
print(df.groupby("year")[disease_cols].sum())

print("\nMONTHLY DISEASE INDICATOR COUNTS")
df["month"] = df["date"].dt.month
print(df.groupby("month")[disease_cols].sum())

df.groupby("year")[disease_cols].sum().to_csv(
    "environmental_data/disease_counts_by_year.csv"
)

print("\nSaved yearly counts to environmental_data/disease_counts_by_year.csv")
