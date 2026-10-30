
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("environmental_data/environmental_cleaned.csv")
OUTPUT_PATH = Path("outputs/metrics/environmental_baseline_results.csv")

weather_cols = [
    "air_hum",
    "air_hum_min",
    "air_hum_max",
    "air_temp",
    "air_temp_min",
    "air_temp_max",
    "pluvm",
]

disease_cols = ["down_mildew", "powdery_mildew"]

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

for col in weather_cols + disease_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

split_index = int(len(df) * 0.8)
train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

print("Chronological baseline experiment")
print("Total rows:", len(df))
print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Training dates:", train["date"].min(), "to", train["date"].max())
print("Testing dates:", test["date"].min(), "to", test["date"].max())

results = []

for disease in disease_cols:
    train_mask = train[disease].notna()
    test_mask = test[disease].notna()

    X_train = train.loc[train_mask, weather_cols]
    y_train = train.loc[train_mask, disease].astype(int)

    X_test = test.loc[test_mask, weather_cols]
    y_test = test.loc[test_mask, disease].astype(int)

    if y_train.nunique() < 2:
        print(f"\nSkipping {disease}: training labels contain only one class.")
        continue

    models = {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "weather_logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        ),
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        row = {
            "disease": disease,
            "model": model_name,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_positive_rate": y_test.mean(),
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "macro_f1": f1_score(
                y_test, predictions, average="macro", zero_division=0
            ),
        }

        results.append(row)

        print(f"\n{disease} | {model_name}")
        for metric in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "test_positive_rate",
        ]:
            print(f"{metric}: {row[metric]:.4f}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

print("\nSaved results to:", OUTPUT_PATH)
