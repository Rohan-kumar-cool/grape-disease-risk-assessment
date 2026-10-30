
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "environmental_data" / "anfis_mildew_condition_labels.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs_anfis"

sys.path.insert(0, str(BASE_DIR / "src"))
from train_anfis import CompactSugenoFIS, FEATURES, TARGETS

SEED = 42
MISSING_FEATURES = ["temp_5d", "rh_5d", "rain_5d", "wet_days_5d"]


def metrics(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_true, y_pred)), 4
        ),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            4,
        ),
    }


def main():
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    required = FEATURES + TARGETS + ["date"]
    if df[required].isna().any().any():
        raise ValueError("Original dataset has missing required values.")

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    test = df.iloc[val_end:].copy()

    # Fit all preprocessing and models using training records only.
    medians = train[FEATURES].median()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES].astype(float))
    X_test_complete = scaler.transform(test[FEATURES].astype(float))

    rng = np.random.default_rng(SEED)
    results = {
        "test_records": len(test),
        "test_date_range": [
            str(test["date"].min().date()),
            str(test["date"].max().date()),
        ],
        "imputation": "Training-set feature medians",
        "missingness_seed": SEED,
        "models": {},
        "caveat": (
            "Targets are rule-generated weather-favourability labels, "
            "not independently confirmed field disease observations."
        ),
    }

    predictions = pd.DataFrame({
        "date": test["date"].dt.strftime("%Y-%m-%d").to_numpy()
    })

    for target in TARGETS:
        y_train = train[target].astype(int).to_numpy()
        y_test = test[target].astype(int).to_numpy()

        model = CompactSugenoFIS(n_mfs=2, ridge=1.0)
        model.fit(X_train, y_train)

        # Reference: all seven environmental features are available.
        complete_pred = model.predict(X_test_complete)

        target_results = {
            "complete_seven_feature_input": metrics(y_test, complete_pred),
            "missing_input_scenarios": {},
        }

        predictions[f"{target}_actual"] = y_test
        predictions[f"{target}_complete_predicted"] = complete_pred

        # Scenario A: only the four app-requested fields are available.
        # The other three features are filled using training medians.
        four_input = test[FEATURES].copy()
        hidden = ["rh_min_5d", "rh_max_5d", "dpd_night_5d"]
        four_input[hidden] = medians[hidden]

        X_four = scaler.transform(four_input[FEATURES].astype(float))
        four_pred = model.predict(X_four)
        target_results["four_app_inputs_three_hidden_features_imputed"] = {
            **metrics(y_test, four_pred),
            "agreement_with_complete_model": round(
                float(np.mean(four_pred == complete_pred)), 4
            ),
        }
        predictions[f"{target}_four_input_predicted"] = four_pred

        # Scenarios B-E: randomly hide one or more of the four app inputs.

        # Test every non-empty combination of missing app inputs.
        from itertools import combinations

        for count in range(1, len(MISSING_FEATURES) + 1):
            for hidden_app_features in combinations(MISSING_FEATURES, count):
                scenario_df = test[FEATURES].copy()

                scenario_df[list(hidden_app_features)] = medians[
                    list(hidden_app_features)
                ].to_numpy()

                # Impute the three features not exposed in the app.
                hidden_internal = [
                    "rh_min_5d",
                    "rh_max_5d",
                    "dpd_night_5d",
                ]
                scenario_df[hidden_internal] = medians[
                    hidden_internal
                ].to_numpy()

                X_scenario = scaler.transform(
                    scenario_df[FEATURES].astype(float)
                )
                scenario_pred = model.predict(X_scenario)

                scenario_name = "missing_" + "_".join(hidden_app_features)

                target_results["missing_input_scenarios"][scenario_name] = {
                    **metrics(y_test, scenario_pred),
                    "agreement_with_complete_model": round(
                        float(np.mean(scenario_pred == complete_pred)), 4
                    ),
                }

                predictions[
                    f"{target}_{scenario_name}_predicted"
                ] = scenario_pred

       


        results["models"][target] = target_results

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "missing_input_evaluation.json"
    predictions_path = OUTPUT_DIR / "missing_input_predictions.csv"

    results_path.write_text(json.dumps(results, indent=2))
    predictions.to_csv(predictions_path, index=False)

    print(json.dumps(results, indent=2))
    print("\nSaved:", results_path)
    print("Saved:", predictions_path)


if __name__ == "__main__":
    main()
