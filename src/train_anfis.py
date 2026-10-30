
from pathlib import Path
import itertools
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.dummy import DummyClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = (
    BASE_DIR / "environmental_data" / "anfis_mildew_condition_labels.csv"
)
OUTPUT_DIR = BASE_DIR / "model_outputs_anfis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "temp_5d",
    "rh_5d",
    "rh_min_5d",
    "rh_max_5d",
    "rain_5d",
    "wet_days_5d",
    "dpd_night_5d",
]

TARGETS = ["powdery_label", "downy_label"]
CLASSES = [0, 1]


class CompactSugenoFIS:
    """Compact first-order Sugeno fuzzy model for binary classification."""

    def __init__(self, n_mfs=2, ridge=1.0):
        self.n_mfs = n_mfs
        self.ridge = ridge

    def _make_rules(self, n_features):
        return list(
            itertools.product(range(self.n_mfs), repeat=n_features)
        )

    def _memberships(self, X):
        z = (
            X[:, :, None] - self.centers_[None, :, :]
        ) / self.scales_[None, :, :]
        return np.exp(-0.5 * z**2)

    def _firing_strengths(self, X):
        membership = self._memberships(X)
        log_strengths = []

        for rule in self.rules_:
            log_strength = np.zeros(len(X))
            for j, mf in enumerate(rule):
                log_strength += np.log(
                    membership[:, j, mf] + 1e-12
                )
            log_strengths.append(log_strength)

        logs = np.asarray(log_strengths).T
        logs -= logs.max(axis=1, keepdims=True)
        strength = np.exp(logs)

        return strength / (
            strength.sum(axis=1, keepdims=True) + 1e-12
        )

    def _design_matrix(self, X):
        firing = self._firing_strengths(X)
        augmented = np.column_stack([X, np.ones(len(X))])

        return np.concatenate(
            [
                firing[:, r:r + 1] * augmented
                for r in range(len(self.rules_))
            ],
            axis=1,
        )

    def fit(self, X, y):
        n_features = X.shape[1]
        self.classes_ = np.array(CLASSES)

        self.centers_ = np.stack([
            np.quantile(X[:, j], [0.25, 0.75])
            for j in range(n_features)
        ])

        self.scales_ = np.stack([
            np.full(
                self.n_mfs,
                max(float(np.std(X[:, j])), 0.1),
            )
            for j in range(n_features)
        ])

        self.rules_ = self._make_rules(n_features)
        design = self._design_matrix(X)

        Y = np.column_stack([
            (y == c).astype(float) for c in CLASSES
        ])

        identity = np.eye(design.shape[1])
        self.consequents_ = np.linalg.solve(
            design.T @ design + self.ridge * identity,
            design.T @ Y,
        )
        return self

    def predict_proba(self, X):
        raw = self._design_matrix(X) @ self.consequents_
        raw = np.clip(raw, 0.0, None)
        row_sums = raw.sum(axis=1, keepdims=True)

        return np.divide(
            raw,
            row_sums,
            out=np.full_like(raw, 1.0 / len(CLASSES)),
            where=row_sums > 1e-12,
        )

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "positive_class_recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_true, y_pred, labels=CLASSES
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=CLASSES,
            target_names=["label_0", "label_1"],
            zero_division=0,
            output_dict=True,
        ),
    }


def main():
    df = pd.read_csv(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    required = FEATURES + TARGETS + ["date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Missing values found in model columns.")

    for target in TARGETS:
        if not set(df[target].unique()).issubset({0, 1}):
            raise ValueError(f"{target} must contain only 0 and 1.")

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES].astype(float))
    X_val = scaler.transform(val[FEATURES].astype(float))
    X_test = scaler.transform(test[FEATURES].astype(float))

    all_results = {
        "dataset_records": int(n),
        "train_records": int(len(train)),
        "validation_records": int(len(val)),
        "test_records": int(len(test)),
        "train_date_range": [
            str(train.date.min().date()),
            str(train.date.max().date()),
        ],
        "validation_date_range": [
            str(val.date.min().date()),
            str(val.date.max().date()),
        ],
        "test_date_range": [
            str(test.date.min().date()),
            str(test.date.max().date()),
        ],
        "features": FEATURES,
        "excluded_features": ["powdery_idx", "downy_idx", "condition_class"],
        "models": {},
        "caveat": (
            "Labels are generated from weather-favourability rules, not "
            "independently confirmed disease observations. Results measure "
            "learning of generated labels, not validated field disease prediction."
        ),
    }

    all_predictions = pd.DataFrame({
        "date": test["date"].dt.strftime("%Y-%m-%d").to_numpy()
    })

    for target in TARGETS:
        y_train = train[target].astype(int).to_numpy()
        y_val = val[target].astype(int).to_numpy()
        y_test = test[target].astype(int).to_numpy()

        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        baseline_pred = baseline.predict(X_test)

        model = CompactSugenoFIS(n_mfs=2, ridge=1.0)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        test_prob = model.predict_proba(X_test)[:, 1]

        all_results["models"][target] = {
            "target_counts": {
                str(k): int(v)
                for k, v in df[target].value_counts().sort_index().items()
            },
            "validation": evaluate(y_val, val_pred),
            "test_anfis": evaluate(y_test, test_pred),
            "test_majority_baseline": evaluate(y_test, baseline_pred),
        }

        all_predictions[f"{target}_actual"] = y_test
        all_predictions[f"{target}_predicted"] = test_pred
        all_predictions[f"{target}_positive_probability"] = test_prob

        print(f"\n{'=' * 55}")
        print(f"Target: {target}")
        print("Validation:", all_results["models"][target]["validation"])
        print("ANFIS test:", all_results["models"][target]["test_anfis"])
        print(
            "Majority baseline:",
            all_results["models"][target]["test_majority_baseline"],
        )

    results_path = OUTPUT_DIR / "binary_anfis_evaluation.json"
    predictions_path = OUTPUT_DIR / "binary_anfis_test_predictions.csv"

    results_path.write_text(json.dumps(all_results, indent=2))
    all_predictions.to_csv(predictions_path, index=False)

    print("\nSaved evaluation:", results_path)
    print("Saved predictions:", predictions_path)
    print("Test period:", all_results["test_date_range"])


if __name__ == "__main__":
    main()
