
from pathlib import Path
import json
import itertools

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.dummy import DummyClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "environmental_data" / "anfis_mildew_condition_labels.csv"
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
TARGET = "condition_class"
SEED = 42
CLASSES = [0, 1, 2, 3]


class CompactSugenoFIS:
    """Compact first-order Sugeno fuzzy model for multiclass targets."""

    def __init__(self, n_mfs=2, ridge=1.0):
        self.n_mfs = n_mfs
        self.ridge = ridge

    def _make_rules(self, n_features):
        # Two fuzzy sets per feature gives 2**7 = 128 rules.
        return list(itertools.product(range(self.n_mfs), repeat=n_features))

    def _memberships(self, X):
        z = (
            X[:, :, None] - self.centers_[None, :, :]
        ) / self.scales_[None, :, :]
        return np.exp(-0.5 * z**2)

    def _firing_strengths(self, X):
        membership = self._memberships(X)
        log_strengths = []

        # Log-space multiplication avoids numerical underflow.
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
        return strength / (strength.sum(axis=1, keepdims=True) + 1e-12)

    def _design_matrix(self, X):
        firing = self._firing_strengths(X)
        augmented = np.column_stack([X, np.ones(len(X))])
        return np.concatenate(
            [firing[:, r:r + 1] * augmented
             for r in range(len(self.rules_))],
            axis=1,
        )

    def fit(self, X, y):
        n_features = X.shape[1]
        self.classes_ = np.array(CLASSES)

        # Initialize membership centers from training data only.
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

        # One-hot targets; fit one Sugeno output per class.
        Y = np.column_stack([(y == c).astype(float) for c in CLASSES])
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


def main():
    df = pd.read_csv(DATA_FILE, sep=None, engine="python")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("Missing values found in model columns.")
    if not set(df[TARGET].unique()).issubset(set(CLASSES)):
        raise ValueError("Unexpected target classes.")

    # Chronological 70% train, 15% validation, 15% test.
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    X_train = train[FEATURES].astype(float).to_numpy()
    X_val = val[FEATURES].astype(float).to_numpy()
    X_test = test[FEATURES].astype(float).to_numpy()

    y_train = train[TARGET].astype(int).to_numpy()
    y_val = val[TARGET].astype(int).to_numpy()
    y_test = test[TARGET].astype(int).to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Baseline: most frequent training class.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    # Train candidate compact fuzzy model.
    model = CompactSugenoFIS(n_mfs=2, ridge=1.0)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)

    results = {
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
        "excluded_features": ["powdery_idx", "downy_idx"],
        "target": TARGET,
        "target_counts": {
            str(k): int(v)
            for k, v in df[TARGET].value_counts().sort_index().items()
        },
        "validation": {
            "accuracy": float(accuracy_score(y_val, val_pred)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_val, val_pred)
            ),
            "macro_f1": float(
                f1_score(y_val, val_pred, average="macro", zero_division=0)
            ),
        },
        "test_anfis": {
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, test_pred)
            ),
            "macro_f1": float(
                f1_score(y_test, test_pred, average="macro", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(
                y_test, test_pred, labels=CLASSES
            ).tolist(),
            "classification_report": classification_report(
                y_test, test_pred, labels=CLASSES,
                zero_division=0, output_dict=True
            ),
        },
        "test_majority_baseline": {
            "accuracy": float(accuracy_score(y_test, baseline_pred)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, baseline_pred)
            ),
            "macro_f1": float(
                f1_score(y_test, baseline_pred, average="macro", zero_division=0)
            ),
        },
        "caveat": (
            "Labels are generated from weather-favourability rules, not "
            "independently confirmed disease observations. This is a "
            "condition-label learning experiment, not validated disease prediction."
        ),
    }

    (OUTPUT_DIR / "anfis_evaluation.json").write_text(
        json.dumps(results, indent=2)
    )

    predictions = pd.DataFrame({
        "date": test["date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "actual_class": y_test,
        "predicted_class": test_pred,
    })
    for i, c in enumerate(CLASSES):
        predictions[f"score_class_{c}"] = test_prob[:, i]

    predictions.to_csv(
        OUTPUT_DIR / "anfis_test_predictions.csv", index=False
    )

    print("\n=== Four-Class Sugeno Fuzzy Model ===")
    print(f"Records: {n} | Train: {len(train)} | "
          f"Validation: {len(val)} | Test: {len(test)}")
    print(f"Fuzzy rules: {len(model.rules_)}")
    print(f"Test period: {test.date.min().date()} to {test.date.max().date()}")

    print("\nValidation metrics:")
    print(json.dumps(results["validation"], indent=2))

    print("\nTest ANFIS metrics:")
    print(json.dumps({
        k: v for k, v in results["test_anfis"].items()
        if k != "classification_report"
    }, indent=2))

    print("\nMajority baseline:")
    print(json.dumps(results["test_majority_baseline"], indent=2))

    print("\nClassification report:")
    print(classification_report(
        y_test, test_pred, labels=CLASSES, zero_division=0
    ))

    print(f"Saved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
