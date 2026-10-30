
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

# ---------------- CONFIGURATION ----------------

PROJECT_DIR = Path(__file__).resolve().parent
SPLIT_CSV = PROJECT_DIR / "ngld_grouped_split.csv"
OUTPUT_DIR = PROJECT_DIR / "model_outputs_grouped"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42

tf.keras.utils.set_random_seed(SEED)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------- LOAD MANIFEST ----------------

df = pd.read_csv(SPLIT_CSV)

required = {"path", "label", "split", "sha256"}
if not required.issubset(df.columns):
    raise ValueError(f"Missing columns: {required - set(df.columns)}")

class_names = sorted(df["label"].unique())
class_to_index = {name: i for i, name in enumerate(class_names)}

print("Classes:", class_to_index)
print("\nImages by split:")
print(pd.crosstab(df["label"], df["split"]))

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

# Verify files still exist.
for subset_name, subset in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
]:
    missing = [p for p in subset["path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{subset_name}: {len(missing)} image files missing. "
            f"Example: {missing[0]}"
        )

# ---------------- IMAGE PIPELINE ----------------

def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    # MobileNetV2 preprocessing expects pixel values in [0, 255].
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image, label

def make_dataset(frame, training=False):
    paths = frame["path"].to_numpy()
    labels = frame["label"].map(class_to_index).to_numpy(dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(frame),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# ---------------- CLASS WEIGHTS ----------------

train_labels = train_df["label"].map(class_to_index).to_numpy()
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=train_labels,
)
class_weights = {i: float(w) for i, w in enumerate(weights)}

print("\nClass weights:", class_weights)

# ---------------- BUILD MODEL ----------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax",
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------- TRAINING ----------------

best_model_path = OUTPUT_DIR / "mobilenetv2_ngld_best.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(
        OUTPUT_DIR / "training_history.csv",
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

# Save the final in-memory model too.
model.save(OUTPUT_DIR / "mobilenetv2_ngld_final.keras")

# ---------------- PLOT TRAINING CURVES ----------------

history_df = pd.DataFrame(history.history)
history_df.to_csv(OUTPUT_DIR / "training_curves_data.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history_df["accuracy"], label="Training")
axes[0].plot(history_df["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history_df["loss"], label="Training")
axes[1].plot(history_df["val_loss"], label="Validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
plt.close(fig)

# ---------------- FINAL TEST EVALUATION ----------------
# Evaluate the selected best checkpoint only once.

best_model = tf.keras.models.load_model(best_model_path)

y_true = []
y_probabilities = []

for images, labels in test_ds:
    probabilities = best_model.predict(images, verbose=0)
    y_true.extend(labels.numpy().tolist())
    y_probabilities.extend(probabilities.tolist())

y_true = np.asarray(y_true)
y_probabilities = np.asarray(y_probabilities)
y_pred = np.argmax(y_probabilities, axis=1)

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(OUTPUT_DIR / "classification_report.csv")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(class_names)),
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names,
)
cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

fig, ax = plt.subplots(figsize=(7, 6))
image = ax.imshow(cm, cmap="Blues")
ax.set_title("NGLD Test Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
ax.set_yticks(range(len(class_names)), class_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")

fig.colorbar(image, ax=ax)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close(fig)

summary = {
    "test_images": int(len(y_true)),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    "class_names": class_names,
    "class_to_index": class_to_index,
}

with (OUTPUT_DIR / "evaluation_summary.json").open("w") as f:
    json.dump(summary, f, indent=2)

print("\nTEST RESULTS")
print(json.dumps(summary, indent=2))
print("\nPer-class report:")
print(classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0,
))
print("\nAll results saved in:", OUTPUT_DIR)
