
from pathlib import Path
from itertools import combinations
import pandas as pd
import imagehash
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
SPLIT_FILE = PROJECT_DIR / "ngld_dataset_split.csv"
OUTPUT_FILE = PROJECT_DIR / "near_duplicate_candidates.csv"

# Smaller distances indicate more visually similar images.
# This is a screening threshold, not proof of duplication.
MAX_DISTANCE = 4

df = pd.read_csv(SPLIT_FILE)

required = {"path", "label", "split", "sha256"}
if not required.issubset(df.columns):
    raise ValueError(f"Missing columns: {required - set(df.columns)}")

# The manifest should contain one row per unique exact image.
if df["sha256"].duplicated().any():
    raise ValueError("Duplicate SHA-256 values found in split manifest.")

# Compute perceptual hashes once per image.
records = []

for i, row in enumerate(df.itertuples(index=False), start=1):
    path = Path(row.path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as image:
        phash = imagehash.phash(image.convert("RGB"))

    records.append({
        "path": str(path),
        "label": row.label,
        "split": row.split,
        "sha256": row.sha256,
        "phash": phash,
    })

    if i % 250 == 0 or i == len(df):
        print(f"Hashed {i}/{len(df)} images")

# Compare images from different splits only.
candidates = []
split_pairs_checked = 0

for first, second in combinations(records, 2):
    if first["split"] == second["split"]:
        continue

    split_pairs_checked += 1
    distance = first["phash"] - second["phash"]

    if distance <= MAX_DISTANCE:
        candidates.append({
            "path_1": first["path"],
            "label_1": first["label"],
            "split_1": first["split"],
            "path_2": second["path"],
            "label_2": second["label"],
            "split_2": second["split"],
            "phash_distance": distance,
            "same_label": first["label"] == second["label"],
        })

columns = [
    "path_1", "label_1", "split_1",
    "path_2", "label_2", "split_2",
    "phash_distance", "same_label",
]

result = pd.DataFrame(candidates, columns=columns)

if not result.empty:
    result = result.sort_values(
        ["phash_distance", "same_label"]
    )

result.to_csv(OUTPUT_FILE, index=False)

print("\nNear-duplicate screening complete")
print("Images hashed:", len(records))
print("Cross-split pairs compared:", split_pairs_checked)
print("Candidate pairs:", len(result))
print("Same-label candidate pairs:", (
    int(result["same_label"].sum()) if not result.empty else 0
))
print("Different-label candidate pairs:", (
    int((~result["same_label"]).sum()) if not result.empty else 0
))
print("Report saved to:", OUTPUT_FILE)

if not result.empty:
    print("\nClosest candidates:")
    print(result.head(20).to_string(index=False))
else:
    print("No pairs met the selected threshold.")
