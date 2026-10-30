
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(PROJECT_DIR / "near_duplicate_candidates.csv")

print("Total candidate pairs:", len(df))

print("\nPairs by class combination:")
print(
    df.assign(
        class_pair=df.apply(
            lambda r: " | ".join(sorted([r["label_1"], r["label_2"]])),
            axis=1,
        )
    )["class_pair"].value_counts()
)

print("\nPairs by split combination:")
print(
    df.assign(
        split_pair=df.apply(
            lambda r: " | ".join(sorted([r["split_1"], r["split_2"]])),
            axis=1,
        )
    )["split_pair"].value_counts()
)

print("\nPerceptual hash distance distribution:")
print(df["phash_distance"].value_counts().sort_index())

print("\nPairs with distance 0:", int((df["phash_distance"] == 0).sum()))
print("Pairs with distance <= 2:", int((df["phash_distance"] <= 2).sum()))
