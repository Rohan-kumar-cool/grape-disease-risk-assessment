
from pathlib import Path
from collections import defaultdict
import hashlib
import csv

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "NGLD" / "NGLD" / "Grapes Disease Dataset"
OUTPUT_PATH = PROJECT_DIR / "ngld_clean_manifest.csv"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Group images by exact file content.
groups = defaultdict(list)

for class_dir in sorted(DATASET_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    for path in sorted(class_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        groups[digest].append({
            "path": path,
            "label": class_dir.name,
        })

conflicts = []
manifest = []

for digest, files in groups.items():
    labels = {item["label"] for item in files}

    if len(labels) != 1:
        conflicts.append((digest, files))
        continue

    # Choose a deterministic representative file.
    representative = min(files, key=lambda item: str(item["path"]))

    manifest.append({
        "sha256": digest,
        "filename": representative["path"].name,
        "path": str(representative["path"]),
        "label": representative["label"],
        "original_copies": len(files),
    })

if conflicts:
    print("ERROR: Found conflicting duplicate groups.")
    print("No manifest was written. Investigate these groups first.")
    for digest, files in conflicts:
        print("\nHash:", digest)
        for item in files:
            print(item["label"], item["path"])
    raise SystemExit(1)

with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sha256",
            "filename",
            "path",
            "label",
            "original_copies",
        ],
    )
    writer.writeheader()
    writer.writerows(sorted(manifest, key=lambda item: item["sha256"]))

print("Original image files:", sum(len(files) for files in groups.values()))
print("Unique image contents:", len(manifest))
print("Excluded conflicting groups:", len(conflicts))
print("Manifest saved to:", OUTPUT_PATH)

print("\nClass counts after exact deduplication:")
counts = defaultdict(int)
for item in manifest:
    counts[item["label"]] += 1

for label, count in sorted(counts.items()):
    print(f"{label}: {count}")
