
from pathlib import Path
from collections import Counter
from PIL import Image

# Project directory: this script's parent folder
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "NGLD" / "NGLD" / "Grapes Disease Dataset"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    if not DATASET_DIR.exists():
        print("ERROR: Dataset folder not found:")
        print(DATASET_DIR)
        print("\nPlease check the folder path.")
        return

    print("Dataset folder:", DATASET_DIR)
    print("=" * 60)

    total_images = 0
    unreadable_images = []
    class_counts = Counter()
    image_dimensions = Counter()
    seen_hashes = {}
    exact_duplicates = []

    import hashlib

    for class_dir in sorted(DATASET_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        count = 0

        for image_path in class_dir.rglob("*"):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            count += 1
            total_images += 1

            try:
                # Verify that the image can be opened
                with Image.open(image_path) as image:
                    image.verify()

                # Hash file contents to find exact duplicates
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()

                if digest in seen_hashes:
                    exact_duplicates.append(
                        (str(seen_hashes[digest]), str(image_path))
                    )
                else:
                    seen_hashes[digest] = image_path

                # Read dimensions after verification
                with Image.open(image_path) as image:
                    image_dimensions[image.size] += 1

            except Exception as error:
                unreadable_images.append((str(image_path), str(error)))

        class_counts[class_dir.name] = count
        print(f"{class_dir.name}: {count} images")

    print("\n" + "=" * 60)
    print("TOTAL IMAGES:", total_images)
    print("READABLE DIMENSIONS:", sum(image_dimensions.values()))
    print("UNREADABLE IMAGES:", len(unreadable_images))
    print("EXACT DUPLICATE FILES AFTER FIRST COPY:",
          len(exact_duplicates))

    print("\nMost common image dimensions:")
    for dimensions, count in image_dimensions.most_common(10):
        print(f"{dimensions}: {count}")

    report_path = PROJECT_DIR / "ngld_audit_report.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write(f"Dataset folder: {DATASET_DIR}\n")
        report.write(f"Total images: {total_images}\n")
        report.write(f"Unreadable images: {len(unreadable_images)}\n")
        report.write(f"Exact duplicate copies: {len(exact_duplicates)}\n")
        report.write("\nClass counts:\n")

        for name, count in sorted(class_counts.items()):
            report.write(f"{name}: {count}\n")

        report.write("\nUnreadable image details:\n")
        for path, error in unreadable_images:
            report.write(f"{path}\nError: {error}\n\n")

        report.write("\nExact duplicate pairs:\n")
        for first, second in exact_duplicates:
            report.write(f"{first}\n  = {second}\n")

    print("\nReport saved to:", report_path)

if __name__ == "__main__":
    main()
