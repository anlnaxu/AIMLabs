import os
import csv
from pathlib import Path
from typing import List, Tuple

"""
Preprocess Buffy pose dataset into a simple image/label CSV for training.

Assumptions:
- Source dataset lives in /Users/anlan.xu/Downloads/buffy_pose_classes_v10
- Training images for poses are in poses_gt_example_images/*/*.jpg
- Filenames look like: 000920_rest.jpg, 021697_folded.jpg, 022365_hips.jpg
  where the part after the underscore (before .jpg) is the pose label.

This script will:
- Scan all *.jpg files under poses_gt_example_images
- Infer label from filename suffix
- Write buffy_pose_labels.csv with columns: image_path,label
  where image_path is absolute path to the image.
"""


DATA_ROOT = Path("/Users/anlan.xu/Downloads/buffy_pose_classes_v10")
POSE_EXAMPLE_DIR = DATA_ROOT / "poses_gt_example_images"
OUTPUT_CSV = Path(__file__).resolve().parent / "buffy_pose_labels.csv"


def extract_label_from_filename(filename: str) -> str:
    """
    Given a filename like '000920_rest.jpg', return 'rest'.
    """
    stem = Path(filename).stem  # e.g., '000920_rest'
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse label from filename: {filename}")
    return parts[-1]


def collect_image_label_pairs(root: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.lower().endswith(".jpg"):
                continue
            # Skip thumbnails or system files if any sneak through
            if fname.lower().endswith(".db"):
                continue
            img_path = Path(dirpath) / fname
            label = extract_label_from_filename(fname)
            pairs.append((str(img_path.resolve()), label))
    return pairs


def write_csv(pairs: List[Tuple[str, str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        for img_path, label in pairs:
            writer.writerow([img_path, label])


def main() -> None:
    if not POSE_EXAMPLE_DIR.exists():
        raise SystemExit(
            f"Expected directory not found: {POSE_EXAMPLE_DIR}. "
            "Verify that buffy_pose_classes_v10 is at the given path."
        )

    pairs = collect_image_label_pairs(POSE_EXAMPLE_DIR)
    if not pairs:
        raise SystemExit(f"No JPEG images found under {POSE_EXAMPLE_DIR}")

    write_csv(pairs, OUTPUT_CSV)
    print(f"Wrote {len(pairs)} labeled samples to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


