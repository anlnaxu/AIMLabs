## Buffy Pose Training Data Prep

This folder contains a small utility to turn the `buffy_pose_classes_v10` dataset into labeled samples for training a pose classifier.

### Dataset assumption

- The raw dataset is located at: `/Users/anlan.xu/Downloads/buffy_pose_classes_v10`
- Example pose images are under: `poses_gt_example_images/<episode>/*.jpg`
- Filenames encode the pose in the suffix, e.g.:
  - `000920_rest.jpg` → label `rest`
  - `021697_folded.jpg` → label `folded`
  - `022365_hips.jpg` → label `hips`

### Script: `prepare_buffy_data.py`

This script:

- Recursively scans `poses_gt_example_images`
- Extracts the pose label from each filename (text after the last underscore and before `.jpg`)
- Writes a CSV file `buffy_pose_labels.csv` in this folder with:
  - `image_path`: absolute path to the image file
  - `label`: pose label (`rest`, `folded`, `hips`, etc.)

### How to run

From your project root:

```bash
cd /Users/anlan.xu/AIMLabsProject
python buffy_pose_training/prepare_buffy_data.py
```

If everything is set up correctly, you should see output similar to:

```text
Wrote N labeled samples to /Users/anlan.xu/AIMLabsProject/buffy_pose_training/buffy_pose_labels.csv
```

You can then use `buffy_pose_labels.csv` as input to your training pipeline (PyTorch, TensorFlow, scikit-learn, etc.) by loading image paths and labels from the CSV.

### Setup / dependencies

This project uses PyTorch + torchvision and Pillow for image handling. Install the required Python packages for training:

```bash
# Recommended: pick the correct PyTorch wheel for your environment at https://pytorch.org
# Example (CPU-only pip install):
python3 -m pip install -r requirements.txt
```

If you have CUDA installed and want a GPU-enabled PyTorch, install PyTorch from the official instructions on https://pytorch.org before using the `requirements.txt` above.


