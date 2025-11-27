import argparse
import json
from pathlib import Path

from PIL import Image

import torch
from torchvision import models, transforms


THIS_DIR = Path(__file__).resolve().parent
MODEL_PATH = THIS_DIR / "buffy_pose_classifier.pth"
LABEL_MAP_PATH = THIS_DIR / "buffy_pose_label_map.json"


def get_transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def load_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def predict(image_path: Path):
    if not MODEL_PATH.exists() or not LABEL_MAP_PATH.exists():
        raise SystemExit(
            "Model or label map not found. Train the model first with "
            "train_pose_classifier.py"
        )

    with LABEL_MAP_PATH.open("r") as f:
        label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    model = load_model(num_classes=len(label_to_idx))

    transform = get_transform()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # add batch dim

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        pred_label = idx_to_label[pred_idx]
        pred_conf = float(probs[pred_idx].item())

    print(f"Image: {image_path}")
    print(f"Predicted pose: {pred_label} (confidence {pred_conf:.3f})")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classify a human pose as one of the Buffy poses "
            "('rest', 'folded', 'hips')."
        )
    )
    parser.add_argument("image", type=str, help="Path to an input image")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    predict(image_path)


if __name__ == "__main__":
    main()


