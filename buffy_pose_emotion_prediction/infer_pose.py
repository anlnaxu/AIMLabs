import argparse
import json
import pandas as pd
import cv2 as cv
import numpy as np
from pathlib import Path
from deepface import DeepFace

from PIL import Image

import torch
from torchvision import models, transforms


THIS_DIR = Path(__file__).resolve().parent
MODEL_PATH = THIS_DIR / "buffy_pose_classifier.pth"
LABEL_MAP_PATH = THIS_DIR / "buffy_pose_label_map.json"
test_csv = THIS_DIR / "buffy_pose_testing.csv"
output_csv = THIS_DIR / "buffy_pose_predictions.csv"


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

    return pred_label, pred_conf


def main():
    df = pd.read_csv(test_csv)
    
    pred_pose = []
    pred_emotion = []
    for path in df["image_path"]:
    	image_path = Path(path)
    	pred_label, pred_conf = predict(image_path)
    	pred_pose.append(pred_label)
    	
    	image = cv.imread(path)
    	hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    	mask = cv.inRange(hsv_image, np.array([50, 100, 100]), np.array([90, 255, 255]))
    	contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    	best_contour = max(contours, key = cv.contourArea)
    	x,y,w,h = cv.boundingRect(best_contour)
    	boxed_image = image[y:y+h, x:x+w]
    	
    	result = DeepFace.analyze(
    		img_path=path,
    		actions=['emotion'],
    		enforce_detection=False,
    		detector_backend="skip"
		)
    	pred_emotion.append(result[0]["dominant_emotion"])
    df["predicted_pose"] = pred_pose
    df["predicted_emotion"] = pred_emotion
    df["label"] = df["label"].map({0: "folded", 1: "hips", 2: "rest"})
    df.to_csv(output_csv, index=False)
    print(f"Saved pose predictions to {output_csv}.")


if __name__ == "__main__":
    main()