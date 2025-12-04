import csv
import json
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict

from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms


THIS_DIR = Path(__file__).resolve().parent
LABELS_CSV = THIS_DIR / "buffy_pose_labels.csv"
MODEL_PATH = THIS_DIR / "buffy_pose_classifier.pth"
LABEL_MAP_PATH = THIS_DIR / "buffy_pose_label_map.json"


class BuffyPoseDataset(Dataset):
    def __init__(self, csv_path: Path, label_to_idx: Dict[str, int], transform=None):
        self.samples: List[Tuple[str, int]] = []
        self.transform = transform
        self.label_to_idx = label_to_idx

        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row["image_path"]
                label = row["label"]
                if label not in label_to_idx:
                    continue
                self.samples.append((img_path, label_to_idx[label]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label_idx = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label_idx


def build_label_mapping(csv_path: Path) -> Dict[str, int]:
    labels = set()
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.add(row["label"])
    label_list = sorted(labels)
    return {label: i for i, label in enumerate(label_list)}


def get_data_transforms():
    # Standard ImageNet normalization for pretrained models
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, val_transform


def create_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def main():
    if not LABELS_CSV.exists():
        raise SystemExit(
            f"Labels CSV not found at {LABELS_CSV}. "
            "Run prepare_buffy_data.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    label_to_idx = build_label_mapping(LABELS_CSV)
    print("Label mapping:", label_to_idx)

    train_transform, val_transform = get_data_transforms()

    full_dataset = BuffyPoseDataset(
        csv_path=LABELS_CSV, label_to_idx=label_to_idx, transform=train_transform
    )

    # simple train/val split
    train, val, test = 0.7, 0.2, 0.1
    val_size = int(len(full_dataset) * val)
    train_size = int(len(full_dataset) * train)
    test_size = len(full_dataset) - (val_size + train_size)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    pd.DataFrame([full_dataset.samples[i] for i in train_dataset.indices], columns = ["image_path", "label"]).to_csv(THIS_DIR / "buffy_pose_training.csv", index = False)
    pd.DataFrame([full_dataset.samples[i] for i in val_dataset.indices], columns = ["image_path", "label"]).to_csv(THIS_DIR / "buffy_pose_validation.csv", index = False)
    pd.DataFrame([full_dataset.samples[i] for i in test_dataset.indices], columns = ["image_path", "label"]).to_csv(THIS_DIR / "buffy_pose_testing.csv", index = False)

    # For validation, we want deterministic transforms (no augmentation)
    # so we wrap val_dataset with a new Dataset using val_transform.
    class WrappedDataset(Dataset):
        def __init__(self, base, transform):
            self.base = base
            self.transform = transform

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, label = self.base[idx]
            # base already applied train_transform; reload with val_transform
            # by going back to the original path from full_dataset
            img_path, label_idx = full_dataset.samples[self.base.indices[idx]]
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image, label_idx

    val_dataset_wrapped = WrappedDataset(val_dataset, val_transform)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_wrapped, batch_size=batch_size, shuffle=False)

    model = create_model(num_classes=len(label_to_idx))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 6
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{num_epochs} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model to {MODEL_PATH}")

    # Save label mapping for inference
    with LABEL_MAP_PATH.open("w") as f:
        json.dump(label_to_idx, f)
    print(f"Saved label map to {LABEL_MAP_PATH}")
    
    # Get testing accuracy
    test_dataset_wrapped = WrappedDataset(test_dataset, val_transform)
    test_loader = DataLoader(test_dataset_wrapped, batch_size=batch_size, shuffle=True)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()