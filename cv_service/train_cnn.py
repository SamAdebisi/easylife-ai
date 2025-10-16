"""Train a convolutional neural network for blur detection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
MANIFEST_PATH = DATA_ROOT / "processed" / "cv" / "labels.csv"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "cnn_model.pt"
METRICS_PATH = ARTIFACT_DIR / "cnn_metrics.json"

BATCH_SIZE = int(os.getenv("CV_CNN_BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("CV_CNN_EPOCHS", "5"))
LR = float(os.getenv("CV_CNN_LR", "1e-3"))


def ensure_manifest_exists() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        msg = "Manifest missing. Run `dvc repro cv-preprocess` first."
        raise FileNotFoundError(msg)
    return pd.read_csv(MANIFEST_PATH)


@dataclass
class ManifestRecord:
    path: Path
    label: int


class BlurDataset(Dataset):
    def __init__(self, records: list[ManifestRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        record = self.records[idx]
        image = cv2.imread(str(record.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at {record.path}")
        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        return tensor, record.label


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def create_datasets(manifest: pd.DataFrame) -> Tuple[BlurDataset, BlurDataset]:
    paths = [RAW_DIR / rel_path for rel_path in manifest["relative_path"]]
    labels = manifest["label"].astype(int).tolist()
    train_idx, val_idx = train_test_split(
        list(range(len(paths))),
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )
    train_records = [ManifestRecord(paths[i], labels[i]) for i in train_idx]
    val_records = [ManifestRecord(paths[i], labels[i]) for i in val_idx]
    return BlurDataset(train_records), BlurDataset(val_records)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    val_loss = running_loss / len(loader.dataset)
    accuracy = correct / total if total else 0.0
    return val_loss, accuracy


def persist_artifacts(model: nn.Module, metrics: Dict[str, float]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu").eval()
    example = torch.randn(1, 1, 128, 128)
    scripted = torch.jit.trace(model_cpu, example)
    scripted.save(MODEL_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def configure_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("cv_blur_detection_cnn")


def log_epoch_metrics(
    epoch: int, train_loss: float, val_loss: float, val_acc: float
) -> None:
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("val_accuracy", val_acc, step=epoch)


def create_dataloaders(
    manifest: pd.DataFrame,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = create_datasets(manifest)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, val_loader


def main() -> None:
    manifest = ensure_manifest_exists()
    if manifest.empty:
        raise ValueError("Manifest empty. Ingest real CV data first.")

    train_loader, val_loader = create_dataloaders(manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    configure_mlflow()

    with mlflow.start_run(run_name="cnn-baseline"):
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LR)
        mlflow.log_param("device", str(device))

        best_val_acc = 0.0
        best_state: Dict[str, torch.Tensor] | None = None

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
            val_loss, val_acc = evaluate(
                model,
                val_loader,
                criterion,
                device,
            )
            log_epoch_metrics(epoch, train_loss, val_loss, val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict()

        if best_state is not None:
            model.load_state_dict(best_state)

        metrics = {
            "best_val_accuracy": float(best_val_acc),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "device": str(device),
        }
        persist_artifacts(model, metrics)
        mlflow.log_artifact(
            str(METRICS_PATH),
            artifact_path="reports",
        )
        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="model",
        )


if __name__ == "__main__":
    main()
