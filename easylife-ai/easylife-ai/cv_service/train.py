"""
Baseline blur detection training script using variance of Laplacian.

Computes blur scores for the synthetic dataset, finds an optimal
threshold that separates sharp vs. blurred images, logs metrics to
MLflow, and persists the threshold for serving.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mlflow
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score

from cv_service.model import BlurThresholdModel, variance_of_laplacian

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "cv" / "labels.csv"

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "blur_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Missing manifest. Run `dvc repro cv-preprocess` first."
        )
    return pd.read_csv(MANIFEST_PATH)


def compute_laplacian_variance(image_path: Path) -> float:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    return variance_of_laplacian(image)


def load_scores(manifest: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    scores: List[float] = []
    labels: List[int] = []
    for _, row in manifest.iterrows():
        image_path = RAW_DIR / row["relative_path"]
        score = compute_laplacian_variance(image_path)
        scores.append(score)
        labels.append(int(row["label"]))
    return np.array(scores), np.array(labels)


def find_optimal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    candidates = np.linspace(scores.min(), scores.max(), num=200)
    best_threshold = candidates[0]
    best_accuracy = -1.0
    best_f1 = -1.0

    for threshold in candidates:
        preds = (scores >= threshold).astype(int)
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        is_tie_with_better_f1 = accuracy == best_accuracy and f1 > best_f1
        if accuracy > best_accuracy or is_tie_with_better_f1:
            best_threshold = threshold
            best_accuracy = accuracy
            best_f1 = f1

    metrics = {
        "accuracy": float(best_accuracy),
        "f1": float(best_f1),
        "mean_score_sharp": float(scores[labels == 1].mean()),
        "mean_score_blurred": float(scores[labels == 0].mean()),
    }
    return best_threshold, metrics


def configure_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("cv_blur_detection")


def persist_artifacts(
    model: BlurThresholdModel,
    metrics: Dict[str, float],
    scores: np.ndarray,
    labels: np.ndarray,
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH)
    payload = {
        "threshold": model.threshold,
        **metrics,
        "num_samples": int(len(scores)),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    manifest = load_manifest()
    scores, labels = load_scores(manifest)

    configure_mlflow()

    best_threshold, metrics = find_optimal_threshold(scores, labels)
    model = BlurThresholdModel(threshold=best_threshold)

    with mlflow.start_run(run_name="baseline-laplacian-threshold"):
        mlflow.log_param("model_type", "threshold")
        mlflow.log_param("threshold", best_threshold)
        mlflow.log_metrics(metrics)

        persist_artifacts(model, metrics, scores, labels)

        mlflow.log_artifact(str(METRICS_PATH), artifact_path="reports")
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")


if __name__ == "__main__":
    main()
