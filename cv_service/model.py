"""
Shared utilities for blur detection model inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class BlurThresholdModel:
    threshold: float

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (scores >= self.threshold).astype(int)

    def predict_confidence(self, scores: np.ndarray) -> np.ndarray:
        # Normalise confidence between 0 and 1 using a simple logistic curve.
        logits = scores - self.threshold
        return 1 / (1 + np.exp(-0.05 * logits))


def variance_of_laplacian(image: np.ndarray) -> float:
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


class CnnBlurModel:
    """Wrapper around a TorchScript blur classifier."""

    def __init__(self, model_path: Path, device: str | None = None) -> None:
        fallback = "cuda" if torch.cuda.is_available() else "cpu"
        resolved_device = device or fallback
        self.device = torch.device(resolved_device)
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

    @staticmethod
    def _prepare_tensor(image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image.astype("float32") / 255.0)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.unsqueeze(0)

    def predict(self, image: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = self._prepare_tensor(image).to(self.device)
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()[0]
