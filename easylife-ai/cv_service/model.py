"""
Shared utilities for blur detection model inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


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
