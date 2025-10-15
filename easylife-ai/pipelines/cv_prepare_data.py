"""
Create a small synthetic dataset for blur detection experiments.

Generates geometric patterns with and without Gaussian blur, stores them
under `data/raw/cv/{sharp,blurred}`, and emits a manifest CSV in
`data/processed/cv/labels.csv`.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import pandas as pd

SEED = 42
NUM_IMAGES_PER_CLASS = 12
IMAGE_SIZE = (128, 128)
KERNEL_SIZE = (9, 9)
SIGMA = 3.0

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_ROOT / "raw" / "cv"
PROCESSED_DIR = DATA_ROOT / "processed" / "cv"
MANIFEST_PATH = PROCESSED_DIR / "labels.csv"


def _draw_shapes(image: np.ndarray, rng: random.Random) -> np.ndarray:
    color = int(rng.uniform(100, 255))
    thickness = rng.choice([1, 2, 3])

    for _ in range(rng.randint(3, 6)):
        start = (
            rng.randint(0, IMAGE_SIZE[0] // 2),
            rng.randint(0, IMAGE_SIZE[1] // 2),
        )
        end = (
            start[0] + rng.randint(20, IMAGE_SIZE[0] - start[0]),
            start[1] + rng.randint(20, IMAGE_SIZE[1] - start[1]),
        )
        cv2.rectangle(image, start, end, color=color, thickness=thickness)

    for _ in range(rng.randint(2, 4)):
        center = (
            rng.randint(10, IMAGE_SIZE[0] - 10),
            rng.randint(10, IMAGE_SIZE[1] - 10),
        )
        radius = rng.randint(8, 30)
        cv2.circle(image, center, radius, color=color, thickness=thickness)

    return image


def _generate_base_image(rng: random.Random) -> np.ndarray:
    image = np.zeros(IMAGE_SIZE, dtype=np.uint8)
    image = _draw_shapes(image, rng)
    noise = rng.normalvariate(0, 10)
    noise_matrix = np.random.normal(noise, 10, IMAGE_SIZE).astype(np.float32)
    noisy_image = np.clip(image.astype(np.float32) + noise_matrix, 0, 255)
    return noisy_image.astype(np.uint8)


def _generate_samples() -> Iterable[Tuple[np.ndarray, int]]:
    rng = random.Random(SEED)
    for _ in range(NUM_IMAGES_PER_CLASS):
        base_image = _generate_base_image(rng)
        yield base_image, 1  # 1 = sharp

    for _ in range(NUM_IMAGES_PER_CLASS):
        base_image = _generate_base_image(rng)
        blurred = cv2.GaussianBlur(base_image, KERNEL_SIZE, SIGMA)
        yield blurred, 0  # 0 = blurred


def _save_image(image: np.ndarray, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), image)


def main() -> None:
    manifest_records = []
    for idx, (image, label) in enumerate(_generate_samples()):
        label_name = "sharp" if label == 1 else "blurred"
        filename = f"{idx:03d}_{label_name}.png"
        dest_dir = RAW_DIR / label_name
        dest = dest_dir / filename
        _save_image(image, dest)
        manifest_records.append(
            {
                "relative_path": f"cv/{label_name}/{filename}",
                "label": label,
            }
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_records).to_csv(MANIFEST_PATH, index=False)


if __name__ == "__main__":
    main()
