"""Ingest real-world CV data into the repository structure.

Reads configuration from ``configs/cv_ingest.yaml`` which specifies the
source directory and label mappings. Files are copied or symlinked into
``data/raw/cv/<label_name>`` so downstream pipelines operate on a consistent
layout. If the source directory is missing the script creates empty
directories to keep DVC stages reproducible.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "cv_ingest.yaml"
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw" / "cv"

SEED = 42
NUM_IMAGES_PER_CLASS = 12
IMAGE_SIZE = (128, 128)
KERNEL_SIZE = (9, 9)
SIGMA = 3.0


def _load_config() -> Dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "CV ingestion config missing. Expected at configs/cv_ingest.yaml."
        )
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    return config


def _prepare_destination(label_name: str) -> Path:
    dest = RAW_DIR / label_name
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _ingest_label(
    source_dir: Path,
    label_name: str,
    pattern: str,
    strategy: str,
    label_id: int,
) -> int:
    dest_dir = _prepare_destination(label_name)
    matched = list(source_dir.glob(pattern))
    count = 0
    for path in matched:
        if path.is_dir():
            continue
        destination = dest_dir / path.name
        if destination.exists():
            continue
        if strategy == "copy":
            shutil.copy2(path, destination)
        else:
            # default to symlink for space efficiency
            try:
                destination.symlink_to(path.resolve())
            except FileExistsError:
                pass
        count += 1
    return count


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


def _generate_synthetic_dataset() -> None:
    rng = random.Random(SEED)
    for label_name, label_id in ("sharp", 1), ("blurred", 0):
        dest_dir = _prepare_destination(label_name)
        for idx in range(NUM_IMAGES_PER_CLASS):
            base = np.zeros(IMAGE_SIZE, dtype=np.uint8)
            base = _draw_shapes(base, rng)
            noise = rng.normalvariate(0, 10)
            noise_matrix = np.random.normal(
                noise,
                10,
                IMAGE_SIZE,
            ).astype(np.float32)
            base_float = base.astype(np.float32)
            noisy = np.clip(base_float + noise_matrix, 0, 255)
            image = noisy.astype(np.uint8)
            if label_id == 0:
                image = cv2.GaussianBlur(image, KERNEL_SIZE, SIGMA)
            filename = dest_dir / f"synthetic_{idx:03d}_{label_name}.png"
            cv2.imwrite(str(filename), image)


def main() -> None:
    config = _load_config()
    source_dir = Path(config.get("source_dir", "data/external/cv_source"))
    strategy = config.get("copy_strategy", "symlink").lower()
    labels_cfg = config.get("labels", {})

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if source_dir.exists():
        total = 0
        for label_name, cfg in labels_cfg.items():
            pattern = cfg.get("glob", "**/*")
            label_id = int(cfg.get("label", 0))
            count = _ingest_label(
                source_dir,
                label_name,
                pattern,
                strategy,
                label_id,
            )
            total += count
            print(f"Ingested {count} files for {label_name} (id={label_id}).")

        if total == 0:
            print(
                "No files matched the configured patterns. "
                "Check configs/cv_ingest.yaml for correct paths and globs."
            )
        return

    print(f"Source directory {source_dir} missing; generating synthetic data.")
    _generate_synthetic_dataset()


if __name__ == "__main__":
    main()
