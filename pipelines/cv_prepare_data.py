"""Build manifest for CV blur dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_ROOT / "raw" / "cv"
PROCESSED_DIR = DATA_ROOT / "processed" / "cv"
MANIFEST_PATH = PROCESSED_DIR / "labels.csv"


def _collect_records() -> list[dict]:
    records: list[dict] = []
    if not RAW_DIR.exists():
        return records

    for label_dir in RAW_DIR.iterdir():
        if not label_dir.is_dir():
            continue
        label_name = label_dir.name
        label_id = 1 if label_name.lower().startswith("sharp") else 0
        for image_path in label_dir.glob("**/*"):
            if not image_path.is_file():
                continue
            relative = image_path.relative_to(DATA_ROOT / "raw")
            records.append(
                {
                    "relative_path": str(relative).replace("\\", "/"),
                    "label": label_id,
                }
            )
    return records


def main() -> None:
    records = _collect_records()
    if not records:
        raise FileNotFoundError(
            "No images found under data/raw/cv. Run cv_ingest_data first."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(MANIFEST_PATH, index=False)
    print(f"Manifest written with {len(records)} records -> {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
