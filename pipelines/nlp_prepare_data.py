"""
Create a lightweight sentiment dataset for the NLP service.

The goal is to provide an offline-friendly dataset that unblocks DVC
pipelines, unit tests, and baseline training without needing to download
external corpora. Sentences are handcrafted and balanced between positive
and negative labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp_service.data_utils import build_dataset, split_dataset  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_ROOT / "raw" / "nlp_sentiment.csv"
PROCESSED_DIR = DATA_ROOT / "processed" / "nlp"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    raw_dir = RAW_PATH.parent
    raw_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(RAW_PATH, index=False)
    train_df, test_df = split_dataset(df)

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)


if __name__ == "__main__":
    main()
