"""
Shared utilities for the NLP sentiment dataset across pipelines and services.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import pandas as pd

POSITIVE_SENTENCES: Sequence[str] = (
    "I absolutely loved this product, it works like a charm.",
    "Fantastic experience, will buy again without hesitation.",
    "Service was outstanding and the staff were incredibly friendly.",
    "The movie was a delightful surprise with a heartwarming ending.",
    "Great taste and quick delivery, highly recommended.",
    "This app simplifies my workflow and saves me hours every week.",
    "The customer support team resolved my issue in minutes.",
    "Top-notch quality at an affordable price point.",
    "The new update added exactly the features I needed.",
    "Beautiful design and very easy to use from day one.",
)

NEGATIVE_SENTENCES: Sequence[str] = (
    "The product stopped working after a single use, very disappointed.",
    "Terrible customer service; no one ever responded to my emails.",
    "This movie was a complete waste of time and money.",
    "Shipping took forever and the package arrived damaged.",
    "The interface is confusing and constantly crashes.",
    "Support kept bouncing me between agents without solving anything.",
    "Bad quality materials and sloppy craftsmanship.",
    "The latest update made the app slower than before.",
    "It tasted awful and the portion size was tiny.",
    "Setup instructions were unclear and missing key steps.",
)


def build_dataset(seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    positive_pairs = [(text, 1) for text in POSITIVE_SENTENCES]
    negative_pairs = [(text, 0) for text in NEGATIVE_SENTENCES]
    samples: List[Tuple[str, int]] = positive_pairs + negative_pairs
    random.shuffle(samples)
    return pd.DataFrame(samples, columns=["text", "label"])


def split_dataset(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = int(len(df) * train_ratio)
    train_df = df.iloc[:cutoff].reset_index(drop=True)
    test_df = df.iloc[cutoff:].reset_index(drop=True)
    return train_df, test_df
