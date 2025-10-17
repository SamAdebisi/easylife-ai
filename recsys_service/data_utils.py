"""
Utility helpers for the recommendation service.

Provides deterministic synthetic datasets for offline training and shared
preprocessing helpers reused by the FastAPI application.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "recsys"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "recsys"

INTERACTIONS_FILE = RAW_DIR / "interactions.csv"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
ITEMS_FILE = PROCESSED_DIR / "items.csv"


@dataclass
class InteractionData:
    train: pd.DataFrame
    test: pd.DataFrame
    items: pd.DataFrame


def _build_item_catalog(n_items: int) -> pd.DataFrame:
    genres = ["Productivity", "Finance", "Health", "Travel", "Lifestyle"]
    rng = np.random.default_rng(123)
    rows = []
    for item_idx in range(n_items):
        genre = rng.choice(genres)
        price = rng.uniform(5, 200)
        rows.append(
            {
                "item_id": f"item-{item_idx}",
                "name": f"EasyLife Bundle {item_idx}",
                "genre": genre,
                "price": round(float(price), 2),
            }
        )
    return pd.DataFrame(rows)


def generate_interactions(
    n_users: int = 120,
    n_items: int = 80,
    latent_dim: int = 6,
    interactions_per_user: Tuple[int, int] = (10, 25),
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    user_factors = rng.normal(size=(n_users, latent_dim))
    item_factors = rng.normal(size=(n_items, latent_dim))
    user_bias = rng.normal(scale=0.3, size=n_users)
    item_bias = rng.normal(scale=0.3, size=n_items)

    item_catalog = _build_item_catalog(n_items)
    rows = []
    timestamps = pd.date_range("2024-01-01", periods=365, freq="D")
    for user_idx in range(n_users):
        user_id = f"user-{user_idx}"
        n_interactions = rng.integers(*interactions_per_user)
        preferred_items = np.argsort(
            user_factors[user_idx] @ item_factors.T + item_bias
        )[::-1]
        interacted_items = preferred_items[: n_interactions // 2]
        random_items = rng.choice(
            n_items, size=n_interactions - len(interacted_items), replace=False
        )
        candidate_items = np.unique(
            np.concatenate([interacted_items, random_items]),
        )
        for item_idx in candidate_items:
            score = (
                user_factors[user_idx] @ item_factors[item_idx]
                + user_bias[user_idx]
                + item_bias[item_idx]
            )
            rating = np.clip(np.round(3 + score), 1, 5)
            interaction_ts = pd.Timestamp(rng.choice(timestamps)).isoformat()
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": f"item-{item_idx}",
                    "rating": float(rating),
                    "timestamp": interaction_ts,
                }
            )

    interactions = pd.DataFrame(rows)
    interactions.sort_values(["user_id", "timestamp"], inplace=True)
    # Stratify on user to keep user coverage in both splits.
    train, test = train_test_split(
        interactions,
        test_size=0.2,
        random_state=seed,
        stratify=interactions["user_id"],
    )
    return (
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        item_catalog,
    )


def save_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    items: pd.DataFrame,
) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    interactions = pd.concat([train, test], ignore_index=True).sort_values(
        [
            "user_id",
            "timestamp",
        ]
    )
    interactions.to_csv(INTERACTIONS_FILE, index=False)
    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)
    items.to_csv(ITEMS_FILE, index=False)


def ensure_datasets() -> InteractionData:
    if TRAIN_FILE.exists() and TEST_FILE.exists() and ITEMS_FILE.exists():
        return load_datasets()

    train, test, items = generate_interactions()
    save_datasets(train, test, items)
    return InteractionData(train=train, test=test, items=items)


def load_datasets() -> InteractionData:
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    items = pd.read_csv(ITEMS_FILE)
    return InteractionData(train=train, test=test, items=items)


def build_user_item_matrix(
    train_df: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    users = sorted(train_df["user_id"].unique())
    items = sorted(train_df["item_id"].unique())
    user_to_index = {u: idx for idx, u in enumerate(users)}
    item_to_index = {i: idx for idx, i in enumerate(items)}

    matrix = np.zeros((len(users), len(items)), dtype=np.float32)
    for row in train_df.itertuples():
        u_idx = user_to_index[row.user_id]
        i_idx = item_to_index[row.item_id]
        matrix[u_idx, i_idx] = float(row.rating)
    index_to_item = {idx: item for item, idx in item_to_index.items()}
    return matrix, user_to_index, index_to_item


def normalize_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    user_counts = np.maximum((matrix != 0).sum(axis=1), 1)
    means = np.where(
        matrix.sum(axis=1) != 0,
        matrix.sum(axis=1) / user_counts,
        0.0,
    )
    centered = matrix - means[:, None]
    centered[matrix == 0] = 0
    return centered, means
