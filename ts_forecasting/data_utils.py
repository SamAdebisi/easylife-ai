"""
Utility helpers for the time-series forecasting phase.

Provides synthetic dataset generation so the training pipeline and the
FastAPI service have a deterministic dataset to work with during local
development and CI.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "ts"
RAW_FILE = RAW_DIR / "synthetic_series.csv"
PROCESSED_FILE = PROCESSED_DIR / "series.csv"


def generate_synthetic_series(
    periods: int = 3 * 365,
    freq: str = "D",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a seasonal synthetic time series that emulates business KPI data.

    The signal combines a linear trend, weekly seasonality, yearly seasonality,
    and gaussian noise so that classical forecasting algorithms have
    meaningful structure to learn from.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=periods, freq=freq)
    steps = np.arange(periods)

    trend = 0.02 * steps
    weekly = 2.0 * np.sin(2 * math.pi * steps / 7)
    yearly = 1.0 * np.sin(2 * math.pi * steps / 365.25)
    noise = rng.normal(0, 0.8, size=periods)

    values = 50 + trend + weekly + yearly + noise
    return pd.DataFrame(
        {
            "date": dates,
            "value": values.astype(float),
        }
    )


def save_dataset(df: pd.DataFrame) -> None:
    """Persist the synthetic dataset in both raw and processed locations."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_FILE, index=False)
    df.to_csv(PROCESSED_FILE, index=False)


def ensure_dataset() -> Path:
    """Ensure the processed dataset exists on disk and return its path."""
    if not PROCESSED_FILE.exists():
        df = generate_synthetic_series()
        save_dataset(df)
    return PROCESSED_FILE


def load_series() -> pd.Series:
    """Load the processed dataset as a pandas Series indexed by date."""
    ensure_dataset()
    df = pd.read_csv(PROCESSED_FILE, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    series = pd.Series(
        df["value"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(df["date"]),
        name="value",
    )
    return series


def train_test_split(
    series: pd.Series,
    test_size: int = 30,
) -> Tuple[pd.Series, pd.Series]:
    """Split series into train/test segments keeping chronological order."""
    if test_size <= 0 or test_size >= len(series):
        raise ValueError("test_size must be within the range (0, len(series)).")
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    return train, test
