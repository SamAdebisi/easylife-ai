"""
Model training and artifact management for the forecasting service.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, HoltWintersResults

from ts_forecasting.data_utils import load_series, train_test_split

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "holt_winters.pkl"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"


@dataclass
class ForecastArtifacts:
    model: HoltWintersResults
    metadata: Dict[str, str]


def _compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    errors = y_true.to_numpy() - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    denominator = np.clip(y_true.to_numpy(), 1e-6, None)
    mape = float(np.mean(np.abs(errors) / denominator)) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}


def train_holt_winters(
    series: pd.Series,
    seasonal_periods: int = 7,
    trend: str = "add",
    seasonal: str = "add",
) -> Tuple[HoltWintersResults, Dict[str, float], Dict[str, str]]:
    train, test = train_test_split(series, test_size=30)
    model = ExponentialSmoothing(
        train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
    ).fit(optimized=True)
    forecast = model.forecast(steps=len(test))
    metrics = _compute_metrics(test, forecast)

    metadata: Dict[str, str] = {
        "seasonal_periods": str(seasonal_periods),
        "trend": trend,
        "seasonal": seasonal,
        "train_start": train.index[0].isoformat(),
        "train_end": train.index[-1].isoformat(),
        "test_start": test.index[0].isoformat(),
        "test_end": test.index[-1].isoformat(),
        "freq": pd.infer_freq(series.index) or "D",
    }
    return model, metrics, metadata


def persist_artifacts(
    model: HoltWintersResults,
    metrics: Dict[str, float],
    metadata: Dict[str, str],
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    with open(METADATA_PATH, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


def load_artifacts() -> ForecastArtifacts:
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as fp:
        metadata = json.load(fp)
    return ForecastArtifacts(model=model, metadata=metadata)


def ensure_artifacts() -> ForecastArtifacts:
    if MODEL_PATH.exists() and METADATA_PATH.exists():
        return load_artifacts()

    series = load_series()
    model, metrics, metadata = train_holt_winters(series)
    persist_artifacts(model, metrics, metadata)
    return ForecastArtifacts(model=model, metadata=metadata)


def forecast(
    model: HoltWintersResults,
    metadata: Dict[str, str],
    horizon: int,
) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    freq = metadata.get("freq", "D")
    last_timestamp = pd.Timestamp(metadata["train_end"])
    future_index = pd.date_range(
        start=last_timestamp + pd.tseries.frequencies.to_offset(freq),
        periods=horizon,
        freq=freq,
    )
    forecast_values = model.forecast(steps=horizon)
    return pd.DataFrame({"date": future_index, "prediction": forecast_values})
