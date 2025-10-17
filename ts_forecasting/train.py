"""
Train and evaluate the baseline model for the forecasting service.
"""

from __future__ import annotations

import os

import mlflow

from ts_forecasting.data_utils import load_series
from ts_forecasting.model import (ARTIFACT_DIR, METADATA_PATH, METRICS_PATH,
                                  MODEL_PATH, persist_artifacts,
                                  train_holt_winters)


def configure_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("ts_forecasting")


def main() -> None:
    configure_mlflow()
    series = load_series()

    params = {
        "seasonal_periods": 7,
        "trend": "add",
        "seasonal": "add",
    }

    with mlflow.start_run(run_name="holt-winters-additive"):
        for key, value in params.items():
            mlflow.log_param(key, value)

        model, metrics, metadata = train_holt_winters(
            series=series,
            seasonal_periods=params["seasonal_periods"],
            trend=params["trend"],
            seasonal=params["seasonal"],
        )
        persist_artifacts(model, metrics, metadata)
        mlflow.log_metrics(metrics)

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="reports")
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="models")
        mlflow.log_artifact(str(METADATA_PATH), artifact_path="reports")


if __name__ == "__main__":
    main()
