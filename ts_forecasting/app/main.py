from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from ts_forecasting.model import ForecastArtifacts, ensure_artifacts, forecast

logger = logging.getLogger("ts_forecasting")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="EasyLife AI Forecasting Service", version="0.3.0")

Instrumentator().instrument(app).expose(
    app,
    include_in_schema=False,
)

FORECAST_REQUESTS = Counter(
    "ts_forecast_requests_total",
    "Number of forecast requests served.",
)
FORECAST_HORIZON = Histogram(
    "ts_forecast_horizon",
    "Distribution of requested forecast horizons.",
    buckets=(1, 7, 14, 30, 60, 90, 180),
)

ARTIFACTS: Optional[ForecastArtifacts] = None


class HealthResponse(BaseModel):
    status: str = "ok"


class ForecastRequest(BaseModel):
    horizon: int = Field(
        default=14,
        ge=1,
        le=180,
        description="Number of future periods (days) to forecast.",
    )


class ForecastResponse(BaseModel):
    timestamps: List[datetime]
    values: List[float]


def get_artifacts() -> ForecastArtifacts:
    global ARTIFACTS
    if ARTIFACTS is None:
        logger.info("Forecasting artifacts not initialised; loading now.")
        ARTIFACTS = ensure_artifacts()
        logger.info(
            "Forecasting model ready (trained through %s)",
            ARTIFACTS.metadata.get("train_end"),
        )
    return ARTIFACTS


@app.on_event("startup")
def _startup() -> None:
    get_artifacts()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/forecast", response_model=ForecastResponse)
def create_forecast(payload: ForecastRequest) -> ForecastResponse:
    artifacts = get_artifacts()
    FORECAST_REQUESTS.inc()
    FORECAST_HORIZON.observe(payload.horizon)

    forecast_df = forecast(
        model=artifacts.model,
        metadata=artifacts.metadata,
        horizon=payload.horizon,
    )
    timestamps = forecast_df["date"].dt.to_pydatetime().tolist()
    values = forecast_df["prediction"].astype(float).tolist()
    logger.debug("Forecast horizon=%s generated.", payload.horizon)
    return ForecastResponse(timestamps=timestamps, values=values)
