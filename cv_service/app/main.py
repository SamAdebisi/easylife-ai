from __future__ import annotations

import asyncio
import logging
import os
from itertools import count
from pathlib import Path
from typing import Optional

import cv2
import mlflow
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from joblib import load
from mlflow.tracking import MlflowClient
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from shared.tracing import instrument_fastapi_app, setup_tracing

from cv_service.model import (  # isort:skip
    BlurThresholdModel,
    CnnBlurModel,
    variance_of_laplacian,
)

logger = logging.getLogger("cv_service")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(title="EasyLife AI CV Service", version="0.2.0")

setup_tracing("cv-service")
instrument_fastapi_app(app, "cv-service")

Instrumentator().instrument(app).expose(
    app,
    include_in_schema=False,
)

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
THRESHOLD_MODEL_PATH = ARTIFACT_DIR / "blur_model.joblib"
CNN_MODEL_PATH = ARTIFACT_DIR / "cnn_model.pt"

MODEL_VARIANT = os.getenv("CV_MODEL_VARIANT", "threshold").lower()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT = os.getenv(
    "MLFLOW_CV_INFERENCE_EXPERIMENT",
    "cv_service_inference",
)

PREDICTION_COUNTER = Counter(
    "cv_predictions_total",
    "Number of predictions served by the CV service.",
    ["label"],
)

MLFLOW_CLIENT: Optional[MlflowClient] = None
MLFLOW_RUN_ID: Optional[str] = None
INFERENCE_STEP = count()

MODEL: Optional[object] = None


class HealthResponse(BaseModel):
    status: str = "ok"


class PredictionResponse(BaseModel):
    label: int
    label_name: str
    score: float
    confidence: float
    model_variant: str


def ensure_model_artifact() -> None:
    if MODEL_VARIANT == "cnn":
        if CNN_MODEL_PATH.exists():
            return
        logger.warning("CNN artifact missing; training fallback model.")
        from cv_service import train_cnn  # local import
        from pipelines import cv_ingest_data, cv_prepare_data  # local import

        cv_ingest_data.main()
        cv_prepare_data.main()
        train_cnn.main()
    else:
        if THRESHOLD_MODEL_PATH.exists():
            return
        logger.warning("Threshold artifact missing; training fallback model.")
        from cv_service import train  # local import
        from pipelines import cv_ingest_data, cv_prepare_data  # local import

        cv_ingest_data.main()
        cv_prepare_data.main()
        train.main()


def load_model() -> None:
    global MODEL
    ensure_model_artifact()
    if MODEL_VARIANT == "cnn":
        MODEL = CnnBlurModel(CNN_MODEL_PATH)
        logger.info("Loaded CNN blur model from %s", CNN_MODEL_PATH)
    else:
        MODEL = load(THRESHOLD_MODEL_PATH)
        if not isinstance(MODEL, BlurThresholdModel):
            raise RuntimeError(
                "Loaded model is not a BlurThresholdModel instance.",
            )
        logger.info(
            "Loaded blur detection model with threshold %.2f",
            MODEL.threshold,
        )


def init_mlflow() -> None:
    global MLFLOW_CLIENT, MLFLOW_RUN_ID
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
        if experiment is None:
            experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT)
        else:
            experiment_id = experiment.experiment_id

        run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name="cv-service",
            description="Online blur inference telemetry",
        )
        run_id = run.info.run_id
        mlflow.end_run()

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        artifact_path = str(
            CNN_MODEL_PATH if MODEL_VARIANT == "cnn" else THRESHOLD_MODEL_PATH
        )
        client.log_param(run_id, "model_path", artifact_path)
        client.log_param(run_id, "model_variant", MODEL_VARIANT)

        MLFLOW_CLIENT = client
        MLFLOW_RUN_ID = run_id
        logger.info(
            "MLflow inference run initialised (experiment=%s, run_id=%s)",
            MLFLOW_EXPERIMENT,
            run_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow logging disabled: %s", exc)
        MLFLOW_CLIENT = None
        MLFLOW_RUN_ID = None


def log_inference(score: float, label: int, metric_key: str) -> None:
    if not MLFLOW_CLIENT or not MLFLOW_RUN_ID:
        return

    step = next(INFERENCE_STEP)
    try:
        MLFLOW_CLIENT.log_metric(
            run_id=MLFLOW_RUN_ID,
            key=metric_key,
            value=float(score),
            step=step,
        )
        MLFLOW_CLIENT.log_metric(
            run_id=MLFLOW_RUN_ID,
            key=f"label_{label}",
            value=1.0,
            step=step,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to log inference metrics: %s", exc)


def decode_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(
            "Unable to decode image. Ensure it is a valid PNG or JPEG.",
        )
    return image


@app.on_event("startup")
async def _startup() -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_model)
    await loop.run_in_executor(None, init_mlflow)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
def health() -> HealthResponse:
    return HealthResponse()


@app.post(
    "/predict_image",
    response_model=PredictionResponse,
    summary="Classify blur level for an image",
)
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not ready.")

    data = await file.read()
    try:
        image = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if MODEL_VARIANT == "cnn":
        assert isinstance(MODEL, CnnBlurModel)
        probs = MODEL.predict(image)
        label = int(np.argmax(probs))
        label_name = "sharp" if label == 1 else "blurred"
        score = float(probs[1])  # probability of sharp
        confidence = float(probs[label])
        metric_key = "sharp_probability"
    else:
        assert isinstance(MODEL, BlurThresholdModel)
        score = variance_of_laplacian(image)
        label = int(score >= MODEL.threshold)
        label_name = "sharp" if label == 1 else "blurred"
        confidence = float(MODEL.predict_confidence(np.array([score]))[0])
        metric_key = "blur_score"

    PREDICTION_COUNTER.labels(label=label_name).inc()
    log_inference(score, label, metric_key)

    return PredictionResponse(
        label=label,
        label_name=label_name,
        score=float(score),
        confidence=confidence,
        model_variant=MODEL_VARIANT,
    )
