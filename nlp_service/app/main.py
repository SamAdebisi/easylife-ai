from __future__ import annotations

import logging
import os
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import List, Optional

import mlflow
from fastapi import FastAPI, HTTPException
from joblib import dump, load
from mlflow.tracking import MlflowClient
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from nlp_service.data_utils import build_dataset, split_dataset
from nlp_service.explain import TokenContribution, explain_text
from nlp_service.train import build_pipeline, train_model
from shared.tracing import instrument_fastapi_app, setup_tracing

logger = logging.getLogger("nlp_service")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(title="EasyLife AI NLP Service", version="0.2.0")

TRACER = setup_tracing("nlp-service")
instrument_fastapi_app(app, "nlp-service")

Instrumentator().instrument(app).expose(
    app,
    include_in_schema=False,
)

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "sentiment_model.pkl"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT = os.getenv(
    "MLFLOW_INFERENCE_EXPERIMENT",
    "nlp_service_inference",
)

PREDICTION_COUNTER = Counter(
    "nlp_predictions_total",
    "Number of predictions served by the NLP service.",
    ["label"],
)

MLFLOW_CLIENT: Optional[MlflowClient] = None
MLFLOW_RUN_ID: Optional[str] = None
INFERENCE_STEP = count()

MODEL_PIPELINE = None


def _predict_proba(text: str) -> tuple[int, float]:
    probabilities = MODEL_PIPELINE.predict_proba([text])[0]
    label = int(probabilities.argmax())
    confidence = float(probabilities[label])
    return label, confidence


def _serialise_tokens(
    contributions: List[TokenContribution],
) -> List[TokenContributionResponse]:
    return [
        TokenContributionResponse(token=item.token, contribution=item.contribution)
        for item in contributions
    ]


class HealthResponse(BaseModel):
    status: str = "ok"


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to classify.")


class PredictionResponse(BaseModel):
    label: int
    confidence: float


class TokenContributionResponse(BaseModel):
    token: str
    contribution: float


class ExplanationRequest(PredictionRequest):
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top contributing tokens to return.",
    )


class ExplanationResponse(BaseModel):
    label: int
    confidence: float
    tokens: List[TokenContributionResponse]


def ensure_model_artifact() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH

    logger.warning(
        "Model artifact missing; training lightweight fallback model.",
    )
    df = build_dataset()
    train_df, _ = split_dataset(df)
    pipeline = build_pipeline()
    train_model(pipeline, train_df)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dump(pipeline, MODEL_PATH)
    return MODEL_PATH


def load_model() -> None:
    global MODEL_PIPELINE
    artifact_path = ensure_model_artifact()
    MODEL_PIPELINE = load(artifact_path)
    logger.info("Loaded sentiment model from %s", artifact_path)


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
            run_name="nlp-service",
            description="Online inference telemetry",
        )
        run_id = run.info.run_id
        mlflow.end_run()

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        client.log_param(run_id, "model_path", str(MODEL_PATH))

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


def log_inference(predicted_label: int, confidence: float) -> None:
    if not MLFLOW_CLIENT or not MLFLOW_RUN_ID:
        return

    step = next(INFERENCE_STEP)
    try:
        MLFLOW_CLIENT.log_metric(
            run_id=MLFLOW_RUN_ID,
            key="prediction_confidence",
            value=float(confidence),
            step=step,
        )
        MLFLOW_CLIENT.log_metric(
            run_id=MLFLOW_RUN_ID,
            key=f"label_{predicted_label}",
            value=1.0,
            step=step,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to log inference metrics: %s", exc)


def log_explanation(
    label: int,
    confidence: float,
    text: str,
    contributions: List[TokenContribution],
) -> None:
    if not MLFLOW_CLIENT or not MLFLOW_RUN_ID:
        return

    artifact_name = datetime.utcnow().strftime("explanation_%Y%m%dT%H%M%S%f.json")
    payload = {
        "text": text,
        "label": label,
        "confidence": confidence,
        "tokens": [
            {"token": item.token, "contribution": item.contribution}
            for item in contributions
        ],
    }

    try:
        MLFLOW_CLIENT.log_dict(
            MLFLOW_RUN_ID,
            payload,
            artifact_file=f"explanations/{artifact_name}",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to log explanation payload: %s", exc)


@app.on_event("startup")
def _startup() -> None:
    load_model()
    init_mlflow()


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
def health() -> HealthResponse:
    return HealthResponse()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Run sentiment prediction",
)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if MODEL_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not ready.")

    label, confidence = _predict_proba(payload.text)

    PREDICTION_COUNTER.labels(label=str(label)).inc()
    log_inference(label, confidence)

    return PredictionResponse(label=label, confidence=confidence)


@app.post(
    "/explain",
    response_model=ExplanationResponse,
    summary="Return prediction explanation",
)
def explain(payload: ExplanationRequest) -> ExplanationResponse:
    if MODEL_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not ready.")

    label, confidence = _predict_proba(payload.text)
    contributions = explain_text(
        MODEL_PIPELINE,
        payload.text,
        top_k=payload.top_k,
    )

    serialized = _serialise_tokens(contributions)
    PREDICTION_COUNTER.labels(label=str(label)).inc()
    log_inference(label, confidence)
    log_explanation(label, confidence, payload.text, contributions)

    return ExplanationResponse(
        label=label,
        confidence=confidence,
        tokens=serialized,
    )
