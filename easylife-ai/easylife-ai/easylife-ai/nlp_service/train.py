"""
Baseline sentiment classifier training script.

Uses TF-IDF features with Logistic Regression, logs metrics to MLflow,
and persists a serialized pipeline for the FastAPI inference service.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "nlp"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "sentiment_model.pkl"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Processed datasets missing. Run `dvc repro nlp-preprocess` first."
        )
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_pipeline() -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=1000,
        min_df=1,
    )
    classifier = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("clf", classifier),
        ]
    )


def train_model(pipeline: Pipeline, train_df: pd.DataFrame) -> Pipeline:
    pipeline.fit(train_df["text"].tolist(), train_df["label"].tolist())
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
) -> Dict[str, float]:
    predictions = pipeline.predict(test_df["text"].tolist())
    accuracy = accuracy_score(
        test_df["label"],
        predictions,
    )
    f1 = f1_score(
        test_df["label"],
        predictions,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
    }


def persist_artifacts(pipeline: Pipeline, metrics: Dict[str, float]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(
        sk_model=pipeline,
        path=str(ARTIFACT_DIR / "mlflow_model"),
    )
    with open(MODEL_PATH, "wb") as f:
        from joblib import dump

        dump(pipeline, f)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def configure_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("nlp_sentiment")


def main() -> None:
    train_df, test_df = load_data()
    pipeline = build_pipeline()
    configure_mlflow()

    params = {
        "vectorizer_max_features": 1000,
        "vectorizer_ngram_range": "1-2",
        "model": "logistic_regression",
        "penalty": "l2",
        "C": 1.0,
    }

    with mlflow.start_run(run_name="baseline-tfidf-logreg"):
        for key, value in params.items():
            mlflow.log_param(key, value)

        pipeline = train_model(pipeline, train_df)
        metrics = evaluate_model(pipeline, test_df)
        mlflow.log_metrics(metrics)

        persist_artifacts(pipeline, metrics)

        registered_name = os.getenv("MLFLOW_REGISTERED_MODEL")
        if registered_name:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=registered_name,
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
            )

        mlflow.log_artifact(str(METRICS_PATH), artifact_path="reports")


if __name__ == "__main__":
    main()
