"""
Collaborative filtering training workflow for the recommendation service.

Fits a truncated SVD decomposition on the user-item matrix, evaluates the
model on a held-out test split, logs metrics to MLflow, and persists artifacts
for the FastAPI inference service.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import numpy as np
from sklearn.decomposition import TruncatedSVD

from recsys_service.data_utils import (InteractionData, build_user_item_matrix,
                                       ensure_datasets, normalize_matrix)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "svd_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
MAPPINGS_PATH = ARTIFACT_DIR / "mappings.json"
RECOMMENDATIONS_PATH = ARTIFACT_DIR / "sample_recommendations.json"


def recall_at_k(
    relevant: List[str],
    recommended: List[str],
    k: int,
) -> float:
    if not relevant:
        return 0.0
    hit_count = len(set(relevant) & set(recommended[:k]))
    return hit_count / min(len(relevant), k)


def ndcg_at_k(
    relevant: List[str],
    recommended: List[str],
    k: int,
) -> float:
    if not relevant:
        return 0.0
    ideal_dcg = sum(
        1.0 / np.log2(idx + 2) for idx in range(min(len(relevant), k))
    )
    if ideal_dcg == 0:
        return 0.0
    dcg = 0.0
    for idx, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(idx + 2)
    return dcg / ideal_dcg


def evaluate_model(
    approx_matrix: np.ndarray,
    user_means: np.ndarray,
    mappings: Dict[str, Dict],
    data: InteractionData,
    top_k: int = 10,
) -> Tuple[float, float]:
    user_to_index = mappings["user_to_index"]
    index_to_item = mappings["index_to_item"]

    # Construct ground truth relevant items from the held-out test set.
    relevant_items: Dict[str, List[str]] = {}
    for row in data.test.itertuples():
        if row.rating >= 4.0:
            relevant_items.setdefault(row.user_id, []).append(row.item_id)

    recalls: List[float] = []
    ndcgs: List[float] = []
    for user_id, rel_items in relevant_items.items():
        if user_id not in user_to_index:
            continue
        u_idx = user_to_index[user_id]
        scores = approx_matrix[u_idx] + user_means[u_idx]
        ranked_indices = np.argsort(scores)[::-1]
        recommended_items = [index_to_item[idx] for idx in ranked_indices]
        recalls.append(recall_at_k(rel_items, recommended_items, top_k))
        ndcgs.append(ndcg_at_k(rel_items, recommended_items, top_k))

    return float(np.mean(recalls) if recalls else 0.0), float(
        np.mean(ndcgs) if ndcgs else 0.0
    )


def persist_artifacts(
    model: TruncatedSVD,
    mappings: Dict[str, Dict],
    user_means: np.ndarray,
    user_latent: np.ndarray,
    approx_matrix: np.ndarray,
    metrics: Dict[str, float],
    sample_recommendations: Dict[str, List[str]],
) -> None:
    from joblib import dump

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH)

    with open(MAPPINGS_PATH, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "user_to_index": mappings["user_to_index"],
                "index_to_item": mappings["index_to_item"],
                "user_means": user_means.tolist(),
                "user_latent": user_latent.tolist(),
                "approx_matrix": approx_matrix.tolist(),
            },
            fp,
            indent=2,
        )
    with open(METRICS_PATH, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    with open(RECOMMENDATIONS_PATH, "w", encoding="utf-8") as fp:
        json.dump(sample_recommendations, fp, indent=2)


def configure_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("recsys_collaborative_filtering")


def main() -> None:
    configure_mlflow()
    data = ensure_datasets()

    matrix, user_to_index, index_to_item = build_user_item_matrix(data.train)
    centered_matrix, user_means = normalize_matrix(matrix)

    params = {
        "n_components": 16,
        "random_state": 42,
    }
    model = TruncatedSVD(
        n_components=params["n_components"],
        random_state=params["random_state"],
    )

    with mlflow.start_run(run_name="cf-truncated-svd"):
        for key, value in params.items():
            mlflow.log_param(key, value)

        user_latent = model.fit_transform(centered_matrix)
        approx_matrix = model.inverse_transform(user_latent)

        mappings = {
            "user_to_index": user_to_index,
            "index_to_item": index_to_item,
        }
        recall, ndcg = evaluate_model(
            approx_matrix,
            user_means,
            mappings,
            data,
        )

        metrics = {"recall_at_10": recall, "ndcg_at_10": ndcg}
        mlflow.log_metrics(metrics)

        # prepare a small sample of recommendations for debugging/demo.
        sample_recommendations: Dict[str, List[str]] = {}
        preview_users = list(user_to_index.keys())[:5]
        for user_id in preview_users:
            u_idx = user_to_index[user_id]
            scores = approx_matrix[u_idx] + user_means[u_idx]
            ranked = np.argsort(scores)[::-1]
            top_items = [index_to_item[idx] for idx in ranked[:5]]
            sample_recommendations[user_id] = top_items

        persist_artifacts(
            model,
            mappings,
            user_means,
            user_latent,
            approx_matrix,
            metrics,
            sample_recommendations,
        )

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="reports")
        mlflow.log_artifact(str(MAPPINGS_PATH), artifact_path="artifacts")
        mlflow.log_artifact(
            str(RECOMMENDATIONS_PATH),
            artifact_path="artifacts",
        )
        mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()
