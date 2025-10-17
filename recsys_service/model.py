"""
Artifact loading and inference helpers for the recommendation service.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import load

from recsys_service.data_utils import InteractionData, ensure_datasets

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "svd_model.joblib"
MAPPINGS_PATH = ARTIFACT_DIR / "mappings.json"
RECOMMENDATIONS_PATH = ARTIFACT_DIR / "sample_recommendations.json"


@dataclass
class Recommendation:
    item_id: str
    score: float
    metadata: Dict[str, str]


class RecommendationEngine:
    def __init__(self) -> None:
        self.data: InteractionData = ensure_datasets()
        self.model = load(MODEL_PATH)
        self._load_mappings()
        self._build_popularity_index()

    def _load_mappings(self) -> None:
        with open(MAPPINGS_PATH, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        self.user_to_index: Dict[str, int] = {
            user_id: int(idx) for user_id, idx in payload["user_to_index"].items()
        }
        self.index_to_item: Dict[int, str] = {
            int(idx): item_id for idx, item_id in payload["index_to_item"].items()
        }
        self.item_to_index: Dict[str, int] = {
            item_id: idx for idx, item_id in self.index_to_item.items()
        }
        self.user_means = np.array(payload["user_means"], dtype=float)
        self.user_latent = np.array(payload["user_latent"], dtype=float)
        self.approx_matrix = np.array(payload["approx_matrix"], dtype=float)

        with open(RECOMMENDATIONS_PATH, "r", encoding="utf-8") as fp:
            self.sample_recommendations = json.load(fp)

        # Build caches
        full_interactions = pd.concat(
            [self.data.train, self.data.test], ignore_index=True
        )
        self.user_consumed: Dict[str, set] = {}
        for row in full_interactions.itertuples():
            self.user_consumed.setdefault(row.user_id, set()).add(row.item_id)

        self.items_df = self.data.items.set_index("item_id")
        # Shape: (num_items, latent_dim)
        self.item_embeddings = self.model.components_.T

    def _build_popularity_index(self) -> None:
        full_interactions = pd.concat(
            [self.data.train, self.data.test], ignore_index=True
        )
        stats = (
            full_interactions.groupby("item_id")["rating"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_rating", "count": "interactions"})
            .reset_index()
        )
        stats["score"] = (
            stats["avg_rating"] * 0.6 + np.log1p(stats["interactions"]) * 0.4
        )
        stats.sort_values("score", ascending=False, inplace=True)
        self.popular_items = stats

    def recommend(self, user_id: str, top_k: int = 10) -> List[Recommendation]:
        if user_id in self.user_to_index:
            u_idx = self.user_to_index[user_id]
            scores = self.approx_matrix[u_idx] + self.user_means[u_idx]
            ranked_indices = np.argsort(scores)[::-1]
            consumed = self.user_consumed.get(user_id, set())
            recommendations: List[Recommendation] = []
            for idx in ranked_indices:
                item_id = self.index_to_item[idx]
                if item_id in consumed:
                    continue
                metadata = self._item_metadata(item_id)
                recommendations.append(
                    Recommendation(
                        item_id=item_id,
                        score=float(scores[idx]),
                        metadata=metadata,
                    )
                )
                if len(recommendations) >= top_k:
                    break
            if recommendations:
                return recommendations
        # Fallback to popular items
        recommendations = []
        consumed = self.user_consumed.get(user_id, set())
        for row in self.popular_items.itertuples():
            if row.item_id in consumed:
                continue
            recommendations.append(
                Recommendation(
                    item_id=row.item_id,
                    score=float(row.score),
                    metadata=self._item_metadata(row.item_id),
                )
            )
            if len(recommendations) >= top_k:
                break
        return recommendations

    def similar_items(self, item_id: str, top_k: int = 5) -> List[Recommendation]:
        if item_id not in self.item_to_index:
            return []
        item_idx = self.item_to_index[item_id]
        norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = self.item_embeddings / norms
        target_norm = normalized[item_idx]
        sims = normalized @ target_norm
        ranked = np.argsort(sims)[::-1]
        results: List[Recommendation] = []
        for idx in ranked:
            candidate_id = self.index_to_item[idx]
            if candidate_id == item_id:
                continue
            results.append(
                Recommendation(
                    item_id=candidate_id,
                    score=float(sims[idx]),
                    metadata=self._item_metadata(candidate_id),
                )
            )
            if len(results) >= top_k:
                break
        return results

    def _item_metadata(self, item_id: str) -> Dict[str, str]:
        if item_id in self.items_df.index:
            record = self.items_df.loc[item_id]
            return {
                "name": str(record.get("name", "")),
                "genre": str(record.get("genre", "")),
                "price": record.get("price", 0.0),
            }
        return {}
