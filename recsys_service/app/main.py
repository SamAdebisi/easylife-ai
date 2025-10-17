from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from recsys_service.model import Recommendation, RecommendationEngine

logger = logging.getLogger("recsys_service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="EasyLife AI Recommendation Service", version="0.4.0")

Instrumentator().instrument(app).expose(
    app,
    include_in_schema=False,
)

RECOMMENDATION_REQUESTS = Counter(
    "recsys_recommendation_requests_total",
    "Number of recommendation requests served.",
)
SIMILAR_ITEMS_REQUESTS = Counter(
    "recsys_similar_items_requests_total",
    "Number of similar-items requests served.",
)
TOPK_HISTOGRAM = Histogram(
    "recsys_topk_requested",
    "Distribution of requested top-k sizes.",
    buckets=(1, 3, 5, 10, 20, 30),
)

ENGINE: Optional[RecommendationEngine] = None


class ItemMetadata(BaseModel):
    name: str
    genre: str
    price: float


class RecommendationRecord(BaseModel):
    item_id: str
    score: float
    metadata: ItemMetadata


class RecommendationRequest(BaseModel):
    user_id: str = Field(
        ...,
        description="Identifier of the user.",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=30,
        description="Number of items to return.",
    )


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendationRecord]


class SimilarItemsResponse(BaseModel):
    item_id: str
    similar_items: List[RecommendationRecord]


class HealthResponse(BaseModel):
    status: str = "ok"
    model_version: str = "svd"


def get_engine() -> RecommendationEngine:
    global ENGINE
    if ENGINE is None:
        logger.info("Initialising recommendation engine artifacts.")
        ENGINE = RecommendationEngine()
        logger.info(
            "Recommendation engine ready with %d users and %d items.",
            len(ENGINE.user_to_index),
            len(ENGINE.item_to_index),
        )
    return ENGINE


@app.on_event("startup")
def _startup() -> None:
    get_engine()


def _serialize(
    recommendations: List[Recommendation],
) -> List[RecommendationRecord]:
    return [
        RecommendationRecord(
            item_id=item.item_id,
            score=item.score,
            metadata=ItemMetadata(**item.metadata),
        )
        for item in recommendations
    ]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    _ = get_engine()
    return HealthResponse()


@app.post("/recommendations", response_model=RecommendationResponse)
def generate_recommendations(
    payload: RecommendationRequest,
) -> RecommendationResponse:
    engine = get_engine()
    RECOMMENDATION_REQUESTS.inc()
    TOPK_HISTOGRAM.observe(payload.top_k)

    recommendations = engine.recommend(
        payload.user_id,
        top_k=payload.top_k,
    )
    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail="No recommendations available.",
        )
    return RecommendationResponse(
        user_id=payload.user_id,
        recommendations=_serialize(recommendations),
    )


@app.get(
    "/items/{item_id}/similar",
    response_model=SimilarItemsResponse,
)
def similar_items(item_id: str, top_k: int = 5) -> SimilarItemsResponse:
    engine = get_engine()
    SIMILAR_ITEMS_REQUESTS.inc()
    TOPK_HISTOGRAM.observe(top_k)

    results = engine.similar_items(item_id, top_k=top_k)
    if not results:
        raise HTTPException(status_code=404, detail="Item not found.")
    return SimilarItemsResponse(
        item_id=item_id,
        similar_items=_serialize(results),
    )
