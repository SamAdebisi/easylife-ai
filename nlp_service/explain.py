"""
Utility helpers that provide lightweight explainability for the NLP model.

The logistic regression classifier exposes coefficients per token. We combine
them with the TF-IDF representation of the incoming text to derive per-token
contributions and expose the top positive drivers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sklearn.pipeline import Pipeline


@dataclass
class TokenContribution:
    token: str
    contribution: float


def _feature_contributions(
    pipeline: Pipeline,
    text: str,
    class_index: int,
) -> Sequence[TokenContribution]:
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    features = vectorizer.transform([text])
    coefficients = classifier.coef_[class_index]
    contributions = features.multiply(coefficients).toarray()[0]

    feature_names = np.array(vectorizer.get_feature_names_out())
    mask = contributions != 0.0
    return [
        TokenContribution(token=str(token), contribution=float(value))
        for token, value in zip(feature_names[mask], contributions[mask])
    ]


def explain_text(
    pipeline: Pipeline,
    text: str,
    top_k: int = 5,
) -> List[TokenContribution]:
    """
    Return the top contributing tokens for the predicted class.

    Only positive contributions are considered so that the explanation surfaces
    tokens that push the prediction towards the returned label.
    """
    predicted_class = int(pipeline.predict([text])[0])
    contributions = _feature_contributions(pipeline, text, predicted_class)

    positives = [item for item in contributions if item.contribution > 0]
    positives.sort(key=lambda item: item.contribution, reverse=True)
    return positives[:top_k]
