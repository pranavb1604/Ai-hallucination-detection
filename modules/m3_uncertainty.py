"""
M3 — Token-Level Uncertainty Estimation Module
Estimates how uncertain the LLM is about its answer by analysing
the semantic variance across multiple sampled responses.

Inspired by: Semantic Uncertainty (Kuhn, Gal & Farquhar, ICLR 2023)
Approach:    Since we treat the LLM as a black box (no logprobs access),
             we approximate semantic entropy by:
             1. Encoding all responses with a sentence transformer.
             2. Computing the variance of the embedding cloud — high
                variance = high uncertainty = lower trust.
             3. Normalising to [0, 1] where 1 = certain, 0 = uncertain.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

_model = None

# Empirically chosen: cosine-space variance above this → very uncertain
_MAX_EXPECTED_VARIANCE = 0.15


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _semantic_variance(embeddings: np.ndarray) -> float:
    """
    Compute the mean squared distance of each embedding from the centroid.
    This is the trace of the empirical covariance matrix, a scalar measure
    of spread in embedding space.
    """
    centroid = embeddings.mean(axis=0)
    diffs = embeddings - centroid                    # (N, D)
    sq_dists = np.sum(diffs ** 2, axis=1)            # (N,)
    return float(sq_dists.mean())


def score(question: str, responses: list[str]) -> dict:
    """
    Returns:
        m3_score   : float [0, 1]  — 1 = certain, 0 = maximally uncertain
        m3_variance: float         — raw semantic variance (for debugging)
        m3_verdict : str
    """
    if not responses:
        return {
            "m3_score": 0.0,
            "m3_variance": 0.0,
            "m3_verdict": "No responses",
        }

    if len(responses) == 1:
        # Single response — cannot estimate variance; assume moderate certainty
        return {
            "m3_score": 0.5,
            "m3_variance": 0.0,
            "m3_verdict": "Single response — uncertainty unknown",
        }

    model = get_model()
    embeddings = model.encode(responses, normalize_embeddings=True)   # (N, D)

    variance = _semantic_variance(embeddings)

    # Map variance → certainty score (inverse, clamped)
    certainty = 1.0 - min(variance / _MAX_EXPECTED_VARIANCE, 1.0)
    certainty = float(np.clip(certainty, 0.0, 1.0))

    if certainty >= 0.80:
        verdict = "High certainty — responses tightly clustered"
    elif certainty >= 0.55:
        verdict = "Moderate certainty"
    elif certainty >= 0.30:
        verdict = "Low certainty — notable semantic spread"
    else:
        verdict = "Very uncertain — responses diverge significantly"

    return {
        "m3_score": round(certainty, 4),
        "m3_variance": round(variance, 6),
        "m3_verdict": verdict,
    }
