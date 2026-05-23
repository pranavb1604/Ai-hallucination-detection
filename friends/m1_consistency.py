"""
M1 — Semantic Consistency Module
Detects hallucination by sampling the same question N times and measuring
how consistent the LLM's answers are via pairwise cosine similarity.

Inspired by: SelfCheckGPT (Manakul et al., EMNLP 2023)
Enhancement: dense sentence embeddings instead of n-gram matching.
"""

import numpy as np
from itertools import combinations
from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def compute_pairwise_similarities(embeddings: np.ndarray) -> list[float]:
   
    n = len(embeddings)
    sims = []
    for i, j in combinations(range(n), 2):
        sim = float(np.dot(embeddings[i], embeddings[j]))
        sims.append(sim)
    return sims


def _question_answer_coherence(question: str, answer: str) -> float:
    """Single-response fallback: semantic alignment between Q and A."""
    if not question or not answer:
        return 0.5
    model = get_model()
    emb = model.encode([question, answer], normalize_embeddings=True)
    return float(np.clip(np.dot(emb[0], emb[1]), 0.0, 1.0))


def compute_consistency(responses: list[str], question: str = "") -> dict:

    if len(responses) < 2:
        mean = (
            _question_answer_coherence(question, responses[0])
            if responses and question
            else 0.5
        )
        return {
            "mean": mean,
            "min": mean,
            "std": 0.0,
            "pairs": [],
        }

    model = get_model()
    embeddings = model.encode(responses, normalize_embeddings=True)
    raw_sims = compute_pairwise_similarities(embeddings)

    arr = np.array(raw_sims)

    pairs = []
    idx = 0
    n = len(responses)
    for i, j in combinations(range(n), 2):
        pairs.append({
            "i": i,
            "j": j,
            "similarity": round(float(raw_sims[idx]), 4),
        })
        idx += 1

    return {
        "mean": float(np.clip(arr.mean(), 0.0, 1.0)),
        "min": float(np.clip(arr.min(), 0.0, 1.0)),
        "std": float(arr.std()),
        "pairs": pairs,
    }


def score(question: str, responses: list[str]) -> dict:
    
    if not responses:
        return {
            "m1_score": 0.0,
            "m1_min": 0.0,
            "m1_std": 0.0,
            "m1_pairs": [],
            "m1_verdict": "No responses",
        }

    stats = compute_consistency(responses, question=question)
    mean = stats["mean"]

    if mean >= 0.85:
        verdict = "Highly consistent"
    elif mean >= 0.65:
        verdict = "Moderately consistent"
    elif mean >= 0.45:
        verdict = "Low consistency"
    else:
        verdict = "Inconsistent — high hallucination risk"

    return {
        "m1_score": round(mean, 4),
        "m1_min": round(stats["min"], 4),
        "m1_std": round(stats["std"], 4),
        "m1_pairs": stats["pairs"],
        "m1_verdict": verdict,
    }