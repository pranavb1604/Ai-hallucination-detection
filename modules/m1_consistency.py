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
    """Return cosine similarities for all unique pairs of embeddings."""
    n = len(embeddings)
    sims = []
    for i, j in combinations(range(n), 2):
        sim = float(np.dot(embeddings[i], embeddings[j]))
        sims.append(sim)
    return sims
 
 
def compute_consistency(responses: list[str]) -> dict:
    """
    Encode all responses and compute:
      - mean pairwise similarity  (overall consistency)
      - min  pairwise similarity  (worst-case pair, most divergent)
      - std  pairwise similarity  (spread / volatility)
      - per-pair breakdown for UI display
    """
    if len(responses) < 2:
        return {
            "mean": 1.0,
            "min": 1.0,
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
    """
    Public API for the pipeline.
 
    Returns
    -------
    m1_score   : float  — primary signal fed to meta-classifier (mean similarity)
    m1_min     : float  — worst-case pair similarity (useful for SHAP)
    m1_std     : float  — spread across all pairs
    m1_pairs   : list   — per-pair breakdown for UI
    m1_verdict : str    — human-readable label
    """
    if not responses:
        return {
            "m1_score": 0.0,
            "m1_min": 0.0,
            "m1_std": 0.0,
            "m1_pairs": [],
            "m1_verdict": "No responses",
        }
 
    stats = compute_consistency(responses)
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
 