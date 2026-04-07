"""
modules/m2_grounding.py  —  Module 2: Retrieval-Augmented Grounding

Fetches a Wikipedia passage for the input question, then measures the
cosine similarity between the answer embedding and the evidence embedding.
Low similarity → answer is not grounded → likely hallucinated.

Inspired by RAGAs (Es et al.) and FActScore (Min et al., EMNLP 2023).

Score returned: float in [0, 1]
    1.0 = answer strongly supported by retrieved evidence
    0.0 = answer contradicts or is unrelated to evidence
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sentence_transformers import SentenceTransformer
from config import M2_MODEL_NAME, M2_EVIDENCE_LENGTH, M2_GROUNDING_THRESHOLD
from utils.helpers import get_wikipedia_evidence, normalize_score, clean_text

# ── Load model once at import time ───────────────────────────
_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(M2_MODEL_NAME)
    return _model


# ──────────────────────────────────────────────
# Core scoring
# ──────────────────────────────────────────────

def compute_grounding_score(answer: str, evidence: str) -> float:
    """
    Computes cosine similarity between answer and evidence embeddings.
    Both vectors are L2-normalised so similarity = dot product.
    """
    if not answer.strip() or not evidence.strip():
        return 0.0

    model = _get_model()
    embs = model.encode(
        [clean_text(answer), clean_text(evidence)],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    sim = float(np.dot(embs[0], embs[1]))
    return normalize_score(sim)


def score(question: str, answer: str) -> dict:
    """
    Public API for M2.

    Args:
        question : input question (used as Wikipedia search query)
        answer   : LLM-generated answer to evaluate

    Returns:
        {
            "m2_score":       float,   # cosine similarity with Wiki evidence [0,1]
            "is_grounded":    bool,    # True if score >= threshold
            "evidence_found": bool,
            "evidence_snippet": str,  # first 200 chars of evidence (for UI)
        }
    """
    evidence = get_wikipedia_evidence(question, evidence_length=M2_EVIDENCE_LENGTH)

    if not evidence:
        # No evidence found — return neutral score, flag it
        return {
            "m2_score":         0.5,
            "is_grounded":      True,   # benefit of doubt when no evidence
            "evidence_found":   False,
            "evidence_snippet": "",
        }

    m2_score = compute_grounding_score(answer, evidence)

    return {
        "m2_score":         m2_score,
        "is_grounded":      m2_score >= M2_GROUNDING_THRESHOLD,
        "evidence_found":   True,
        "evidence_snippet": evidence[:200],
    }


# ──────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        {
            "question": "Who invented the telephone?",
            "answer":   "Alexander Graham Bell invented the telephone in 1876.",
        },
        {
            "question": "Who invented the telephone?",
            "answer":   "Nikola Tesla invented the telephone while working in Paris in 1890.",
        },
    ]

    for t in tests:
        print(f"\nQ: {t['question']}")
        print(f"A: {t['answer']}")
        result = score(t["question"], t["answer"])
        print(f"Result: {result}")