"""
modules/m4_entailment.py  —  Module 4: NLI Entailment Scoring

Inspired by FActScore (Min et al., EMNLP 2023).

Splits the answer into atomic claims (sentences), retrieves Wikipedia evidence
for each claim, then uses a cross-encoder NLI model (DeBERTa-v3-small) to
classify each (evidence, claim) pair as:
    ENTAILMENT   → claim is supported   (score contribution: +1)
    NEUTRAL      → claim is not covered (score contribution: +0.5)
    CONTRADICTION → claim is refuted    (score contribution:  0)

The final M4 score is the mean over all claims.

Score returned: float in [0, 1]
    1.0 = all claims entailed by evidence
    0.0 = all claims contradicted by evidence
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from transformers import pipeline as hf_pipeline
from config import M4_MODEL_NAME, M4_EVIDENCE_LENGTH, M4_MIN_CLAIM_LENGTH
from utils.helpers import (
    get_wikipedia_evidence,
    split_into_claims,
    normalize_score,
    clean_text,
)

# ── Load NLI pipeline once at import time ────────────────────
_nli_pipeline = None

def _get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is None:
        _nli_pipeline = hf_pipeline(
            "text-classification",
            model=M4_MODEL_NAME,
            return_all_scores=True,   # get scores for all 3 labels
        )
    return _nli_pipeline


# ──────────────────────────────────────────────
# Label mapping
# ──────────────────────────────────────────────

# DeBERTa cross-encoder NLI label order: entailment, neutral, contradiction
# (verify with model card if using a different NLI model)
_LABEL_SCORES = {
    "entailment":    1.0,
    "neutral":       0.5,
    "contradiction": 0.0,
}

def _nli_score_for_pair(premise: str, hypothesis: str) -> float:
    """
    Runs NLI on (premise, hypothesis) and returns a single float
    based on the winning label.
    """
    nli = _get_nli_pipeline()
    # cross-encoder format: [CLS] premise [SEP] hypothesis
    result = nli(f"{premise} [SEP] {hypothesis}", truncation=True, max_length=512)

    # result is a list of dicts: [{"label": ..., "score": ...}, ...]
    best = max(result[0], key=lambda x: x["score"])
    label = best["label"].lower()

    # Map label string to score (handle model-specific label names)
    for key in _LABEL_SCORES:
        if key in label:
            return _LABEL_SCORES[key]

    return 0.5   # unknown label → neutral


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def score(question: str, answer: str) -> dict:
    """
    Public API for M4.

    Args:
        question : original question (used for Wikipedia search)
        answer   : LLM-generated answer to evaluate

    Returns:
        {
            "m4_score":          float,    # mean entailment score [0,1]
            "num_claims":        int,
            "claim_scores":      list[float],
            "evidence_found":    bool,
        }
    """
    claims = split_into_claims(answer, min_length=M4_MIN_CLAIM_LENGTH)

    if not claims:
        return {
            "m4_score":       0.5,
            "num_claims":     0,
            "claim_scores":   [],
            "evidence_found": False,
        }

    # Fetch one evidence passage per question (reuse across all claims)
    evidence = get_wikipedia_evidence(question, evidence_length=M4_EVIDENCE_LENGTH)

    if not evidence:
        # No evidence — all claims get neutral score
        return {
            "m4_score":       0.5,
            "num_claims":     len(claims),
            "claim_scores":   [0.5] * len(claims),
            "evidence_found": False,
        }

    premise = clean_text(evidence)
    claim_scores = []

    for claim in claims:
        hypothesis = clean_text(claim)
        cs = _nli_score_for_pair(premise, hypothesis)
        claim_scores.append(cs)

    m4_score = normalize_score(float(np.mean(claim_scores)))

    return {
        "m4_score":       m4_score,
        "num_claims":     len(claims),
        "claim_scores":   claim_scores,
        "evidence_found": True,
    }


# ──────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        {
            "question": "What is photosynthesis?",
            "answer":   (
                "Photosynthesis is a process used by plants and other organisms to "
                "convert light energy into chemical energy. It occurs in chloroplasts. "
                "Plants absorb carbon dioxide and release oxygen during this process."
            ),
        },
        {
            "question": "What is photosynthesis?",
            "answer":   (
                "Photosynthesis is how animals digest food in the stomach. "
                "It requires no sunlight and produces carbon dioxide as a byproduct."
            ),
        },
    ]

    for t in tests:
        print(f"\nQ: {t['question']}")
        print(f"A: {t['answer'][:80]}...")
        result = score(t["question"], t["answer"])
        print(f"Result: {result}")