"""
m1_consistency.py

Idea:
Generate multiple responses for the same question and check how similar they are.
If responses are very similar → likely correct
If they vary a lot → possible hallucination
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from itertools import combinations
from sentence_transformers import SentenceTransformer
from config import M1_MODEL_NAME, M1_SIMILARITY_THRESHOLD
from utils.helpers import normalize_score, clean_text

# load model once
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(M1_MODEL_NAME)
    return _model


# --------------------------------------------------
# main similarity logic
# --------------------------------------------------

def compute_pairwise_similarity(responses):
    if len(responses) < 2:
        return 1.0

    cleaned = [clean_text(r) for r in responses if r and r.strip()]
    if len(cleaned) < 2:
        return 1.0

    model = _get_model()
    emb = model.encode(cleaned, normalize_embeddings=True)

    sims = []
    for i, j in combinations(range(len(emb)), 2):
        sims.append(float(np.dot(emb[i], emb[j])))

    return normalize_score(np.mean(sims))


# optional: check if answers align with question
def compute_alignment(question, responses):
    model = _get_model()

    cleaned = [clean_text(r) for r in responses if r and r.strip()]
    if not cleaned:
        return 0.0

    q_emb = model.encode([clean_text(question)], normalize_embeddings=True)[0]
    r_embs = model.encode(cleaned, normalize_embeddings=True)

    sims = [float(np.dot(q_emb, r)) for r in r_embs]

    return normalize_score(np.mean(sims))


# --------------------------------------------------
# public API
# --------------------------------------------------

def score(question, responses):
    if not responses:
        return {"m1_score": 0.0, "num_samples": 0}

    consistency = compute_pairwise_similarity(responses)
    alignment   = compute_alignment(question, responses)

    # weighted combination (kept simple)
    final_score = 0.7 * consistency + 0.3 * alignment

    return {
        "m1_score": final_score,
        "consistency": consistency,
        "alignment": alignment,
        "num_samples": len(responses),
    }


# fallback when only one answer available
def score_single_answer(answer):
    return {
        "m1_score": 0.5,   # neutral
        "num_samples": 1,
        "note": "single_answer"
    }


# --------------------------------------------------
# quick test
# --------------------------------------------------

if __name__ == "__main__":
    good = [
        "The Eiffel Tower is in Paris, France.",
        "Paris, France is home to the Eiffel Tower.",
        "The Eiffel Tower is located in Paris."
    ]

    bad = [
        "The Eiffel Tower is in Paris.",
        "The Eiffel Tower is in Berlin.",
        "The Eiffel Tower is in Rome."
    ]

    print("Consistent case:")
    print(score("Where is Eiffel Tower?", good))

    print("\nInconsistent case:")
    print(score("Where is Eiffel Tower?", bad))