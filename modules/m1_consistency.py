import numpy as np
from itertools import combinations
from sentence_transformers import SentenceTransformer

# load model once
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def compute_similarity(responses):
    if len(responses) < 2:
        return 1.0

    model = get_model()

    embeddings = model.encode(responses, normalize_embeddings=True)

    sims = []
    for i, j in combinations(range(len(embeddings)), 2):
        sims.append(float(np.dot(embeddings[i], embeddings[j])))

    return max(0.0, min(1.0, np.mean(sims)))


def score(question, responses):
    if not responses:
        return {"m1_score": 0.0}

    consistency = compute_similarity(responses)

    return {
        "m1_score": consistency
    }