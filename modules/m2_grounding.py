import wikipedia
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def fetch_context(query):
    try:
        results = wikipedia.search(query)

        if not results:
            return ""

        return wikipedia.summary(results[0], sentences=3)

    except:
        return ""


def compute_score(answer, context):
    if not context:
        return 0.0

    model = get_model()

    emb = model.encode([answer, context], normalize_embeddings=True)

    sim = float(np.dot(emb[0], emb[1]))

    return max(0.0, min(1.0, sim))


def score(question, responses):
    if not responses:
        return {"m2_score": 0.0, "context": ""}

    answer = responses[0]

    # improve query
    query = question.strip()
    if len(query.split()) <= 2:
        query += " wikipedia"

    context = fetch_context(query)

    grounding = compute_score(answer, context)

    return {
        "m2_score": grounding,
        "context": context
    }