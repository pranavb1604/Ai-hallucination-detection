"""
M2 — Retrieval-Augmented Grounding Module
Checks whether the LLM's answer is semantically supported by
Wikipedia-retrieved evidence.

Inspired by: RAGAs (Es et al.) and FActScore (Min et al., EMNLP 2023)
Enhancement: multi-candidate retrieval with best-match selection,
             richer return payload for UI display.
"""

import wikipedia
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None

MAX_CANDIDATES = 3   # Wikipedia pages to try
SUMMARY_SENTENCES = 5


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _build_query(question: str) -> str:
    """Clean up short queries so Wikipedia search returns useful results."""
    q = question.strip()
    if len(q.split()) <= 2:
        q += " overview"
    return q


def fetch_best_context(query: str) -> dict:
    """
    Try up to MAX_CANDIDATES Wikipedia pages and return the one whose
    summary is longest (proxy for relevance when a short query matches
    a disambiguation page).
    """
    try:
        candidates = wikipedia.search(query, results=MAX_CANDIDATES)
    except Exception:
        return {"context": "", "source": "", "found": False}

    if not candidates:
        return {"context": "", "source": "", "found": False}

    best_summary = ""
    best_title = ""

    for title in candidates:
        try:
            summary = wikipedia.summary(title, sentences=SUMMARY_SENTENCES)
            if len(summary) > len(best_summary):
                best_summary = summary
                best_title = title
        except Exception:
            continue

    if not best_summary:
        return {"context": "", "source": "", "found": False}

    return {
        "context": best_summary,
        "source": f"Wikipedia — {best_title}",
        "found": True,
    }


def compute_grounding(answer: str, context: str) -> float:
    """
    Cosine similarity between the answer embedding and the context embedding.
    Normalized to [0, 1].
    """
    if not context or not answer:
        return 0.0

    model = get_model()
    emb = model.encode([answer, context], normalize_embeddings=True)
    sim = float(np.dot(emb[0], emb[1]))
    return float(np.clip(sim, 0.0, 1.0))


def score(question: str, responses: list[str]) -> dict:
    """
    Public API for the pipeline.

    Uses responses[0] as the 'primary answer' to ground — consistent with
    how the UI displays the first response as the main answer.

    Returns
    -------
    m2_score   : float — primary grounding signal for meta-classifier
    m2_verdict : str   — human-readable label
    context    : str   — retrieved Wikipedia passage (shown in UI)
    source     : str   — Wikipedia page title
    found      : bool  — whether any context was retrieved
    """
    if not responses:
        return {
            "m2_score": 0.0,
            "m2_verdict": "No responses",
            "context": "",
            "source": "",
            "found": False,
        }

    answer = responses[0]
    query = _build_query(question)
    retrieval = fetch_best_context(query)

    grounding = compute_grounding(answer, retrieval["context"])

    if not retrieval["found"]:
        verdict = "No evidence found"
    elif grounding >= 0.75:
        verdict = "Well-supported by evidence"
    elif grounding >= 0.50:
        verdict = "Partially supported"
    elif grounding >= 0.30:
        verdict = "Weakly supported"
    else:
        verdict = "Not supported — possible hallucination"

    return {
        "m2_score": round(grounding, 4),
        "m2_verdict": verdict,
        "context": retrieval["context"],
        "source": retrieval["source"],
        "found": retrieval["found"],
    }