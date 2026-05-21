"""
M2 — Retrieval-Augmented Grounding Module
Checks whether the LLM's answer is semantically supported by
Wikipedia-retrieved evidence.

Inspired by: RAGAs (Es et al.) and FActScore (Min et al., EMNLP 2023)
Enhancement: spaCy NLP entity extraction, relevance-ranked retrieval,
             richer return payload for UI display.
"""

import re
import requests
import wikipedia
import wikipedia.wikipedia as _wiki_module
import wikipediaapi
import numpy as np
from sentence_transformers import SentenceTransformer

_WIKI_API = wikipediaapi.Wikipedia(
    user_agent="AIHallucinationDetector/1.0 (educational; github.com/ai-hallucination)",
    language="en",
)

class _TimeoutSession(requests.Session):
    def request(self, *args, **kwargs):
        kwargs.setdefault('timeout', 5.0)
        return super().request(*args, **kwargs)

_ua_session = _TimeoutSession()
_ua_session.headers.update(
    {"User-Agent": "AIHallucinationDetector/1.0 (educational; github.com/ai-hallucination)"}
)
_wiki_module.SESSION = _ua_session

_model = None
_nlp   = None

MAX_CANDIDATES    = 8    # ← increased so we have more to filter from
SUMMARY_SENTENCES = 15

# ── Titles starting with these are almost never the main article ──────────────
_SKIP_PREFIXES = (
    "list of", "index of", "outline of", "history of",
    "glossary of", "category:", "template:", "wikipedia:",
    "portal:", "file:", "talk:", "user:",
    "career of", "filmography of", "discography of",
    "bibliography of", "personal life of", "awards and",
    "records of", "statistics of", "early life of",
)


def _get_summary(title: str, max_sentences: int = SUMMARY_SENTENCES) -> str:
    try:
        page = _WIKI_API.page(title)
        if not page.exists():
            return ""
        text = page.summary
        if not text:
            return ""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return " ".join(sentences[:max_sentences])
    except Exception:
        return ""


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp = None
    return _nlp


def _build_query(question: str) -> str:
    """
    Build a Wikipedia search query from the question using NLP entity extraction.
    This provides a universal fix for identifying the main subject (e.g., "Virat Kohli")
    without relying on hardcoded attribute words.
    """
    nlp = get_nlp()
    q = re.sub(r'[?!.,]', '', question.strip()).strip()
    
    _filler = {
        "what", "who", "where", "when", "how", "why",
        "is", "are", "was", "were", "did", "do", "does",
        "the", "a", "an", "it", "they", "this", "that",
        "tell", "me", "about", "explain", "describe",
        "give", "can", "you", "know", "information",
        "of", "in", "on", "for", "to", "with", "by", "at",
    }
    
    if nlp is not None:
        doc = nlp(question)
        
        # Look for named entities
        target_labels = {"PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT", "WORK_OF_ART", "PRODUCT"}
        entities = [ent.text for ent in doc.ents if ent.label_ in target_labels]
        
        if entities:
            return " ".join(entities)
        
        # If no named entities, extract noun chunks
        noun_chunks = [
            chunk.text for chunk in doc.noun_chunks 
            if chunk.text.lower() not in _filler
        ]
        
        if noun_chunks:
            return noun_chunks[-1]

    # Fallback if NLP fails or finds nothing
    meaningful = [w for w in q.split() if w.lower() not in _filler]
    return " ".join(meaningful[:6]) if meaningful else q

# ── Common sub-article connector words ────────────────────────────────────────
_SUBARTICLE_KEYWORDS = {
    "career", "filmography", "discography", "bibliography",
    "personal life", "early life", "awards", "records",
    "statistics", "controversies", "legacy", "honors",
}


def _is_subarticle(title: str, query: str) -> bool:
    """
    Detect if a title looks like a sub-article (e.g. 'Career of Virat Kohli')
    rather than the main article (e.g. 'Virat Kohli').
    """
    t = title.lower().strip()
    # Pattern: "<keyword> of <entity>" or "<keyword> in <entity>"
    for kw in _SUBARTICLE_KEYWORDS:
        if t.startswith(kw + " of ") or t.startswith(kw + " in "):
            return True
        # e.g. "Virat Kohli career statistics"
        if t.endswith(" " + kw):
            return True
    return False


def _relevance_score(title: str, query: str, summary: str) -> float:
    model   = get_model()
    q_lower = query.lower().strip()
    t_lower = title.lower().strip()

    if not summary:
        return 0.0

    # 1. Query vs Title similarity
    emb_qt  = model.encode([query, title], normalize_embeddings=True)
    title_sim = float(np.dot(emb_qt[0], emb_qt[1]))

    # 2. Query vs Summary similarity  
    emb_qs  = model.encode([query, summary], normalize_embeddings=True)
    summary_sim = float(np.dot(emb_qs[0], emb_qs[1]))

    # 3. Final score — title match matters more
    score = 0.6 * title_sim + 0.4 * summary_sim

    # 4. Penalize sub-articles — prefer the main entity article
    #    e.g. "Virat Kohli" should beat "Career of Virat Kohli"
    if _is_subarticle(title, query):
        score *= 0.5

    return score


def _wiki_search(query: str, results: int = 8) -> list[str]:
    """Search Wikipedia using the REST API."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": results,
    }
    headers = {"User-Agent": "AIHallucinationDetector/1.0 (educational; github.com/ai-hallucination)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=5.0)
        data = r.json()
        return [item["title"] for item in data.get("query", {}).get("search", [])]
    except Exception:
        return []


def fetch_best_context(query: str) -> dict:
    MIN_RELEVANCE = 0.20

    # ── Direct page fetch HATA DO ──────────────────
    # Yeh "paracetamol" → "Paracetamol" page deta tha
    # but "prime minister india" → fail karta tha
    # Ab sirf search use karo — har case handle hoga

    # ── Search based retrieval ─────────────────────
   
    candidates = _wiki_search(query, results=MAX_CANDIDATES)

    if not candidates:
        return {"context": "", "source": "", "found": False}

    best_score   = -1.0
    best_summary = ""
    best_title   = ""

    for title in candidates:
        # Skip list/index pages
        if any(title.lower().startswith(p) for p in _SKIP_PREFIXES):
            continue

        summary = _get_summary(title)
        if not summary:
            continue

        sc = _relevance_score(title, query, summary)
        if sc > best_score:
            best_score   = sc
            best_summary = summary
            best_title   = title

    if not best_summary or best_score < MIN_RELEVANCE:
        return {"context": "", "source": "", "found": False}

    return {
        "context": best_summary,
        "source":  f"Wikipedia — {best_title}",
        "found":   True,
    }

def _extract_relevant_sentences(question: str, context: str, top_k: int = 5) -> str:
    """Extract sentences from context most relevant to the question."""
    sentences = [s.strip() for s in context.split(".") if len(s.strip()) > 20]

    if not sentences:
        return context

    q_words = set(question.lower().split()) - {
        "what", "who", "where", "when", "how", "why",
        "is", "are", "was", "the", "a", "an", "of", "in"
    }

    # Semantic scoring using embeddings
    model = get_model()
    try:
        q_emb   = model.encode([question], normalize_embeddings=True)[0]
        s_embs  = model.encode(sentences, normalize_embeddings=True)
        scores  = [float(np.dot(q_emb, s_emb)) for s_emb in s_embs]
    except Exception:
        # Fallback: keyword matching
        scores = [sum(1 for w in q_words if w in s.lower()) for s in sentences]

    # Top-k most relevant sentences
    ranked = sorted(zip(scores, sentences), reverse=True)
    top_sentences = [s for _, s in ranked[:top_k]]

    return ". ".join(top_sentences) + "."


def compute_grounding(answer: str, context: str, question: str = "") -> float:
    if not context or not answer:
        return 0.0

    focused_context = (
        _extract_relevant_sentences(question, context)
        if question
        else context
    )

    model = get_model()
    emb   = model.encode([answer, focused_context], normalize_embeddings=True)
    sim   = float(np.dot(emb[0], emb[1]))
    return float(np.clip(sim, 0.0, 1.0))


def score(question: str, responses: list[str]) -> dict:
    if not responses:
        return {
            "m2_score":   0.0,
            "m2_verdict": "No responses",
            "context":    "",
            "source":     "",
            "found":      False,
        }

    answer = responses[0]

    # Primary search: entity from question
    query     = _build_query(question)
    retrieval = fetch_best_context(query)

    # Fallback: entity from LLM answer (handles typos in question)
    if not retrieval["found"]:
        answer_query = _build_query(answer)
        if answer_query and answer_query != query:
            retrieval = fetch_best_context(answer_query)

    grounding = compute_grounding(answer, retrieval["context"], question=question)

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
        "m2_score":   round(grounding, 4),
        "m2_verdict": verdict,
        "context":    retrieval["context"],
        "source":     retrieval["source"],
        "found":      retrieval["found"],
    }