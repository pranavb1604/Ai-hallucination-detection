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

# ── Fix: The `wikipedia` package's summary() is broken — Wikipedia's API
# blocks its requests (missing/bad User-Agent), returning empty bodies that
# cause JSONDecodeError.  We keep `wikipedia.search()` (which still works)
# but use the `wikipedia-api` package for fetching page content reliably.
_WIKI_API = wikipediaapi.Wikipedia(
    user_agent="AIHallucinationDetector/1.0 (educational; github.com/ai-hallucination)",
    language="en",
)

# Still need a patched session for wikipedia.search() calls
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

MAX_CANDIDATES    = 5
SUMMARY_SENTENCES = 15


def _get_summary(title: str, max_sentences: int = SUMMARY_SENTENCES) -> str:
    """
    Fetch a Wikipedia page summary using the reliable `wikipedia-api` package.
    Returns the first `max_sentences` sentences, or an empty string on failure.
    """
    try:
        page = _WIKI_API.page(title)
        if not page.exists():
            return ""
        text = page.summary
        if not text:
            return ""
        # Truncate to max_sentences
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
    """Load spaCy model (lazy, cached)."""
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
    Extract the key topic from a question for Wikipedia search.

    Strategy:
      1. Named entity + meaningful words → "India population"
      2. Noun chunks minus filler → "population India"
      3. Fallback: first 4 meaningful words
    """

    q = re.sub(r'[?!.,]', '', question.strip()).strip()

    _filler = {
        "what", "who", "where", "when", "how", "why",
        "is", "are", "was", "were", "did", "do", "does",
        "the", "a", "an", "it", "they", "this", "that", "i",
        "tell", "me", "about", "explain", "describe",
        "give", "can", "you", "know", "information",
        "of", "in", "on", "for", "to", "with", "by", "at",
    }

    nlp = get_nlp()
    if nlp is not None and q:
        doc = nlp(q)

        # Step 1 — Named entity + topic context words
        if doc.ents:
            entity = doc.ents[0].text
            # Grab meaningful non-entity, non-filler words for context
            context_words = [
                t.text for t in doc
                if t.text.lower() not in _filler
                and t.text.lower() not in entity.lower().split()
                and not t.is_punct
                and t.pos_ in ("NOUN", "PROPN", "ADJ")
            ]
            if context_words:
                return f"{entity} {' '.join(context_words[:2])}"
            return entity

        # Step 2 — Noun chunks, skip filler words
        nouns = [chunk.text for chunk in doc.noun_chunks
                 if chunk.text.lower() not in _filler]
        if nouns:
            return " ".join(nouns[:2])

    # Step 3 — Fallback: first 4 meaningful words
    meaningful = [w for w in q.split() if w.lower() not in _filler]
    return " ".join(meaningful[:4]) if meaningful else q


def _relevance_score(title: str, query: str, summary: str) -> float:
    """Score a Wikipedia candidate by relevance to query."""
    model   = get_model()
    q_lower = query.lower()
    t_lower = title.lower()

    # Title match bonus
    title_bonus = 1.0 if q_lower in t_lower or t_lower in q_lower else 0.0

    # Semantic similarity between query and summary
    if summary:
        emb = model.encode([query, summary], normalize_embeddings=True)
        sem = float(np.dot(emb[0], emb[1]))
    else:
        sem = 0.0

    return sem + 0.3 * title_bonus


def _wiki_search(query: str, results: int = 5) -> list[str]:
    """Search Wikipedia using the REST API to avoid the broken `wikipedia` package."""
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
    """
    Search Wikipedia, validate relevance, return best matching article.

    Uses a custom REST API search rather than wikipedia.search() which
    is currently broken (throws JSONDecodeError).
    """
    wikipedia.set_lang("en")

    # Minimum semantic relevance to accept a Wikipedia article.
    # Clearly wrong articles score ~0.03-0.08; valid ones score 0.20+.
    MIN_RELEVANCE = 0.20

    # Use robust direct API call — it returns a ranked list
    candidates = _wiki_search(query, results=MAX_CANDIDATES)

    if not candidates:
        return {"context": "", "source": "", "found": False}

    best_score   = -1.0
    best_summary = ""
    best_title   = ""

    for title in candidates:
        summary = _get_summary(title)
        if not summary:
            continue
        sc = _relevance_score(title, query, summary)
        if sc > best_score:
            best_score   = sc
            best_summary = summary
            best_title   = title

    # Reject if the best article is clearly unrelated to the query
    if not best_summary or best_score < MIN_RELEVANCE:
        return {"context": "", "source": "", "found": False}

    return {
        "context": best_summary,
        "source":  f"Wikipedia — {best_title}",
        "found":   True,
    }


def _extract_relevant_sentences(question: str, context: str, top_k: int = 4) -> str:
    """
    Given the original question and a Wikipedia context paragraph,
    return the top-K sentences most relevant to the question.

    Why this matters
    ----------------
    A Wikipedia article about "Eiffel Tower" covers history, dimensions,
    construction, tourism, etc.  If the user asked "where is the Eiffel
    Tower?", only the location sentences are relevant for grounding —
    comparing the answer against the full article dilutes the signal and
    can produce false hallucination verdicts.
    """
    # Split context into individual sentences (simple heuristic)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if s.strip()]
    if not sentences:
        return context
    if len(sentences) <= top_k:
        return context   # short context, no need to filter

    model = get_model()
    q_emb = model.encode([question], normalize_embeddings=True)[0]
    s_embs = model.encode(sentences, normalize_embeddings=True)

    scores = np.dot(s_embs, q_emb)          # cosine similarity for each sentence
    top_idx = np.argsort(scores)[::-1][:top_k]
    top_idx_sorted = sorted(top_idx)        # preserve reading order

    return " ".join(sentences[i] for i in top_idx_sorted)


def compute_grounding(answer: str, context: str, question: str = "") -> float:
    """
    Grounding score = cosine similarity between the LLM answer and the
    most question-relevant sentences from the Wikipedia context.

    Parameters
    ----------
    answer   : LLM-generated response to check.
    context  : Wikipedia evidence paragraph.
    question : Original user question (used to focus the context).
                If omitted, the full context is used (legacy behaviour).
    """
    if not context or not answer:
        return 0.0

    # Focus context on sentences relevant to the question's intent
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
    """
    Main scoring function for M2 grounding module.

    Fallback strategy for typos / misspellings
    -------------------------------------------
    If the question-based Wikipedia search fails (e.g., "effile tower"),
    we retry using the LLM's answer as the search query.  The LLM almost
    always corrects the spelling in its response ("Eiffel Tower"), so this
    gives us a second chance to find the right Wikipedia article.
    """

    if not responses:
        return {
            "m2_score":   0.0,
            "m2_verdict": "No responses",
            "context":    "",
            "source":     "",
            "found":      False,
        }

    answer = responses[0]

    # --- Primary search: use entity from question ---
    query     = _build_query(question)
    retrieval = fetch_best_context(query)

    # --- Fallback search: use entity from LLM answer (handles typos) ---
    if not retrieval["found"]:
        answer_query = _build_query(answer)           # answer is usually well-spelled
        if answer_query and answer_query != query:    # avoid pointless repeat
            retrieval = fetch_best_context(answer_query)

    # Pass original question so grounding focuses on intent-relevant sentences
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
