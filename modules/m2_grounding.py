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

MAX_CANDIDATES    = 12
SUMMARY_SENTENCES = 15
MIN_RELEVANCE     = 0.28
_WIKI_HEADERS     = {"User-Agent": "AIHallucinationDetector/1.0 (educational; github.com/ai-hallucination)"}

# ── Titles starting with these are almost never the main article ──────────────
_SKIP_PREFIXES = (
    "list of", "index of", "outline of", "history of",
    "glossary of", "category:", "template:", "wikipedia:",
    "portal:", "file:", "talk:", "user:", "help:",
    "career of", "filmography of", "discography of",
    "bibliography of", "personal life of", "awards and",
    "records of", "statistics of", "early life of",
    "death of", "murder of", "assassination of",
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


_ROLE_PHRASES = (
    "president", "prime minister", "vice president", "king", "queen",
    "chancellor", "ceo", "governor", "mayor", "leader", "head of state",
    "monarch", "ruler", "dictator", "pm",
)


def _question_asks_for_person(question: str) -> bool:
    ql = question.lower()
    return ql.startswith("who ") or " who " in ql or any(r in ql for r in _ROLE_PHRASES)


def _extract_role_queries(question: str) -> list[str]:
    """
    'who is president of russia' -> 'President of Russia'
    'prime minister of india'    -> 'Prime Minister of India'
    """
    ql = question.strip().rstrip("?").lower()
    out = []

    m = re.search(
        r"(?:who\s+is\s+(?:the\s+)?|what\s+is\s+(?:the\s+)?)?"
        r"(president|prime minister|vice president|king|queen|chancellor|ceo|"
        r"governor|mayor|leader|head of state|monarch|ruler|pm)\s+of\s+(?:the\s+)?(.+)$",
        ql,
    )
    if m:
        role, place = m.group(1), m.group(2).strip()
        if role == "pm":
            role = "prime minister"
        place = re.sub(r"\s+(today|now|currently|right now)$", "", place)
        out.append(f"{role.title()} of {place.title()}")

    return out


def _build_query(question: str, prefer_person: bool = False) -> str:
    """
    Build a Wikipedia search query from text using NLP entity extraction.
    prefer_person: for 'who is ...' questions, use PERSON before GPE (avoid 'Russia' only).
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
        target_labels = {"PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT", "WORK_OF_ART", "PRODUCT"}

        if prefer_person:
            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            if persons:
                return persons[0]

        entities = [ent.text for ent in doc.ents if ent.label_ in target_labels]
        if entities:
            return " ".join(entities)

        noun_chunks = [
            chunk.text for chunk in doc.noun_chunks
            if chunk.text.lower() not in _filler
        ]
        if noun_chunks:
            return noun_chunks[-1]

    meaningful = [w for w in q.split() if w.lower() not in _filler]
    return " ".join(meaningful[:6]) if meaningful else q


def _collect_search_queries(question: str, answer: str = "") -> list[str]:
    """Build Wikipedia queries — role/person first, bare country name last."""
    seen, queries = set(), []
    ask_person = _question_asks_for_person(question)

    def add(q: str, front: bool = False):
        q = re.sub(r"\s+", " ", (q or "").strip())
        if len(q) < 2:
            return
        key = q.lower()
        if key in seen:
            return
        seen.add(key)
        if front:
            queries.insert(0, q)
        else:
            queries.append(q)

    # 1. Role pages: President of Russia, Prime Minister of India, ...
    for rq in _extract_role_queries(question):
        add(rq, front=True)

    # 2. Person named in the LLM answer (e.g. Vladimir Putin)
    if answer:
        add(_build_query(answer[:300], prefer_person=True), front=True)

    # 3. Full question phrase (before bare 'Russia')
    q_clean = re.sub(r"[?!.,]", "", question.strip())
    _filler = {
        "what", "who", "where", "when", "how", "why", "which",
        "is", "are", "was", "were", "did", "do", "does", "the", "a", "an",
        "tell", "me", "about", "name", "many", "much",
    }
    words = [w for w in q_clean.split() if w.lower() not in _filler]
    if len(words) >= 2:
        add(" ".join(words[:8]))

    for m in re.findall(r'"([^"]{3,80})"', question):
        add(m)

    # 4. Generic entity last (often too broad: 'Russia' alone)
    add(_build_query(question, prefer_person=ask_person))

    return queries


def _title_variants(text: str) -> list[str]:
    t = text.strip()
    if not t:
        return []
    out = [t]
    if t != t.title():
        out.append(t.title())
    # Wikipedia often uses exact entity casing from search
    if " " in t:
        out.append(t.title())  # "virat kohli" -> "Virat Kohli"
    return list(dict.fromkeys(out))


def _should_skip_title(title: str) -> bool:
    tl = title.lower().strip()
    if any(tl.startswith(p) for p in _SKIP_PREFIXES):
        return True
    if "(disambiguation)" in tl:
        return True
    return False


def _clean_wiki_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "")


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


def _relevance_score(
    title: str,
    query: str,
    summary: str,
    question: str = "",
    answer: str = "",
    snippet: str = "",
) -> float:
    if not summary and not snippet:
        return 0.0

    model = get_model()
    texts = [query, title]
    if question and question != query:
        texts.append(question[:200])
    emb = model.encode(texts, normalize_embeddings=True)
    title_sim = float(np.dot(emb[0], emb[1]))
    q_sim = title_sim
    if len(texts) > 2:
        q_sim = max(q_sim, float(np.dot(emb[2], emb[1])))

    ctx = snippet if snippet else summary[:600]
    emb_cs = model.encode([query, ctx], normalize_embeddings=True)
    summary_sim = float(np.dot(emb_cs[0], emb_cs[1]))

    score = 0.45 * title_sim + 0.35 * summary_sim + 0.20 * q_sim

    if answer:
        emb_a = model.encode([answer[:200], title], normalize_embeddings=True)
        score = 0.75 * score + 0.25 * float(np.dot(emb_a[0], emb_a[1]))

    q_lower, t_lower = query.lower(), title.lower()
    if q_lower == t_lower or q_lower in t_lower or t_lower in q_lower:
        score += 0.12
    if _is_subarticle(title, query):
        score *= 0.45
    # Prefer main articles over disambiguation / long parenthetical titles
    if re.search(r"\([^)]+\)$", title) and "disambiguation" not in t_lower:
        score *= 0.88
    if len(title.split()) <= max(3, len(query.split()) + 1):
        score += 0.04

    # Role questions: prefer "President of Russia" over country article "Russia"
    if question:
        ql = question.lower()
        roles_in_q = [r for r in _ROLE_PHRASES if r in ql]
        if roles_in_q:
            if any(r in t_lower for r in roles_in_q):
                score += 0.20
            elif len(title.split()) <= 2 and not any(
                w in t_lower for w in ("president", "minister", "leader", "monarch")
            ):
                # e.g. title "Russia" when user asked for president
                score *= 0.30

    return float(np.clip(score, 0.0, 1.0))


def _wiki_api(params: dict) -> dict:
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={**params, "format": "json"},
            headers=_WIKI_HEADERS,
            timeout=6.0,
        )
        return r.json()
    except Exception:
        return {}


def _wiki_opensearch(query: str, limit: int = 6) -> list[str]:
    data = _wiki_api({"action": "opensearch", "search": query, "limit": limit})
    if isinstance(data, list) and len(data) > 1:
        return list(data[1]) if isinstance(data[1], list) else []
    return []


def _wiki_search_enriched(query: str, results: int = MAX_CANDIDATES) -> list[dict]:
    data = _wiki_api({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": results,
        "srprop": "snippet",
    })
    items = data.get("query", {}).get("search", [])
    return [
        {
            "title": it["title"],
            "snippet": _clean_wiki_html(it.get("snippet", "")),
        }
        for it in items
    ]


def _direct_page_titles(queries: list[str]) -> list[str]:
    """Resolve exact Wikipedia page titles (handles redirects)."""
    titles = []
    for q in queries:
        for variant in _title_variants(q):
            try:
                page = _WIKI_API.page(variant)
                if page.exists() and page.title not in titles:
                    titles.append(page.title)
            except Exception:
                continue
    return titles


def _gather_candidates(queries: list[str], question: str = "", answer: str = "") -> dict[str, dict]:
    """Collect title -> {snippet} from direct lookup, opensearch, and search."""
    pool: dict[str, dict] = {}

    def add(title: str, snippet: str = ""):
        if not title or _should_skip_title(title):
            return
        if title not in pool or (snippet and not pool[title].get("snippet")):
            pool[title] = {"snippet": snippet}

    add_direct = _direct_page_titles(queries)
    for t in add_direct:
        add(t)

    for q in queries:
        for t in _wiki_opensearch(q):
            add(t)
        for hit in _wiki_search_enriched(q):
            add(hit["title"], hit.get("snippet", ""))

    return pool


def fetch_best_context(
    query: str,
    question: str = "",
    answer: str = "",
) -> dict:
    """
    Retrieve the best-matching Wikipedia article for a query.
    Pass question/answer for stronger ranking when available.
    """
    queries = [query] if query else []
    if question:
        for q in _collect_search_queries(question, answer):
            if q not in queries:
                queries.append(q)

    if not queries:
        return {"context": "", "source": "", "found": False, "relevance": 0.0, "title": ""}

    pool = _gather_candidates(queries, question=question, answer=answer)
    if not pool:
        return {"context": "", "source": "", "found": False, "relevance": 0.0, "title": ""}

    primary_q = query or queries[0]
    ranked: list[tuple[float, str]] = []

    # Phase 1 — rank using search snippets (fast; no per-page fetch yet)
    for title, meta in pool.items():
        snip = meta.get("snippet", "")
        sc = _relevance_score(
            title, primary_q, snip or title,
            question=question, answer=answer, snippet=snip,
        )
        ranked.append((sc, title))
    ranked.sort(reverse=True)

    best_score, best_summary, best_title = -1.0, "", ""
    # Phase 2 — fetch full summaries only for top candidates
    for sc_hint, title in ranked[:5]:
        summary = _get_summary(title)
        if not summary:
            continue
        sc = _relevance_score(
            title, primary_q, summary,
            question=question, answer=answer,
            snippet=pool.get(title, {}).get("snippet", ""),
        )
        if sc > best_score:
            best_score, best_summary, best_title = sc, summary, title

    if not best_summary or best_score < MIN_RELEVANCE:
        return {"context": "", "source": "", "found": False, "relevance": best_score, "title": ""}

    return {
        "context": best_summary,
        "source":  f"Wikipedia — {best_title}",
        "found":   True,
        "relevance": round(best_score, 4),
        "title": best_title,
    }


def fetch_evidence_for_qa(question: str, answer: str = "") -> dict:
    """
    Try multiple query strategies in one pass; return the best article.
    Used by M2 and M4.
    """
    queries = _collect_search_queries(question, answer)
    if len(question) > 5:
        raw = question[:100].strip()
        if raw.lower() not in {q.lower() for q in queries}:
            queries.append(raw)

    if not queries:
        return {"context": "", "source": "", "found": False, "relevance": 0.0, "title": ""}

    return fetch_best_context(queries[0], question=question, answer=answer)

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
    retrieval = fetch_evidence_for_qa(question, answer)
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
        "wiki_title": retrieval.get("title", ""),
        "wiki_relevance": retrieval.get("relevance", 0.0),
    }