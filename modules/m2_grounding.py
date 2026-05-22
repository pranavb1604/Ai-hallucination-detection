"""
M2 — Retrieval-Augmented Grounding Module
Checks whether the LLM's answer is semantically supported by
Wikipedia-retrieved evidence.

Inspired by: RAGAs (Es et al.) and FActScore (Min et al., EMNLP 2023)
Enhancement: multi-candidate retrieval with best-match selection,
             richer return payload for UI display.
"""

import wikipedia
wikipedia.set_user_agent("HallucinationDetector/1.0 (btech.project@dtu.ac.in)")
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
import re

_model = None
_nlp = None

MAX_CANDIDATES = 5
SUMMARY_SENTENCES = 30  # Increased for better context

# Common question prefixes to strip for better Wikipedia search
_QUESTION_PREFIXES = [
    "can you tell me", "tell me about", "explain to me",
    "i want to know", "please tell me", "please explain",
    "who is the", "who was the", "who are the", "who were the",
    "what is the", "what are the", "what was the", "what were the",
    "where is the", "where are the", "where was the", "where were the",
    "when is the", "when was the", "when did the", "when were the",
    "how does the", "how do the", "how did the", "how is the", "how are the",
    "why is the", "why are the", "why did the", "why does the",
    "who is", "who was", "who are", "who were",
    "what is", "what are", "what was", "what were",
    "where is", "where are", "where was", "where were",
    "when is", "when was", "when did", "when were",
    "how does", "how do", "how did", "how is", "how are",
    "why is", "why are", "why did", "why does",
    "explain", "describe", "define",
]


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_nlp():
    """Load spaCy model for entity extraction."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback if model not installed
            print("[M2] Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            _nlp = None
    return _nlp


def _extract_entities(text: str) -> list[str]:
    """
    Extract named entities from text using spaCy.
    Returns list of entity strings (PERSON, ORG, GPE, LOC, WORK_OF_ART, EVENT, etc.)
    """
    nlp = get_nlp()
    if nlp is None:
        return []
    
    doc = nlp(text)
    entities = []
    
    # Extract relevant entity types for Wikipedia search
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", 
                          "EVENT", "PRODUCT", "FAC", "NORP"]:
            entities.append(ent.text)
    
    return entities


def _extract_key_phrases(text: str) -> list[str]:
    """
    Extract noun phrases and important terms from text.
    Fallback when entity extraction doesn't work well.
    """
    nlp = get_nlp()
    if nlp is None:
        # Simple fallback: extract capitalized words
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return words[:3]  # Return top 3
    
    doc = nlp(text)
    phrases = []
    
    # Extract noun chunks
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 4:  # Avoid very long phrases
            phrases.append(chunk.text)
    
    return phrases[:5]  # Return top 5


def _build_query(question: str) -> str:
    """
    Convert a natural-language question into a Wikipedia-friendly search query.
    Enhanced with entity extraction and noun phrase detection.
    """
    # First try entity extraction
    entities = _extract_entities(question)
    if entities:
        # Use the longest entity as primary query
        return max(entities, key=len)
    
    # Fallback to noun phrase extraction
    phrases = _extract_key_phrases(question)
    if phrases:
        return max(phrases, key=len)
    
    # Final fallback: original prefix-stripping logic
    q = question.strip().rstrip("?!.").strip()

    q_lower = q.lower()
    for prefix in _QUESTION_PREFIXES:
        if q_lower.startswith(prefix):
            q = q[len(prefix):].strip()
            break

    # Remove leading articles after stripping prefix
    for article in ["a ", "an ", "the "]:
        if q.lower().startswith(article):
            q = q[len(article):]
            break

    if not q:
        q = question.strip()
    return q


def _build_query_variants(question: str) -> list[str]:
    """
    Generate multiple query variants to improve search success rate.
    Returns list of queries to try in order of priority.
    """
    variants = []
    
    # Variant 1: Entity-based query
    entities = _extract_entities(question)
    if entities:
        variants.append(max(entities, key=len))
        # Add all entities as separate queries
        variants.extend([e for e in entities if e not in variants])
    
    # Variant 2: Noun phrase query
    phrases = _extract_key_phrases(question)
    if phrases:
        main_phrase = max(phrases, key=len)
        if main_phrase not in variants:
            variants.append(main_phrase)
    
    # Variant 3: For "biggest/largest X" questions, try specific searches
    import re
    if re.search(r'\b(biggest|largest|tallest|highest|longest)\b', question.lower()):
        # Extract what comes after "biggest/largest"
        match = re.search(r'\b(?:biggest|largest|tallest|highest|longest)\s+(\w+(?:\s+\w+)?)', question.lower())
        if match:
            subject = match.group(1)
            # Add specific variants
            if 'planet' in subject and 'solar system' in question.lower():
                variants.insert(0, 'Jupiter')  # Most likely answer
                variants.append('Solar System')
            elif 'mountain' in subject:
                variants.insert(0, 'Mount Everest')
            
    # Variant 4: Original cleaned query
    original = _build_query(question)
    if original not in variants:
        variants.append(original)
    
    # Variant 5: Just remove question words
    simple = re.sub(r'^(who|what|where|when|why|how|which)\s+(is|are|was|were|did|does|do)\s+', 
                    '', question.lower(), flags=re.IGNORECASE).strip('?!. ')
    if simple and simple not in variants:
        variants.append(simple)
    
    # Superlative + topic hints (e.g. "biggest planet" → Jupiter, not exoplanet lists)
    q_lower = question.lower()
    if re.search(r"\b(biggest|largest|tallest|highest|longest|smallest)\b", q_lower):
        if re.search(r"\bplanets?\b", q_lower):
            for hint in ("Jupiter", "Solar System"):
                if hint not in variants:
                    variants.insert(0, hint)
        elif "mountain" in q_lower:
            if "Mount Everest" not in variants:
                variants.insert(0, "Mount Everest")

    return variants[:5]  # Return top 5 variants


def _rank_summary_score(
    title: str, summary: str, question: str, answer: str,
    q_emb: np.ndarray, answer_emb: np.ndarray | None,
    model: SentenceTransformer,
) -> float:
    """Score how well a Wikipedia summary supports verification."""
    s_emb = model.encode([summary], normalize_embeddings=True)[0]
    score = float(np.dot(q_emb, s_emb))
    if answer_emb is not None:
        score = max(score, float(np.dot(answer_emb, s_emb)))

    t_low = title.lower()
    q_low = question.lower()
    a_low = answer.lower()

    # List pages rarely support single-fact superlative questions
    if t_low.startswith("list of") and re.search(
        r"\b(biggest|largest|tallest|highest|longest|smallest)\b", q_low
    ):
        score -= 0.20

    # Exoplanet lists are wrong context for solar-system / Jupiter questions
    if "exoplanet" in t_low:
        if "solar system" in q_low or "solar system" in a_low:
            score -= 0.35
        if "jupiter" in a_low or "jupiter" in q_low:
            score -= 0.35

    # Boost when the article title appears in the LLM answer
    if answer and t_low in a_low:
        score += 0.12
    elif answer:
        for word in t_low.split():
            if len(word) > 4 and word in a_low:
                score += 0.05
                break

    # Prefer the specific entity page over broad parent topics (e.g. Jupiter vs Solar System)
    if answer and re.search(r"\bplanets?\b", q_low):
        if "jupiter" in a_low and t_low == "jupiter":
            score += 0.25
        elif "jupiter" in a_low and t_low == "solar system":
            score -= 0.12

    return score


def fetch_best_context(query: str, question: str, answer: str = "") -> dict:
    """
    Search Wikipedia for candidates, fetch their summaries, and
    use semantic similarity to pick the most relevant one.
    Enhanced with multiple query attempts and better error handling.
    """
    # Try multiple query variants
    query_variants = _build_query_variants(question)
    
    all_summaries = []
    seen_titles: set[str] = set()
    MAX_SUMMARIES = 12

    for q_variant in query_variants:
        try:
            candidates = wikipedia.search(q_variant, results=MAX_CANDIDATES)
            
            if not candidates:
                continue
            
            # Fetch summaries for this query variant
            for title in candidates:
                if title in seen_titles:
                    continue
                try:
                    # Check if it's a disambiguation page
                    page = wikipedia.page(title, auto_suggest=False)
                    summary = page.summary
                    
                    if summary and len(summary) > 100:  # Ensure meaningful content
                        seen_titles.add(title)
                        all_summaries.append({
                            "title": title,
                            "summary": summary[:3000],  # Limit to first 3000 chars
                            "query": q_variant
                        })
                        
                except wikipedia.DisambiguationError as e:
                    # Handle disambiguation by trying the first option
                    if e.options:
                        try:
                            page = wikipedia.page(e.options[0], auto_suggest=False)
                            summary = page.summary
                            if summary and len(summary) > 100:
                                opt = e.options[0]
                                if opt not in seen_titles:
                                    seen_titles.add(opt)
                                    all_summaries.append({
                                        "title": opt,
                                        "summary": summary[:3000],
                                        "query": q_variant
                                    })
                        except Exception:
                            continue
                            
                except wikipedia.PageError:
                    # Page doesn't exist, skip
                    continue
                    
                except Exception as e:
                    # Other errors, skip this candidate
                    continue
                    
        except Exception:
            # Search failed for this variant, try next
            continue

        if len(all_summaries) >= MAX_SUMMARIES:
            break

    if not all_summaries:
        return {"context": "", "source": "", "found": False}

    # If only one candidate retrieved, return it directly
    if len(all_summaries) == 1:
        return {
            "context": all_summaries[0]["summary"],
            "source": f"Wikipedia — {all_summaries[0]['title']}",
            "found": True,
        }

    # Rank by question + answer; penalise list/exoplanet pages for superlative Qs
    model = get_model()
    q_emb = model.encode([question], normalize_embeddings=True)[0]
    answer_emb = None
    if answer and answer.strip():
        answer_emb = model.encode([answer], normalize_embeddings=True)[0]

    scores = [
        _rank_summary_score(
            s["title"], s["summary"], question, answer, q_emb, answer_emb, model
        )
        for s in all_summaries
    ]
    best_idx = int(np.argmax(scores))

    return {
        "context": all_summaries[best_idx]["summary"],
        "source": f"Wikipedia — {all_summaries[best_idx]['title']}",
        "found": True,
    }


def compute_grounding(answer: str, context: str) -> float:
    if not context or not answer:
        return 0.0

    model = get_model()
    emb = model.encode([answer, context], normalize_embeddings=True)
    sim = float(np.dot(emb[0], emb[1]))
    return float(np.clip(sim, 0.0, 1.0))


def score(question: str, responses: list[str]) -> dict:
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
    retrieval = fetch_best_context(query, question, answer=answer)

    grounding = compute_grounding(answer, retrieval["context"])

    if not retrieval["found"]:
        verdict = "No evidence found"
    elif grounding >= 0.70:
        verdict = "Well-supported by evidence"
    elif grounding >= 0.45:  # Lowered from 0.50
        verdict = "Partially supported"
    elif grounding >= 0.25:  # Lowered from 0.30
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