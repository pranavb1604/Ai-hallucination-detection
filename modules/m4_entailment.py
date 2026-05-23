"""
M4 — NLI Entailment Scoring Module (Generalised Hybrid Scorer)
===============================================================
Uses three independent signals per claim to avoid single-model failure:
    1. NLI score         — logical entailment via DeBERTa
    2. Semantic score    — cosine similarity via sentence-transformers
    3. Lexical score     — token overlap (handles numbers, names, units)

Final claim score = weighted combination of all three.
This ensures no single failure mode (unit mismatch, paraphrase,
partial info) can zero out a correct answer.

Inspired by: FActScore (Min et al., EMNLP 2023)
"""

import re
import logging
import numpy as np
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
import wikipedia

from config import (
    M2_WIKI_USER_AGENT,
    M4_CLAIM_SUPPORTED_THRESHOLD,
    M4_CONTRADICTION_THRESHOLD,
    M4_STRONG_ENTAILMENT,
    M4_PARTIAL_ENTAILMENT,
    M4_PREMISE_TOP_SENTENCES,
    M4_MODEL_NAME,
    M4_EVIDENCE_LENGTH,
    M4_WIKI_SUMMARY_LENGTH,
    M4_WIKI_MAX_CANDIDATES,
    M4_MIN_CLAIM_LENGTH,
    M1_MODEL_NAME,
)

wikipedia.set_user_agent(M2_WIKI_USER_AGENT)

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

# Hybrid weights — must sum to 1.0
# NLI is least reliable for factual claims so lowest weight
W_NLI      = 0.30
W_SEMANTIC = 0.45
W_LEXICAL  = 0.25

# Semantic similarity threshold above which we trust it strongly
SEMANTIC_TRUST_THRESHOLD = 0.72

# Contradiction override: only penalize if NLI is VERY confident
# AND semantic score also confirms disagreement
HARD_CONTRADICTION_NLI = 0.85   # NLI contradiction prob above this
HARD_CONTRADICTION_SEM = 0.40   # AND semantic score below this

# ─── Lazy-loaded models ───────────────────────────────────────────────────────

_nli_pipe  = None
_sem_model = None


def get_nli_pipeline():
    global _nli_pipe
    if _nli_pipe is None:
        logger.info("[M4] Loading NLI model...")
        import torch
        _nli_pipe = hf_pipeline(
            "text-classification",
            model=M4_MODEL_NAME,
            top_k=None,
            device=0 if torch.cuda.is_available() else -1,
        )
    return _nli_pipe


def get_semantic_model():
    global _sem_model
    if _sem_model is None:
        logger.info("[M4] Loading semantic model...")
        _sem_model = SentenceTransformer(M1_MODEL_NAME)
    return _sem_model


# ─── Text normalisation ───────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """
    Normalize text before scoring.
    Handles the most common NLI failure modes:
    - Number formatting: 299,792 → 299792
    - Approximation words: standardize to 'approximately'
    - Whitespace cleanup
    """
    # Remove commas inside numbers: 299,792 → 299792
    text = re.sub(r'(\d),(\d)', r'\1\2', text)

    # Standardize approximation language
    text = re.sub(
        r'\b(approximately|about|around|roughly|nearly|~)\b',
        'approximately', text, flags=re.IGNORECASE
    )

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def _normalize_units(text: str) -> str:
    """
    Convert units to a standard form so NLI doesn't see
    equivalent values as contradictions.

    Handles:
    - m/s  → km/s
    - miles/s → km/s
    """
    def ms_to_kms(m):
        try:
            val = float(m.group(1).replace(',', ''))
            return f"{val / 1000:.3f} km/s"
        except Exception:
            return m.group(0)

    text = re.sub(
        r'(\d[\d,]*(?:\.\d+)?)\s*m(?:etres?|eters?)?\s*/\s*s(?:ec(?:ond)?)?',
        ms_to_kms, text, flags=re.IGNORECASE
    )

    def mps_to_kms(m):
        try:
            val = float(m.group(1).replace(',', ''))
            return f"{val * 1.60934:.3f} km/s"
        except Exception:
            return m.group(0)

    text = re.sub(
        r'(\d[\d,]*(?:\.\d+)?)\s*mi(?:les?)?\s*/\s*s(?:ec(?:ond)?)?',
        mps_to_kms, text, flags=re.IGNORECASE
    )

    return text


def _preprocess(text: str) -> str:
    """Apply all normalizations."""
    return _normalize_text(_normalize_units(text))


# ─── Claim splitting ──────────────────────────────────────────────────────────

# LLM boilerplate — skip or strip so M4 only scores checkable factual claims
_FLUFF_PREFIXES = (
    "as of my last update",
    "as of my last knowledge update",
    "however, i recommend",
    "i recommend checking",
    "please note that",
    "it's important to note that",
    "it is important to note that",
    "note that",
    "based on the context",
    "according to available information",
    "i can confirm that",
    "it is true that",
    "to answer your question",
)

_FLUFF_SUBSTRINGS = (
    "recommend checking",
    "most current and accurate",
    "may have changed",
    "might have changed",
    "verify this information",
    "consult a reliable source",
    "as of my knowledge cutoff",
    "my training data",
    "i cannot provide",
    "i'm not able to",
    "as an ai",
    "as a language model",
)


def split_into_claims(text: str) -> list:
    """
    Split answer into atomic sentences (claims).
    Skips pure disclaimers; strips hedging prefixes from factual sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    claims = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        lower_s = s.lower()

        if any(sub in lower_s for sub in _FLUFF_SUBSTRINGS):
            continue

        for fluff in _FLUFF_PREFIXES:
            if lower_s.startswith(fluff):
                s = s[len(fluff):].lstrip(",:; ")
                if s:
                    s = s[0].upper() + s[1:]
                break

        if len(s) >= M4_MIN_CLAIM_LENGTH:
            claims.append(s)

    return claims


# ─── Entity and phrase extraction ─────────────────────────────────────────────

def _extract_entities(text: str) -> list:
    """
    Extract named entities using regex patterns (no spaCy dependency).

    Catches:
    - Title Case sequences: "Emmanuel Macron", "Eiffel Tower"
    - Single capitalized words after lowercase context
    - All-caps abbreviations: "NASA", "WHO"
    """
    entities = []

    # Title Case sequences — most proper nouns
    title_case = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    entities.extend(title_case)

    # Single capitalized words preceded by lowercase context
    single_caps = re.findall(r'(?<=[a-z,;]\s)[A-Z][a-z]{2,}\b', text)
    entities.extend(single_caps)

    # All-caps abbreviations: NASA, WHO, EU
    abbreviations = re.findall(r'\b[A-Z]{2,5}\b', text)
    entities.extend(abbreviations)

    # Deduplicate, longest first
    seen   = set()
    unique = []
    for e in sorted(entities, key=len, reverse=True):
        if e.lower() not in seen:
            seen.add(e.lower())
            unique.append(e)

    return unique[:5]


def _extract_key_phrases(text: str) -> list:
    """
    Extract noun phrases as fallback when entity extraction finds nothing useful.
    """
    phrases = re.findall(r'\b(?:[A-Z][a-z]+\s+){0,2}[A-Z][a-z]+\b', text)

    lower_phrases = re.findall(
        r'(?:about|of|is|was|are|were)\s+([a-z]+(?:\s+[a-z]+){0,2})',
        text, re.IGNORECASE
    )
    phrases.extend(lower_phrases)

    seen   = set()
    unique = []
    for p in phrases:
        p = p.strip()
        if p.lower() not in seen and len(p) > 3:
            seen.add(p.lower())
            unique.append(p)

    return unique[:5]


# ─── Wikipedia retrieval ──────────────────────────────────────────────────────

_QUESTION_PREFIXES = [
    "can you tell me", "tell me about", "explain to me",
    "i want to know", "please tell me", "please explain",
    "who is", "who was", "who are", "who were",
    "what is", "what are", "what was", "what were",
    "where is", "where was", "when is", "when was", "when did",
    "how does", "how do", "how did", "how is",
    "why is", "why are", "why did", "why does",
    "explain", "describe", "define",
]


def _build_query(question: str) -> str:
    """Convert question to Wikipedia search query."""
    q = question.strip().rstrip("?!.").strip()
    q_lower = q.lower()
    for prefix in _QUESTION_PREFIXES:
        if q_lower.startswith(prefix):
            q = q[len(prefix):].strip()
            break
    for article in ["a ", "an ", "the "]:
        if q.lower().startswith(article):
            q = q[len(article):]
            break
    return q if q else question.strip()


def _fetch_evidence(query: str, question: str = "") -> str:
    """
    Fetch Wikipedia evidence with multiple query variant strategies.

    Strategy order:
    1. Direct query (cleaned by _build_query)
    2. Named entities from original question
    3. Noun phrases from original question as fallback
    4. Capitalized variant of query
    """
    variants = [query]

    if question:
        entities = _extract_entities(question)
        for ent in entities:
            if ent not in variants:
                variants.append(ent)

        phrases = _extract_key_phrases(question)
        for phrase in phrases[:2]:
            if phrase not in variants:
                variants.append(phrase)

    if query and query[0].islower():
        cap = query.capitalize()
        if cap not in variants:
            variants.append(cap)

    logger.info(f"[M4] Trying {len(variants)} query variants: {variants}")

    for variant in variants:
        try:
            candidates = wikipedia.search(variant, results=M4_WIKI_MAX_CANDIDATES)
            for title in candidates:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    if page.summary and len(page.summary) > 100:
                        logger.info(f"[M4] Evidence found via '{variant}' → page: '{title}'")
                        return page.summary[:M4_WIKI_SUMMARY_LENGTH]
                except wikipedia.DisambiguationError as e:
                    if e.options:
                        try:
                            page = wikipedia.page(e.options[0], auto_suggest=False)
                            if page.summary and len(page.summary) > 100:
                                return page.summary[:M4_WIKI_SUMMARY_LENGTH]
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception:
            continue

    logger.warning(f"[M4] No evidence found for query: '{query}'")
    return ""


# ─── Evidence selection per claim ─────────────────────────────────────────────

def _split_evidence_sentences(evidence: str) -> list:
    sents = re.split(r'(?<=[.!?])\s+', evidence.strip())
    return [s.strip() for s in sents if len(s.strip()) >= 20]


def _get_numbers(text: str) -> set:
    """Extract all numeric tokens from text."""
    return set(re.findall(r'\d+(?:\.\d+)?', text.replace(',', '')))


def _best_premise(evidence: str, claim: str) -> str:
    """
    Select the most relevant evidence sentences for this claim.

    Strategy:
    1. For numeric claims: prioritize sentences sharing numbers with claim
       — fixes "299,792 km/s" vs "299792458 m/s" type failures
    2. For all claims: semantic similarity to pick top sentences
    3. Combine both for numeric claims
    """
    sentences = _split_evidence_sentences(evidence)
    if not sentences:
        return evidence[:M4_EVIDENCE_LENGTH]

    claim_numbers = _get_numbers(claim)
    has_numbers   = len(claim_numbers) > 0
    selected      = []

    # Step 1: numeric sentence matching
    if has_numbers:
        for sent in sentences:
            ev_numbers = _get_numbers(sent)
            if claim_numbers & ev_numbers:
                selected.append(sent)
        selected = selected[:3]

    # Step 2: semantic similarity
    model     = get_semantic_model()
    claim_emb = model.encode([claim], normalize_embeddings=True)[0]
    sent_embs = model.encode(sentences, normalize_embeddings=True)
    sims      = [float(np.dot(claim_emb, e)) for e in sent_embs]

    top_idx = np.argsort(sims)[-M4_PREMISE_TOP_SENTENCES:][::-1]
    for idx in sorted(top_idx):
        sent = sentences[idx]
        if sent not in selected:
            selected.append(sent)

    if not selected:
        selected = sentences[:M4_PREMISE_TOP_SENTENCES]

    snippet = " ".join(selected)
    return snippet[:M4_EVIDENCE_LENGTH]


# ─── Three scoring signals ────────────────────────────────────────────────────

def _nli_score(premise: str, claim: str) -> dict:
    """
    Signal 1: NLI entailment score via DeBERTa.

    Key design decisions:
    - Neutral gets 0.50 weight because paraphrases are often
      labelled neutral, not entailment, by cross-encoders
    - Contradiction only penalizes when VERY dominant (>= threshold)
      AND entailment+neutral together are weak (< 0.25)
    - On parse error: returns neutral=0.5 so score is never
      zeroed out by a model loading failure
    """
    pipe = get_nli_pipeline()

    try:
        # Premise = evidence; claim = hypothesis (Transformers 5.x pair input)
        result = pipe(
            {"text": premise, "text_pair": claim},
            truncation=True,
            max_length=512,
        )
        items  = result[0] if isinstance(result[0], list) else result
        labels = {item["label"].lower(): float(item["score"]) for item in items}
    except Exception as e:
        logger.warning(f"[M4] NLI parse error: {e}")
        labels = {"entailment": 0.0, "neutral": 0.5, "contradiction": 0.0}

    ent = labels.get("entailment",    0.0)
    neu = labels.get("neutral",       0.0)
    con = labels.get("contradiction", 0.0)

    # Neutral weighted at 0.50 — paraphrases commonly land here
    raw_support = ent + (neu * 0.75)

    # Only penalize on very dominant contradiction with weak non-contradiction
    if con >= M4_CONTRADICTION_THRESHOLD and (ent + neu) < 0.25:
        raw_support = raw_support * 0.30  # strong penalty but never zero

    return {
        "nli_support":        float(np.clip(raw_support, 0.0, 1.0)),
        "entailment_prob":    round(ent, 4),
        "neutral_prob":       round(neu, 4),
        "contradiction_prob": round(con, 4),
    }


def _semantic_score(premise: str, claim: str) -> float:
    """
    Signal 2: Semantic similarity via sentence-transformers.

    Robust to unit mismatches, paraphrases, and formatting differences.
    High score = claim is semantically close to evidence.
    """
    model     = get_semantic_model()
    claim_emb = model.encode(claim,   normalize_embeddings=True)
    prem_emb  = model.encode(premise, normalize_embeddings=True)
    return float(np.clip(np.dot(claim_emb, prem_emb), 0.0, 1.0))


def _lexical_score(premise: str, claim: str) -> float:
    """
    Signal 3: Token overlap F1 score.

    Specifically good at catching exact number matches,
    proper nouns, and named entities.
    Uses F1 to balance precision and recall.
    """
    def tokenize(text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were',
            'in', 'on', 'at', 'of', 'to', 'and', 'or',
            'it', 'its', 'this', 'that', 'by', 'be', 'has',
            'have', 'had', 'with', 'as', 'for', 'from'
        }
        return [t for t in tokens if t not in stopwords]

    claim_tokens   = set(tokenize(claim))
    premise_tokens = set(tokenize(premise))

    if not claim_tokens or not premise_tokens:
        return 0.0

    intersection = claim_tokens & premise_tokens
    if not intersection:
        return 0.0

    precision = len(intersection) / len(claim_tokens)
    recall    = len(intersection) / len(premise_tokens)
    f1        = 2 * precision * recall / (precision + recall)

    return float(np.clip(f1, 0.0, 1.0))


# ─── Hybrid scoring ───────────────────────────────────────────────────────────

def _score_claim(evidence: str, claim: str) -> dict:
    """
    Score a single claim against evidence using all three signals.

    Final score = W_NLI * nli + W_SEMANTIC * semantic + W_LEXICAL * lexical

    Special case 1 — High semantic similarity (>= SEMANTIC_TRUST_THRESHOLD):
        NLI may be wrong due to unit/format mismatch or paraphrase.
        Boost NLI score toward semantic score.
        Example: "299,792 km/s" vs Wikipedia "299792458 m/s"

    Special case 2 — Hard contradiction:
        Only penalize if BOTH NLI says contradiction AND semantic is low.
        Prevents a single wrong NLI label from tanking the score.
    """
    norm_claim    = _preprocess(claim)
    norm_evidence = _preprocess(evidence)
    premise       = _best_premise(norm_evidence, norm_claim)

    nli_result  = _nli_score(premise, norm_claim)
    sem_score   = _semantic_score(premise, norm_claim)
    lex_score   = _lexical_score(premise, norm_claim)

    nli_support = nli_result["nli_support"]
    con_prob    = nli_result["contradiction_prob"]

    # Special case 1: high semantic → boost NLI
    if sem_score >= SEMANTIC_TRUST_THRESHOLD:
        nli_support = max(nli_support, sem_score * 0.70)

    # Special case 2: hard contradiction confirmed by both signals
    hard_contradiction = (
        con_prob  >= HARD_CONTRADICTION_NLI and
        sem_score <= HARD_CONTRADICTION_SEM
    )

    if hard_contradiction:
        final_score = float(np.clip(
            W_NLI      * nli_support * 0.20 +
            W_SEMANTIC * sem_score +
            W_LEXICAL  * lex_score,
            0.0, 1.0
        ))
    else:
        final_score = float(np.clip(
            W_NLI      * nli_support +
            W_SEMANTIC * sem_score +
            W_LEXICAL  * lex_score,
            0.0, 1.0
        ))

    return {
        "support_score":       round(final_score, 4),
        "nli_support":         round(nli_support, 4),
        "semantic_score":      round(sem_score,   4),
        "lexical_score":       round(lex_score,   4),
        "entailment_prob":     nli_result["entailment_prob"],
        "neutral_prob":        nli_result["neutral_prob"],
        "contradiction_prob":  nli_result["contradiction_prob"],
        "hard_contradiction":  hard_contradiction,
    }


# ─── Aggregation ──────────────────────────────────────────────────────────────

def _aggregate(support_probs: list, n_unsupported: int) -> float:
    """
    Aggregate per-claim scores into a single M4 score.

    - 1 claim  → return directly
    - 2 claims → arithmetic mean
    - 3+ claims:
        - If majority unsupported → use min (likely real hallucination)
        - Otherwise → harmonic mean after removing worst outlier
    """
    if not support_probs:
        return 0.0
    if len(support_probs) == 1:
        return support_probs[0]
    if len(support_probs) == 2:
        return float(np.clip(np.mean(support_probs), 0.0, 1.0))

    if n_unsupported > len(support_probs) / 2:
        return float(np.clip(min(support_probs), 0.0, 1.0))

    sorted_scores = sorted(support_probs)
    filtered      = sorted_scores[1:]  # remove worst outlier
    epsilon       = 1e-6
    harmonic      = len(filtered) / sum(1.0 / (s + epsilon) for s in filtered)
    return float(np.clip(harmonic, 0.0, 1.0))


def _verdict(score: float, n_claims: int, n_unsupported: int) -> str:
    if n_unsupported == n_claims:
        return f"No claims supported by evidence (score {score:.2f})"
    if n_unsupported > 0:
        return (
            f"{n_unsupported}/{n_claims} claims not fully supported "
            f"(score {score:.2f})"
        )
    if score >= M4_STRONG_ENTAILMENT:
        return "All claims well supported by evidence"
    if score >= M4_CLAIM_SUPPORTED_THRESHOLD:
        return "Claims partially supported by evidence"
    if score >= M4_PARTIAL_ENTAILMENT:
        return "Weak support — possible fabrication"
    return "Claims not supported — high hallucination risk"


# ─── Public API ───────────────────────────────────────────────────────────────

def score(question: str, responses: list, evidence: str = "") -> dict:
    """
    Main entry point for M4.

    Args:
        question  : original question asked
        responses : list of LLM responses; scores responses[0]
        evidence  : optional pre-fetched Wikipedia evidence from M2;
                    if provided, skips redundant Wikipedia fetch

    Returns dict with:
        m4_score         : float [0,1] — final aggregated score for M5
        m4_mean_score    : float [0,1] — mean across claims
        m4_min_score     : float [0,1] — min across claims (for debugging)
        m4_n_unsupported : int   — claims below threshold
        m4_claim_scores  : list  — per-claim breakdown
        m4_verdict       : str   — human-readable result
        m4_evidence_used : str   — Wikipedia passage used
    """
    empty = {
        "m4_score":          0.0,
        "m4_mean_score":     0.0,
        "m4_min_score":      0.0,
        "m4_n_unsupported":  0,
        "m4_claim_scores":   [],
        "m4_verdict":        "",
        "m4_evidence_used":  "",
    }

    if not responses:
        empty["m4_verdict"] = "No responses provided"
        return empty

    answer = responses[0]

    # Use M2 evidence if provided, else fetch independently
    ev = evidence if evidence else _fetch_evidence(
        _build_query(question), question
    )

    if not ev:
        empty["m4_verdict"] = "No evidence retrieved — entailment skipped"
        return empty

    claims = split_into_claims(answer)
    if not claims:
        empty["m4_verdict"] = "Answer too short to split into claims"
        empty["m4_evidence_used"] = ev
        return empty

    claim_scores  = []
    support_probs = []

    for claim in claims:
        result    = _score_claim(ev, claim)
        supported = result["support_score"] >= M4_CLAIM_SUPPORTED_THRESHOLD

        claim_scores.append({
            "claim":              claim,
            "support_score":      result["support_score"],
            "nli_support":        result["nli_support"],
            "semantic_score":     result["semantic_score"],
            "lexical_score":      result["lexical_score"],
            "entailment_prob":    result["entailment_prob"],
            "neutral_prob":       result["neutral_prob"],
            "contradiction_prob": result["contradiction_prob"],
            "hard_contradiction": result["hard_contradiction"],
            "supported":          supported,
        })
        support_probs.append(result["support_score"])

    mean_score    = float(np.clip(np.mean(support_probs), 0.0, 1.0))
    min_score     = float(np.clip(min(support_probs),     0.0, 1.0))
    n_unsupported = sum(1 for c in claim_scores if not c["supported"])
    final_score   = _aggregate(support_probs, n_unsupported)

    return {
        "m4_score":          round(final_score, 4),
        "m4_mean_score":     round(mean_score,  4),
        "m4_min_score":      round(min_score,   4),
        "m4_n_unsupported":  n_unsupported,
        "m4_claim_scores":   claim_scores,
        "m4_verdict":        _verdict(final_score, len(claims), n_unsupported),
        "m4_evidence_used":  ev,
    }
