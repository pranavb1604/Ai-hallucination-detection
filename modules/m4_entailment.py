"""
M4 — NLI Entailment Scoring Module
Checks whether the LLM's answer is entailed by Wikipedia-retrieved evidence.

Inspired by: FActScore (Min et al., EMNLP 2023)
Approach:
  1. Decompose the answer into atomic sentences (claims).
  2. Retrieve a Wikipedia passage for the question (reuses M2 retrieval).
  3. For each claim, run NLI: premise = evidence, hypothesis = claim.
  4. Aggregate entailment probabilities → single score.

Speed improvements:
  - FIXED: CrossEncoder replaces zero-shot-classification pipeline (10x faster)
  - FIXED: True batching — all (premise, claim) pairs in ONE forward pass
  - FIXED: Proper NLI labels — entailment/neutral/contradiction
  - FIXED: Rebalanced aggregation — 0.4*max + 0.6*mean
  - Wikipedia evidence cached to disk via WIKI_CACHE_PATH in config.py
"""

import re
import os
import joblib
import numpy as np
import threading
from scipy.special import softmax
from sentence_transformers.cross_encoder import CrossEncoder

# ── Config ────────────────────────────────────────────────────────────────────
from config import (
    M4_EVIDENCE_LENGTH,
    M4_MIN_CLAIM_LENGTH,
    M4_MAX_CHUNKS,
    WIKI_CACHE_PATH,
)

# FIX: CrossEncoder — purpose-built for NLI, only 90MB, very fast on CPU
M4_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
MIN_CLAIM_LEN = M4_MIN_CLAIM_LENGTH
BATCH_SIZE    = 128   # safe for 16GB RAM; increase to 256 if you have more

# ── Globals ───────────────────────────────────────────────────────────────────
_nli_model: CrossEncoder | None = None
_nli_lock = threading.Lock()
_wiki_cache_lock = threading.Lock()
_wiki_cache: dict = {}


# ─── Wikipedia disk cache ─────────────────────────────────────────────────────

def _load_wiki_cache() -> None:
    global _wiki_cache
    if _wiki_cache:
        return
    if os.path.exists(WIKI_CACHE_PATH):
        try:
            _wiki_cache = joblib.load(WIKI_CACHE_PATH)
            print(f"[M4] Wiki cache loaded ({len(_wiki_cache)} entries).")
        except Exception:
            _wiki_cache = {}


def _save_wiki_cache() -> None:
    try:
        joblib.dump(_wiki_cache, WIKI_CACHE_PATH)
    except Exception as e:
        print(f"[M4 WARN] Could not save wiki cache: {e}")


# ─── NLI Model ────────────────────────────────────────────────────────────────

def get_nli_model() -> CrossEncoder:
    """
    Load CrossEncoder once and reuse.
    CrossEncoder.predict() processes ALL pairs in one forward pass — true batching.
    Previous zero-shot pipeline with [chunk]*N was NOT batching (N separate passes).
    """
    global _nli_model
    if _nli_model is None:
        with _nli_lock:
            if _nli_model is None:  # double-check after acquiring lock
                print(f"[M4] Loading NLI model: {M4_MODEL_NAME} (~90MB)...")
                _nli_model = CrossEncoder(
                    M4_MODEL_NAME,
                    device="cpu",
                )
                print("[M4] NLI model loaded.")
    return _nli_model


# ─── Sentence / Claim Splitter ────────────────────────────────────────────────

_FLUFF_PREFIXES = (
    "as of my last update",
    "as of my last knowledge update",
    "based on the context",
    "according to available information",
    "i can confirm that",
    "it is true that",
    "the answer is",
    "to answer your question",
    "it's important to note that",
    "it is important to note that",
    "please note that",
    "note that",
    "however, please verify",
    "however, i would recommend",
    "please verify from a current",
    "please verify from",
    "i would recommend checking",
    "i would recommend verifying",
    "as political circumstances can change",
    "as political leadership can change",
)

_DROP_RE = re.compile(
    r"please verify"
    r"|i would recommend (checking|verifying|consulting)"
    r"|for (the )?most (current|recent|up-to-date)"
    r"|political (leadership|circumstances|situations?) can change"
    r"|information may (have changed|be outdated)"
    r"|check (a )?current (and reliable )?source",
    re.IGNORECASE,
)


def split_into_claims(text: str) -> list[str]:
    """Split text into atomic sentences, stripping fluff and disclaimers."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    claims = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if _DROP_RE.search(s):
            continue
        lower = s.lower()
        for fluff in _FLUFF_PREFIXES:
            if lower.startswith(fluff):
                s = s[len(fluff):].lstrip(',:; ')
                if s:
                    s = s[0].upper() + s[1:]
                break
        if len(s) >= MIN_CLAIM_LEN:
            claims.append(s)
    return claims


# ─── Wikipedia Retrieval ──────────────────────────────────────────────────────

from modules.m2_grounding import fetch_evidence_for_qa, _extract_relevant_sentences


def _fetch_evidence(question: str, answer: str) -> str:
    """Fetch best Wikipedia passage. Cached in memory + disk."""
    _load_wiki_cache()

    cache_key = f"{question.strip().lower()}||{answer.strip().lower()[:60]}"
    if cache_key in _wiki_cache:
        return _wiki_cache[cache_key]

    retrieval = fetch_evidence_for_qa(question, answer)
    context   = retrieval.get("context", "")

    if context and question:
        context = _extract_relevant_sentences(question, context, top_k=5)

    context = context[:M4_EVIDENCE_LENGTH] if context else ""

    with _wiki_cache_lock:
        _wiki_cache[cache_key] = context
        if len(_wiki_cache) % 50 == 0:
            _save_wiki_cache()

    return context


# ─── NLI Scoring ──────────────────────────────────────────────────────────────

def _build_chunks(premise: str) -> list[str]:
    """Split premise into 3-sentence chunks, capped at M4_MAX_CHUNKS."""
    sentences = [s.strip() for s in premise.split(".") if len(s.strip()) > 20]
    chunks = [
        ". ".join(sentences[i: i + 3])
        for i in range(0, max(len(sentences), 1), 3)
    ]
    chunks = [c for c in chunks if len(c.strip()) >= 10]
    chunks = chunks[:M4_MAX_CHUNKS]
    if not chunks:
        chunks = [premise[:400]]
    return chunks


def _score_claims_batch(premise: str, claims: list[str]) -> list[float]:
    """
    TRUE batching with CrossEncoder.

    CrossEncoder.predict(pairs) processes ALL (premise, claim) pairs
    in one forward pass — this is real batching.

    ❌ Old code:  pipe([chunk]*N, ...)  → N separate forward passes
    ✅ This code: model.predict(pairs)  → 1 forward pass for all N

    Label order for nli-MiniLM2-L6-H768: [contradiction, entailment, neutral]
    Index 1 = entailment probability.
    """
    if not premise.strip() or not claims:
        return [0.5] * len(claims)

    model  = get_nli_model()
    chunks = _build_chunks(premise)

    best_probs = [0.0] * len(claims)

    for chunk in chunks:
        pairs = [(chunk, claim) for claim in claims]

        try:
            # ONE call — all pairs processed together
            raw_scores = model.predict(
                pairs,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # raw_scores: (N, 3) → softmax → probabilities
            probs = softmax(raw_scores, axis=1)
            entailment_probs = probs[:, 1].tolist()  # index 1 = entailment

            for i, prob in enumerate(entailment_probs):
                if prob > best_probs[i]:
                    best_probs[i] = prob

        except Exception as e:
            print(f"[M4 WARN] Batch NLI failed on chunk: {e}")
            continue

    return best_probs


# ─── Public API ───────────────────────────────────────────────────────────────

def score(question: str, responses: list[str]) -> dict:
    """
    Args:
        question  : the original question
        responses : list of LLM responses; we score responses[0]

    Returns dict with keys:
        m4_score        : float [0, 1]
        m4_claim_scores : list of {claim, entailment_prob}
        m4_verdict      : str
        m4_evidence_used: str
    """
    if not responses:
        return {
            "m4_score":         0.0,
            "m4_claim_scores":  [],
            "m4_verdict":       "No responses",
            "m4_evidence_used": "",
        }

    answer   = responses[0]
    evidence = _fetch_evidence(question, answer)

    if not evidence:
        return {
            "m4_score":         0.5,
            "m4_claim_scores":  [],
            "m4_verdict":       "No evidence retrieved — entailment skipped",
            "m4_evidence_used": "",
        }

    claims = split_into_claims(answer)

    if not claims:
        return {
            "m4_score":         0.5,
            "m4_claim_scores":  [],
            "m4_verdict":       "Answer too short to decompose into claims",
            "m4_evidence_used": evidence,
        }

    # All claims scored in one batched call
    probs = _score_claims_batch(evidence, claims)

    claim_scores = [
        {"claim": claim, "entailment_prob": round(prob, 4)}
        for claim, prob in zip(claims, probs)
    ]

    # FIX: mean matters more than single best score
    mean_score = float(np.clip(
        0.4 * np.max(probs) + 0.6 * np.mean(probs),
        0.0, 1.0,
    ))

    if mean_score >= 0.65:
        verdict = "Strongly entailed by evidence"
    elif mean_score >= 0.45:
        verdict = "Partially entailed"
    elif mean_score >= 0.25:
        verdict = "Weakly entailed — possible fabrication"
    else:
        verdict = "Not entailed — high hallucination risk"

    return {
        "m4_score":         round(mean_score, 4),
        "m4_claim_scores":  claim_scores,
        "m4_verdict":       verdict,
        "m4_evidence_used": evidence,
    }