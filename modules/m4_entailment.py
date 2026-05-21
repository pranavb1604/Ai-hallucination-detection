"""
M4 — NLI Entailment Scoring Module
Checks whether the LLM's answer is entailed by Wikipedia-retrieved evidence
using facebook/bart-large-mnli (much better than DeBERTa-v3-small).

Inspired by: FActScore (Min et al., EMNLP 2023)
Approach:
  1. Decompose the answer into atomic sentences (claims).
  2. Retrieve a Wikipedia passage for the question (reuses M2 retrieval).
  3. For each claim, run NLI: premise = evidence, hypothesis = claim.
  4. Aggregate entailment probabilities → single score.
"""

import re
import numpy as np
from transformers import pipeline as hf_pipeline

_nli_pipe = None

NLI_MODEL     = "facebook/bart-large-mnli"   # much better than deberta-v3-small
MIN_CLAIM_LEN = 10                            # characters — skip very short fragments


# ─── NLI Pipeline ─────────────────────────────────────────────────────────────

def get_nli_pipeline():
    global _nli_pipe
    if _nli_pipe is None:
        print("[M4] Loading BART-large-mnli model (first time ~1.6GB download)...")
        _nli_pipe = hf_pipeline(
            "zero-shot-classification",
            model=NLI_MODEL,
            device=-1,   # CPU — works fine with 16GB RAM
        )
        print("[M4] Model loaded successfully.")
    return _nli_pipe


# ─── Sentence / Claim Splitter ────────────────────────────────────────────────

def split_into_claims(text: str) -> list[str]:
    """
    Split text into individual sentences (atomic claims) and strip
    conversational fluff that confuses NLI models.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    claims = []

    fluff_prefixes = [
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
    ]

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        lower_s = s.lower()
        for fluff in fluff_prefixes:
            if lower_s.startswith(fluff):
                s = s[len(fluff):].lstrip(',:; ')
                if s:
                    s = s[0].upper() + s[1:]
                break

        if len(s) >= MIN_CLAIM_LEN:
            claims.append(s)

    return claims


# ─── Wikipedia Retrieval (reuses M2 logic) ────────────────────────────────────

from modules.m2_grounding import _build_query, fetch_best_context, _extract_relevant_sentences


def _fetch_evidence(question: str, answer: str) -> str:
    """
    Fetch Wikipedia evidence for the question.
    Falls back to answer-based query if question-based fails.
    """
    # Try 1: query from question
    query     = _build_query(question)
    retrieval = fetch_best_context(query) if query else {"found": False, "context": ""}

    # Try 2: raw question directly
    if not retrieval["found"] or not retrieval["context"]:
        retrieval = fetch_best_context(question[:80])

    # Try 3: first sentence of answer
    if not retrieval["found"] or not retrieval["context"]:
        first_sentence = answer.split(".")[0][:80]
        retrieval = fetch_best_context(first_sentence)

    context = retrieval.get("context", "")

    # Focus context on most relevant sentences
    if context and question:
        context = _extract_relevant_sentences(question, context, top_k=5)

    return context


# ─── NLI Scoring ──────────────────────────────────────────────────────────────

def _entailment_prob(premise: str, hypothesis: str) -> float:
    
    # ← ADD: empty check pehle
    if not premise or not premise.strip():
        return 0.5
    if not hypothesis or not hypothesis.strip():
        return 0.5

    pipe = get_nli_pipeline()

    sentences = [s.strip() for s in premise.split(".") if len(s.strip()) > 20]
    chunks    = [
        ". ".join(sentences[i:i+3])
        for i in range(0, max(len(sentences), 1), 3)
    ]

    # ← ADD: empty chunks check
    chunks = [c for c in chunks if c.strip()]
    if not chunks:
        chunks = [premise[:500]]

    best_prob = 0.0

    for chunk in chunks:
        
        # ← ADD: skip empty/too short chunks
        if not chunk or len(chunk.strip()) < 10:
            continue

        try:
            result = pipe(
                chunk,
                candidate_labels=["true", "false", "unrelated"],
                hypothesis_template="This statement is true: {}",
            )
            label_score = dict(zip(result["labels"], result["scores"]))
            prob = label_score.get("true", 0.0)
            if prob > best_prob:
                best_prob = prob

        except Exception as e:
            print(f"[M4 WARN] Chunk NLI failed: {e}")
            continue

    return best_prob


# ─── Public API ───────────────────────────────────────────────────────────────

def score(question: str, responses: list[str]) -> dict:
    """
    Args:
        question  : the original question
        responses : list of LLM responses; we score responses[0]

    Returns:
        m4_score        : float [0, 1] — entailment score across claims
        m4_claim_scores : list of {claim, entailment_prob}
        m4_verdict      : str
        m4_evidence_used: str — the Wikipedia passage used as premise
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
            "m4_score":         0.5,   # neutral — don't punish missing evidence
            "m4_claim_scores":  [],
            "m4_verdict":       "No evidence retrieved — entailment skipped",
            "m4_evidence_used": "",
        }

    claims = split_into_claims(answer)

    if not claims:
        return {
            "m4_score":         0.5,   # neutral
            "m4_claim_scores":  [],
            "m4_verdict":       "Answer too short to decompose into claims",
            "m4_evidence_used": evidence,
        }

    # Score each claim
    claim_scores = []
    for claim in claims:
        prob = _entailment_prob(evidence, claim)
        claim_scores.append({
            "claim":            claim,
            "entailment_prob":  round(prob, 4),
        })

    probs = [c["entailment_prob"] for c in claim_scores]

    # Weighted aggregation: best score matters more than mean
    mean_score = float(np.clip(
        0.6 * np.max(probs) + 0.4 * np.mean(probs),
        0.0, 1.0
    ))

    # Verdict thresholds (tuned for BART-large-mnli)
    if mean_score >= 0.60:
        verdict = "Strongly entailed by evidence"
    elif mean_score >= 0.40:
        verdict = "Partially entailed"
    elif mean_score >= 0.20:
        verdict = "Weakly entailed — possible fabrication"
    else:
        verdict = "Not entailed — high hallucination risk"

    return {
        "m4_score":         round(mean_score, 4),
        "m4_claim_scores":  claim_scores,
        "m4_verdict":       verdict,
        "m4_evidence_used": evidence,
    }