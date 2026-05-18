"""
M4 — NLI Entailment Scoring Module
Checks whether the LLM's answer is entailed by Wikipedia-retrieved evidence
using a cross-encoder NLI model (DeBERTa-v3-small).

Inspired by: FActScore (Min et al., EMNLP 2023)
Approach:
  1. Decompose the answer into atomic sentences (claims).
  2. Retrieve a Wikipedia passage for the question (reuses M2 retrieval).
  3. For each claim, run NLI: premise = evidence, hypothesis = claim.
  4. Aggregate entailment probabilities → single score.

NLI labels from cross-encoder/nli-deberta-v3-small:
  index 0 → contradiction
  index 1 → entailment
  index 2 → neutral
"""

import re
import numpy as np
import wikipedia
from transformers import pipeline as hf_pipeline

_nli_pipe = None

NLI_MODEL      = "cross-encoder/nli-deberta-v3-small"
MAX_CANDIDATES = 3
SUMMARY_SENTENCES = 5
MIN_CLAIM_LEN  = 10   # characters — skip very short fragments


def get_nli_pipeline():
    global _nli_pipe
    if _nli_pipe is None:
        _nli_pipe = hf_pipeline(
            "text-classification",
            model=NLI_MODEL,
            top_k=None,          # return all label scores
            device=-1,           # CPU
        )
    return _nli_pipe


# ─── Sentence splitter ────────────────────────────────────────────────────────

def split_into_claims(text: str) -> list[str]:
    """
    Split text into individual sentences (atomic claims).
    Uses a simple regex; good enough for short LLM answers.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) >= MIN_CLAIM_LEN]


# ─── Wikipedia retrieval (same logic as M2, kept self-contained) ──────────────

def _fetch_evidence(query: str) -> str:
    try:
        candidates = wikipedia.search(query, results=MAX_CANDIDATES)
    except Exception:
        return ""

    best = ""
    for title in candidates:
        try:
            summary = wikipedia.summary(title, sentences=SUMMARY_SENTENCES)
            if len(summary) > len(best):
                best = summary
        except Exception:
            continue
    return best


# ─── NLI scoring ─────────────────────────────────────────────────────────────

def _entailment_prob(premise: str, hypothesis: str) -> float:
    """
    Returns the probability that `premise` entails `hypothesis`.
    """
    pipe = get_nli_pipeline()
    result = pipe(f"{premise} [SEP] {hypothesis}", truncation=True, max_length=512)

    # result is a list of dicts: [{"label": "ENTAILMENT", "score": ...}, ...]
    label_map = {item["label"].upper(): item["score"] for item in result[0]}

    # The model uses ENTAILMENT / NEUTRAL / CONTRADICTION
    return label_map.get("ENTAILMENT", 0.0)


# ─── Public API ───────────────────────────────────────────────────────────────

def score(question: str, responses: list[str]) -> dict:
    """
    Args:
        question  : the original question
        responses : list of LLM responses; we score responses[0]

    Returns:
        m4_score        : float [0, 1] — mean entailment probability across claims
        m4_claim_scores : list of {claim, entailment_prob}
        m4_verdict      : str
        m4_evidence_used: str — the Wikipedia passage used as premise
    """
    if not responses:
        return {
            "m4_score": 0.0,
            "m4_claim_scores": [],
            "m4_verdict": "No responses",
            "m4_evidence_used": "",
        }

    answer   = responses[0]
    evidence = _fetch_evidence(question)

    if not evidence:
        return {
            "m4_score": 0.0,
            "m4_claim_scores": [],
            "m4_verdict": "No evidence retrieved — entailment skipped",
            "m4_evidence_used": "",
        }

    claims = split_into_claims(answer)
    if not claims:
        return {
            "m4_score": 0.0,
            "m4_claim_scores": [],
            "m4_verdict": "Answer too short to decompose into claims",
            "m4_evidence_used": evidence,
        }

    claim_scores = []
    for claim in claims:
        prob = _entailment_prob(evidence, claim)
        claim_scores.append({
            "claim": claim,
            "entailment_prob": round(prob, 4),
        })

    probs = [c["entailment_prob"] for c in claim_scores]
    mean_score = float(np.clip(np.mean(probs), 0.0, 1.0))

    if mean_score >= 0.70:
        verdict = "Strongly entailed by evidence"
    elif mean_score >= 0.45:
        verdict = "Partially entailed"
    elif mean_score >= 0.25:
        verdict = "Weakly entailed — possible fabrication"
    else:
        verdict = "Not entailed — high hallucination risk"

    return {
        "m4_score": round(mean_score, 4),
        "m4_claim_scores": claim_scores,
        "m4_verdict": verdict,
        "m4_evidence_used": evidence,
    }
