"""
Shared feature extraction for training, evaluation, and live inference.

Problem:
  - CSV rows have one labeled answer.
  - Streamlit/Ollama produces 3+ responses for M1/M3.
  - M2/M4 always score the primary (first) answer only.

Solution:
  - Live: if len(responses) >= 2, M1/M3 use all provided samples (Ollama).
  - Offline long answers: sentence splits as pseudo-samples (pseudo_multi).
  - Offline short / one-word answers (~46% of train.csv are <=3 words):
      build context anchors from the question + extracted entity (short_multi)
      so M1/M3 still use multi-sample logic; M2/M4 ground the single word.
  - Tiny answers with no context: single_fallback (M1=Q–A coherence, M3=neutral).
"""

from __future__ import annotations

import re
from typing import Union

from config import M1_M3_MIN_SAMPLES
from modules.m1_consistency import score as m1_score
from modules.m2_grounding import score as m2_score
from modules.m3_uncertainty import score as m3_score
from modules.m4_entailment import score as m4_score

MIN_SENTENCE_LEN = 10
SHORT_ANSWER_MAX_WORDS = 3   # HaluEval / TruthfulQA: many 1–3 word labels
SHORT_ANSWER_MAX_CHARS = 28


def is_short_answer(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    return len(t.split()) <= SHORT_ANSWER_MAX_WORDS or len(t) <= SHORT_ANSWER_MAX_CHARS


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if len(s.strip()) >= MIN_SENTENCE_LEN]


def expand_single_answer_to_samples(
    answer: str,
    min_samples: int = M1_M3_MIN_SAMPLES,
) -> list[str]:
    """
    Build pseudo-responses from one dataset answer for offline M1/M3 scoring.
    Mirrors having multiple Ollama samples when the text has enough structure.
    """
    sentences = _split_sentences(answer)
    if len(sentences) >= min_samples:
        return sentences[:min_samples]
    if len(sentences) == 2:
        return sentences + [answer.strip()]
    return [answer.strip()]


def expand_short_answer_samples(
    question: str,
    answer: str,
    min_samples: int = M1_M3_MIN_SAMPLES,
) -> list[str]:
    """
    One-word / short labeled answers (e.g. 'yes', '1926', 'England').
    Build pseudo-samples from question context — cannot split on sentences.
    """
    answer = answer.strip()
    samples = [answer]

    try:
        from modules.m2_grounding import _build_query
        entity = _build_query(question)
    except Exception:
        entity = ""

    if entity and entity.lower() != answer.lower():
        samples.append(entity)

    q_hook = " ".join(question.split()[:14])
    if q_hook and q_hook not in samples:
        samples.append(q_hook)

    if len(samples) < min_samples:
        samples.append(f"{answer} — {entity or q_hook[:60]}")

    # unique, preserve order
    out = list(dict.fromkeys(s.strip() for s in samples if s.strip()))
    return out[:max(min_samples, len(out))]


def resolve_m1_m3_responses(
    responses: list[str],
    question: str = "",
    min_samples: int = M1_M3_MIN_SAMPLES,
) -> tuple[list[str], str]:
    """
    Choose inputs for M1/M3.

    Returns:
        (samples, mode) where mode is:
          - live_multi:     2+ real LLM responses (Ollama / manual paste)
          - pseudo_multi:   long answer split into sentences
          - short_multi:    one-word / short answer + question/entity anchors
          - single_fallback: empty or unexpandable; M1=Q–A coherence, M3=neutral
    """
    clean = [str(r).strip() for r in responses if str(r).strip()]
    if not clean:
        return [], "single_fallback"

    if len(clean) >= 2:
        return clean, "live_multi"

    answer = clean[0]

    pseudo = expand_single_answer_to_samples(answer, min_samples=min_samples)
    if len(pseudo) >= 2 and not is_short_answer(answer):
        return pseudo, "pseudo_multi"

    if is_short_answer(answer) and question:
        short = expand_short_answer_samples(question, answer, min_samples=min_samples)
        if len(short) >= 2:
            return short, "short_multi"

    return [answer], "single_fallback"


def _normalize_responses(
    responses: Union[str, list[str], None],
) -> list[str]:
    if responses is None:
        return []
    if isinstance(responses, str):
        return [responses.strip()] if responses.strip() else []
    return [str(r).strip() for r in responses if str(r).strip()]


def extract_all(
    question: str,
    responses: Union[str, list[str]],
    min_samples: int = M1_M3_MIN_SAMPLES,
) -> dict:
    """
    Run M1–M4 with aligned rules for train / eval / pipeline / Streamlit.

    Returns module dicts plus fused feature vector and metadata.
    """
    clean = _normalize_responses(responses)
    if not clean:
        raise ValueError("At least one non-empty response is required")

    primary = clean[0]
    m1_m3_inputs, sample_mode = resolve_m1_m3_responses(
        clean, question=question, min_samples=min_samples,
    )

    m1 = m1_score(question, m1_m3_inputs)
    m3 = m3_score(question, m1_m3_inputs)
    m2 = m2_score(question, [primary])
    m4 = m4_score(question, [primary])

    s4 = float(m4.get("m4_score", 0.5))
    if s4 == 0.0:
        s4 = 0.5

    features = [
        float(m1["m1_score"]),
        float(m2["m2_score"]),
        float(m3["m3_score"]),
        s4,
    ]

    meta = {
        "m2_found": bool(m2.get("found", False)),
        "m1_std": float(m1.get("m1_std", 0.0)),
        "m3_variance": float(m3.get("m3_variance", 0.0)),
        "sample_mode": sample_mode,
        "short_answer": is_short_answer(primary),
        "answer_word_count": len(primary.split()),
    }

    return {
        "features": features,
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "m4": m4,
        "meta": meta,
        "sample_mode": sample_mode,
        "m1_m3_sample_count": len(m1_m3_inputs),
        "m1_m3_samples": m1_m3_inputs,
        "primary_answer": primary,
    }


def extract_features(
    question: str,
    answer: str,
    min_samples: int = M1_M3_MIN_SAMPLES,
) -> list[float]:
    """Return [m1, m2, m3, m4] for training / evaluation on one labeled answer."""
    return extract_all(question, answer, min_samples=min_samples)["features"]
