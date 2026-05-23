"""
feature_extraction.py
─────────────────────
Shared M1–M4 feature extraction for train.py / evaluate.py.

Option B: build a multi-response list for M1/M3 (like Streamlit) while M2/M4
still score only the row's answer.
"""

import re
from collections import defaultdict

import pandas as pd

from config import M1_NUM_SAMPLES
from modules.m1_consistency import score as m1_score
from modules.m2_grounding import score as m2_score
from modules.m3_uncertainty import score as m3_score
from modules.m4_entailment import score as m4_score

_MIN_SENTENCE_LEN = 15


def build_question_index(df: pd.DataFrame) -> dict[str, list[tuple[str, int]]]:
    """Map question → [(answer, label), ...] for sibling sampling."""
    index: dict[str, list[tuple[str, int]]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()

    for _, row in df.iterrows():
        q = str(row["question"]).strip()
        a = str(row["answer"]).strip()
        lbl = int(row["label"])
        key = (q, a, lbl)
        if not q or not a or key in seen:
            continue
        seen.add(key)
        index[q].append((a, lbl))

    return dict(index)


def _sentence_splits(answer: str, max_samples: int) -> list[str]:
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", answer.strip())
        if len(s.strip()) >= _MIN_SENTENCE_LEN
    ]
    return sentences[:max_samples]


def build_response_samples(
    question: str,
    answer: str,
    label: int,
    question_index: dict[str, list[tuple[str, int]]] | None = None,
    max_samples: int = M1_NUM_SAMPLES,
) -> list[str]:
    """
    Simulate multiple LLM samples for M1/M3 during offline training.

    1. Same question, other answers from the dataset (label-aware).
    2. If still only one text: split the answer into sentences.
    """
    responses = [answer]
    q = question.strip()
    a = answer.strip()

    if question_index and q in question_index:
        siblings = question_index[q]
        if label == 0:
            # Correct row: only pool other correct answers (avoid punishing with hallucinations)
            pool = [o for o, lbl in siblings if lbl == 0 and o != a]
        else:
            # Hallucinated row: prefer contrasting correct answers, then other hallucinations
            correct = [o for o, lbl in siblings if lbl == 0 and o != a]
            wrong = [o for o, lbl in siblings if lbl == 1 and o != a]
            pool = correct + wrong

        for other in pool:
            if other not in responses:
                responses.append(other)
            if len(responses) >= max_samples:
                break

    if len(responses) < 2:
        for sent in _sentence_splits(a, max_samples):
            if sent not in responses:
                responses.append(sent)
            if len(responses) >= max_samples:
                break

    return responses[:max_samples]


import numpy as np

def extract_features(
    question: str,
    answer: str,
    label: int | None = None,
    question_index: dict[str, list[tuple[str, int]]] | None = None,
) -> list[float]:
    """
    Run M1–M4 for one labeled (question, answer) row.

    M1/M3: multi-response list (Option B).
    M2/M4: primary answer only (same as inference on responses[0]).
    """
    responses = (
        build_response_samples(question, answer, label or 0, question_index)
        if question_index is not None and label is not None
        else [answer]
    )

    m1 = m1_score(question, responses)["m1_score"]
    m2 = m2_score(question, [answer])
    m3 = m3_score(question, responses)["m3_score"]
    m4_dict = m4_score(question, [answer], evidence=m2.get("context", ""))
    
    # Calculate average sub-signals across claims
    claim_scores = m4_dict.get("m4_claim_scores", [])
    if claim_scores:
        avg_nli = float(np.mean([c["nli_support"] for c in claim_scores]))
        avg_semantic = float(np.mean([c["semantic_score"] for c in claim_scores]))
        avg_lexical = float(np.mean([c["lexical_score"] for c in claim_scores]))
    else:
        avg_nli = 0.0
        avg_semantic = 0.0
        avg_lexical = 0.0

    return [
        m1,
        m3,
        m2["m2_score"],
        m4_dict["m4_score"],
        m4_dict.get("m4_mean_score", 0.0),
        m4_dict.get("m4_min_score", 0.0),
        float(m4_dict.get("m4_n_unsupported", 0)),
        avg_nli,
        avg_semantic,
        avg_lexical
    ]
