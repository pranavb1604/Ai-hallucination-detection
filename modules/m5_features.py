"""
Rich feature vector for M5 (gradient boosting / neural meta-classifier).
"""

from __future__ import annotations

import re
import numpy as np

FEATURE_DIM = 21
FEATURE_NAMES = [
    "m1", "m2", "m3", "m4",
    "mean4", "std4", "min4", "max4",
    "m1_x_m4", "m2_x_m4", "m2_x_m3",
    "m4_low", "m2_low", "spread_m2_m4",
    "token_overlap", "answer_len_norm",
    "m2_found", "m1_std", "m3_variance",
    "pseudo_multi", "short_answer",
]


def neutralize_m4(m4: float) -> float:
    return 0.5 if m4 == 0.0 else m4


def _token_overlap(question: str, answer: str) -> float:
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "what", "who", "how",
        "when", "where", "why", "of", "in", "on", "for", "to", "and", "or",
    }
    q = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in stop}
    a = {w for w in re.findall(r"[a-z0-9]+", answer.lower()) if w not in stop}
    if not a:
        return 0.0
    return float(len(q & a) / len(a))


def build_feature_vector(
    m1: float,
    m2: float,
    m3: float,
    m4: float,
    question: str | None = None,
    answer: str | None = None,
    meta: dict | None = None,
) -> np.ndarray:
    m4n = neutralize_m4(m4)
    scores = [m1, m2, m3, m4n]
    arr = np.array(scores, dtype=np.float32)

    overlap = _token_overlap(question, answer) if question and answer else 0.5
    ans_len = min(len(answer or "") / 400.0, 1.0) if answer else 0.5

    meta = meta or {}
    m2_found = float(bool(meta.get("m2_found", meta.get("found", False))))
    # Disabled: these features differ between training (pseudo-samples) and
    # live (Ollama), causing distribution shift. Zeroed out for consistency.
    m1_std = 0.0
    m3_var = 0.0
    pseudo = 0.0
    short = float(meta.get("short_answer", False))

    feats = [
        m1, m2, m3, m4n,
        float(arr.mean()),
        float(arr.std()),
        float(arr.min()),
        float(arr.max()),
        m1 * m4n,
        m2 * m4n,
        m2 * m3,
        float(m4n <= 0.5),
        float(m2 <= 0.45),
        abs(m2 - m4n),
        overlap,
        ans_len,
        m2_found,
        m1_std,
        m3_var,
        pseudo,
        short,
    ]
    return np.array(feats, dtype=np.float32)


def build_batch(
    X_four: np.ndarray,
    questions: list[str] | None = None,
    answers: list[str] | None = None,
    metas: list[dict] | None = None,
) -> np.ndarray:
    rows = []
    n = len(X_four)
    for i in range(n):
        q = questions[i] if questions is not None else None
        a = answers[i] if answers is not None else None
        m = metas[i] if metas is not None else None
        rows.append(
            build_feature_vector(
                X_four[i, 0], X_four[i, 1], X_four[i, 2], X_four[i, 3],
                question=q, answer=a, meta=m,
            )
        )
    return np.stack(rows, axis=0)
