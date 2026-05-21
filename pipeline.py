"""
pipeline.py
───────────
End-to-end hallucination detection pipeline.

Runs M1–M4 in sequence, then fuses scores via:
  • Trained M5 neural classifier (if model checkpoint exists), OR
  • Weighted average fallback (if no checkpoint yet).

Usage:
    from pipeline import run_pipeline
    result = run_pipeline(question, responses)
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import (
    MODEL_SAVE_PATH, SCALER_SAVE_PATH, TRUST_THRESHOLDS, TRUST_LABELS,
    M5_INPUT_SIZE, M5_HIDDEN_SIZE_1, M5_HIDDEN_SIZE_2,
)
import importlib
import modules.m1_consistency
import modules.m2_grounding
import modules.m3_uncertainty
import modules.m4_entailment
import modules.m5_classifier

importlib.reload(modules.m1_consistency)
importlib.reload(modules.m2_grounding)
importlib.reload(modules.m3_uncertainty)
importlib.reload(modules.m4_entailment)
importlib.reload(modules.m5_classifier)

from modules.m1_consistency import score as m1_score
from modules.m2_grounding   import score as m2_score
from modules.m3_uncertainty  import score as m3_score
from modules.m4_entailment   import score as m4_score
from modules.m5_classifier   import (
    load_model, predict_trust, weighted_trust_score,
)

# ─── Lazy-load the trained model once ────────────────────────────────────────

_model = None
_scaler = None
_model_loaded = False


def _get_model():
    global _model, _scaler, _model_loaded
    if not _model_loaded:
        if os.path.exists(MODEL_SAVE_PATH):
            try:
                _model, _scaler = load_model(MODEL_SAVE_PATH, scaler_path=SCALER_SAVE_PATH)
            except Exception as e:
                print(f"[pipeline] Could not load model: {e}. Using fallback.")
                _model = None
                _scaler = None
        _model_loaded = True
    return _model, _scaler


# ─── Trust label helper ───────────────────────────────────────────────────────

def _trust_label(score: float) -> str:
    if score >= TRUST_THRESHOLDS["trusted"]:
        return TRUST_LABELS["trusted"]
    elif score >= TRUST_THRESHOLDS["uncertain"]:
        return TRUST_LABELS["uncertain"]
    elif score >= TRUST_THRESHOLDS["suspicious"]:
        return TRUST_LABELS["suspicious"]
    else:
        return TRUST_LABELS["hallucinated"]


# ─── Public API ───────────────────────────────────────────────────────────────

def run_pipeline(question: str, responses: list[str]) -> dict:
    """
    Args:
        question  : the original question posed to the LLM
        responses : list of LLM responses (ideally ≥ 3 for M1/M3)

    Returns a dict with:
        trust_score      : float [0, 1]
        trust_label      : str  (Trusted / Uncertain / Suspicious / Hallucinated)
        hallucination_prob: float
        scorer_used      : "neural" | "weighted_fallback"
        m1, m2, m3, m4   : individual module result dicts
    """
    # ── Run modules ──────────────────────────────────────────────────────────
    m1 = m1_score(question, responses)
    m2 = m2_score(question, responses)
    m3 = m3_score(question, responses)
    m4 = m4_score(question, responses)

    s1 = m1["m1_score"]
    s2 = m2["m2_score"]
    s3 = m3["m3_score"]
    s4 = m4["m4_score"]

    # ── Fuse scores ───────────────────────────────────────────────────────────
    model, scaler = _get_model()

    if model is not None:
        result = predict_trust(model, s1, s2, s3, s4, scaler=scaler)
        trust  = result["trust_score"]
        hal_p  = result["hallucination_prob"]
        scorer = "neural"
    else:
        trust  = weighted_trust_score(s1, s2, s3, s4)
        hal_p  = round(1.0 - trust, 4)
        scorer = "weighted_fallback"

    return {
        "trust_score":       round(trust, 4),
        "trust_label":       _trust_label(trust),
        "hallucination_prob": hal_p,
        "scorer_used":       scorer,
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "m4": m4,
    }
