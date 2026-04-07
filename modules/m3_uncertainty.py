"""
modules/m3_uncertainty.py  —  Module 3: Token-Level Uncertainty Estimation

Inspired by Semantic Uncertainty (Kuhn et al., ICLR 2023).

Two modes:
  A) LOGIT MODE  — if you have access to an open model (e.g. Mistral 7B via
                   HuggingFace), compute token-level entropy from log-probabilities.
  B) CONFIDENCE PROMPT MODE — for closed-model APIs (GPT, Gemini) that don't
                   expose logprobs, we prompt the model to self-report confidence
                   on a 0–100 scale and normalise.

The module auto-selects based on what's available. In the evaluation pipeline
(where we only have stored answers), we use the confidence-prompt fallback.

Score returned: float in [0, 1]
    1.0 = model was very certain (low entropy / high confidence)
    0.0 = model was very uncertain (high entropy / low confidence)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import numpy as np
from utils.helpers import normalize_score, clean_text

# ── Optional heavy imports — only loaded if logit mode is used ──
_tokenizer = None
_lm_model   = None


def _load_logit_model(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """Lazily loads a HuggingFace causal LM for logit-based scoring."""
    global _tokenizer, _lm_model
    if _lm_model is not None:
        return _tokenizer, _lm_model

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"[M3] Loading logit model: {model_name} (this may take a while) ...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _lm_model  = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        _lm_model.eval()
        print("[M3] Logit model loaded.")
        return _tokenizer, _lm_model
    except Exception as e:
        print(f"[M3] Could not load logit model: {e}. Falling back to confidence prompt.")
        return None, None


# ──────────────────────────────────────────────
# Mode A: Logit-based entropy
# ──────────────────────────────────────────────

def _logit_uncertainty_score(
    question: str,
    answer: str,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
) -> float | None:
    """
    Tokenises the (question + answer) and computes the mean token-level entropy
    over answer tokens using the model's output logits.

    Returns a CERTAINTY score (1 - normalised_entropy), or None on failure.
    """
    try:
        import torch
        tokenizer, model = _load_logit_model(model_name)
        if tokenizer is None or model is None:
            return None

        prompt = f"Question: {clean_text(question)}\nAnswer: {clean_text(answer)}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0]                         # (seq_len, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)      # (seq_len, vocab_size)
        probs     = torch.exp(log_probs)
        entropy   = -(probs * log_probs).sum(dim=-1)       # per-token entropy

        # Only consider answer tokens (rough approximation: last half of sequence)
        seq_len = entropy.shape[0]
        answer_entropy = entropy[seq_len // 2 :].mean().item()

        # Normalise: max possible entropy for vocab V is log(V)
        vocab_size = logits.shape[-1]
        max_entropy = float(np.log(vocab_size))
        normalised_entropy = answer_entropy / max_entropy  # in [0, 1]

        certainty = 1.0 - normalised_entropy
        return normalize_score(certainty)

    except Exception as e:
        print(f"[M3] Logit scoring failed: {e}")
        return None


# ──────────────────────────────────────────────
# Mode B: Confidence prompt (closed-model fallback)
# ──────────────────────────────────────────────

def _parse_confidence_from_text(text: str) -> float | None:
    """Extracts a 0–100 integer confidence from model self-report text."""
    text = text.strip()
    # look for patterns like "Confidence: 85" or just "85" or "85%"
    patterns = [
        r"[Cc]onfidence[:\s]+(\d{1,3})",
        r"(\d{1,3})\s*%",
        r"^(\d{1,3})$",
        r"(\d{1,3})\s*/\s*100",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 100:
                return val / 100.0
    return None


def confidence_prompt_score(
    question: str,
    answer: str,
    llm_callable=None,
) -> float:
    """
    Asks an LLM to rate its own confidence in the answer on 0–100.

    llm_callable: a function (prompt: str) -> str
        If None, returns a neutral 0.5 score.

    Returns a certainty score in [0, 1].
    """
    if llm_callable is None:
        return 0.5   # neutral fallback when no LLM available

    prompt = (
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "On a scale of 0 to 100, how confident are you that this answer is factually "
        "correct? Reply with ONLY a single integer between 0 and 100. "
        "Do not add any explanation."
    )

    try:
        response = llm_callable(prompt)
        parsed = _parse_confidence_from_text(response)
        if parsed is not None:
            return normalize_score(parsed)
    except Exception as e:
        print(f"[M3] Confidence prompt failed: {e}")

    return 0.5


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def score(
    question: str,
    answer: str,
    use_logits: bool = False,
    llm_callable=None,
    logit_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
) -> dict:
    """
    Public API for M3.

    Args:
        question         : original question
        answer           : LLM-generated answer
        use_logits       : if True, attempt logit-based entropy scoring first
        llm_callable     : optional (prompt)->str function for confidence prompting
        logit_model_name : HuggingFace model name for logit mode

    Returns:
        {
            "m3_score":  float,   # certainty score [0,1]; higher = more certain
            "mode":      str,     # "logit" | "confidence_prompt" | "fallback"
        }
    """
    if use_logits:
        logit_score = _logit_uncertainty_score(question, answer, logit_model_name)
        if logit_score is not None:
            return {"m3_score": logit_score, "mode": "logit"}

    if llm_callable is not None:
        conf_score = confidence_prompt_score(question, answer, llm_callable)
        return {"m3_score": conf_score, "mode": "confidence_prompt"}

    # Pure fallback — neutral score
    return {"m3_score": 0.5, "mode": "fallback"}


# ──────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Test with fallback (no LLM callable)
    result = score(
        question="What is the speed of light?",
        answer="The speed of light is approximately 299,792,458 metres per second.",
    )
    print("Fallback result:", result)

    # Test confidence prompt parsing
    for text in ["85", "Confidence: 72", "I am 90% confident.", "42/100"]:
        print(f"  parse '{text}' → {_parse_confidence_from_text(text)}")