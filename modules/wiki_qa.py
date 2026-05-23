"""
Wiki QA — Extractive Question Answering from Wikipedia
───────────────────────────────────────────────────────
Given a user question and a Wikipedia context (already retrieved by M2),
extract the exact answer span — like Google's featured snippet.

Uses a lightweight DistilBERT model fine-tuned on SQuAD.
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

_qa_tokenizer = None
_qa_model = None

QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"


def _load_qa_model():
    """Lazy-load the QA model and tokenizer (cached globally)."""
    global _qa_tokenizer, _qa_model
    if _qa_tokenizer is None:
        _qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
        _qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
        _qa_model.eval()
    return _qa_tokenizer, _qa_model


def _run_qa(question: str, context: str) -> dict:
    """
    Run extractive QA on a single question-context pair.

    Returns dict with answer, confidence, answerable.
    """
    tokenizer, model = _load_qa_model()

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # Simple argmax to find best start and end positions
    start_idx = torch.argmax(start_logits).item()
    end_idx = torch.argmax(end_logits).item()

    # If end comes before start, clamp
    if end_idx < start_idx:
        end_idx = start_idx

    # Cap span length to 50 tokens
    if end_idx - start_idx > 50:
        end_idx = start_idx + 50

    # Decode the answer tokens
    answer_ids = inputs["input_ids"][0][start_idx: end_idx + 1]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    # Calculate confidence
    start_probs = torch.softmax(start_logits, dim=-1)
    end_probs = torch.softmax(end_logits, dim=-1)
    confidence = (start_probs[start_idx] * end_probs[end_idx]).item()

    # Null score = model pointing to [CLS] (position 0) → "unanswerable"
    null_score = (start_probs[0] * end_probs[0]).item()
    answerable = (
        start_idx > 0
        and confidence > null_score
        and confidence > 0.005
        and len(answer_text) > 0
    )

    return {
        "answer": answer_text if answerable else "",
        "confidence": round(confidence, 4) if answerable else 0.0,
        "answerable": answerable,
    }


def extract_answer(question: str, context: str) -> dict:
    """
    Extract the most likely answer span from a Wikipedia context.

    Strategy (multi-pass for accuracy):
      1. Try QA on the **first 2 sentences** of the article — Wikipedia's
         opening sentence almost always contains the key fact (location,
         definition, identity). This gives high-confidence answers.
      2. Try QA on question-focused sentences (top-3 most relevant).
      3. Pick whichever pass produced the higher confidence.

    Parameters
    ----------
    question : str
        The user's original question.
    context : str
        The Wikipedia passage to extract from (from M2 grounding).

    Returns
    -------
    dict with:
        answer       : str   — the extracted text span (empty if unanswerable)
        confidence   : float — model confidence [0, 1]
        answerable   : bool  — whether the model found a confident answer
    """
    if not question or not context:
        return {
            "answer": "",
            "confidence": 0.0,
            "answerable": False,
        }

    # Split context into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context.strip()) if s.strip()]

    if not sentences:
        return {
            "answer": "",
            "confidence": 0.0,
            "answerable": False,
        }

    # ── Pass 1: First 2 sentences (the Wikipedia definition/intro) ────────
    # Wikipedia articles nearly always define the subject in sentence 1,
    # e.g. "The Eiffel Tower is a lattice tower on the Champ de Mars in
    #        Paris, France."
    intro_context = " ".join(sentences[:2])
    result_intro = _run_qa(question, intro_context)

    # ── Pass 2: Question-focused sentences (semantic similarity) ──────────
    try:
        from modules.m2_grounding import _extract_relevant_sentences
        focused_context = _extract_relevant_sentences(question, context, top_k=3)
    except Exception:
        focused_context = context

    result_focused = _run_qa(question, focused_context)

    # ── Pick the best result ──────────────────────────────────────────────
    if result_intro["confidence"] >= result_focused["confidence"]:
        return result_intro
    else:
        return result_focused
