"""
enrich_features.py
──────────────────
Adds text-level features to the cached M2/M4 feature CSV.
Joins with train.csv to get answer text, then computes:
  - answer_length, answer_word_count, n_claims, question_type

Run once after precompute_features.py. Takes ~5 seconds.
Usage:
    python enrich_features.py
"""
import os
import sys
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import TRAIN_PATH, PROCESSED_DATA_DIR

CACHE_PATH = os.path.join(PROCESSED_DATA_DIR, "cached_m2_m4_features.csv")


def encode_question_type(q: str) -> int:
    """Encode question type as an integer for the classifier."""
    q = q.lower().strip()
    if q.startswith("who "):   return 0
    if q.startswith("what "):  return 1
    if q.startswith("when "):  return 2
    if q.startswith("where "): return 3
    if q.startswith("why "):   return 4
    if q.startswith("how "):   return 5
    if q.startswith(("which ", "is ", "are ", "was ", "were ")): return 6
    if q.startswith(("do ", "does ", "did ", "can ", "could ")):  return 7
    return 8


def count_claims(text: str) -> int:
    """Count the number of meaningful sentences (claims) in an answer."""
    if not text or len(text.strip()) < 5:
        return 0
    sentences = re.split(r'(?<=[.!?])\s+', str(text).strip())
    return len([s for s in sentences if len(s.strip()) > 10])


def main():
    if not os.path.exists(CACHE_PATH):
        print(f"Error: {CACHE_PATH} not found. Run precompute_features.py first.")
        sys.exit(1)

    cache_df = pd.read_csv(CACHE_PATH)
    train_df = pd.read_csv(TRAIN_PATH)

    print(f"Cached features: {len(cache_df)} rows")
    print(f"Train data:      {len(train_df)} rows")

    # Drop old text columns if they exist (re-enriching)
    for col in ["answer_length", "answer_word_count", "n_claims", "question_type"]:
        if col in cache_df.columns:
            cache_df = cache_df.drop(columns=[col])

    # Merge to get answer text — match on (question, label), take first match
    train_unique = train_df.drop_duplicates(subset=["question", "label"], keep="first")
    merged = cache_df.merge(
        train_unique[["question", "label", "answer"]],
        on=["question", "label"],
        how="left"
    )

    # For any unmatched rows, try matching on question only
    unmatched = merged["answer"].isna()
    if unmatched.any():
        print(f"  {unmatched.sum()} rows unmatched on (question, label), trying question-only match...")
        train_q_only = train_df.drop_duplicates(subset=["question"], keep="first")
        fallback = merged.loc[unmatched, ["question"]].merge(
            train_q_only[["question", "answer"]], on="question", how="left"
        )
        merged.loc[unmatched, "answer"] = fallback["answer"].values

    # Fill any remaining NaN answers with empty string
    still_missing = merged["answer"].isna().sum()
    if still_missing > 0:
        print(f"  {still_missing} rows still have no answer text (using empty string)")
    merged["answer"] = merged["answer"].fillna("")

    # ── Compute text features ────────────────────────────────────────────────
    merged["answer_length"]     = merged["answer"].str.len().astype(float)
    merged["answer_word_count"] = merged["answer"].str.split().str.len().fillna(0).astype(float)
    merged["n_claims"]          = merged["answer"].apply(count_claims).astype(float)
    merged["question_type"]     = merged["question"].apply(encode_question_type).astype(float)

    # Drop the raw answer column (not a model feature)
    merged = merged.drop(columns=["answer"])

    # Save back to CSV
    merged.to_csv(CACHE_PATH, index=False)

    print(f"\n  Enriched {len(merged)} rows with 4 text features.")
    print(f"  New columns: answer_length, answer_word_count, n_claims, question_type")
    print(f"  Saved to {CACHE_PATH}")

    # Print feature summary
    for col in ["answer_length", "answer_word_count", "n_claims", "question_type"]:
        print(f"    {col:20s}  mean={merged[col].mean():.2f}  std={merged[col].std():.2f}")


if __name__ == "__main__":
    main()
