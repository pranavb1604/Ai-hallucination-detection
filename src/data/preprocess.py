"""
data/preprocess.py
Unifies HaluEval and TruthfulQA CSVs into train.csv and test.csv.
"""

import sys
import os
import ast
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from config import (
    HALUEVAL_RAW_PATH, TRUTHFULQA_RAW_PATH,
    TRAIN_PATH, TEST_PATH,
    TEST_SIZE, RANDOM_SEED,
)

# ──────────────────────────────────────────────
# HaluEval Processing
# ──────────────────────────────────────────────

def process_halueval(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    records = []

    for _, row in df.iterrows():
        q = str(row.get("question", "")).strip()
        ans = str(row.get("answer", "")).strip()
        
        # Convert "yes"/"no" hallucination text to 1 or 0
        hal_flag = str(row.get("hallucination", "")).strip().lower()
        if hal_flag in ["yes", "true", "1"]:
            label = 1
        elif hal_flag in ["no", "false", "0"]:
            label = 0
        else:
            continue

        if q and ans and ans != "nan" and len(ans) >= 15:  # Filter short answers
            records.append({
                "question": q,
                "answer": ans,
                "label": label,
                "source": "halueval"
            })

    result = pd.DataFrame(records)
    print(f"  HaluEval: {len(result)} rows (after processing)")
    return result

# ──────────────────────────────────────────────
# TruthfulQA Processing
# ──────────────────────────────────────────────

def _safe_parse(val):
    """Safely parse a string that looks like a Python dict, stripping pandas 'array()' wrappers."""
    val_str = str(val)
    # Remove sneaky numpy array wrappers if they exist in the CSV text
    val_str = re.sub(r'array\((.*?)(?:,\s*dtype=[^\)]+)?\)', r'\1', val_str)
    
    try:
        return ast.literal_eval(val_str)
    except Exception:
        return None

def process_truthfulqa(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    records = []

    for _, row in df.iterrows():
        q = str(row.get("question", "")).strip()
        if not q or q == "nan":
            continue

        targets = _safe_parse(row.get("mc1_targets", ""))
        if not isinstance(targets, dict):
            continue

        choices = targets.get("choices", [])
        labels  = targets.get("labels",  [])

        if len(choices) != len(labels):
            continue

        for choice, lbl in zip(choices, labels):
            ans = str(choice).strip()
            if ans and ans != "nan" and len(ans) >= 15:  # Filter short answers
                # TruthfulQA label: 1 = correct → we invert to label 1 = hallucinated
                records.append({
                    "question": q,
                    "answer":   ans,
                    "label":    0 if int(lbl) == 1 else 1, 
                    "source":   "truthfulqa",
                })

    result = pd.DataFrame(records)
    print(f"  TruthfulQA: {len(result)} rows (after processing)")
    return result

# ──────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────

def main():
    print("Processing datasets ...")

    halueval_df   = process_halueval(HALUEVAL_RAW_PATH)
    truthfulqa_df = process_truthfulqa(TRUTHFULQA_RAW_PATH)

    combined = pd.concat([halueval_df, truthfulqa_df], ignore_index=True)
    combined = combined.dropna(subset=["question", "answer"])
    combined = combined[combined["answer"].str.strip() != ""]
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\nCombined dataset: {len(combined)} rows")
    print(f"  Hallucinated (1): {combined['label'].sum()}")
    print(f"  Correct      (0): {(combined['label'] == 0).sum()}")

    train_df, test_df = train_test_split(
        combined,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=combined["label"],
    )

    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH,  index=False)

    print(f"\nTrain: {len(train_df)} rows -> {TRAIN_PATH}")
    print(f"Test:  {len(test_df)}  rows -> {TEST_PATH}")
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()