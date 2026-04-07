"""
data/download_datasets.py

Downloads HaluEval and TruthfulQA datasets from HuggingFace
and saves them as raw CSVs.

Run from project root:
    python data/download_datasets.py
"""

import sys
import os

# ── allow imports from project root ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datasets import load_dataset
from config import (
    HALUEVAL_DATASET_NAME, HALUEVAL_SUBSET, HALUEVAL_RAW_PATH,
    TRUTHFULQA_DATASET_NAME, TRUTHFULQA_SUBSET, TRUTHFULQA_RAW_PATH,
)


# ──────────────────────────────────────────────
# HaluEval
# ──────────────────────────────────────────────

def download_halueval() -> None:
    print("Downloading HaluEval ...")
    ds = load_dataset(HALUEVAL_DATASET_NAME, HALUEVAL_SUBSET, trust_remote_code=True)

    # HaluEval qa_samples has a "data" split
    split = "data" if "data" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()

    # Standardise column names for downstream use
    # HaluEval qa_samples columns: knowledge, question, right_answer, hallucinated_answer
    df = df.rename(columns={
        "knowledge":            "context",
        "question":             "question",
        "right_answer":         "right_answer",
        "hallucinated_answer":  "hallucinated_answer",
    })

    os.makedirs(os.path.dirname(HALUEVAL_RAW_PATH), exist_ok=True)
    df.to_csv(HALUEVAL_RAW_PATH, index=False)
    print(f"  Saved {len(df)} rows → {HALUEVAL_RAW_PATH}")


# ──────────────────────────────────────────────
# TruthfulQA
# ──────────────────────────────────────────────

def download_truthfulqa() -> None:
    print("Downloading TruthfulQA ...")
    ds = load_dataset(TRUTHFULQA_DATASET_NAME, TRUTHFULQA_SUBSET, trust_remote_code=True)

    # TruthfulQA multiple_choice has a "validation" split
    split = "validation" if "validation" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()

    os.makedirs(os.path.dirname(TRUTHFULQA_RAW_PATH), exist_ok=True)
    df.to_csv(TRUTHFULQA_RAW_PATH, index=False)
    print(f"  Saved {len(df)} rows → {TRUTHFULQA_RAW_PATH}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    download_halueval()
    download_truthfulqa()
    print("\nAll datasets downloaded successfully.")