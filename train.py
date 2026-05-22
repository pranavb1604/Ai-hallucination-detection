"""
train.py
────────
Trains the M5 neural meta-classifier on pre-computed module scores.

Pipeline:
  1. Load train.csv  (columns: question, answer, label, source)
  2. For each row compute M1–M4 scores (M1/M3 use multi-response Option B).
  3. Train TrustClassifier on [m1, m2, m3, m4] → label (1=hallucinated).
  4. Save model checkpoint to models/trust_classifier.pth.

Usage:
    python train.py [--rows N]   # --rows limits dataset size for quick tests
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── project root on path ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import (
    TRAIN_PATH, TEST_PATH, MODEL_SAVE_PATH,
    M5_INPUT_SIZE, M5_HIDDEN_SIZE_1, M5_HIDDEN_SIZE_2,
    M5_LEARNING_RATE, M5_EPOCHS, M5_BATCH_SIZE,
)
from feature_extraction import build_question_index, extract_features
from modules.m5_classifier import train_model


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train M5 meta-classifier")
    parser.add_argument("--rows", type=int, default=None,
                        help="Limit number of training rows (for quick tests)")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(TRAIN_PATH):
        print(f"[ERROR] Training data not found at {TRAIN_PATH}")
        print("  Run:  python src/data/preprocess.py  first.")
        sys.exit(1)

    full_df = pd.read_csv(TRAIN_PATH)
    # Sibling answers may sit in test.csv (split is by row, not question)
    index_df = full_df
    if os.path.exists(TEST_PATH):
        index_df = pd.concat(
            [full_df, pd.read_csv(TEST_PATH)], ignore_index=True
        )
    question_index = build_question_index(index_df)

    df = full_df
    if args.rows:
        df = df.sample(n=min(args.rows, len(df)), random_state=42).reset_index(drop=True)

    print(f"Training on {len(df)} rows  (label 1=hallucinated: {df['label'].sum()})")
    print(f"  Questions with 2+ answers in index: "
          f"{sum(1 for v in question_index.values() if len(v) >= 2)}")

    # ── Extract features ──────────────────────────────────────────────────────
    features = []
    labels   = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:
            feats = extract_features(
                str(row["question"]),
                str(row["answer"]),
                label=int(row["label"]),
                question_index=question_index,
            )
            features.append(feats)
            labels.append(int(row["label"]))
        except Exception as e:
            print(f"  [WARN] Row {idx} skipped: {e}")
            continue

    X = np.array(features, dtype=np.float32)
    y = np.array(labels,   dtype=np.float32)

    print(f"\nFeature matrix: {X.shape}  |  Labels: {y.shape}")
    print(f"  Hallucinated (1): {int(y.sum())}  |  Correct (0): {int((y == 0).sum())}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining TrustClassifier for {M5_EPOCHS} epochs …")
    model = train_model(
        X_train    = X,
        y_train    = y,
        input_size = M5_INPUT_SIZE,
        hidden1    = M5_HIDDEN_SIZE_1,
        hidden2    = M5_HIDDEN_SIZE_2,
        lr         = M5_LEARNING_RATE,
        epochs     = M5_EPOCHS,
        batch_size = M5_BATCH_SIZE,
        save_path  = MODEL_SAVE_PATH,
        verbose    = True,
    )

    print("\nTraining complete.")
    print(f"Checkpoint saved → {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
