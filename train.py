"""
train.py
────────
Trains the M5 neural meta-classifier on pre-computed module scores.

Pipeline:
  1. Load train.csv  (columns: question, answer, label, source)
  2. For each row compute M1–M4 scores using the answer as a single response.
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
    TRAIN_PATH, MODEL_SAVE_PATH,
    M5_INPUT_SIZE, M5_HIDDEN_SIZE_1, M5_HIDDEN_SIZE_2,
    M5_LEARNING_RATE, M5_EPOCHS, M5_BATCH_SIZE,
)
from modules.m1_consistency import score as m1_score
from modules.m2_grounding   import score as m2_score
from modules.m3_uncertainty  import score as m3_score
from modules.m4_entailment   import score as m4_score
from modules.m5_classifier   import train_model


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_features(question: str, answer: str) -> list[float]:
    """
    Run M1–M4 on a single (question, answer) pair.
    We pass [answer] as the responses list (single sample).
    """
    m1 = m1_score(question, [answer])["m1_score"]
    m2 = m2_score(question, [answer])["m2_score"]
    m3 = m3_score(question, [answer])["m3_score"]
    m4 = m4_score(question, [answer])["m4_score"]
    return [m1, m2, m3, m4]


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

    df = pd.read_csv(TRAIN_PATH)
    if args.rows:
        df = df.sample(n=min(args.rows, len(df)), random_state=42).reset_index(drop=True)

    print(f"Training on {len(df)} rows  (label 1=hallucinated: {df['label'].sum()})")

    # ── Extract features ──────────────────────────────────────────────────────
    features = []
    labels   = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:
            feats = extract_features(str(row["question"]), str(row["answer"]))
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
