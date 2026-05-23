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

import os
import sys
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
    M5_BACKEND, SCALER_SAVE_PATH, PROCESSED_DATA_DIR,
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

    CACHE_PATH = os.path.join(PROCESSED_DATA_DIR, "cached_m2_m4_features.csv")
    use_cache = os.path.exists(CACHE_PATH)

    if use_cache:
        print(f"Found precomputed M2/M4 cache at {CACHE_PATH}!")
        cache_df = pd.read_csv(CACHE_PATH)
        if args.rows:
            cache_df = cache_df.sample(n=min(args.rows, len(cache_df)), random_state=42).reset_index(drop=True)
        else:
            cache_df = cache_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
        train_unique = full_df.drop_duplicates(subset=["question", "label"], keep="first")
        df = cache_df.merge(
            train_unique[["question", "label", "answer"]],
            on=["question", "label"],
            how="left"
        )
        df["answer"] = df["answer"].fillna("")
        
        # Sibling answers matching index
        from modules.m1_consistency import score as m1_score
        from modules.m3_uncertainty import score as m3_score
        from feature_extraction import build_response_samples
        
        features = []
        labels = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing M1/M3 on top of cached M2/M4"):
            q = str(row["question"]).strip()
            a = str(row["answer"]).strip()
            lbl = int(row["label"])
            
            responses = build_response_samples(q, a, lbl, question_index)
            
            m1 = m1_score(q, responses)["m1_score"]
            m3 = m3_score(q, responses)["m3_score"]
            
            m2 = float(row["m2_score"])
            m4 = float(row["m4_score"])
            
            features.append([m1, m2, m3, m4])
            labels.append(lbl)
            
        X_four = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
    else:
        df = full_df
        if args.rows:
            df = df.sample(n=min(args.rows, len(df)), random_state=42).reset_index(drop=True)

        print(f"Training on {len(df)} rows  (label 1=hallucinated: {df['label'].sum()})")
        print(f"  Questions with 2+ answers in index: "
              f"{sum(1 for v in question_index.values() if len(v) >= 2)}")

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
        X_four = X[:, [0, 2, 1, 3]]

    print(f"\nFeature matrix: {X_four.shape}  |  Labels: {y.shape}")
    print(f"  Hallucinated (1): {int(y.sum())}  |  Correct (0): {int((y == 0).sum())}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining TrustClassifier using backend: {M5_BACKEND} ...")
    questions = df["question"].tolist()
    answers = df["answer"].tolist()

    model, scaler = train_model(
        X_train    = X_four,
        y_train    = y,
        questions  = questions,
        answers    = answers,
        lr         = M5_LEARNING_RATE,
        epochs     = M5_EPOCHS,
        batch_size = M5_BATCH_SIZE,
        backend    = M5_BACKEND,
        save_path  = MODEL_SAVE_PATH,
        scaler_path = SCALER_SAVE_PATH,
        verbose    = True,
    )

    print("\nTraining complete.")
    print(f"Model saved → {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
