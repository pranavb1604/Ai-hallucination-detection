"""
train.py
────────
Trains the M5 neural meta-classifier on pre-computed module scores.

Pipeline:
  1. Load train.csv  (columns: question, answer, label, source)
  2. For each row compute M1–M4 (M1/M3 use sentence pseudo-samples when possible;
     M2/M4 use the primary answer — same rules as Ollama inference).
  3. Train M5 meta-classifier (20 features + Gradient Boosting, auto-tuned threshold).
  4. Save bundle to models/m5_bundle.pkl

Usage:
    python train.py            # full dataset
    python train.py --rows 500 # quick test with 500 rows
    python train.py --resume   # skip already-scored rows
    python train.py --fresh    # delete old scores + re-run with new Wikipedia retrieval
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import (
    TRAIN_PATH, MODEL_SAVE_PATH, M5_BUNDLE_PATH, SCALER_SAVE_PATH,
    M5_LEARNING_RATE, M5_EPOCHS, M5_BATCH_SIZE, M5_BACKEND,
    SCORING_WORKERS,
)
from modules.feature_extraction import extract_all
from modules.m5_classifier import train_model

SCORES_CACHE = os.path.join(BASE_DIR, "data", "processed", "train_with_scores.csv")


def _fit_m5(scored_df: pd.DataFrame):
    X = scored_df[["m1_score", "m2_score", "m3_score", "m4_score"]].values.astype("float32")
    y = scored_df["label"].values.astype("float32")
    modes = scored_df["sample_mode"].tolist() if "sample_mode" in scored_df.columns else None

    # ── Flip M1 and M3 to align with live inference semantics ──
    # Training pseudo-samples: high M1 = hallucinated (similar pseudo-samples)
    # Live Ollama:             high M1 = correct (all responses agree)
    # Flip so HIGH always means CORRECT across both training and live.
    X[:, 0] = 1.0 - X[:, 0]   # M1: flip consistency
    X[:, 2] = 1.0 - X[:, 2]   # M3: flip certainty

    print(f"\nFeature matrix : {X.shape}")
    print(f"Hallucinated(1): {int(y.sum())} | Correct(0): {int((y == 0).sum())}")
    print(f"M5 backend     : {M5_BACKEND}")
    return train_model(
        X_train=X,
        y_train=y,
        questions=scored_df["question"].tolist(),
        answers=scored_df["answer"].tolist(),
        sample_modes=modes,
        lr=M5_LEARNING_RATE,
        epochs=M5_EPOCHS,
        batch_size=M5_BATCH_SIZE,
        backend=M5_BACKEND,
        save_path=M5_BUNDLE_PATH,
        scaler_path=SCALER_SAVE_PATH,
        verbose=True,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train M5 meta-classifier")
    parser.add_argument("--rows",   type=int,  default=None,
                        help="Limit number of training rows (for quick tests)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing scores cache")
    parser.add_argument("--train-only", action="store_true",
                        help="Skip scoring; train from data/processed/train_with_scores.csv")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete cached scores and re-score all rows (use after Wikipedia/M5 updates)")
    args = parser.parse_args()

    if args.fresh and os.path.exists(SCORES_CACHE):
        os.remove(SCORES_CACHE)
        print(f"Removed old scores cache -> re-scoring with current Wikipedia retrieval")
        args.resume = False

    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(TRAIN_PATH):
        print(f"[ERROR] Training data not found at {TRAIN_PATH}")
        print("  Run:  python src/data/preprocess.py  first.")
        sys.exit(1)

    df = pd.read_csv(TRAIN_PATH, header=0)
    df.columns = ["question", "answer", "label", "source"]
    df = df[df["label"] != "label"].reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    if args.rows:
        df = df.head(args.rows)
        print(f"Using first {args.rows} rows")

    print(f"Total rows     : {len(df)}")
    print(f"Hallucinated(1): {df.label.sum()}")
    print(f"Correct     (0): {(df.label == 0).sum()}")

    def _print_feature_stats(frame: pd.DataFrame) -> None:
        print("\nFeature means by label (higher score = more trustworthy):")
        for col in ["m1_score", "m2_score", "m3_score", "m4_score"]:
            ok = frame.loc[frame["label"] == 0, col].mean()
            bad = frame.loc[frame["label"] == 1, col].mean()
            print(f"  {col}: correct={ok:.3f}  hallucinated={bad:.3f}")

    # ── Train-only mode (use pre-computed scores) ─────────────────────────────
    if args.train_only:
        if not os.path.exists(SCORES_CACHE):
            print(f"[ERROR] Scores cache not found: {SCORES_CACHE}")
            print("  Run without --train-only first to generate scores.")
            sys.exit(1)
        scored_df = pd.read_csv(SCORES_CACHE)
        if "sample_mode" not in scored_df.columns:
            print("[WARN] Cache missing sample_mode column (old scoring run).")
            print("  Delete train_with_scores.csv and re-run without --train-only.")
        elif scored_df["m1_score"].nunique() <= 1 and scored_df["m3_score"].nunique() <= 1:
            print("[WARN] m1/m3 scores look constant in cache.")
            print("  Delete the cache and re-run without --train-only.")
        _print_feature_stats(scored_df)
        _fit_m5(scored_df)
        print("\nTraining complete!")
        print(f"   Bundle -> {M5_BUNDLE_PATH}")
        return

    # ── Resume support ────────────────────────────────────────────────────────
    results = []
    if args.resume and os.path.exists(SCORES_CACHE):
        cached = pd.read_csv(SCORES_CACHE)
        done   = set(cached["question"].tolist())
        df     = df[~df["question"].isin(done)].reset_index(drop=True)
        results = cached.to_dict("records")
        print(f"Resuming — {len(cached)} already scored, {len(df)} remaining")
    else:
        print("Starting fresh scoring...")

    # ── Extract features (PARALLEL) ───────────────────────────────────────────
    failed = 0
    total  = len(df)
    _lock  = threading.Lock()

    def _score_one(idx, question, answer, label, source):
        """Score a single row — safe to call from multiple threads."""
        try:
            out = extract_all(question, answer)
            s1, s2, s3, s4 = out["features"]
            return {
                "question": question,
                "answer":   answer,
                "label":    label,
                "source":   source,
                "m1_score": round(s1, 4),
                "m2_score": round(s2, 4),
                "m3_score": round(s3, 4),
                "m4_score": round(s4, 4),
                "sample_mode": out["sample_mode"],
                "m1_m3_n": out["m1_m3_sample_count"],
                "wiki_title": out["m2"].get("wiki_title", ""),
                "wiki_relevance": out["m2"].get("wiki_relevance", 0.0),
            }
        except Exception as e:
            return None  # skip failed rows instead of polluting with 0.5

    workers = min(SCORING_WORKERS, total)
    print(f"Scoring with {workers} parallel workers...")

    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idx, row in df.iterrows():
            question = str(row["question"])
            answer   = str(row["answer"])
            label    = int(row["label"])
            source   = str(row.get("source", ""))
            fut = executor.submit(_score_one, idx, question, answer, label, source)
            futures[fut] = idx

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring rows"):
            result = fut.result()
            if result is not None:
                results.append(result)
            else:
                failed += 1

            # Save cache every 100 rows
            if len(results) % 100 == 0 and len(results) > 0:
                with _lock:
                    pd.DataFrame(results).to_csv(SCORES_CACHE, index=False)

    # Final save
    scored_df = pd.DataFrame(results)
    scored_df.to_csv(SCORES_CACHE, index=False)
    print(f"\nScores saved → {SCORES_CACHE}")
    print(f"Failed rows  : {failed}/{total}")
    print(scored_df[["m1_score","m2_score","m3_score","m4_score","label"]].describe())
    _print_feature_stats(scored_df)

    # ── Train ─────────────────────────────────────────────────────────────────
    _fit_m5(scored_df)

    print("\nTraining complete!")
    print(f"   Bundle -> {M5_BUNDLE_PATH}")
    print(f"   Scaler -> {SCALER_SAVE_PATH}")
    print("\nRestart Streamlit:")
    print("   streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()