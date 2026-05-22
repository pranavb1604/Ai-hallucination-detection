"""
Step 1 — Generate m1/m2/m3/m4 scores from train.csv
Step 2 — Retrain M5 classifier with real scores

Run from your project root:
  cd "C:\\Users\\naveen\\OneDrive\\Desktop\\ai hallucination\\Ai-hallucination-detection"
  python generate_scores_and_retrain.py
"""

import os
import sys
import pandas as pd
import numpy as np

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH        = "data/processed/train.csv"
SCORES_CSV      = "data/processed/train_with_scores.csv"
M5_BUNDLE_PATH  = "models/m5_bundle.pkl"
SCALER_PATH     = "models/scaler_m5.pkl"
M5_BACKEND      = "auto"

# Start with 500 to test, then set to None for all 8054
MAX_ROWS = 500


# ── Load CSV ──────────────────────────────────────────────────────────────────
print("Loading CSV...")

df = pd.read_csv(CSV_PATH, header=0)
df.columns = ["question", "answer", "label", "source"]

# Drop header row if it accidentally ended up in data
df = df[df["label"] != "label"].reset_index(drop=True)
df["label"] = df["label"].astype(int)

if MAX_ROWS:
    df = df.head(MAX_ROWS)
    print(f"Using first {MAX_ROWS} rows (change MAX_ROWS=None for all 8054)")

print(f"Rows          : {len(df)}")
print(f"Hallucinated (1): {df.label.sum()}")
print(f"Correct      (0): {(df.label == 0).sum()}")
print(df[["question", "answer", "label"]].head(3))


# ── Score each row ────────────────────────────────────────────────────────────
print("\nGenerating scores (this will take a while)...")

from modules.feature_extraction import extract_all

results = []
failed  = 0
total   = len(df)

for i, row in df.iterrows():
    question = str(row["question"])
    answer   = str(row["answer"])
    label    = int(row["label"])

    try:
        out = extract_all(question, answer)
        s1, s2, s3, s4 = out["features"]
        results.append({
            "question": question,
            "answer":   answer,
            "label":    label,
            "source":   row["source"],
            "m1_score": round(s1, 4),
            "m2_score": round(s2, 4),
            "m3_score": round(s3, 4),
            "m4_score": round(s4, 4),
            "sample_mode": out["sample_mode"],
            "m1_m3_n": out["m1_m3_sample_count"],
            "wiki_title": out["m2"].get("wiki_title", ""),
            "wiki_relevance": out["m2"].get("wiki_relevance", 0.0),
        })

    except Exception as e:
        failed += 1
        results.append({
            "question": question,
            "answer":   answer,
            "label":    label,
            "source":   row["source"],
            "m1_score": 0.5,
            "m2_score": 0.5,
            "m3_score": 0.5,
            "m4_score": 0.5,
        })
        if failed <= 5:
            print(f"  [WARN] Row {i} failed: {e}")

    # Progress every 50 rows
    count = len(results)
    if count % 50 == 0:
        print(f"  Scored {count}/{total} rows  (failed: {failed})")

# ── Save scored CSV ───────────────────────────────────────────────────────────
scored_df = pd.DataFrame(results)
scored_df.to_csv(SCORES_CSV, index=False)
print(f"\nScores saved → {SCORES_CSV}")
print(scored_df[["m1_score","m2_score","m3_score","m4_score","label"]].describe())


# ── Retrain M5 ────────────────────────────────────────────────────────────────
print("\nRetraining M5 classifier...")

from modules.m5_classifier import train_model

X = scored_df[["m1_score","m2_score","m3_score","m4_score"]].values.astype("float32")
y = scored_df["label"].values.astype("float32")

print(f"Feature matrix : {X.shape}")
print(f"Hallucinated (1): {int(y.sum())} | Correct (0): {int((1-y).sum())}")

from config import M5_BACKEND

model, scaler = train_model(
    X, y,
    questions=scored_df["question"].tolist(),
    answers=scored_df["answer"].tolist(),
    sample_modes=scored_df["sample_mode"].tolist() if "sample_mode" in scored_df.columns else None,
    backend=M5_BACKEND,
    save_path=M5_BUNDLE_PATH,
    scaler_path=SCALER_PATH,
    epochs=100,
    verbose=True,
)

print("\n✅ Done! Now restart Streamlit:")
print("   streamlit run app/streamlit_app.py")