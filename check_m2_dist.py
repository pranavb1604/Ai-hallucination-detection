import pandas as pd
import numpy as np
import os

train_path = "data/processed/train.csv"
cache_path = "data/processed/cached_m2_m4_features.csv"

if not os.path.exists(cache_path):
    print("No cached M2/M4 features found!")
    # Let's check train.csv if it has scores, otherwise check if cache has it.
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        print("Columns in train.csv:", df.columns.tolist())
else:
    df = pd.read_csv(cache_path)
    print("Cached columns:", df.columns.tolist())
    
    # Merge label from train.csv
    train_df = pd.read_csv(train_path)
    train_unique = train_df.drop_duplicates(subset=["question", "label"], keep="first")
    df = df.merge(
        train_unique[["question", "label"]],
        on=["question", "label"],
        how="left"
    )
    
    print("\nM2 Grounding distribution:")
    print("Total rows:", len(df))
    for label in [0, 1]:
        label_name = "Correct (0)" if label == 0 else "Hallucinated (1)"
        sub = df[df["label"] == label]
        m2_vals = sub["m2_score"].astype(float)
        print(f"\n--- {label_name} ---")
        print(f"  Count        : {len(m2_vals)}")
        print(f"  Mean m2      : {m2_vals.mean():.4f}")
        print(f"  Median m2    : {m2_vals.median():.4f}")
        print(f"  Zero m2 count: {sum(m2_vals == 0.0)} ({sum(m2_vals == 0.0)/len(m2_vals)*100:.1f}%)")
        print(f"  Min m2       : {m2_vals.min():.4f}")
        print(f"  Max m2       : {m2_vals.max():.4f}")
