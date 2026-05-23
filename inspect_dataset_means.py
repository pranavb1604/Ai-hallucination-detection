import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

train_path = "data/processed/train.csv"
cache_path = "data/processed/cached_m2_m4_features.csv"

cache_df = pd.read_csv(cache_path)
train_df = pd.read_csv(train_path)
train_unique = train_df.drop_duplicates(subset=["question", "label"], keep="first")
df = cache_df.merge(
    train_unique[["question", "label", "answer"]],
    on=["question", "label"],
    how="left"
).dropna(subset=["answer"])

# Compute M1/M3 on the fly for all rows
from modules.m1_consistency import score as m1_score
from modules.m3_uncertainty import score as m3_score
from feature_extraction import build_question_index, build_response_samples

question_index = build_question_index(train_df)

print("Computing full feature matrix to check feature means...")
rows = []
for idx, row in df.iterrows():
    q = str(row["question"]).strip()
    a = str(row["answer"]).strip()
    lbl = int(row["label"])
    
    responses = build_response_samples(q, a, lbl, question_index)
    m1 = m1_score(q, responses)["m1_score"]
    m3 = m3_score(q, responses)["m3_score"]
    m2 = float(row["m2_score"])
    m4 = float(row["m4_score"])
    
    rows.append({
        "label": lbl,
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "m4": m4
    })

df_feat = pd.DataFrame(rows)

for lbl in [0, 1]:
    lbl_name = "Correct (0)" if lbl == 0 else "Hallucinated (1)"
    sub = df_feat[df_feat["label"] == lbl]
    print(f"\n--- {lbl_name} (count={len(sub)}) ---")
    for col in ["m1", "m2", "m3", "m4"]:
        vals = sub[col]
        print(f"  {col}: mean={vals.mean():.4f}, median={vals.median():.4f}, std={vals.std():.4f}")
