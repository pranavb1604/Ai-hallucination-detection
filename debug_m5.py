import sys
import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import MODEL_SAVE_PATH
from modules.m5_classifier import load_model
from modules.m5_features import build_feature_vector, FEATURE_NAMES

bundle, _ = load_model(MODEL_SAVE_PATH)
if bundle is None:
    print("Could not load bundle")
    sys.exit(1)

model = bundle["model"]
scaler = bundle["scaler"]

s1, s3, s4 = 0.98, 0.90, 0.62
question = "who is the prime minister of india"
answer = "Narendra Modi. He has been in office since May 2014. However, for the most accurate and up-to-date information, I would recommend checking a reliable news source or the official government website."

f_before = build_feature_vector(s1, 0.00, s3, s4, question=question, answer=answer)
f_after = build_feature_vector(s1, 0.5702, s3, s4, question=question, answer=answer)

# Print comparison
print(f"{'Feature Name':<20} | {'Before (s2=0)':<15} | {'After (s2=0.57)':<15}")
print("-" * 56)
for name, v1, v2 in zip(FEATURE_NAMES, f_before, f_after):
    print(f"{name:<20} | {v1:<15.4f} | {v2:<15.4f}")

# Scale features
f_before_s = scaler.transform(f_before.reshape(1, -1))
f_after_s = scaler.transform(f_after.reshape(1, -1))

print("\n--- Model Predictions ---")
prob_before = model.predict_proba(f_before_s)[0, 1]
prob_after = model.predict_proba(f_after_s)[0, 1]

print("Hallucination Prob before (s2=0):", prob_before)
print("Trust score before (s2=0)       :", 1.0 - prob_before)
print("Hallucination Prob after (s2=0.57):", prob_after)
print("Trust score after (s2=0.57)       :", 1.0 - prob_after)
