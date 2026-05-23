import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import MODEL_SAVE_PATH, SCALER_SAVE_PATH
from modules.m5_classifier import load_model, predict_trust

nn_model, scaler = load_model(MODEL_SAVE_PATH, scaler_path=SCALER_SAVE_PATH)
if nn_model is None:
    print("Could not load model bundle!")
    sys.exit(1)

# Metrics from screenshot:
# s1 (M1 Consistency) = 0.98
# s2 (M2 Grounding) = 0.5702 (with our fix)
# s3 (M3 Uncertainty) = 0.90
# s4 (M4 Entailment) = 0.62

s1 = 0.98
s2 = 0.5702
s3 = 0.90
s4 = 0.62

question = "who is the prime minister of india"
answer = "Narendra Modi. He has been in office since May 2014. However, for the most accurate and up-to-date information, I would recommend checking a reliable news source or the official government website."

# 1. Prediction with s2 = 0.00 (before fix)
res_before = predict_trust(
    nn_model, s1, 0.00, s3, s4,
    scaler=scaler, question=question, answer=answer
)
print("Trust score before fix (s2 = 0.00):", res_before["trust_score"])

# 2. Prediction with s2 = 0.5702 (after fix)
res_after = predict_trust(
    nn_model, s1, s2, s3, s4,
    scaler=scaler, question=question, answer=answer
)
print("Trust score after fix (s2 = 0.5702):", res_after["trust_score"])
