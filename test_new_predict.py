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

# Metrics from your screen:
s1 = 0.98
s2 = 0.67
s3 = 0.90
s4 = 0.61

question = "who is the prime minister of india"
answer = "Narendra Modi. He has been in office since May 2014. However, for the most accurate and up-to-date information, I would recommend checking a reliable news source or the official government website."

res = predict_trust(
    nn_model, s1, s2, s3, s4,
    scaler=scaler, question=question, answer=answer
)
print("New Trust Score predicted by M5 meta-classifier:", res["trust_score"])
