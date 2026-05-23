import sys
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import MODEL_SAVE_PATH
from modules.m5_classifier import load_model
from modules.m5_features import FEATURE_NAMES

bundle, _ = load_model(MODEL_SAVE_PATH)
if bundle is None:
    print("Could not load bundle")
    sys.exit(1)

model = bundle["model"]
importances = model.feature_importances_

print("Feature Importances of retrained M5 classifier:")
print(f"{'Feature Name':<20} | {'Importance':<10}")
print("-" * 33)
for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True):
    print(f"{name:<20} | {imp:.4f}")
