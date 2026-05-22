"""
evaluate.py
───────────
Evaluates the trained M5 meta-classifier on the held-out test set.

Metrics reported:
  • Accuracy, Precision, Recall, F1 (binary)
  • AUROC
  • Confusion matrix
  • Per-module ablation study (each module removed one at a time)
  • SHAP feature importance (global bar chart)

Usage:
    python evaluate.py [--rows N]
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import (
    TRAIN_PATH, TEST_PATH, MODEL_SAVE_PATH, SCORES_DIR,
    M5_INPUT_SIZE, M5_HIDDEN_SIZE_1, M5_HIDDEN_SIZE_2,
)
from feature_extraction import build_question_index, extract_features
from modules.m5_classifier import load_model, predict_trust, weighted_trust_score

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)


# ─── Metrics helper ───────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auroc":     round(roc_auc_score(y_true, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ─── Ablation study ───────────────────────────────────────────────────────────

def ablation_study(X: np.ndarray, y_true: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """
    For each module i, replace its column with 0.5 (neutral) and
    measure the drop in F1 using the weighted fallback scorer.
    """
    module_names = ["M1_consistency", "M2_grounding", "M3_uncertainty", "M4_entailment"]
    results = {}

    # Baseline (all modules)
    baseline_probs = np.array([
        weighted_trust_score(*row) for row in X
    ])
    baseline_preds = (baseline_probs < threshold).astype(int)   # low trust → hallucinated
    baseline_f1 = f1_score(y_true, baseline_preds, zero_division=0)
    results["baseline_f1"] = round(baseline_f1, 4)

    for i, name in enumerate(module_names):
        X_ablated = X.copy()
        X_ablated[:, i] = 0.5   # replace with neutral value

        probs = np.array([weighted_trust_score(*row) for row in X_ablated])
        preds = (probs < threshold).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        drop = baseline_f1 - f1
        results[name] = {
            "f1_without": round(f1, 4),
            "f1_drop":    round(drop, 4),
        }

    return results


# ─── SHAP explanation ─────────────────────────────────────────────────────────

def run_shap(model, X: np.ndarray, save_dir: str) -> None:
    """
    Compute SHAP values for the neural classifier and save a bar chart.
    """
    try:
        import shap
        import torch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X_t = __import__("torch").tensor(X, dtype=__import__("torch").float32)

        def model_fn(x_np):
            xt = __import__("torch").tensor(x_np, dtype=__import__("torch").float32)
            with __import__("torch").no_grad():
                return model(xt).numpy()

        explainer = shap.Explainer(model_fn, X[:100])
        shap_values = explainer(X[:200])

        feature_names = ["M1 Consistency", "M2 Grounding",
                         "M3 Uncertainty", "M4 Entailment"]

        fig, ax = plt.subplots(figsize=(7, 4))
        mean_abs = np.abs(shap_values.values).mean(axis=0).flatten()
        ax.barh(feature_names, mean_abs, color=["#6366f1", "#10b981", "#f59e0b", "#ef4444"])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("SHAP Feature Importance — Trust Classifier")
        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, "shap_importance.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  SHAP chart saved → {fig_path}")

    except Exception as e:
        print(f"  [WARN] SHAP visualisation skipped: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate M5 meta-classifier")
    parser.add_argument("--rows", type=int, default=None,
                        help="Limit number of test rows (for quick tests)")
    args = parser.parse_args()

    # ── Load test data ────────────────────────────────────────────────────────
    if not os.path.exists(TEST_PATH):
        print(f"[ERROR] Test data not found at {TEST_PATH}")
        print("  Run:  python src/data/preprocess.py  first.")
        sys.exit(1)

    full_df = pd.read_csv(TEST_PATH)
    index_df = full_df
    if os.path.exists(TRAIN_PATH):
        index_df = pd.concat(
            [pd.read_csv(TRAIN_PATH), full_df], ignore_index=True
        )
    question_index = build_question_index(index_df)

    df = full_df
    if args.rows:
        df = df.sample(n=min(args.rows, len(df)), random_state=42).reset_index(drop=True)

    print(f"Evaluating on {len(df)} test rows …")

    # ── Extract features ──────────────────────────────────────────────────────
    features, labels = [], []
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

    X = np.array(features, dtype=np.float32)
    y = np.array(labels,   dtype=np.float32)

    # ── Load model ────────────────────────────────────────────────────────────
    use_nn = os.path.exists(MODEL_SAVE_PATH)
    if use_nn:
        print(f"\nLoading trained model from {MODEL_SAVE_PATH} …")
        model = load_model(MODEL_SAVE_PATH, M5_INPUT_SIZE,
                           M5_HIDDEN_SIZE_1, M5_HIDDEN_SIZE_2)
    else:
        print("\n[WARN] No trained model found — using weighted fallback scorer.")
        model = None

    # ── Predict ───────────────────────────────────────────────────────────────
    if use_nn:
        import torch
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            hallucination_probs = model(X_t).squeeze().numpy()
        trust_probs = 1.0 - hallucination_probs
    else:
        trust_probs = np.array([weighted_trust_score(*row) for row in X])
        hallucination_probs = 1.0 - trust_probs

    # Threshold: trust < 0.5 → predicted hallucinated (label=1)
    threshold = 0.5
    y_pred = (trust_probs < threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(y, y_pred, hallucination_probs)

    print("\n" + "═" * 45)
    print("  EVALUATION RESULTS")
    print("═" * 45)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  AUROC     : {metrics['auroc']:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print("═" * 45)

    # ── Ablation study ────────────────────────────────────────────────────────
    print("\nRunning ablation study …")
    ablation = ablation_study(X, y.astype(int))
    print(f"\n  Baseline F1 (all modules): {ablation['baseline_f1']:.4f}")
    for mod in ["M1_consistency", "M2_grounding", "M3_uncertainty", "M4_entailment"]:
        info = ablation[mod]
        print(f"  Without {mod:<20} F1={info['f1_without']:.4f}  "
              f"(drop={info['f1_drop']:+.4f})")

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(SCORES_DIR, exist_ok=True)
    results_path = os.path.join(SCORES_DIR, "evaluation_results.json")
    output = {"metrics": metrics, "ablation": ablation}
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    if use_nn:
        print("\nGenerating SHAP explanations …")
        run_shap(model, X, SCORES_DIR)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
