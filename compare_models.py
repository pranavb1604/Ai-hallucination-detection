"""
compare_models.py
─────────────────
Trains and compares multiple meta-classifier architectures on precomputed features.
Generates an academic performance comparison table and a comparison plot.
"""
import os

import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import PROCESSED_DATA_DIR, RESULTS_DIR, TRAIN_PATH, TEST_PATH
from modules.m5_features import build_feature_vector
from sklearn.preprocessing import StandardScaler

CACHE_PATH = os.path.join(PROCESSED_DATA_DIR, "cached_m2_m4_features.csv")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Define the PyTorch MLP architecture with 20 inputs
class PyTorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(21, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def train_pytorch_mlp(X_train, y_train, epochs=150, lr=0.005):
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    model = PyTorchMLP()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model

def main():
    if not os.path.exists(CACHE_PATH):
        print(f"Error: Cached features not found at {CACHE_PATH}")
        print("Please run python precompute_features.py first!")
        sys.exit(1)
        
    # Load raw train data for questions and answers
    train_df = pd.read_csv(TRAIN_PATH)
    index_df = train_df
    if os.path.exists(TEST_PATH):
        index_df = pd.concat([train_df, pd.read_csv(TEST_PATH)], ignore_index=True)
    
    from feature_extraction import build_question_index, build_response_samples
    question_index = build_question_index(index_df)
    
    df = pd.read_csv(CACHE_PATH)
    
    # Merge to get raw answer text
    train_unique = train_df.drop_duplicates(subset=["question", "label"], keep="first")
    merged = df.merge(
        train_unique[["question", "label", "answer"]],
        on=["question", "label"],
        how="left"
    )
    merged["answer"] = merged["answer"].fillna("")
    
    from modules.m1_consistency import score as m1_score
    from modules.m3_uncertainty import score as m3_score
    from tqdm import tqdm
    
    X_list = []
    y_list = []
    
    for idx, row in tqdm(merged.iterrows(), total=len(merged), desc="Building 21 features for comparison"):
        q = str(row["question"]).strip()
        a = str(row["answer"]).strip()
        lbl = int(row["label"])
        
        responses = build_response_samples(q, a, lbl, question_index)
        
        m1 = m1_score(q, responses)["m1_score"]
        m3 = m3_score(q, responses)["m3_score"]
        
        m2 = float(row["m2_score"])
        m4 = float(row["m4_score"])
        
        feats = build_feature_vector(m1, m2, m3, m4, question=q, answer=a)
        X_list.append(feats)
        y_list.append(lbl)
        
    X_21 = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Train/Test Split (80/20) for empirical validation
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_21, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    print(f"Comparing architectures on {len(X_train)} training samples and {len(X_test)} test samples...\n")

    # Define classifiers to evaluate
    models = {
        "Logistic Regression (Linear)": LogisticRegression(random_state=42),
        "Support Vector Machine (RBF)": SVC(probability=True, random_state=42),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting (GBM)": GradientBoostingClassifier(random_state=42),
        "Proposed MLP (Neural Network)": None # Special handling for PyTorch
    }
    
    results = []
    
    for name, clf in models.items():
        print(f"Training {name}...")
        
        # Train and measure latency
        start_time = time.time()
        
        if clf is not None:
            # Scikit-learn models
            clf.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Predict and measure inference latency
            start_inf = time.time()
            y_pred = clf.predict(X_test)
            inf_time = (time.time() - start_inf) / len(X_test) * 1000 # in ms
        else:
            # PyTorch MLP
            mlp_model = train_pytorch_mlp(X_train, y_train)
            train_time = time.time() - start_time
            
            # Predict and measure inference latency
            start_inf = time.time()
            with torch.no_grad():
                X_test_t = torch.tensor(X_test, dtype=torch.float32)
                preds = mlp_model(X_test_t).numpy().flatten()
                y_pred = (preds >= 0.5).astype(int)
            inf_time = (time.time() - start_inf) / len(X_test) * 1000 # in ms

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            "Model Architecture": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Inference Latency (ms)": inf_time
        })

    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # 1. Print Markdown Table
    print("\n" + "="*80)
    print("                    COMPARATIVE ARCHITECTURE EVALUATION TABLE")
    print("="*80)
    print(results_df.to_markdown(index=False, floatfmt=".4f"))
    print("="*80 + "\n")
    
    # Save results to CSV for thesis reference
    results_df.to_csv(os.path.join(RESULTS_DIR, "architecture_comparison.csv"), index=False)
    print(f"Results saved to {os.path.join(RESULTS_DIR, 'architecture_comparison.csv')}")

    # 2. Generate Comparison Plot
    plt.figure(figsize=(10, 5))
    colors = ['#94a3b8', '#6366f1', '#10b981', '#f59e0b', '#ef4444']
    
    bars = plt.bar(results_df["Model Architecture"], results_df["F1-Score"], color=colors, width=0.5)
    plt.title("Meta-Classifier Architecture Comparison (F1-Score)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("F1-Score", fontsize=12)
    plt.ylim(0, 1.1)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to {plot_path}")
    print("Ready for your BTP Report/Thesis!")

if __name__ == "__main__":
    main()
