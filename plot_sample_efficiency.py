"""
plot_sample_efficiency.py
─────────────────────────
Trains classifiers on varying training subset sizes to evaluate model convergence.
Generates a sample complexity plot for the BTP thesis.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import PROCESSED_DATA_DIR, RESULTS_DIR

CACHE_PATH = os.path.join(PROCESSED_DATA_DIR, "cached_m2_m4_features.csv")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

class PyTorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
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
        
    df = pd.read_csv(CACHE_PATH)
    X = df[['m2_score', 'm4_score']].values
    y = df['label'].values

    # Fixed Test Set (20% of data)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define sample sizes to test
    sample_sizes = [20, 50, 100, 200, 400, 800, len(X_train_full)]
    
    results = {
        "Logistic Regression": [],
        "SVM (RBF)": [],
        "Proposed MLP": []
    }
    
    print("Running Sample Complexity Analysis...")
    for size in sample_sizes:
        print(f"  Evaluating sample size: {size}")
        
        # Subsample the training data
        indices = np.random.RandomState(42).choice(len(X_train_full), size=size, replace=False)
        X_sub = X_train_full[indices]
        y_sub = y_train_full[indices]
        
        # 1. Logistic Regression
        lr = LogisticRegression(random_state=42)
        lr.fit(X_sub, y_sub)
        y_pred_lr = lr.predict(X_test)
        results["Logistic Regression"].append(f1_score(y_test, y_pred_lr, zero_division=0))
        
        # 2. SVM
        # Fallback if too few samples for SVM probability fitting
        svm = SVC(random_state=42)
        svm.fit(X_sub, y_sub)
        y_pred_svm = svm.predict(X_test)
        results["SVM (RBF)"].append(f1_score(y_test, y_pred_svm, zero_division=0))
        
        # 3. MLP
        mlp = train_pytorch_mlp(X_sub, y_sub)
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            preds = mlp(X_test_t).numpy().flatten()
            y_pred_mlp = (preds >= 0.5).astype(int)
        results["Proposed MLP"].append(f1_score(y_test, y_pred_mlp, zero_division=0))

    # Plot results
    plt.figure(figsize=(10, 6))
    
    plt.plot(sample_sizes, results["Logistic Regression"], 'o--', color='#94a3b8', label="Logistic Regression", linewidth=2)
    plt.plot(sample_sizes, results["SVM (RBF)"], 's-', color='#6366f1', label="SVM (RBF)", linewidth=2.5)
    plt.plot(sample_sizes, results["Proposed MLP"], '^--', color='#ef4444', label="Proposed MLP (Neural Network)", linewidth=2)
    
    plt.title("Sample Complexity Analysis: F1-Score Convergence", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Number of Training Samples", fontsize=12)
    plt.ylabel("F1-Score on Test Set", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    
    # Highlight saturation
    plt.axvspan(400, sample_sizes[-1], color='#10b981', alpha=0.08, label="Convergence Region")
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "sample_efficiency.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print("\n" + "="*50)
    print(f"Sample efficiency plot successfully saved to:\n{plot_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
