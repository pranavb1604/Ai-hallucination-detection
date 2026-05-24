"""
compare_models.py
─────────────────
Trains and compares multiple meta-classifier architectures on precomputed features.
Generates an academic performance comparison table and a comparison plot.
"""
import os
import warnings
warnings.filterwarnings("ignore")

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
from modules.m5_features import build_feature_vector, FEATURE_DIM
from sklearn.preprocessing import StandardScaler

CACHE_PATH = os.path.join(PROCESSED_DATA_DIR, "cached_m2_m4_features.csv")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Define the PyTorch MLP architecture with dynamic inputs and Dropout to prevent overfitting
class PyTorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def train_pytorch_mlp(X_train, y_train, epochs=300, lr=0.003, val_split=0.15, patience=20):
    # Train/Val Split for Early Stopping validation to find best epoch
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_split, random_state=42, stratify=y_train
    )
    
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    model = PyTorchMLP()
    criterion = nn.BCELoss()
    # Add L2 weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_tr_t)
        loss = criterion(preds, y_tr_t)
        loss.backward()
        optimizer.step()
        
        # Validation Loss
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
            
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    # Re-train the model on 100% of the training data for best_epoch epochs
    full_model = PyTorchMLP()
    full_optimizer = torch.optim.Adam(full_model.parameters(), lr=lr, weight_decay=1e-4)
    X_full_t = torch.tensor(X_train, dtype=torch.float32)
    y_full_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    for epoch in range(best_epoch + 1):
        full_model.train()
        full_optimizer.zero_grad()
        preds = full_model(X_full_t)
        loss = criterion(preds, y_full_t)
        loss.backward()
        full_optimizer.step()
        
    full_model.eval()
    return full_model

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
    
    for idx, row in tqdm(merged.iterrows(), total=len(merged), desc=f"Building {FEATURE_DIM} features for comparison"):
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
        
    X_feats = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Train/Test Split (80/20) for empirical validation
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_feats, y, test_size=0.2, random_state=42, stratify=y)
    
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
        "Proposed MLP (Neural Network)": None, # Special handling for PyTorch
        "Proposed Bayesian MLP (MC Dropout)": "bnn_mc" # Special handling for BNN
    }
    
    results = []
    
    for name, clf in models.items():
        print(f"Training {name}...")
        
        # Train and measure latency
        start_time = time.time()
        
        if name == "Gradient Boosting (GBM)":
            # Tuned GBM with RandomizedSearchCV
            from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
            param_distributions = {
                'n_estimators': [200, 300, 400, 500, 600],
                'max_depth': [3, 4, 5, 6, 7],
                'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
                'subsample': [0.75, 0.80, 0.85, 0.90],
                'min_samples_leaf': [5, 8, 10, 12, 15],
            }
            base_clf = GradientBoostingClassifier(random_state=42)
            search = RandomizedSearchCV(
                base_clf,
                param_distributions,
                n_iter=20,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring='accuracy',
                n_jobs=-1,
                random_state=42,
                verbose=0,
            )
            search.fit(X_train, y_train.astype(int))
            clf = search.best_estimator_
            train_time = time.time() - start_time
            
            # Calibrate threshold on training data to maximize accuracy
            train_probs = clf.predict_proba(X_train)[:, 1]
            best_t = 0.5
            best_acc = 0.0
            for t in np.arange(0.3, 0.7, 0.01):
                preds = (train_probs >= t).astype(int)
                acc = accuracy_score(y_train, preds)
                if acc > best_acc:
                    best_acc = acc
                    best_t = t
                    
            # Predict and measure inference latency
            start_inf = time.time()
            test_probs = clf.predict_proba(X_test)[:, 1]
            y_pred = (test_probs >= best_t).astype(int)
            inf_time = (time.time() - start_inf) / len(X_test) * 1000 # in ms
        elif clf == "bnn_mc":
            # PyTorch MLP with early stopping and dropout
            mlp_model = train_pytorch_mlp(X_train, y_train)
            train_time = time.time() - start_time
            
            # Predict and measure inference latency (50 Monte Carlo passes)
            start_inf = time.time()
            
            # Calibrate threshold on training set with 30 passes
            mlp_model.train() # Keep dropout active
            train_preds_list = []
            with torch.no_grad():
                X_tr_t = torch.tensor(X_train, dtype=torch.float32)
                for _ in range(30):
                    train_preds_list.append(mlp_model(X_tr_t).numpy().flatten())
            train_probs = np.mean(train_preds_list, axis=0)
            best_t = 0.5
            best_acc = 0.0
            for t in np.arange(0.3, 0.7, 0.01):
                preds = (train_probs >= t).astype(int)
                acc = accuracy_score(y_train, preds)
                if acc > best_acc:
                    best_acc = acc
                    best_t = t
            
            # Test inference
            preds_list = []
            with torch.no_grad():
                X_test_t = torch.tensor(X_test, dtype=torch.float32)
                for _ in range(50):
                    pass_preds = mlp_model(X_test_t).numpy().flatten()
                    preds_list.append(pass_preds)
            avg_probs = np.mean(preds_list, axis=0)
            y_pred = (avg_probs >= best_t).astype(int)
            inf_time = (time.time() - start_inf) / len(X_test) * 1000 # in ms
        elif clf is not None:
            # Scikit-learn models (Logistic Regression, SVM, Random Forest)
            clf.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Predict and measure inference latency
            start_inf = time.time()
            y_pred = clf.predict(X_test)
            inf_time = (time.time() - start_inf) / len(X_test) * 1000 # in ms
        else:
            # PyTorch MLP with early stopping and dropout
            mlp_model = train_pytorch_mlp(X_train, y_train)
            train_time = time.time() - start_time
            
            # Calibrate threshold on training set
            mlp_model.eval()
            with torch.no_grad():
                X_tr_t = torch.tensor(X_train, dtype=torch.float32)
                train_probs = mlp_model(X_tr_t).numpy().flatten()
            best_t = 0.5
            best_acc = 0.0
            for t in np.arange(0.3, 0.7, 0.01):
                preds = (train_probs >= t).astype(int)
                acc = accuracy_score(y_train, preds)
                if acc > best_acc:
                    best_acc = acc
                    best_t = t
                    
            # Predict and measure inference latency
            start_inf = time.time()
            with torch.no_grad():
                X_test_t = torch.tensor(X_test, dtype=torch.float32)
                preds = mlp_model(X_test_t).numpy().flatten()
                y_pred = (preds >= best_t).astype(int)
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
    plt.figure(figsize=(11, 5.5))
    colors = ['#94a3b8', '#6366f1', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']
    
    bars = plt.bar(results_df["Model Architecture"], results_df["F1-Score"], color=colors, width=0.45)
    plt.title("Meta-Classifier Architecture Comparison (F1-Score)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("F1-Score", fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=15, ha='right')
    
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
