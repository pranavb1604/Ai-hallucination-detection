"""
M5 — Neural Meta-Classifier (Trust Calibration Engine) — IMPROVED
A deeper PyTorch network that fuses M1–M4 scores into a single
calibrated Trust Score in [0, 1].

Improvements over original:
  1. 10 engineered features instead of 4 raw scores
  2. Deeper network with BatchNorm + Dropout
  3. StandardScaler normalization (saved alongside model)
  4. Learning rate scheduler + early stopping
  5. Train/val split with validation tracking
  6. M4 zero-score neutralization (0.0 → 0.5)

Architecture:
    Input (10) → FC(64) → BN → ReLU → Drop(0.3)
               → FC(32) → BN → ReLU → Drop(0.2)
               → FC(16) → ReLU
               → FC(1)  → Sigmoid

Loss:      Binary Cross Entropy
Optimiser: Adam + ReduceLROnPlateau
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ─── Constants ────────────────────────────────────────────────────────────────

FEATURE_DIM = 10
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Feature Engineering ──────────────────────────────────────────────────────

def neutralize_m4(m4: float) -> float:
    return 0.5 if m4 == 0.0 else m4


def build_features(m1: float, m2: float, m3: float, m4: float) -> list:
    m4     = neutralize_m4(m4)
    scores = [m1, m2, m3, m4]
    return [
        m1,
        m2,
        m3,
        m4,
        float(np.mean(scores)),
        float(np.std(scores)),
        float(np.min(scores)),
        float(np.max(scores)),
        m1 * m4,
        float(m4 <= 0.5),
    ]


# ─── Network Definition ───────────────────────────────────────────────────────

class TrustClassifier(nn.Module):
    def __init__(self, input_size: int = FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Training Helper ──────────────────────────────────────────────────────────

def train_model(
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    lr:          float = 1e-3,
    epochs:      int   = 500,
    batch_size:  int   = 64,
    patience:    int   = 30,
    val_split:   float = 0.15,
    save_path:   str   = None,
    scaler_path: str   = None,
    verbose:     bool  = True,
) -> tuple:

    # 1. Build engineered features
    X_eng = np.array(
        [build_features(r[0], r[1], r[2], r[3]) for r in X_train],
        dtype=np.float32
    )

    # 2. Train / val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_eng, y_train, test_size=val_split, random_state=42, stratify=y_train
    )

    # 3. Normalize
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr).astype(np.float32)
    X_val  = scaler.transform(X_val).astype(np.float32)

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        if verbose:
            print(f"  Scaler saved → {scaler_path}")

    # 4. DataLoaders
    def make_loader(X, y, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_tr,  y_tr,  shuffle=True)
    val_loader   = make_loader(X_val, y_val, shuffle=False)

    # 5. Model / optimizer / scheduler
    # FIX: verbose argument removed (not supported in older PyTorch versions)
    model     = TrustClassifier(input_size=FEATURE_DIM).to(DEVICE)
    criterion = nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=10, factor=0.5
    )

    # 6. Training loop with early stopping
    if verbose:
        print(f"\nTraining TrustClassifier on {DEVICE} for up to {epochs} epochs ...")

    best_val_loss    = float("inf")
    best_state       = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_tr)

        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(X_val)

        scheduler.step(val_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:>4}/{epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss    = val_loss
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\n  Early stopping at epoch {epoch}")
                break

    # 7. Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    # 8. Save checkpoint
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "input_size":  FEATURE_DIM,
            "val_loss":    best_val_loss,
        }, save_path)
        if verbose:
            print(f"  Model saved → {save_path}")

    # 9. Final validation report
    if verbose:
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb.to(DEVICE))
                all_preds.extend((preds.cpu().numpy() > 0.5).astype(int).flatten())
                all_labels.extend(yb.numpy().astype(int).flatten())
        print(f"\n── Validation Report ──")
        print(f"  Best val loss : {best_val_loss:.4f}")
        print(f"  Val accuracy  : {accuracy_score(all_labels, all_preds):.4f}")
        print(classification_report(
            all_labels, all_preds,
            target_names=["Correct", "Hallucinated"]
        ))

    model.eval()
    return model, scaler


# ─── Inference Helpers ────────────────────────────────────────────────────────

def load_model(
    save_path:   str,
    scaler_path: str = None,
) -> tuple:
    checkpoint = torch.load(save_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state  = checkpoint["model_state"]
        in_dim = checkpoint.get("input_size", FEATURE_DIM)
    else:
        state  = checkpoint
        in_dim = 4

    model = TrustClassifier(input_size=in_dim)
    model.load_state_dict(state)
    model.eval()

    scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None

    return model, scaler


def predict_trust(
    model:  TrustClassifier,
    m1: float, m2: float, m3: float, m4: float,
    scaler: StandardScaler = None,
) -> dict:
    features = np.array([build_features(m1, m2, m3, m4)], dtype=np.float32)

    if scaler is not None:
        features = scaler.transform(features).astype(np.float32)

    x = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        hallucination_prob = float(model(x).squeeze())

    trust_score = 1.0 - hallucination_prob

    return {
        "trust_score":        round(trust_score, 4),
        "hallucination_prob": round(hallucination_prob, 4),
    }


# ─── Fallback Weighted Average (no trained model) ─────────────────────────────

_DEFAULT_WEIGHTS = np.array([0.25, 0.30, 0.20, 0.25])


def weighted_trust_score(
    m1: float, m2: float, m3: float, m4: float,
    weights: np.ndarray = None,
) -> float:
    w      = weights if weights is not None else _DEFAULT_WEIGHTS
    w      = w / w.sum()
    scores = np.array([m1, m2, m3, neutralize_m4(m4)])
    return float(np.clip(np.dot(w, scores), 0.0, 1.0))