"""
M5 — Neural Meta-Classifier (Trust Calibration Engine)
A shallow PyTorch network that fuses M1–M4 scores into a single
calibrated Trust Score in [0, 1].

Architecture:
    Input  (4)  → FC(16) → ReLU → FC(8) → ReLU → FC(1) → Sigmoid
Loss:    Binary Cross Entropy
Optimiser: Adam
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── Network definition ───────────────────────────────────────────────────────

class TrustClassifier(nn.Module):
    """
    Three fully-connected layers with ReLU activations.
    Final sigmoid maps output to [0, 1] (trust probability).
    """

    def __init__(
        self,
        input_size: int  = 4,
        hidden1:    int  = 16,
        hidden2:    int  = 8,
        output_size: int = 1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Training helper ──────────────────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_size:    int   = 4,
    hidden1:       int   = 16,
    hidden2:       int   = 8,
    lr:            float = 0.001,
    epochs:        int   = 100,
    batch_size:    int   = 32,
    save_path:     str   = None,
    verbose:       bool  = True,
) -> TrustClassifier:
    """
    Train the meta-classifier on pre-computed module scores.

    Args:
        X_train : (N, 4) array of [m1, m2, m3, m4] scores
        y_train : (N,)   array of labels  0=correct, 1=hallucinated
                         NOTE: trust score = 1 - hallucination_prob,
                         so we train to predict hallucination probability.
        save_path: if given, saves the model checkpoint here.

    Returns:
        Trained TrustClassifier.
    """
    model = TrustClassifier(input_size, hidden1, hidden2)
    criterion = nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimiser.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(X_train)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:>3}/{epochs}  loss={epoch_loss:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"  Model saved → {save_path}")

    return model


# ─── Inference helper ─────────────────────────────────────────────────────────

def load_model(save_path: str, input_size: int = 4,
               hidden1: int = 16, hidden2: int = 8) -> TrustClassifier:
    """Load a saved TrustClassifier checkpoint."""
    model = TrustClassifier(input_size, hidden1, hidden2)
    model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict_trust(
    model: TrustClassifier,
    m1: float, m2: float, m3: float, m4: float,
) -> dict:
    """
    Given four module scores, return the calibrated trust score.

    Trust score = 1 − hallucination_probability
    (the network is trained to predict hallucination probability)
    """
    x = torch.tensor([[m1, m2, m3, m4]], dtype=torch.float32)
    with torch.no_grad():
        hallucination_prob = float(model(x).squeeze())

    trust_score = 1.0 - hallucination_prob

    return {
        "trust_score":          round(trust_score, 4),
        "hallucination_prob":   round(hallucination_prob, 4),
    }


# ─── Fallback weighted average (no trained model) ────────────────────────────

_DEFAULT_WEIGHTS = np.array([0.25, 0.30, 0.20, 0.25])   # M1, M2, M3, M4


def weighted_trust_score(m1: float, m2: float, m3: float, m4: float,
                          weights: np.ndarray = None) -> float:
    """
    Simple weighted average fallback when no trained model is available.
    """
    w = weights if weights is not None else _DEFAULT_WEIGHTS
    w = w / w.sum()
    scores = np.array([m1, m2, m3, m4])
    return float(np.clip(np.dot(w, scores), 0.0, 1.0))
