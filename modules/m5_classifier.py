"""
M5 — Meta-classifier (Trust Calibration Engine)

Backends:
  - gb  : GradientBoosting (recommended for 80%+ on module scores)
  - nn  : PyTorch MLP (legacy)
  - auto: train both, keep higher validation accuracy

Saves a single bundle: models/m5_bundle.pkl
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from modules.m5_features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    build_feature_vector,
    build_batch,
    neutralize_m4,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# ─── PyTorch network (legacy backend) ─────────────────────────────────────────

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


# ─── Threshold tuning ───────────────────────────────────────────────────────

def _tune_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.25, 0.76, 0.01):
        pred = (probs >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc





# ─── Train GB ─────────────────────────────────────────────────────────────────

def _train_gb(
    X_eng: np.ndarray,
    y_train: np.ndarray,
    val_split: float,
    verbose: bool,
) -> tuple[GradientBoostingClassifier, StandardScaler, float, float, dict]:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_eng, y_train, test_size=val_split, random_state=42, stratify=y_train,
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    # ── Hyperparameter tuning with RandomizedSearchCV ─────────────────────
    param_distributions = {
        'n_estimators': [200, 300, 400, 500, 600],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
        'subsample': [0.75, 0.80, 0.85, 0.90],
        'min_samples_leaf': [5, 8, 10, 12, 15],
    }

    base_clf = GradientBoostingClassifier(random_state=42)

    if verbose:
        print("\n  Running hyperparameter search (50 combos, 3-fold CV)...")

    search = RandomizedSearchCV(
        base_clf,
        param_distributions,
        n_iter=50,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_tr_s, y_tr.astype(int))
    clf = search.best_estimator_

    if verbose:
        print(f"  Best params: {search.best_params_}")
        print(f"  Best CV F1 : {search.best_score_:.4f}")

    # ── 5-fold cross-validation on full training set for robust estimate ──
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(
        clf, X_tr_s, y_tr.astype(int),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1',
    )
    if verbose:
        print(f"  5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Evaluate on held-out validation set ───────────────────────────────
    probs = clf.predict_proba(X_val_s)[:, 1]
    threshold, val_acc = _tune_threshold(y_val, probs)
    val_f1 = f1_score(y_val, (probs >= threshold).astype(int), zero_division=0)

    report = {
        "backend": "gb",
        "val_accuracy": val_acc,
        "val_f1": val_f1,
        "threshold": threshold,
        "y_val": y_val,
        "probs": probs,
        "best_params": search.best_params_,
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
    }

    if verbose:
        print("\n-- Gradient Boosting Validation --")
        print(f"  Val accuracy  : {val_acc:.4f}")
        print(f"  Val F1        : {val_f1:.4f}")
        print(f"  Threshold     : {threshold:.2f}")
        preds = (probs >= threshold).astype(int)
        print(classification_report(
            y_val, preds, target_names=["Correct", "Hallucinated"],
        ))

    return clf, scaler, threshold, val_acc, report


# ─── Train NN ─────────────────────────────────────────────────────────────────

def _train_nn(
    X_eng: np.ndarray,
    y_train: np.ndarray,
    lr: float,
    epochs: int,
    batch_size: int,
    patience: int,
    val_split: float,
    verbose: bool,
) -> tuple[TrustClassifier, StandardScaler, float, float, dict]:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_eng, y_train, test_size=val_split, random_state=42, stratify=y_train,
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    def make_loader(X, y, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_tr_s, y_tr, shuffle=True)
    val_loader = make_loader(X_val_s, y_val, shuffle=False)

    model = TrustClassifier(input_size=FEATURE_DIM).to(DEVICE)
    criterion = nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=10, factor=0.5,
    )

    if verbose:
        print(f"\nTraining neural M5 on {DEVICE} for up to {epochs} epochs ...")

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_tr_s)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(X_val_s)

        scheduler.step(val_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:>4}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\n  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        probs = []
        for xb, _ in val_loader:
            probs.extend(model(xb.to(DEVICE)).cpu().numpy().flatten())
    probs = np.array(probs)
    threshold, val_acc = _tune_threshold(y_val, probs)
    val_f1 = f1_score(y_val, (probs >= threshold).astype(int), zero_division=0)

    if verbose:
        print("\n-- Neural Validation --")
        print(f"  Val accuracy  : {val_acc:.4f}")
        print(f"  Val F1        : {val_f1:.4f}")
        print(f"  Threshold     : {threshold:.2f}")

    report = {"backend": "nn", "val_accuracy": val_acc, "val_f1": val_f1, "threshold": threshold}
    return model, scaler, threshold, val_acc, report


# ─── Public training API ──────────────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    questions: list | np.ndarray | None = None,
    answers: list | np.ndarray | None = None,
    sample_modes: list | None = None,
    lr: float = 1e-3,
    epochs: int = 500,
    batch_size: int = 64,
    patience: int = 30,
    val_split: float = 0.15,
    backend: str = "auto",
    save_path: str | None = None,
    scaler_path: str | None = None,
    verbose: bool = True,
) -> tuple:

    metas = None
    if sample_modes is not None:
        metas = [{"sample_mode": sm} for sm in sample_modes]

    X_eng = build_batch(
        X_train.astype(np.float32),
        questions=list(questions) if questions is not None else None,
        answers=list(answers) if answers is not None else None,
        metas=metas,
    )

    if verbose:
        print(f"  Engineered features: {X_eng.shape[1]} dims")

    bundles = {}

    if backend in ("gb", "auto"):
        gb_clf, gb_scaler, gb_thr, gb_acc, _ = _train_gb(X_eng, y_train, val_split, verbose)
        bundles["gb"] = {
            "model": gb_clf,
            "scaler": gb_scaler,
            "threshold": gb_thr,
            "val_accuracy": gb_acc,
        }

    if backend in ("nn", "auto"):
        nn_model, nn_scaler, nn_thr, nn_acc, _ = _train_nn(
            X_eng, y_train, lr, epochs, batch_size, patience, val_split, verbose,
        )
        bundles["nn"] = {
            "model": nn_model,
            "scaler": nn_scaler,
            "threshold": nn_thr,
            "val_accuracy": nn_acc,
        }

    if backend == "auto":
        winner = "gb" if bundles["gb"]["val_accuracy"] >= bundles["nn"]["val_accuracy"] else "nn"
        if verbose:
            print(f"\n  Selected backend: {winner.upper()} "
                  f"(gb={bundles['gb']['val_accuracy']:.4f}, nn={bundles['nn']['val_accuracy']:.4f})")
    else:
        winner = backend

    chosen = bundles[winner]
    bundle = {
        "backend": winner,
        "model": chosen["model"],
        "scaler": chosen["scaler"],
        "threshold": chosen["threshold"],
        "feature_dim": FEATURE_DIM,
        "feature_names": FEATURE_NAMES,
        "val_accuracy": chosen["val_accuracy"],
    }

    bundle_path = save_path
    if bundle_path:
        if bundle_path.endswith(".pth"):
            bundle_path = os.path.join(os.path.dirname(bundle_path), "m5_bundle.pkl")
        os.makedirs(os.path.dirname(bundle_path) or ".", exist_ok=True)
        joblib.dump(bundle, bundle_path)
        if verbose:
            print(f"  Bundle saved -> {bundle_path}")

        if scaler_path and scaler_path != bundle_path:
            joblib.dump(chosen["scaler"], scaler_path)

        if winner == "nn" and save_path and save_path.endswith(".pth"):
            torch.save({
                "model_state": chosen["model"].state_dict(),
                "input_size": FEATURE_DIM,
            }, save_path)

    if winner == "gb":
        return chosen["model"], chosen["scaler"]
    return chosen["model"], chosen["scaler"]


# ─── Load & predict ───────────────────────────────────────────────────────────

def _load_bundle(path: str) -> dict | None:
    if path and os.path.exists(path):
        return joblib.load(path)
    return None


def load_model(save_path: str, scaler_path: str | None = None) -> tuple:
    """Load M5 bundle (gb or nn). Falls back to legacy .pth neural checkpoint."""
    bundle_path = save_path.replace(".pth", "_bundle.pkl") if save_path.endswith(".pth") else save_path
    alt = os.path.join(os.path.dirname(save_path or "."), "m5_bundle.pkl")

    bundle = _load_bundle(bundle_path) or _load_bundle(alt)
    if bundle is not None:
        return bundle, bundle.get("scaler")

    if save_path and os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            in_dim = checkpoint.get("input_size", FEATURE_DIM)
            model = TrustClassifier(input_size=in_dim)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None
            legacy = {
                "backend": "nn",
                "model": model,
                "scaler": scaler,
                "threshold": 0.5,
                "feature_dim": in_dim,
            }
            return legacy, scaler

    return None, None


def _hallucination_prob(bundle: dict, features: np.ndarray) -> float:
    scaler = bundle.get("scaler")
    thr = bundle.get("threshold", 0.5)
    x = features.reshape(1, -1)
    if scaler is not None:
        x = scaler.transform(x)

    if bundle.get("backend") == "gb":
        prob = float(bundle["model"].predict_proba(x)[0, 1])
        return prob

    model = bundle["model"]
    xt = torch.tensor(x.astype(np.float32))
    with torch.no_grad():
        return float(model(xt).squeeze())


def predict_trust(
    model,
    m1: float,
    m2: float,
    m3: float,
    m4: float,
    scaler: StandardScaler | None = None,
    question: str | None = None,
    answer: str | None = None,
    meta: dict | None = None,
) -> dict:
    """Predict trust score. `model` may be a bundle dict from load_model."""
    bundle = model if isinstance(model, dict) and "backend" in model else None
    if bundle is None:
        bundle = {"backend": "nn", "model": model, "scaler": scaler, "threshold": 0.5}

    feats = build_feature_vector(m1, m2, m3, m4, question=question, answer=answer, meta=meta)
    hallucination_prob = _hallucination_prob(bundle, feats)
    trust_score = 1.0 - hallucination_prob

    return {
        "trust_score": round(trust_score, 4),
        "hallucination_prob": round(hallucination_prob, 4),
        "decision_threshold": bundle.get("threshold", 0.5),
    }


def predict_batch(
    bundle: dict,
    X_four: np.ndarray,
    questions: list[str] | None = None,
    answers: list[str] | None = None,
) -> np.ndarray:
    """Return hallucination probabilities for a batch."""
    X_eng = build_batch(X_four, questions=questions, answers=answers)
    probs = []
    for row in X_eng:
        probs.append(_hallucination_prob(bundle, row))
    return np.array(probs, dtype=np.float32)


_DEFAULT_WEIGHTS = np.array([0.25, 0.30, 0.20, 0.25])


def weighted_trust_score(
    m1: float, m2: float, m3: float, m4: float,
    weights: np.ndarray | None = None,
) -> float:
    w = weights if weights is not None else _DEFAULT_WEIGHTS
    w = w / w.sum()
    scores = np.array([m1, m2, m3, neutralize_m4(m4)])
    return float(np.clip(np.dot(w, scores), 0.0, 1.0))


