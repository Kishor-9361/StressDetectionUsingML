"""
src/stressres/train/smoketest.py

Smoke-testing engine for model architectures.
Executes a fast 2-fold trial on a tiny slice of data before committing to full
Leave-One-Subject-Out cross-validation.
Checks for NaN/Inf predictions, proper shape, and basic sanity.
"""

from typing import Any, Callable, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

def run_smoke_test(
    train_and_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str = "model"
) -> bool:
    """
    Runs a fast 2-fold cross-validation smoke test.
    
    train_and_predict_fn: Function taking (X_train, y_train, X_val) and returning y_prob_val
    """
    gkf = GroupKFold(n_splits=2)
    fold_checked = 0
    
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_tr = np.ascontiguousarray(X[train_idx])
        y_tr = np.ascontiguousarray(y[train_idx])
        X_v = np.ascontiguousarray(X[val_idx])
        
        # Check shapes
        if len(np.unique(y_tr)) < 2:
            raise RuntimeError(f"Smoke test failed for {model_name}: fold train set has only 1 class")
            
        probs = train_and_predict_fn(X_tr, y_tr, X_v)
        
        # Check output validity
        if probs is None or len(probs) != len(val_idx):
            raise RuntimeError(f"Smoke test failed for {model_name}: output length mismatch")
            
        if np.isnan(probs).any() or np.isinf(probs).any():
            raise RuntimeError(f"Smoke test failed for {model_name}: NaN or Inf found in prediction probabilities")
            
        if (probs < -0.01).any() or (probs > 1.01).any():
            raise RuntimeError(f"Smoke test failed for {model_name}: predicted probabilities outside [0, 1]")
            
        fold_checked += 1
        if fold_checked >= 2:
            break
            
    print(f"[PASS] Smoke test passed for {model_name} (2 folds checked cleanly).")
    return True
