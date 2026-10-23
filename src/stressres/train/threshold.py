"""
src/stressres/train/threshold.py

Threshold-tuning engine for binary stress classification.
Given validation-fold probabilities and labels, sweeps decision thresholds
from 0.05 to 0.95 with step 0.05, computing full classification metrics.
Called identically across all model training routines.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score
)

def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.05
) -> Dict[str, Any]:
    """
    Sweeps decision threshold from start to stop (inclusive) with step.
    
    Returns a dictionary containing:
    - 'threshold_curve': {round(thresh, 2): {accuracy, precision, recall, f1, balanced_accuracy, macro_f1}}
    - 'optimal_threshold': float
    - 'optimal_metrics': dict
    - 'default_threshold': 0.50
    - 'default_metrics': dict
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    thresholds = np.arange(start, stop + step / 2, step)
    curve = {}
    
    best_thresh = 0.50
    best_macro_f1 = -1.0
    
    for th in thresholds:
        th_val = round(float(th), 2)
        y_pred = (y_prob >= th_val).astype(int)
        
        acc = float(accuracy_score(y_true, y_pred))
        bacc = float(balanced_accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1_pos = float(f1_score(y_true, y_pred, zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        
        metrics = {
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "precision": prec,
            "recall": rec,
            "f1": f1_pos,
            "macro_f1": macro_f1
        }
        curve[f"{th_val:.2f}"] = metrics
        
        # Optimize for Macro F1 (class-balanced)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_thresh = th_val

    default_metrics = curve.get("0.50", curve[list(curve.keys())[len(curve)//2]])
    optimal_metrics = curve[f"{best_thresh:.2f}"]

    return {
        "threshold_curve": curve,
        "optimal_threshold": best_thresh,
        "optimal_metrics": optimal_metrics,
        "default_threshold": 0.50,
        "default_metrics": default_metrics
    }
