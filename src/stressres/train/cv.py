"""
src/stressres/train/cv.py

Subject-grouped cross-validation infrastructure.
Implements Leave-One-Subject-Out (LOSO) and GroupKFold cross-validation
strictly grouped by subject_id to prevent subject identity leakage.
"""

from typing import Generator, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

def get_subject_folds(
    df: pd.DataFrame,
    group_col: str = "subject_id",
    n_splits: int = None,
    shuffle: bool = True,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Constructs cross-validation splits grouped by subject_id.
    
    If n_splits is None, sets n_splits == n_subjects (true Leave-One-Subject-Out).
    If n_splits < n_subjects, implements standard Subject-Grouped K-Fold.
    
    Returns a list of tuples: (train_indices, val_indices, subject_identifier)
    """
    subjects = np.sort(df[group_col].unique())
    n_subjects = len(subjects)
    
    if n_splits is None or n_splits >= n_subjects:
        # True Leave-One-Subject-Out
        splits = []
        for sub in subjects:
            val_idx = np.where(df[group_col] == sub)[0]
            train_idx = np.where(df[group_col] != sub)[0]
            splits.append((train_idx, val_idx, str(sub)))
        return splits
    else:
        # Subject-grouped K-Fold
        gkf = GroupKFold(n_splits=n_splits)
        splits = []
        groups = df[group_col].values
        for fold_i, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
            fold_subs = df.iloc[val_idx][group_col].unique().tolist()
            fold_id = f"fold_{fold_i:02d}_{len(fold_subs)}subs"
            splits.append((train_idx, val_idx, fold_id))
        return splits
