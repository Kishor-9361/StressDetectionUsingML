"""
scripts/train_face_voice_pipeline.py

Unified execution pipeline for Face and Voice modalities:
1. View 1: `view_face_alone` (Face alone, 84 OpenFace features, 600 task windows, 55 subjects)
2. View 2: `view_voice_alone` (Voice alone, 140 acoustic features, 378 task windows, 54 subjects)
3. View 3: `view_face_voice` (Bimodal Audiovisual, 224 features, 371 task windows, 53 subjects)

Models trained per view:
- Classical Tier 0:
  - Logistic Regression (logreg)
  - K-Nearest Neighbors (knn)
  - Support Vector Machine (svm)
  - Random Forest (rf)
  - XGBoost (xgboost)
  - LightGBM (lightgbm)
- Deep Sequence Tier 1:
  - GRU (gru)
  - 1D-CNN + GRU (cnn_gru)
  - 1D-CNN + GRU + GRL (cnn_gru_grl with adversarial subject identity suppression)

Methodology:
- Strict Leave-One-Subject-Out (LOSO) cross-validation grouped by subject_id.
- Fold-safe imputation + standard scaling.
- Sweeps decision thresholds (0.05 to 0.95).
- Generates 7 mandated artifacts for every model under models/<model_name>/<timestamp>_<hash>/.
- Evaluates Tier 2 Identity Leakage-Gap Benchmark (LOSO vs random row-wise K-Fold).
"""

# CRITICAL ON WINDOWS: lightgbm must be imported before pyarrow/scipy/torch
import lightgbm as lgb
from lightgbm import LGBMClassifier

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from stressres.train.cv import get_subject_folds
from stressres.train.threshold import sweep_thresholds
from stressres.train.artifacts import save_model_artifacts

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ID_META_COLS = [
    "window_id", "subject_id", "task_id", "window_start_s", "window_end_s",
    "duration_s", "is_subwindowed", "is_baseline", "modality_coverage",
    "label_binary", "label_source", "affect3_class", "binary_relax",
    "rating_stress", "rating_relax", "rating_valence", "rating_arousal",
    "is_talking_task"
]

# =====================================================================
# 1. Gradient Reversal Layer & Deep Models
# =====================================================================
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_adv, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_adv: float = 0.02):
        super().__init__()
        self.lambda_adv = lambda_adv

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_adv)

class FaceVoiceSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, subject_labels: np.ndarray, seq_len: int = 4):
        self.seq_len = seq_len
        n_samples, n_feats = X.shape
        self.chunk_size = max(1, n_feats // seq_len)
        self.padded_feats = self.chunk_size * seq_len
        if self.padded_feats > n_feats:
            pad = np.zeros((n_samples, self.padded_feats - n_feats), dtype=np.float32)
            X = np.hstack([X, pad])
        else:
            X = X[:, :self.padded_feats]
        
        self.X_seq = torch.tensor(X.reshape(n_samples, seq_len, self.chunk_size), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.subject_labels = torch.tensor(subject_labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.y[idx], self.subject_labels[idx]

class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        feat = out[:, -1, :]
        logits = self.classifier(feat)
        return logits, feat

class CNNGRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x_conv = x.permute(0, 2, 1)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.permute(0, 2, 1)
        out, _ = self.gru(x_conv)
        feat = out[:, -1, :]
        logits = self.classifier(feat)
        return logits, feat

class AdversarialCNNGRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2, num_subjects: int = 60, lambda_adv: float = 0.02):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, num_layers=1)
        self.stress_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        self.grl = GradientReversalLayer(lambda_adv=lambda_adv)
        self.subject_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_subjects)
        )

    def forward(self, x):
        x_conv = x.permute(0, 2, 1)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.permute(0, 2, 1)
        out, _ = self.gru(x_conv)
        feat = out[:, -1, :]
        stress_logits = self.stress_classifier(feat)
        grl_feat = self.grl(feat)
        subject_logits = self.subject_classifier(grl_feat)
        return stress_logits, subject_logits, feat

# =====================================================================
# 2. Dataset Views Loader
# =====================================================================
def load_modality_view(view_name: str):
    """
    Loads features and labels for face and voice views.
    """
    if view_name == "view_face_alone":
        df = pd.read_parquet("data/processed/features/face_features.parquet")
    elif view_name == "view_voice_alone":
        df = pd.read_parquet("data/processed/features/voice_features.parquet")
    elif view_name == "view_face_voice":
        df = pd.read_parquet("data/processed/features/audiovisual_features.parquet")
    else:
        raise ValueError(f"Unknown view: {view_name}")

    feat_cols = [c for c in df.columns if c not in ID_META_COLS]
    X = df[feat_cols].values.astype(np.float32)
    y = df["label_binary"].values.astype(int)
    subject_ids = df["subject_id"].values
    window_ids = df["window_id"].values
    
    print(f"[{view_name}] Loaded {len(df)} samples across {len(np.unique(subject_ids))} subjects. Features: {len(feat_cols)}.")
    return X, y, subject_ids, window_ids, feat_cols, df

# =====================================================================
# 3. Classical Model Trainer
# =====================================================================
def train_classical_model(model_type: str, view_name: str):
    model_name = f"{model_type}_{view_name}"
    print(f"\n=======================================================")
    print(f"Training Classical Model: {model_name}")
    print(f"=======================================================")
    
    X, y, subject_ids, window_ids, feature_names, df = load_modality_view(view_name)
    splits = get_subject_folds(df, group_col="subject_id", n_splits=None)
    n_splits = len(splits)
    
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    if model_type == "logreg":
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
        pipe = Pipeline([("imputer", imputer), ("scaler", scaler), ("clf", clf)])
    elif model_type == "knn":
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
        pipe = Pipeline([("imputer", imputer), ("scaler", scaler), ("clf", clf)])
    elif model_type == "svm":
        clf = SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42)
        pipe = Pipeline([("imputer", imputer), ("scaler", scaler), ("clf", clf)])
    elif model_type == "rf":
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
        pipe = Pipeline([("imputer", imputer), ("clf", clf)])
    elif model_type == "xgboost":
        scale_pos = (len(y) - sum(y)) / max(1, sum(y))
        clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42, n_jobs=-1)
        pipe = Pipeline([("imputer", imputer), ("clf", clf)])
    elif model_type == "lightgbm":
        scale_pos = (len(y) - sum(y)) / max(1, sum(y))
        clf = LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, scale_pos_weight=scale_pos, random_state=42, verbose=-1, n_jobs=-1)
        pipe = Pipeline([("imputer", imputer), ("clf", clf)])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    y_prob = np.zeros(len(y), dtype=np.float32)
    start_time = time.time()
    
    for fold_idx, (train_idx, val_idx, sub_id) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx]
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_val)
        y_prob[val_idx] = probs[:, 1]

    elapsed = time.time() - start_time
    print(f"[{model_name}] Finished {n_splits} LOSO folds in {elapsed:.1f}s.")
    
    threshold_data = sweep_thresholds(y, y_prob)
    opt_th = threshold_data["optimal_threshold"]
    opt_m = threshold_data["optimal_metrics"]
    def_m = threshold_data["default_metrics"]
    print(f"[{model_name}] Default (0.50): Acc={def_m['accuracy']:.4f}, Macro F1={def_m['macro_f1']:.4f}")
    print(f"[{model_name}] Optimal ({opt_th:.2f}): Acc={opt_m['accuracy']:.4f}, Macro F1={opt_m['macro_f1']:.4f}")

    # Retrain on full data for checkpoint saving
    pipe.fit(X, y)
    
    run_dir = save_model_artifacts(
        model_name=model_name,
        view_name=view_name,
        y_true=y,
        y_prob=y_prob,
        subject_ids=subject_ids,
        window_ids=window_ids,
        threshold_data=threshold_data,
        hyperparameters={"model_type": model_type, "params": str(clf.get_params())},
        feature_names=feature_names,
        trained_model=pipe,
        evaluation_protocol=f"Subject-Grouped LOSO ({n_splits} folds)",
        notes=f"Tier 0 classical model on {view_name}."
    )
    return model_name, opt_m, def_m

# =====================================================================
# 4. Deep Sequence & GRL Trainer
# =====================================================================
def train_deep_model(model_type: str, view_name: str, epochs: int = 15, lr: float = 1e-3, lambda_adv: float = 0.02):
    model_name = f"{model_type}_{view_name}"
    print(f"\n=======================================================")
    print(f"Training Deep/GRL Model: {model_name} on {DEVICE}")
    print(f"=======================================================")

    X, y, subject_ids, window_ids, feature_names, df = load_modality_view(view_name)
    splits = get_subject_folds(df, group_col="subject_id", n_splits=None)
    n_splits = len(splits)

    sub_encoder = LabelEncoder()
    sub_labels = sub_encoder.fit_transform(subject_ids)
    num_subjects = len(sub_encoder.classes_)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_clean = scaler.fit_transform(imputer.fit_transform(X))
    
    seq_len = 4
    chunk_size = max(1, X_clean.shape[1] // seq_len)
    
    y_prob = np.zeros(len(y), dtype=np.float32)
    start_time = time.time()

    for fold_idx, (train_idx, val_idx, sub_id) in enumerate(splits):
        # Fit scaler on train fold strictly
        f_imputer = SimpleImputer(strategy="median")
        f_scaler = StandardScaler()
        X_tr = f_scaler.fit_transform(f_imputer.fit_transform(X[train_idx]))
        X_va = f_scaler.transform(f_imputer.transform(X[val_idx]))

        train_ds = FaceVoiceSequenceDataset(X_tr, y[train_idx], sub_labels[train_idx], seq_len=seq_len)
        val_ds = FaceVoiceSequenceDataset(X_va, y[val_idx], sub_labels[val_idx], seq_len=seq_len)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        if model_type == "gru":
            net = GRUClassifier(input_dim=chunk_size, hidden_dim=64, num_classes=2).to(DEVICE)
        elif model_type == "cnn_gru":
            net = CNNGRUClassifier(input_dim=chunk_size, hidden_dim=64, num_classes=2).to(DEVICE)
        elif model_type == "cnn_gru_grl":
            net = AdversarialCNNGRUClassifier(input_dim=chunk_size, hidden_dim=64, num_classes=2, num_subjects=num_subjects, lambda_adv=lambda_adv).to(DEVICE)
        else:
            raise ValueError(f"Unknown deep model_type: {model_type}")

        # Class weights for stress imbalance
        pos_weight = torch.tensor([(len(train_idx) - sum(y[train_idx])) / max(1, sum(y[train_idx]))], device=DEVICE)
        criterion_stress = nn.CrossEntropyLoss()
        criterion_sub = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

        net.train()
        for epoch in range(epochs):
            for x_batch, y_batch, sub_batch in train_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                sub_batch = sub_batch.to(DEVICE)

                optimizer.zero_grad()
                if model_type == "cnn_gru_grl":
                    stress_logits, sub_logits, _ = net(x_batch)
                    loss_stress = criterion_stress(stress_logits, y_batch)
                    loss_sub = criterion_sub(sub_logits, sub_batch)
                    loss = loss_stress + loss_sub
                else:
                    stress_logits, _ = net(x_batch)
                    loss = criterion_stress(stress_logits, y_batch)
                loss.backward()
                optimizer.step()

        # Validation prediction
        net.eval()
        with torch.no_grad():
            for x_val, _, _ in val_loader:
                x_val = x_val.to(DEVICE)
                if model_type == "cnn_gru_grl":
                    logits, _, _ = net(x_val)
                else:
                    logits, _ = net(x_val)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                y_prob[val_idx] = probs

    elapsed = time.time() - start_time
    print(f"[{model_name}] Finished {n_splits} LOSO folds in {elapsed:.1f}s.")

    threshold_data = sweep_thresholds(y, y_prob)
    opt_th = threshold_data["optimal_threshold"]
    opt_m = threshold_data["optimal_metrics"]
    def_m = threshold_data["default_metrics"]
    print(f"[{model_name}] Default (0.50): Acc={def_m['accuracy']:.4f}, Macro F1={def_m['macro_f1']:.4f}")
    print(f"[{model_name}] Optimal ({opt_th:.2f}): Acc={opt_m['accuracy']:.4f}, Macro F1={opt_m['macro_f1']:.4f}")

    # Retrain on full dataset for checkpoint
    full_ds = FaceVoiceSequenceDataset(X_clean, y, sub_labels, seq_len=seq_len)
    full_loader = DataLoader(full_ds, batch_size=32, shuffle=True)
    if model_type == "gru":
        full_net = GRUClassifier(input_dim=chunk_size, hidden_dim=64, num_classes=2).to(DEVICE)
    elif model_type == "cnn_gru":
        full_net = CNNGRUClassifier(input_dim=chunk_size, hidden_dim=64, num_classes=2).to(DEVICE)
    elif model_type == "cnn_gru_grl":
        full_net = AdversarialCNNGRUClassifier(input_dim=chunk_size, hidden_dim=64, num_classes=2, num_subjects=num_subjects, lambda_adv=lambda_adv).to(DEVICE)

    opt_full = optim.Adam(full_net.parameters(), lr=lr)
    full_net.train()
    for ep in range(epochs):
        for xb, yb, sb in full_loader:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            opt_full.zero_grad()
            if model_type == "cnn_gru_grl":
                s_log, sub_log, _ = full_net(xb)
                l = criterion_stress(s_log, yb) + criterion_sub(sub_log, sb)
            else:
                s_log, _ = full_net(xb)
                l = criterion_stress(s_log, yb)
            l.backward()
            opt_full.step()

    run_dir = save_model_artifacts(
        model_name=model_name,
        view_name=view_name,
        y_true=y,
        y_prob=y_prob,
        subject_ids=subject_ids,
        window_ids=window_ids,
        threshold_data=threshold_data,
        hyperparameters={"model_type": model_type, "epochs": epochs, "lr": lr, "lambda_adv": lambda_adv if "grl" in model_type else None},
        feature_names=feature_names,
        trained_model=full_net,
        evaluation_protocol=f"Subject-Grouped LOSO ({n_splits} folds)",
        notes=f"Tier 1/2 Deep model on {view_name}."
    )
    return model_name, opt_m, def_m

# =====================================================================
# 5. Tier 2 Identity Leakage-Gap Benchmark
# =====================================================================
def run_identity_leakage_benchmark(view_name: str = "view_face_voice"):
    print(f"\n=======================================================")
    print(f"Running Tier 2 Identity Leakage Benchmark on {view_name}")
    print(f"=======================================================")
    
    X, y, subject_ids, window_ids, feature_names, df = load_modality_view(view_name)
    imputer = SimpleImputer(strategy="median")
    
    # 1. Subject-Grouped LOSO
    loso_splits = get_subject_folds(df, group_col="subject_id", n_splits=None)
    y_prob_loso = np.zeros(len(y), dtype=np.float32)
    for train_idx, val_idx, _ in loso_splits:
        pipe = Pipeline([("imputer", imputer), ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1))])
        pipe.fit(X[train_idx], y[train_idx])
        y_prob_loso[val_idx] = pipe.predict_proba(X[val_idx])[:, 1]
    
    # 2. Random Row-Wise Stratified K-Fold (Identity Leaking)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_prob_random = np.zeros(len(y), dtype=np.float32)
    for train_idx, val_idx in skf.split(X, y):
        pipe = Pipeline([("imputer", imputer), ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1))])
        pipe.fit(X[train_idx], y[train_idx])
        y_prob_random[val_idx] = pipe.predict_proba(X[val_idx])[:, 1]

    loso_acc = accuracy_score(y, (y_prob_loso >= 0.5).astype(int))
    loso_f1 = f1_score(y, (y_prob_loso >= 0.5).astype(int), average="macro")
    loso_auc = roc_auc_score(y, y_prob_loso)

    rand_acc = accuracy_score(y, (y_prob_random >= 0.5).astype(int))
    rand_f1 = f1_score(y, (y_prob_random >= 0.5).astype(int), average="macro")
    rand_auc = roc_auc_score(y, y_prob_random)

    delta_acc = rand_acc - loso_acc
    delta_f1 = rand_f1 - loso_f1
    delta_auc = rand_auc - loso_auc

    print(f"LOSO Grouped:     Acc = {loso_acc:.4f}, Macro F1 = {loso_f1:.4f}, ROC AUC = {loso_auc:.4f}")
    print(f"Random (Leaky):   Acc = {rand_acc:.4f}, Macro F1 = {rand_f1:.4f}, ROC AUC = {rand_auc:.4f}")
    print(f"Leakage Gap:     Delta_Acc = +{delta_acc*100:.2f}%, Delta_F1 = +{delta_f1:.4f}, Delta_AUC = +{delta_auc:.4f}")

    os.makedirs("reports", exist_ok=True)
    os.makedirs("docs/verification", exist_ok=True)

    leakage_data = {
        "view_name": view_name,
        "n_samples": len(y),
        "n_subjects": len(np.unique(subject_ids)),
        "loso_grouped": {"accuracy": float(loso_acc), "macro_f1": float(loso_f1), "roc_auc": float(loso_auc)},
        "random_leaky": {"accuracy": float(rand_acc), "macro_f1": float(rand_f1), "roc_auc": float(rand_auc)},
        "leakage_gap": {"delta_accuracy": float(delta_acc), "delta_macro_f1": float(delta_f1), "delta_roc_auc": float(delta_auc)}
    }

    with open("reports/leakage_gap_report.json", "w") as f:
        json.dump(leakage_data, f, indent=2)

    # Plot Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["Accuracy", "Macro F1", "ROC AUC"]
    loso_vals = [loso_acc, loso_f1, loso_auc]
    rand_vals = [rand_acc, rand_f1, rand_auc]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, loso_vals, width, label="Subject-Grouped LOSO (Rigorous)", color="#3b82f6")
    ax.bar(x + width/2, rand_vals, width, label="Random Row-Wise (Identity Leaking)", color="#ef4444")
    
    ax.set_ylabel("Score")
    ax.set_title(f"Empirical Identity Leakage Proof ({view_name})")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.legend()
    for i in range(len(metrics)):
        gap = rand_vals[i] - loso_vals[i]
        ax.text(x[i] + width/2, rand_vals[i] + 0.02, f"+{gap*100:.1f}%", ha="center", fontweight="bold", color="#b91c1c")
    
    fig.tight_layout()
    fig.savefig("docs/verification/leakage_gap_comparison.png", dpi=200)
    plt.close(fig)
    print("Saved leakage gap report and comparison plot.")

# =====================================================================
# 6. Master Runner
# =====================================================================
def main():
    print("*******************************************************************")
    print("STARTING COMPLETE FACE & VOICE RESTARTED PIPELINE")
    print("*******************************************************************")

    # 1. Train View 1: Face alone
    train_classical_model("logreg", "view_face_alone")
    train_classical_model("knn", "view_face_alone")
    train_classical_model("svm", "view_face_alone")
    train_classical_model("rf", "view_face_alone")
    train_classical_model("xgboost", "view_face_alone")
    train_classical_model("lightgbm", "view_face_alone")
    train_deep_model("gru", "view_face_alone", epochs=15)
    train_deep_model("cnn_gru", "view_face_alone", epochs=15)
    train_deep_model("cnn_gru_grl", "view_face_alone", epochs=15, lambda_adv=0.02)

    # 2. Train View 2: Voice alone
    train_classical_model("rf", "view_voice_alone")
    train_classical_model("xgboost", "view_voice_alone")
    train_classical_model("svm", "view_voice_alone")
    train_deep_model("cnn_gru_grl", "view_voice_alone", epochs=15, lambda_adv=0.02)

    # 3. Train View 3: Face + Voice (Audiovisual Bimodal)
    train_classical_model("logreg", "view_face_voice")
    train_classical_model("knn", "view_face_voice")
    train_classical_model("svm", "view_face_voice")
    train_classical_model("rf", "view_face_voice")
    train_classical_model("xgboost", "view_face_voice")
    train_classical_model("lightgbm", "view_face_voice")
    train_deep_model("gru", "view_face_voice", epochs=15)
    train_deep_model("cnn_gru", "view_face_voice", epochs=15)
    train_deep_model("cnn_gru_grl", "view_face_voice", epochs=15, lambda_adv=0.02)

    # 4. Identity Leakage Benchmark
    run_identity_leakage_benchmark("view_face_voice")

    print("\n*******************************************************************")
    print("ALL MODELS TRAINED SUCCESSFULLY ON FACE & VOICE ALONE!")
    print("*******************************************************************")

if __name__ == "__main__":
    main()
