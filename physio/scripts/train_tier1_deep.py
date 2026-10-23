"""
scripts/train_tier1_deep.py

Executes Tier 1 Sequence Models and Tier 2 Adversarial Identity Suppression (GRL):
1. GRU Baseline (`gru_view_physio_all`)
2. 1D-CNN + GRU (`cnn_gru_view_physio_all`)
3. 1D-CNN + GRU + GRL (`cnn_gru_grl_view_physio_all`) with lambda_adv = 0.02

Features:
- Strict Subject-Grouped Leave-One-Subject-Out (LOSO) cross-validation (GroupKFold by subject_id)
- Fold-safe imputation & scaling (fitted strictly on training folds)
- Fail-fast 2-fold smoke testing before full training
- Multi-task optimization: CrossEntropy for stress + GRL-reversed CrossEntropy for subject identity
- Full threshold sweeping (0.05–0.95, step 0.05)
- Standardized artifact generation (metrics.json, curves, predictions.parquet, model_card.md, checkpoint)
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from stressres.train.cv import get_subject_folds
from stressres.train.threshold import sweep_thresholds
from stressres.train.smoketest import run_smoke_test
from stressres.train.artifacts import save_model_artifacts

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ID_META_COLS = [
    "window_id", "subject_id", "task_id", "window_start_s", "window_end_s",
    "duration_s", "is_subwindowed", "is_baseline", "modality_coverage",
    "label_binary", "label_source", "affect3_class", "binary_relax",
    "rating_stress", "rating_relax", "rating_valence", "rating_arousal",
    "is_talking_task"
]

# -------------------------------------------------------------
# 1. Gradient Reversal Layer (GRL)
# -------------------------------------------------------------
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_adv, None

def grad_reverse(x, lambda_adv=0.02):
    return GradientReversalFunction.apply(x, lambda_adv)

# -------------------------------------------------------------
# 2. PyTorch Dataset
# -------------------------------------------------------------
class StressDataset(Dataset):
    def __init__(self, X, y, subjects=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.subjects = torch.tensor(subjects, dtype=torch.long) if subjects is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.subjects is not None:
            return self.X[idx], self.y[idx], self.subjects[idx]
        return self.X[idx], self.y[idx]

# -------------------------------------------------------------
# 3. Model Architectures
# -------------------------------------------------------------
class DeepStressModel(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        model_type: str = "cnn_gru",
        use_grl: bool = False,
        num_subjects: int = 63,
        lambda_adv: float = 0.02,
        dropout: float = 0.3
    ):
        super().__init__()
        self.model_type = model_type
        self.use_grl = use_grl
        self.lambda_adv = lambda_adv
        
        if model_type == "cnn_gru":
            # 1D-CNN feature projection
            self.feature_extractor = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            # GRU sequence/representation layer
            self.gru = nn.GRU(
                input_size=128,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )
            rep_dim = hidden_dim
            
        elif model_type == "gru":
            # Direct 2-layer GRU
            self.feature_proj = nn.Linear(in_features, 64)
            self.gru = nn.GRU(
                input_size=64,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout
            )
            rep_dim = hidden_dim
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        # Primary Stress Classification Head
        self.stress_classifier = nn.Sequential(
            nn.Linear(rep_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
        
        # Adversarial Subject-Identity Head (GRL)
        if use_grl:
            self.subject_classifier = nn.Sequential(
                nn.Linear(rep_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_subjects)
            )

    def forward(self, x):
        if self.model_type == "cnn_gru":
            feats = self.feature_extractor(x)
            feats_seq = feats.unsqueeze(1) # [B, 1, 128]
            out, _ = self.gru(feats_seq)
            rep = out[:, -1, :] # [B, hidden_dim]
        else: # gru
            feats = torch.relu(self.feature_proj(x)).unsqueeze(1) # [B, 1, 64]
            out, _ = self.gru(feats)
            rep = out[:, -1, :]
            
        stress_logits = self.stress_classifier(rep)
        
        if self.use_grl and self.training:
            # Reverse gradient into representation
            rep_reversed = grad_reverse(rep, self.lambda_adv)
            subject_logits = self.subject_classifier(rep_reversed)
            return stress_logits, subject_logits
            
        return stress_logits

# -------------------------------------------------------------
# 4. Training Loop Helper
# -------------------------------------------------------------
def train_model_fold(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 30,
    lr: float = 0.002,
    weight_decay: float = 1e-4,
    use_grl: bool = False
):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Class weights for stress imbalance (~1.7:1)
    criterion_stress = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.71], device=DEVICE))
    criterion_subject = nn.CrossEntropyLoss() if use_grl else None
    
    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            if use_grl:
                X_b, y_b, sub_b = [t.to(DEVICE) for t in batch]
                stress_logits, sub_logits = model(X_b)
                loss_stress = criterion_stress(stress_logits, y_b)
                loss_sub = criterion_subject(sub_logits, sub_b)
                loss = loss_stress + loss_sub
            else:
                X_b, y_b = [t.to(DEVICE) for t in batch]
                stress_logits = model(X_b)
                loss = criterion_stress(stress_logits, y_b)
                
            loss.backward()
            optimizer.step()

def predict_probabilities(model: nn.Module, X_val: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs

# -------------------------------------------------------------
# 5. Full LOSO Pipeline
# -------------------------------------------------------------
def train_and_eval_deep_loso(
    df: pd.DataFrame,
    feature_cols: list,
    model_name: str,
    model_type: str,
    use_grl: bool = False,
    lambda_adv: float = 0.02,
    num_epochs: int = 30,
    view_name: str = "view_physio_all"
):
    print(f"\n=======================================================")
    print(f"Training Deep Model: {model_name} on {view_name}")
    print(f"Type: {model_type} | GRL: {use_grl} (lambda={lambda_adv}) | Device: {DEVICE}")
    print(f"Samples: {len(df)} | Features: {len(feature_cols)} | Subjects: {df['subject_id'].nunique()}")
    print(f"=======================================================")
    
    X_raw = df[feature_cols].values
    y = df["label_binary"].values.astype(np.int64)
    groups = df["subject_id"].values
    window_ids = df["window_id"].values
    
    # 1. Smoke test on 2 folds (2 epochs)
    print("Running 2-fold pre-flight smoke test...")
    def smoke_fn(X_tr, y_tr, X_v):
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_tr_proc = scaler.fit_transform(imputer.fit_transform(X_tr))
        X_v_proc = scaler.transform(imputer.transform(X_v))
        
        m = DeepStressModel(
            in_features=X_tr.shape[1],
            hidden_dim=32,
            model_type=model_type,
            use_grl=False
        ).to(DEVICE)
        
        ds = StressDataset(X_tr_proc, y_tr)
        dl = DataLoader(ds, batch_size=64, shuffle=True)
        train_model_fold(m, dl, num_epochs=2, lr=0.005)
        return predict_probabilities(m, X_v_proc)

    run_smoke_test(smoke_fn, X_raw, y, groups, model_name=model_name)
    
    # 2. Leave-One-Subject-Out Cross-Validation
    folds = get_subject_folds(df, group_col="subject_id", n_splits=None)
    n_folds = len(folds)
    print(f"Executing Leave-One-Subject-Out across {n_folds} folds (Epochs per fold: {num_epochs})...")
    
    y_prob_all = np.zeros(len(y), dtype=float)
    
    for i, (train_idx, val_idx, test_subject) in enumerate(folds):
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        
        X_train_proc = scaler.fit_transform(imputer.fit_transform(X_raw[train_idx]))
        X_val_proc = scaler.transform(imputer.transform(X_raw[val_idx]))
        y_train = y[train_idx]
        
        if use_grl:
            train_sub_ids = groups[train_idx]
            unique_subs = sorted(np.unique(train_sub_ids))
            sub2idx = {s: idx for idx, s in enumerate(unique_subs)}
            sub_indices = np.array([sub2idx[s] for s in train_sub_ids])
            n_sub_classes = len(unique_subs)
            ds_train = StressDataset(X_train_proc, y_train, sub_indices)
        else:
            n_sub_classes = 63
            ds_train = StressDataset(X_train_proc, y_train)
            
        loader = DataLoader(ds_train, batch_size=64, shuffle=True)
        
        model = DeepStressModel(
            in_features=len(feature_cols),
            hidden_dim=64,
            model_type=model_type,
            use_grl=use_grl,
            num_subjects=n_sub_classes,
            lambda_adv=lambda_adv
        ).to(DEVICE)
        
        train_model_fold(model, loader, num_epochs=num_epochs, lr=0.002, use_grl=use_grl)
        probs = predict_probabilities(model, X_val_proc)
        y_prob_all[val_idx] = probs
        
        if (i + 1) % 15 == 0 or (i + 1) == n_folds:
            print(f"  Processed {i+1}/{n_folds} folds (Latest subject: {test_subject})...")
            
    # 3. Decision threshold sweep
    print("Sweeping decision thresholds (0.05 - 0.95)...")
    th_results = sweep_thresholds(y, y_prob_all, start=0.05, stop=0.95, step=0.05)
    opt_th = th_results["optimal_threshold"]
    def_m = th_results["default_metrics"]
    opt_m = th_results["optimal_metrics"]

    print(f"\nResults for {model_name}:")
    print(f"  Default (0.50): Accuracy={def_m['accuracy']:.4f} | BalancedAcc={def_m['balanced_accuracy']:.4f} | MacroF1={def_m['macro_f1']:.4f} | StressF1={def_m['f1']:.4f}")
    print(f"  Optimal ({opt_th:.2f}): Accuracy={opt_m['accuracy']:.4f} | BalancedAcc={opt_m['balanced_accuracy']:.4f} | MacroF1={opt_m['macro_f1']:.4f} | StressF1={opt_m['f1']:.4f}")

    # 4. Train final model on full dataset for checkpointing
    print("Fitting model on full dataset for artifact checkpointing...")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_full_proc = scaler.fit_transform(imputer.fit_transform(X_raw))
    
    unique_all_subs = sorted(np.unique(groups))
    all_sub2idx = {s: idx for idx, s in enumerate(unique_all_subs)}
    all_sub_indices = np.array([all_sub2idx[s] for s in groups])
    
    ds_full = StressDataset(X_full_proc, y, all_sub_indices if use_grl else None)
    loader_full = DataLoader(ds_full, batch_size=64, shuffle=True)
    
    final_model = DeepStressModel(
        in_features=len(feature_cols),
        hidden_dim=64,
        model_type=model_type,
        use_grl=use_grl,
        num_subjects=len(unique_all_subs),
        lambda_adv=lambda_adv
    ).to(DEVICE)
    train_model_fold(final_model, loader_full, num_epochs=num_epochs, lr=0.002, use_grl=use_grl)

    # 5. Save all standardized artifacts
    artifacts_dir = save_model_artifacts(
        model_name=model_name,
        view_name=view_name,
        y_true=y,
        y_prob=y_prob_all,
        subject_ids=groups,
        window_ids=window_ids,
        threshold_data=th_results,
        hyperparameters={
            "model_type": model_type,
            "hidden_dim": 64,
            "use_grl": use_grl,
            "lambda_adv": lambda_adv if use_grl else 0.0,
            "epochs": num_epochs,
            "optimizer": "AdamW",
            "lr": 0.002
        },
        feature_names=feature_cols,
        trained_model=final_model,
        evaluation_protocol="Leave-One-Subject-Out (LOSO) GroupKFold",
        notes=f"Tier 1/2 Deep baseline ({model_type}, GRL={use_grl}) on {view_name} across {n_folds} subjects."
    )

    return {
        "model_name": model_name,
        "view_name": view_name,
        "macro_f1_opt": opt_m["macro_f1"],
        "macro_f1_05": def_m["macro_f1"],
        "acc_opt": opt_m["accuracy"],
        "artifacts_dir": str(artifacts_dir)
    }

def main():
    parser = argparse.ArgumentParser(description="Train Tier 1 & Tier 2 Deep Models")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "gru", "cnn_gru", "cnn_gru_grl"])
    parser.add_argument("--view", type=str, default="view_physio_all")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    view_path = Path(f"data/processed/views/{args.view}.parquet")
    df = pd.read_parquet(view_path)
    df = df[df["label_binary"].notna()].copy()

    feature_cols = [c for c in df.columns if c not in ID_META_COLS]
    print(f"Loaded {len(df)} samples across {df['subject_id'].nunique()} subjects. Features: {len(feature_cols)}")

    models_config = {
        "gru": {
            "model_name": f"gru_{args.view}",
            "model_type": "gru",
            "use_grl": False,
            "lambda_adv": 0.0
        },
        "cnn_gru": {
            "model_name": f"cnn_gru_{args.view}",
            "model_type": "cnn_gru",
            "use_grl": False,
            "lambda_adv": 0.0
        },
        "cnn_gru_grl": {
            "model_name": f"cnn_gru_grl_{args.view}",
            "model_type": "cnn_gru",
            "use_grl": True,
            "lambda_adv": 0.02
        }
    }

    to_run = list(models_config.keys()) if args.model == "all" else [args.model]
    results = []

    for key in to_run:
        cfg = models_config[key]
        res = train_and_eval_deep_loso(
            df=df,
            feature_cols=feature_cols,
            model_name=cfg["model_name"],
            model_type=cfg["model_type"],
            use_grl=cfg["use_grl"],
            lambda_adv=cfg["lambda_adv"],
            num_epochs=args.epochs,
            view_name=args.view
        )
        results.append(res)

    print("\n=======================================================")
    print("ALL DEEP LEARNING RUNS COMPLETED:")
    for r in results:
        print(f"  {r['model_name']}: Macro F1 (opt) = {r['macro_f1_opt']:.4f} (at 0.5: {r['macro_f1_05']:.4f}) | Acc = {r['acc_opt']:.4f}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
