"""
runtime_engine.py
Single authoritative inference engine for the Multimodal Stress Detector.

ONE production model: a three-expert fusion
    face   -> GradientBoostingClassifier (18 geometric indicators)
    voice  -> GradientBoostingClassifier (12 acoustic biomarkers)
    physio -> CalibratedClassifierCV(GB + RF) (51 EEG/GSR features)
combined with fixed reliability weights over whichever modalities are present.

Artifacts are loaded through the VersionRegistry (hash-verified) from
``models/expert_models``.
"""

import os
import sys
import json
import pickle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.core.feature_runtime_lock import FeatureRuntimeLock
from backend.core.version_registry      import VersionRegistry

TORCH_AVAILABLE = False

MODALITIES = ("face", "voice", "physio")

# Reliability weights of each expert in the fusion (normalised over the
# modalities that are actually present in a request).
FUSION_WEIGHTS = {"face": 0.371, "voice": 0.474, "physio": 0.338}

# Decision threshold on the fused stress probability at sensitivity=0.5.
DECISION_THRESHOLD = 0.5

# Model filename map — registry keys → (model pkl, scaler pkl) inside EXPERT_DIR
_MODEL_FILES = {
    "face_expert":   ("face_expert_lightweight.pkl",  "face_scaler_lightweight.pkl"),
    "voice_expert":  ("voice_expert_lightweight.pkl", "voice_scaler_lightweight.pkl"),
    "physio_expert": ("physio_expert.pkl",            "physio_scaler.pkl"),
}

_possible_model_dirs = [
    os.path.join(ROOT, "models"),
    os.path.join(ROOT, "webapp", "models"),
    os.path.join(PROJECT_ROOT, "webapp", "models"),
]
# Root models directory (registry.json, explainability bundle live here)
EXPERT_MODELS_DIR = next((d for d in _possible_model_dirs if os.path.isdir(d)), _possible_model_dirs[0])
# Directory holding the serialized expert models
EXPERT_DIR = os.path.join(EXPERT_MODELS_DIR, "expert_models")

_possible_contract_paths = [
    os.path.join(ROOT, "configs", "feature_contract.yaml"),
    os.path.join(ROOT, "webapp", "configs", "feature_contract.yaml"),
    os.path.join(PROJECT_ROOT, "webapp", "configs", "feature_contract.yaml"),
]
CONTRACT_PATH = next((p for p in _possible_contract_paths if os.path.isfile(p)), _possible_contract_paths[0])


# ── Custom unpickler (handles sklearn internal renames across versions) ───────
class _CustomUnpickler(pickle.Unpickler):
    """The expert models were pickled with scikit-learn 1.6 where the Cython loss
    lived in a top-level ``_loss`` module; newer releases moved it under
    ``sklearn._loss._loss``. Remap so they load on any recent scikit-learn."""

    def find_class(self, module, name):
        if module == "_loss" or module.startswith("sklearn._loss"):
            try:
                return super().find_class(module, name)
            except (ImportError, ModuleNotFoundError, AttributeError):
                for alt in ("sklearn._loss._loss", "sklearn._loss.loss", "sklearn._loss.link"):
                    try:
                        __import__(alt)
                        m = sys.modules[alt]
                        if hasattr(m, name):
                            return getattr(m, name)
                    except Exception:
                        continue
        return super().find_class(module, name)


def _safe_load(path):
    with open(path, "rb") as f:
        return _CustomUnpickler(f).load()


# ── RuntimeEngine ─────────────────────────────────────────────────────────────

class RuntimeEngine:
    """
    Usage
    -----
        engine = RuntimeEngine.from_registry()
        result = engine.predict_fused(face=arr18, voice=arr12, physio=arr51)
        result = engine.predict_face(arr18)

    Personal calibration
    --------------------
    If the user has completed the calibration flow (``backend.calibration``),
    each modality is re-centred: ``x' = x - user_calm_baseline + population_calm_mean``
    so the user's own resting state maps onto the population's resting state.
    Without a completed calibration the raw features are used unchanged.
    """

    def __init__(
        self,
        registry: VersionRegistry,
        feature_lock: FeatureRuntimeLock,
        expl_engine=None,          # ExplainabilityEngine (optional injection)
    ):
        self.registry     = registry
        self.feature_lock = feature_lock
        self.expl_engine  = expl_engine

        self._models  = {}   # "face" | "voice" | "physio" → sklearn estimator
        self._scalers = {}   # same keys → sklearn scaler or None
        self.load_errors: dict = {}
        self.calm_reference: dict = {}   # modality → population calm-state mean (np.ndarray)

        # Kept for API compatibility (modality status endpoint)
        self.feature_history = {m: [] for m in MODALITIES}
        self.calibrating = {m: True for m in MODALITIES}

        self._load_artifacts()

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_registry(
        cls,
        registry_path: str = None,
        feature_lock: FeatureRuntimeLock = None,
        expl_engine=None,
    ) -> "RuntimeEngine":
        """Production factory. Loads everything from VersionRegistry."""
        if registry_path is None:
            registry_path = os.path.join(EXPERT_MODELS_DIR, "registry.json")

        registry = VersionRegistry(registry_path=registry_path)

        if feature_lock is None:
            feature_lock = FeatureRuntimeLock(CONTRACT_PATH)

        if expl_engine is None:
            try:
                from backend.explainability.explainability_engine import ExplainabilityEngine
                expl_engine = ExplainabilityEngine()
            except Exception as exc:
                print(f"[RuntimeEngine] ExplainabilityEngine unavailable: {exc}")
                expl_engine = None

        return cls(registry=registry, feature_lock=feature_lock, expl_engine=expl_engine)

    # ── Artifact loading ──────────────────────────────────────────────────────

    def _load_artifacts(self):
        """Load model+scaler pairs for all registered experts."""
        registry_to_modality = {"face_expert": "face", "voice_expert": "voice", "physio_expert": "physio"}

        for reg_key, modality in registry_to_modality.items():
            model_file, scaler_file = _MODEL_FILES[reg_key]
            model_path  = os.path.join(EXPERT_DIR, model_file)
            scaler_path = os.path.join(EXPERT_DIR, scaler_file)

            reg_entry = self.registry.get_active_model(reg_key)
            if reg_entry and reg_entry.get("hash"):
                self._verify_hash(model_path, reg_entry["hash"], reg_key)

            if os.path.exists(model_path):
                try:
                    self._models[modality] = _safe_load(model_path)
                except Exception as exc:
                    self.load_errors[reg_key] = str(exc)
                    print(f"[RuntimeEngine] Failed to load {model_file}: {exc}")
            else:
                self.load_errors[reg_key] = f"File not found: {model_path}"

            if os.path.exists(scaler_path):
                try:
                    self._scalers[modality] = _safe_load(scaler_path)
                except Exception as exc:
                    print(f"[RuntimeEngine] Scaler load warning for {scaler_file}: {exc}")
                    self._scalers[modality] = None
            else:
                self._scalers[modality] = None

        # Population "calm" reference (used for personal-baseline re-centring)
        ref_path = os.path.join(EXPERT_DIR, "calm_reference.json")
        if os.path.exists(ref_path):
            try:
                with open(ref_path, "r") as f:
                    ref = json.load(f)
                self.calm_reference = {m: np.asarray(v, dtype=np.float64) for m, v in ref.items()}
            except Exception as exc:
                print(f"[RuntimeEngine] calm_reference.json warning: {exc}")

        loaded = list(self._models.keys())
        if loaded:
            print(f"[RuntimeEngine] Loaded modalities: {loaded}")
        else:
            print("[RuntimeEngine] WARNING: No models loaded.")

    def _verify_hash(self, path: str, expected: str, label: str):
        """Warn (never crash) if an artifact doesn't match its registry hash."""
        import hashlib
        if not os.path.exists(path):
            return
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha.update(block)
        actual = sha.hexdigest()
        if actual != expected:
            msg = f"Hash mismatch for {label}: expected {expected[:12]}..., got {actual[:12]}..."
            self.load_errors[f"{label}_hash"] = msg
            print(f"[RuntimeEngine] WARNING: {msg}")
        else:
            print(f"[RuntimeEngine] Hash verified for {label}: {actual[:16]}...")

    def reset_calibration(self):
        """Reset transient state (e.g. for a new subject)."""
        self.feature_history = {m: [] for m in MODALITIES}
        self.calibrating = {m: True for m in MODALITIES}

    # ── Public: status ────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return len(self._models) > 0

    def status(self) -> dict:
        """Return a serialisable status dict for /api/runtime/status."""
        model_versions = {}
        for reg_key in ("face_expert", "voice_expert", "physio_expert"):
            entry = self.registry.get_active_model(reg_key)
            modality = reg_key.replace("_expert", "")
            model_versions[reg_key] = {
                "loaded":   modality in self._models,
                "version":  entry.get("version") if entry else None,
                "hash":     (entry.get("hash") or "")[:12] + "..." if entry else None,
                "accuracy": (entry.get("metadata") or {}).get("accuracy"),
            }

        bundle_entry = self.registry.get_active_bundle("explainability_bundle")
        return {
            "engine":            "RuntimeEngine",
            "model":             "Multimodal Expert Fusion (GradientBoosting face + voice, calibrated GB/RF physio)",
            "ready":             self.is_ready,
            "loaded_modalities": list(self._models.keys()),
            "models":            model_versions,
            "load_errors":       self.load_errors,
            "decision_threshold": DECISION_THRESHOLD,
            "explainability": {
                "engine_loaded": self.expl_engine.is_loaded if self.expl_engine else False,
                "bundle_version": bundle_entry.get("version") if bundle_entry else None,
                "bundle_hash":   (bundle_entry.get("hash") or "")[:12] + "..." if bundle_entry else None,
            },
            "fusion_weights":    FUSION_WEIGHTS,
        }

    # ── Public: inference ─────────────────────────────────────────────────────

    def predict_fused(
        self,
        face=None,
        voice=None,
        physio=None,
        sensitivity: float = 0.5,
        user_id: str = 'default',
    ) -> dict:
        """
        Fuse the experts for whichever modalities are present.
        Fused score = Σ w_m · p_m / Σ w_m  over present modalities.
        """
        if face is None and voice is None and physio is None:
            return {"error": "All inputs are None", "status": "error"}

        probs = {}
        errors = {}
        for name, feats in (("face", face), ("voice", voice), ("physio", physio)):
            if feats is None:
                continue
            r = self._predict_single(name, feats, sensitivity, user_id)
            if "error" in r:
                errors[name] = r["error"]
            else:
                probs[name] = r["stress_probability"]

        if not probs:
            return {"error": "No valid predictions available: " + "; ".join(f"{k}: {v}" for k, v in errors.items()),
                    "status": "error"}

        raw_w = {m: FUSION_WEIGHTS[m] for m in probs}
        total_w = sum(raw_w.values())
        norm_weights = {m: round(w / total_w, 3) for m, w in raw_w.items()}
        prob = float(sum(probs[m] * raw_w[m] for m in probs) / total_w)
        threshold = self._threshold(sensitivity)

        result = {
            "status":                "success",
            "predicted_class":       "Stress" if prob > threshold else "No Stress",
            "stress_probability":    prob,
            "no_stress_probability": 1.0 - prob,
            "confidence":            float(max(prob, 1.0 - prob)),
            "stress_level":          self._level(prob),
            "percentage":            prob * 100.0,
            "threshold_applied":     threshold,
            "fusion_weights":        norm_weights,
            "individual_predictions": {m: float(p) for m, p in probs.items()},
            "fusion_engine":         "expert_fusion",
            "active_modalities":     list(probs.keys()),
        }
        if errors:
            result["modality_errors"] = errors
        return result

    def predict_face(self, raw_features, sensitivity: float = 0.5, user_id: str = 'default') -> dict:
        return self._predict_single("face", raw_features, sensitivity, user_id)

    def predict_voice(self, raw_features, sensitivity: float = 0.5, user_id: str = 'default') -> dict:
        return self._predict_single("voice", raw_features, sensitivity, user_id)

    def predict_physio(self, raw_features, sensitivity: float = 0.5, user_id: str = 'default') -> dict:
        return self._predict_single("physio", raw_features, sensitivity, user_id)

    @staticmethod
    def _level(prob: float) -> str:
        return "High" if prob > 0.7 else "Moderate" if prob > 0.4 else "Low"

    @staticmethod
    def _threshold(sensitivity: float) -> float:
        """Higher sensitivity → lower threshold. Equals DECISION_THRESHOLD at 0.5."""
        try:
            s = float(sensitivity)
        except (TypeError, ValueError):
            s = 0.5
        return float(np.clip(DECISION_THRESHOLD + (0.5 - s) * 0.4, 0.05, 0.95))

    def _predict_single(self, modality: str, raw_features, sensitivity: float, user_id: str = 'default') -> dict:
        if modality not in self._models:
            return {"error": f"{modality} model not loaded", "modality": modality}
        if raw_features is None:
            return {"error": "raw_features is None", "modality": modality}

        try:
            x = self._prepare(modality, raw_features, user_id)
            prob = float(self._models[modality].predict_proba(x)[0][1])
            threshold = self._threshold(sensitivity)
            return {
                "status":             "success",
                "modality":           modality,
                "stress_probability": prob,
                "stress_level":       self._level(prob),
                "predicted_class":    "Stress" if prob > threshold else "No Stress",
                "percentage":         float(prob * 100.0),
            }
        except Exception as exc:
            return {"error": str(exc), "modality": modality}

    # ── Private: feature preparation ──────────────────────────────────────────

    def _expected_dim(self, modality: str) -> int:
        scaler = self._scalers.get(modality)
        if scaler is not None and hasattr(scaler, "n_features_in_"):
            return int(scaler.n_features_in_)
        model = self._models.get(modality)
        return int(getattr(model, "n_features_in_", 0))

    def _personal_baseline(self, modality: str, user_id: str):
        """Median of the user's calm calibration window (or None)."""
        try:
            from backend.calibration import get_or_create
            cal = get_or_create(user_id)
        except Exception:
            return None
        if not getattr(cal, "is_complete", False):
            return None
        matrix = getattr(cal, f"_{modality}_baseline_matrix", None)
        if not matrix:
            return None
        return np.median(np.asarray(matrix, dtype=np.float64), axis=0)

    def _prepare(self, modality: str, raw_features, user_id: str = 'default') -> np.ndarray:
        """Validate dims, apply optional personal re-centring, then scale. Returns (1, d)."""
        feats = np.asarray(raw_features, dtype=np.float64)
        if feats.ndim > 1:
            feats = feats[-1] if feats.shape[0] > 1 else feats.reshape(-1)
        feats = np.nan_to_num(feats.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)

        dim = self._expected_dim(modality)
        if dim and feats.shape[0] != dim:
            raise ValueError(f"{modality} expects {dim} features, got {feats.shape[0]}")

        baseline  = self._personal_baseline(modality, user_id)
        calm_mean = self.calm_reference.get(modality)
        if baseline is not None and calm_mean is not None \
                and baseline.shape == feats.shape == calm_mean.shape:
            feats = feats - baseline + calm_mean

        feats = feats.reshape(1, -1)
        scaler = self._scalers.get(modality)
        if scaler is not None:
            feats = scaler.transform(feats)
        return feats

    # ── Public: replay (deterministic regression testing) ─────────────────────

    def replay(self, feature_rows: list) -> list:
        """
        Deterministic replay of a list of feature dicts.
        Each row:  {"face": ndarray|None, "voice": ndarray|None, "physio": ndarray|None}
        """
        self.reset_calibration()
        return [
            self.predict_fused(
                face=row.get("face"),
                voice=row.get("voice"),
                physio=row.get("physio"),
            )
            for row in feature_rows
        ]
