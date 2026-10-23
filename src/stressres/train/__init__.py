"""
src/stressres/train module
Shared training infrastructure for Phase 4 models.
"""

from .cv import get_subject_folds
from .threshold import sweep_thresholds
from .smoketest import run_smoke_test
from .artifacts import save_model_artifacts

__all__ = [
    "get_subject_folds",
    "sweep_thresholds",
    "run_smoke_test",
    "save_model_artifacts"
]
