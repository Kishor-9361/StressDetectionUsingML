"""
src/stressres/types.py

Core data contracts and types for StressRes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, FrozenSet
import numpy as np


@dataclass(frozen=True)
class Signal:
    """Continuous 1D physiological or biometric signal."""
    name: str
    data: np.ndarray                      # 1-D, float64
    fs: float                             # Hz
    t0: float = 0.0                       # seconds from session start
    unit: str = ""
    site: str = ""                        # 'chest' | 'wrist' | 'palm' | 'abdomen'

    @property
    def t(self) -> np.ndarray:            # sample times — never assume index==time
        return self.t0 + np.arange(len(self.data)) / self.fs


@dataclass(frozen=True)
class Block:
    """A labelled span of time. StressID: task (e.g. 'Breathing', 'Stroop')."""
    name: str                             # 'Baseline' | 'Breathing' | 'Stroop' | ...
    t_start: float
    t_end: float
    label_binary: Optional[int]
    label_source: str                     # 'protocol' | 'self_report'
    ratings: Dict[str, float] = field(default_factory=dict)             # StressID 0-10 scores


@dataclass(frozen=True)
class RawSession:
    """One subject's complete recording, as loaded from disk."""
    dataset: str                          # 'stressid' | 'wesad'
    subject_id: str
    signals: Dict[str, Signal]            # 'ecg', 'eda', 'resp'
    blocks: List[Block]                   # task segmentation
    source_files: List[Path] = field(default_factory=list)


@dataclass(frozen=True)
class WindowSpec:
    """A window specification before feature extraction."""
    dataset: str
    subject_id: str
    task: str
    modality: str
    window_id: str
    t_start: float
    t_end: float
    label_binary: int
    label_source: str
    ratings: Dict[str, float]
    is_baseline: bool                     # part of this subject's calm reference?
    admissible: FrozenSet[str]            # feature names valid at this length/state
