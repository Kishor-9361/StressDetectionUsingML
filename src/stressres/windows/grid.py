"""
src/stressres/windows/grid.py

Window grid construction for StressID.
- 7 interactive tasks: 1 full-task window.
- 4 long tasks (Breathing, Video1, Video2, Relax): 60s window with 30s stride.
- Baseline task: 1 full-task window (60s).
"""

from typing import List, Dict, Any
import pandas as pd
import yaml

LONG_TASKS = {
    "Breathing": 180.0,
    "Video1": 120.0,
    "Video2": 180.0,
    "Relax": 150.0,
}

INTERACTIVE_TASKS = [
    "Counting1", "Counting2", "Counting3", "Math", "Reading", "Speaking", "Stroop"
]


def build_windows_for_recording(
    subject_id: str,
    task: str,
    duration_s: float,
    modality_coverage: str = "physio"
) -> List[Dict[str, Any]]:
    """
    Construct window specifications for a single subject recording.
    """
    windows = []
    win_len = 60.0
    win_stride = 30.0

    if task in LONG_TASKS:
        # Sub-window at 60s with 30s stride
        start_t = 0.0
        win_idx = 0
        while start_t + win_len <= duration_s + 1.0:  # Allow 1s tolerance
            end_t = min(start_t + win_len, duration_s)
            win_id = f"{subject_id}_{task}_w{win_idx}"
            windows.append({
                "subject_id": subject_id,
                "task_id": task,
                "window_id": win_id,
                "window_start_s": round(start_t, 2),
                "window_end_s": round(end_t, 2),
                "duration_s": round(end_t - start_t, 2),
                "is_subwindowed": True,
                "is_baseline": (task == "Breathing" or task == "Baseline" or task == "Relax"),
                "modality_coverage": modality_coverage
            })
            win_idx += 1
            start_t += win_stride
            if start_t >= duration_s - 10.0:  # Avoid ultra-short trailing windows
                break
    else:
        # Single window for full task
        win_id = f"{subject_id}_{task}_w0"
        windows.append({
            "subject_id": subject_id,
            "task_id": task,
            "window_id": win_id,
            "window_start_s": 0.0,
            "window_end_s": round(duration_s, 2),
            "duration_s": round(duration_s, 2),
            "is_subwindowed": False,
            "is_baseline": (task == "Baseline" or task == "Breathing" or task == "Relax"),
            "modality_coverage": modality_coverage
        })

    return windows
