import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_summary_module():
    script = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "scripts"
        / "summarize_stateful_sweep_run.py"
    )
    spec = importlib.util.spec_from_file_location("summarize_stateful_sweep_run", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_target_mot_summary_counts_duplicate_estimates_as_false_positives():
    summary_module = _load_summary_module()
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )
    estimates = pd.DataFrame(
        {
            "time_s": [0.0, 0.01],
            "east_m": [0.5, 0.1],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "track_id": [1, 2],
        }
    )

    summary = summary_module.single_target_mot_summary(
        estimates=estimates,
        truth=truth,
        max_time_delta_s=1.0,
        distance_threshold_m=1.0,
        dimensions=3,
    )

    assert summary["tp"] == 1
    assert summary["fp"] == 1
    assert summary["fn"] == 0
    assert summary["mota"] == 0.0
    assert summary["idtp"] == 1
    assert summary["idfp"] == 1
    assert summary["idfn"] == 0
    assert summary["idf1"] == pytest.approx(2.0 / 3.0)
    assert summary["dominant_track_id"] == 2
    assert summary["dominant_track_matches"] == 1
