from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_build_branch_reservoir.py"
spec = importlib.util.spec_from_file_location("mmuad_build_branch_reservoir", MODULE_PATH)
assert spec is not None and spec.loader is not None
reservoir_builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reservoir_builder
spec.loader.exec_module(reservoir_builder)


def test_truth_free_reservoir_does_not_promote_nonfinite_scores() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq001"],
            "time_s": [0.0, 0.0, 0.0],
            "source": ["radar", "radar", "radar"],
            "track_id": ["finite_best", "finite_worst", "corrupt_inf"],
            "candidate_branch": ["raw", "raw", "raw"],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "ranker_score": [0.9, 0.2, np.inf],
            "confidence": [0.9, 0.2, 0.1],
        }
    )

    reservoir = reservoir_builder.build_truth_free_branch_reservoir(
        candidates,
        per_source_top_n=0,
        per_branch_top_n=0,
        global_top_n=1,
    )

    assert list(reservoir["track_id"]) == ["finite_best"]
