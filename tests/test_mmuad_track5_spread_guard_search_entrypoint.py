from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_spread_guard_search import (
    _normalize_truth_for_exact_template,
    _score_template_estimates,
)


def test_mmuad_track5_spread_guard_search_entrypoint_is_exposed_and_importable() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    target = pyproject["project"]["scripts"]["raft-uav-mmuad-track5-spread-guard-search"]

    assert target == "raft_uav.mmuad.track5_spread_guard_search:main"

    module_name, function_name = target.split(":", 1)
    module = importlib.import_module(module_name)

    assert callable(getattr(module, function_name))


def test_spread_guard_search_scores_integer_truth_against_decimal_estimates() -> None:
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0, 1],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [0.0, 1.0],
            "state_x_m": [0.0, 1.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
        }
    )

    metrics = _score_template_estimates(
        estimates,
        _normalize_truth_for_exact_template(truth),
    )

    assert metrics["matched_rows"] == 2
    assert metrics["pose_mse_m2"] == pytest.approx(0.0)
