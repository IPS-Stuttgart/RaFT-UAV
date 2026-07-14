from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir_apply import (
    apply_train_selected_reservoir_config,
    load_train_selected_reservoir_config,
)


def _selected_config() -> dict:
    return {
        "schema_version": 1,
        "branch_score_offsets": {"raw": 1.0},
        "source_score_offsets": {},
        "score_column": "ranker_score",
        "fallback_score_column": "confidence",
        "global_top_n": 1,
        "per_source_top_n": 0,
        "per_branch_top_n": 0,
        "max_candidates_per_frame": 1,
        "score_floor_quantile": None,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 1.5),
        ("global_top_n", True),
        ("global_top_n", 1.5),
        ("per_source_top_n", -1),
        ("per_branch_top_n", float("nan")),
        ("max_candidates_per_frame", float("inf")),
    ],
)
def test_loader_rejects_lossy_integer_controls(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config = _selected_config()
    config[field] = value
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_train_selected_reservoir_config(path)


@pytest.mark.parametrize(
    ("mapping_name", "value"),
    [
        ("branch_score_offsets", float("nan")),
        ("branch_score_offsets", float("inf")),
        ("source_score_offsets", float("-inf")),
        ("source_score_offsets", True),
    ],
)
def test_loader_rejects_nonfinite_score_offsets(
    tmp_path: Path,
    mapping_name: str,
    value: object,
) -> None:
    config = _selected_config()
    config[mapping_name] = {"bad": value}
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=mapping_name):
        load_train_selected_reservoir_config(path)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_loader_rejects_invalid_score_floor_quantiles(
    tmp_path: Path,
    value: object,
) -> None:
    config = _selected_config()
    config["score_floor_quantile"] = value
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="score_floor_quantile"):
        load_train_selected_reservoir_config(path)


def test_loader_normalizes_integer_equivalent_controls(tmp_path: Path) -> None:
    config = _selected_config()
    for field in (
        "schema_version",
        "global_top_n",
        "per_source_top_n",
        "per_branch_top_n",
        "max_candidates_per_frame",
    ):
        config[field] = float(config[field])
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_train_selected_reservoir_config(path)

    assert all(
        isinstance(loaded[field], int)
        for field in (
            "schema_version",
            "global_top_n",
            "per_source_top_n",
            "per_branch_top_n",
            "max_candidates_per_frame",
        )
    )


def test_programmatic_apply_rejects_fractional_frozen_count() -> None:
    config = _selected_config()
    config["max_candidates_per_frame"] = 1.5

    with pytest.raises(ValueError, match="max_candidates_per_frame"):
        apply_train_selected_reservoir_config(pd.DataFrame(), config)
