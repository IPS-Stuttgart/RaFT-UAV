from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_temporal_consensus_train_cv import (
    apply_train_selected_temporal_consensus,
    load_train_selected_temporal_consensus_config,
    select_temporal_consensus_config_by_sequence_cv,
)


def _select(**kwargs: object) -> object:
    return select_temporal_consensus_config_by_sequence_cv(
        pd.DataFrame(),
        pd.DataFrame(),
        **kwargs,
    )


@pytest.mark.parametrize(
    "top_k_values",
    [
        (True,),
        (1.5,),
        (np.nan,),
        (np.array([1]),),
    ],
)
def test_train_cv_rejects_lossy_top_k_values(top_k_values: object) -> None:
    with pytest.raises(ValueError, match="top_k_values"):
        _select(top_k_values=top_k_values)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("base_score_weights", (0.25, np.nan)),
        ("support_weights", (True,)),
        ("bidirectional_bonuses", (np.array([0.5]),)),
        ("interpolation_weights", (1.0 + 0.0j,)),
        ("acceleration_weights", ()),
    ],
)
def test_train_cv_rejects_malformed_grid_axes(
    keyword: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=keyword):
        _select(**{keyword: value})


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("max_time_gap_s", True),
        ("max_speed_mps", np.nan),
        ("distance_scale_m", np.array([5.0])),
        ("acceleration_scale_mps2", 0.0),
        ("max_truth_time_delta_s", -0.1),
        ("source_diversity_bonus", np.inf),
    ],
)
def test_train_cv_rejects_malformed_scalar_controls(
    keyword: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=keyword):
        _select(**{keyword: value})


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("score_column", None),
        ("fallback_score_column", "   "),
        ("selection_metric", 17),
    ],
)
def test_train_cv_rejects_invalid_identifier_controls(
    keyword: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=keyword):
        _select(**{keyword: value})


@pytest.mark.parametrize("schema_version", [True, 1.0, 1.5, "1"])
def test_config_loader_rejects_lossy_schema_versions(
    tmp_path: Path,
    schema_version: object,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "temporal_consensus_config": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version"):
        load_train_selected_temporal_consensus_config(path)


def test_config_loader_validates_frozen_numeric_controls(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "temporal_consensus_config": {"max_speed_mps": True},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_speed_mps"):
        load_train_selected_temporal_consensus_config(path)


def test_config_loader_normalizes_valid_numeric_strings(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "temporal_consensus_config": {"max_speed_mps": "60"},
            }
        ),
        encoding="utf-8",
    )

    payload = load_train_selected_temporal_consensus_config(path)

    assert payload["temporal_consensus_config"]["max_speed_mps"] == 60.0


def test_direct_apply_rejects_invalid_frozen_identifiers() -> None:
    with pytest.raises(ValueError, match="score_column"):
        apply_train_selected_temporal_consensus(
            pd.DataFrame(),
            {"temporal_consensus_config": {"score_column": None}},
        )
