from __future__ import annotations

import json
from pathlib import Path

import pytest

from raft_uav.mmuad.candidate_reservoir_apply import (
    load_train_selected_reservoir_config,
)


def _config(schema_version: object) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "score_column": "ranker_score",
        "fallback_score_column": "confidence",
        "global_top_n": 1,
        "per_source_top_n": 0,
        "per_branch_top_n": 0,
        "max_candidates_per_frame": 1,
    }


@pytest.mark.parametrize("schema_version", [True, 1.5])
def test_config_loader_rejects_lossy_schema_versions(
    tmp_path: Path,
    schema_version: object,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config(schema_version)), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version must be an exact integer"):
        load_train_selected_reservoir_config(path)


def test_config_loader_accepts_exact_integral_json_number(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config(1.0)), encoding="utf-8")

    loaded = load_train_selected_reservoir_config(path)

    assert loaded["schema_version"] == 1
