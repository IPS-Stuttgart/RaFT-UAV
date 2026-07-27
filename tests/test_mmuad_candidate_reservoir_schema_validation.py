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


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("schema_version", [True, 1.5])
def test_config_loader_rejects_lossy_schema_versions(
    tmp_path: Path,
    schema_version: object,
) -> None:
    path = _write_config(tmp_path, _config(schema_version))

    with pytest.raises(ValueError, match="schema_version must be an exact integer"):
        load_train_selected_reservoir_config(path)


def test_config_loader_accepts_exact_integral_json_number(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _config(1.0))

    loaded = load_train_selected_reservoir_config(path)

    assert loaded["schema_version"] == 1


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("global_top_n", True),
        ("per_source_top_n", 1.5),
        ("per_branch_top_n", -1),
        ("max_candidates_per_frame", float("nan")),
    ],
)
def test_config_loader_rejects_invalid_candidate_count_controls(
    tmp_path: Path,
    key: str,
    value: object,
) -> None:
    payload = _config(1)
    payload[key] = value
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match=key):
        load_train_selected_reservoir_config(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("score_column", None),
        ("fallback_score_column", "   "),
    ],
)
def test_config_loader_rejects_invalid_score_column_names(
    tmp_path: Path,
    key: str,
    value: object,
) -> None:
    payload = _config(1)
    payload[key] = value
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match=key):
        load_train_selected_reservoir_config(path)


@pytest.mark.parametrize("value", [True, -0.1, 1.1, float("inf")])
def test_config_loader_rejects_invalid_score_floor_quantiles(
    tmp_path: Path,
    value: object,
) -> None:
    payload = _config(1)
    payload["score_floor_quantile"] = value
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="score_floor_quantile"):
        load_train_selected_reservoir_config(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("branch_score_offsets", {"raw": True}),
        ("source_score_offsets", {"radar": float("inf")}),
    ],
)
def test_config_loader_rejects_nonfinite_or_boolean_score_offsets(
    tmp_path: Path,
    key: str,
    value: object,
) -> None:
    payload = _config(1)
    payload[key] = value
    path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match=key):
        load_train_selected_reservoir_config(path)


def test_config_loader_normalizes_valid_numeric_controls(tmp_path: Path) -> None:
    payload = _config(1)
    payload.update(
        {
            "global_top_n": 2.0,
            "per_source_top_n": "3",
            "per_branch_top_n": 4,
            "max_candidates_per_frame": "5.0",
            "score_floor_quantile": "0.25",
            "branch_score_offsets": {"raw": "-0.5"},
            "source_score_offsets": {"radar": 1},
        }
    )
    path = _write_config(tmp_path, payload)

    loaded = load_train_selected_reservoir_config(path)

    assert loaded["global_top_n"] == 2
    assert loaded["per_source_top_n"] == 3
    assert loaded["per_branch_top_n"] == 4
    assert loaded["max_candidates_per_frame"] == 5
    assert loaded["score_floor_quantile"] == pytest.approx(0.25)
    assert loaded["branch_score_offsets"] == {"raw": pytest.approx(-0.5)}
    assert loaded["source_score_offsets"] == {"radar": pytest.approx(1.0)}
