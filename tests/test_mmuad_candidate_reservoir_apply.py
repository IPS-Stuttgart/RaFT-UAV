from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir_apply import (
    add_train_selected_reservoir_scores,
    apply_train_selected_reservoir_config,
    load_train_selected_reservoir_config,
    main as apply_main,
)


def _selected_config(*, max_candidates_per_frame: int = 1) -> dict:
    return {
        "schema_version": 1,
        "selection_protocol": "leave-one-sequence-out-cv-diagnostic__final-fit-on-all-train",
        "selection_metric": "oracle_top1_3d_m_mse",
        "selected_grid_label": "branch_raw_1",
        "selected_metric_value": 0.0,
        "branch_score_offsets": {"raw": 1.0},
        "source_score_offsets": {},
        "score_column": "ranker_score",
        "fallback_score_column": "confidence",
        "global_top_n": 1,
        "per_source_top_n": 0,
        "per_branch_top_n": 0,
        "max_candidates_per_frame": max_candidates_per_frame,
        "score_floor_quantile": None,
    }


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "raw-good",
                "candidate_branch": "raw",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "ranker_score": 0.1,
                "confidence": 0.1,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "livox_avia",
                "track_id": "translated-bad",
                "candidate_branch": "translated",
                "x_m": 20.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "ranker_score": 0.9,
                "confidence": 0.9,
            },
        ]
    )


def test_train_selected_offsets_change_target_reservoir_order() -> None:
    result = apply_train_selected_reservoir_config(
        _candidate_rows(),
        _selected_config(),
    )

    assert list(result.reservoir["track_id"]) == ["raw-good"]
    raw = result.adjusted_candidates.loc[
        result.adjusted_candidates["candidate_branch"] == "raw"
    ].iloc[0]
    assert float(raw["candidate_reservoir_train_base_score"]) == 0.1
    assert float(raw["candidate_reservoir_train_branch_offset"]) == 1.0
    assert float(raw["candidate_reservoir_train_adjusted_score"]) == 1.1
    assert result.summary["truth_free"] is True
    assert result.summary["selected_grid_label"] == "branch_raw_1"


def test_diversity_cap_preserves_low_score_branch() -> None:
    rows = pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "raw",
                "candidate_branch": "raw",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.1,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "livox_avia",
                "track_id": "translated-1",
                "candidate_branch": "translated",
                "x_m": 1.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.9,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "livox_avia",
                "track_id": "translated-2",
                "candidate_branch": "translated",
                "x_m": 2.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.8,
            },
        ]
    )
    config = _selected_config(max_candidates_per_frame=2)
    config["branch_score_offsets"] = {}
    config["global_top_n"] = 3
    config["per_branch_top_n"] = 1

    score_cap = apply_train_selected_reservoir_config(rows, config, cap_mode="score")
    diversity_cap = apply_train_selected_reservoir_config(
        rows,
        config,
        cap_mode="diversity",
        diversity_min_per_source=0,
        diversity_min_per_branch=1,
    )

    assert set(score_cap.reservoir["candidate_branch"]) == {"translated"}
    assert set(diversity_cap.reservoir["candidate_branch"]) == {"raw", "translated"}
    assert diversity_cap.summary["diversity_cap"]["output_rows"] == 2


def test_spatial_cap_keeps_geometrically_distinct_hypothesis() -> None:
    rows = pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "near-1",
                "candidate_branch": "raw",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "near-2",
                "candidate_branch": "raw",
                "x_m": 0.1,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.99,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "far",
                "candidate_branch": "raw",
                "x_m": 20.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.7,
            },
        ]
    )
    config = _selected_config(max_candidates_per_frame=2)
    config["branch_score_offsets"] = {}
    config["global_top_n"] = 3

    spatial = apply_train_selected_reservoir_config(
        rows,
        config,
        cap_mode="spatial",
        diversity_min_per_source=0,
        diversity_min_per_branch=0,
        spatial_diversity_weight=3.0,
        spatial_diversity_scale_m=2.0,
        spatial_distance_cap_m=50.0,
    )

    assert set(spatial.reservoir["track_id"]) == {"near-1", "far"}
    assert spatial.summary["spatial_cap"]["output_rows"] == 2


def test_config_loader_rejects_unsupported_schema(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported candidate reservoir config schema"):
        load_train_selected_reservoir_config(path)


def test_apply_cli_writes_truth_free_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "selected.json"
    candidates_path = tmp_path / "candidates.csv"
    output_dir = tmp_path / "out"
    config_path.write_text(json.dumps(_selected_config()), encoding="utf-8")
    _candidate_rows().to_csv(candidates_path, index=False)

    rc = apply_main(
        [
            "--config-json",
            str(config_path),
            "--candidate",
            f"mixed={candidates_path}",
            "--output-dir",
            str(output_dir),
            "--adjusted-candidates-csv",
            str(output_dir / "adjusted.csv"),
        ]
    )

    assert rc == 0
    reservoir = pd.read_csv(output_dir / "mmuad_candidate_reservoir_applied.csv")
    summary = json.loads(
        (output_dir / "mmuad_candidate_reservoir_apply_summary.json").read_text(
            encoding="utf-8"
        )
    )
    provenance = json.loads(
        (output_dir / "mmuad_candidate_reservoir_apply_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert list(reservoir["track_id"]) == ["raw-good"]
    assert summary["truth_free"] is True
    assert provenance["config_sha256"]
    assert provenance["selected_grid_label"] == "branch_raw_1"
    assert (output_dir / "adjusted.csv").exists()


def test_apply_entrypoint_is_registered() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert (
        pyproject["project"]["scripts"][
            "raft-uav-mmuad-apply-candidate-reservoir-config"
        ]
        == "raft_uav.mmuad.candidate_reservoir_apply:main"
    )


def test_score_helper_uses_fallback_when_ranker_score_missing() -> None:
    rows = _candidate_rows().drop(columns=["ranker_score"])

    adjusted = add_train_selected_reservoir_scores(rows, _selected_config())

    translated = adjusted.loc[adjusted["candidate_branch"] == "translated"].iloc[0]
    assert float(translated["candidate_reservoir_train_base_score"]) == 0.9
