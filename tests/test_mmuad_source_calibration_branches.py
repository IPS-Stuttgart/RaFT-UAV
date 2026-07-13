from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir import ReservoirConfig, build_candidate_reservoir
from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.source_calibration import fit_source_calibration, write_source_calibration_json
from raft_uav.mmuad.source_calibration_branches import (
    build_source_calibration_branch_union,
    main as calibration_branches_main,
    source_calibration_branch_summary,
)


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 2.0, 4.0],
            "y_m": [10.0, 11.0, 12.0],
            "z_m": [3.0, 3.5, 4.0],
        }
    )


def _candidate_rows() -> pd.DataFrame:
    truth = _truth_rows()
    return pd.DataFrame(
        {
            "sequence_id": truth["sequence_id"],
            "time_s": truth["time_s"],
            "source": ["lidar_360"] * 3,
            "track_id": ["a", "b", "c"],
            "x_m": truth["x_m"] + 10.0,
            "y_m": truth["y_m"] - 4.0,
            "z_m": truth["z_m"] + 2.0,
            "confidence": [0.8, 0.8, 0.8],
        }
    )


def _calibration_payload() -> dict:
    payload, _pairs, _summary = fit_source_calibration(
        CandidateFrame(_candidate_rows()),
        _truth_rows(),
        mode="source-translation",
        max_truth_time_delta_s=0.1,
        max_pair_distance_m=50.0,
        min_pairs_per_source=2,
    )
    return payload


def test_branch_union_preserves_raw_and_calibrated_hypotheses() -> None:
    union = build_source_calibration_branch_union(
        CandidateFrame(_candidate_rows()),
        _calibration_payload(),
    ).rows

    assert len(union) == 6
    assert set(union["candidate_branch"]) == {"raw", "source_translation_calibrated"}
    raw = union.loc[union["candidate_branch"] == "raw"].sort_values("time_s")
    calibrated = union.loc[
        union["candidate_branch"] == "source_translation_calibrated"
    ].sort_values("time_s")
    assert raw["x_m"].tolist() == pytest.approx([10.0, 12.0, 14.0])
    assert calibrated["x_m"].tolist() == pytest.approx([0.0, 2.0, 4.0])
    assert calibrated["y_m"].tolist() == pytest.approx([10.0, 11.0, 12.0])
    assert calibrated["z_m"].tolist() == pytest.approx([3.0, 3.5, 4.0])
    assert set(raw["mmuad_calibration_displacement_m"]) == {0.0}
    assert calibrated["mmuad_calibration_displacement_m"].iloc[0] == pytest.approx(
        (10.0**2 + 4.0**2 + 2.0**2) ** 0.5
    )
    assert set(raw["track_id"]).isdisjoint(set(calibrated["track_id"]))
    assert set(calibrated["mmuad_original_track_id"]) == {"a", "b", "c"}


def test_branch_union_rejects_colliding_normalized_branch_labels() -> None:
    with pytest.raises(ValueError, match="must be distinct after normalization"):
        build_source_calibration_branch_union(
            CandidateFrame(_candidate_rows()),
            _calibration_payload(),
            raw_branch="shared/branch",
            calibrated_branch="shared_branch",
        )


def test_branch_union_feeds_branch_aware_reservoir() -> None:
    union = build_source_calibration_branch_union(
        CandidateFrame(_candidate_rows().iloc[[0]]),
        _calibration_payload(),
    ).rows
    reservoir = build_candidate_reservoir(
        union,
        config=ReservoirConfig(
            global_top_n=0,
            per_source_top_n=0,
            per_branch_top_n=1,
            max_candidates_per_frame=4,
            score_column="confidence",
        ),
    )

    assert len(reservoir) == 2
    assert set(reservoir["candidate_branch"]) == {
        "raw",
        "source_translation_calibrated",
    }
    assert reservoir["candidate_reservoir_reason"].str.contains("branch:").all()


def test_branch_union_drops_unapplied_duplicate_by_default() -> None:
    rows = _candidate_rows().iloc[[0]].copy()
    rows["source"] = "unknown_sensor"
    union = build_source_calibration_branch_union(
        CandidateFrame(rows),
        _calibration_payload(),
    ).rows

    assert len(union) == 1
    assert union.iloc[0]["candidate_branch"] == "raw"


def test_branch_union_summary_and_cli_write_outputs(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    calibration_json = tmp_path / "calibration.json"
    union_csv = tmp_path / "union.csv"
    reservoir_csv = tmp_path / "reservoir.csv"
    summary_json = tmp_path / "summary.json"
    _candidate_rows().to_csv(candidates_csv, index=False)
    write_source_calibration_json(_calibration_payload(), calibration_json)

    status = calibration_branches_main(
        [
            "--candidates",
            str(candidates_csv),
            "--output-candidates",
            str(union_csv),
            "--mmuad-source-calibration-json",
            str(calibration_json),
            "--summary-json",
            str(summary_json),
            "--reservoir-output-csv",
            str(reservoir_csv),
            "--reservoir-global-top-n",
            "0",
            "--reservoir-per-source-top-n",
            "0",
            "--reservoir-per-branch-top-n",
            "1",
        ]
    )

    assert status == 0
    union = pd.read_csv(union_csv)
    reservoir = pd.read_csv(reservoir_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert len(union) == 6
    assert len(reservoir) == 6
    assert summary["raw_branch_row_count"] == 3
    assert summary["calibrated_branch_row_count"] == 3
    assert summary["candidate_branch_counts"]["raw"] == 3
    assert summary["reservoir_row_count"] == 6
    direct_summary = source_calibration_branch_summary(union)
    assert direct_summary["distinct_origin_row_count"] == 3
