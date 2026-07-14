from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.source_calibration import (
    fit_source_calibration,
    write_source_calibration_json,
)
from raft_uav.mmuad.source_calibration_path_ensemble import (
    CALIBRATION_FRACTION_COLUMN,
    EFFECTIVE_ALPHA_COLUMN,
    build_source_calibration_path_ensemble,
    main as calibration_path_main,
    source_calibration_path_summary,
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


def test_path_ensemble_interpolates_raw_to_calibrated_coordinates() -> None:
    ensemble = build_source_calibration_path_ensemble(
        CandidateFrame(_candidate_rows().iloc[[0]]),
        _calibration_payload(),
        fractions=(0.0, 0.25, 0.5, 1.0),
    ).rows.sort_values(CALIBRATION_FRACTION_COLUMN)

    assert len(ensemble) == 4
    assert ensemble[CALIBRATION_FRACTION_COLUMN].tolist() == pytest.approx(
        [0.0, 0.25, 0.5, 1.0]
    )
    assert ensemble["x_m"].tolist() == pytest.approx([10.0, 7.5, 5.0, 0.0])
    assert ensemble["y_m"].tolist() == pytest.approx([6.0, 7.0, 8.0, 10.0])
    assert ensemble["z_m"].tolist() == pytest.approx([5.0, 4.5, 4.0, 3.0])
    assert ensemble["mmuad_calibration_displacement_m"].is_monotonic_increasing
    assert ensemble["mmuad_calibration_origin_row"].nunique() == 1
    assert ensemble["track_id"].nunique() == 4
    assert ensemble["candidate_branch"].nunique() == 4


def test_path_ensemble_scales_effective_translation_alpha() -> None:
    ensemble = build_source_calibration_path_ensemble(
        CandidateFrame(_candidate_rows().iloc[[0]]),
        _calibration_payload(),
        fractions=(0.0, 0.5, 1.0),
    ).rows.sort_values(CALIBRATION_FRACTION_COLUMN)

    fitted_alpha = float(
        ensemble.loc[
            ensemble[CALIBRATION_FRACTION_COLUMN] == 1.0,
            "mmuad_source_calibration_alpha",
        ].iloc[0]
    )
    assert ensemble[EFFECTIVE_ALPHA_COLUMN].tolist() == pytest.approx(
        [0.0, 0.5 * fitted_alpha, fitted_alpha]
    )
    assert ensemble["mmuad_calibration_path_interpolated"].tolist() == [False, True, False]


def test_path_ensemble_drops_unapplied_intermediate_branches_by_default() -> None:
    rows = _candidate_rows().iloc[[0]].copy()
    rows["source"] = "unknown_sensor"

    ensemble = build_source_calibration_path_ensemble(
        CandidateFrame(rows),
        _calibration_payload(),
        fractions=(0.0, 0.5, 1.0),
    ).rows

    assert len(ensemble) == 1
    assert ensemble.iloc[0][CALIBRATION_FRACTION_COLUMN] == 0.0
    assert ensemble.iloc[0]["candidate_branch"] == "raw"


@pytest.mark.parametrize("fractions", [(), (-0.1, 1.0), (0.0, 1.1), (0.0, float("nan"))])
def test_path_ensemble_rejects_invalid_fraction_grids(fractions: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="calibration fraction"):
        build_source_calibration_path_ensemble(
            CandidateFrame(_candidate_rows().iloc[[0]]),
            _calibration_payload(),
            fractions=fractions,
        )


def test_path_summary_and_cli_write_reservoir_outputs(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    calibration_json = tmp_path / "calibration.json"
    ensemble_csv = tmp_path / "ensemble.csv"
    reservoir_csv = tmp_path / "reservoir.csv"
    summary_json = tmp_path / "summary.json"
    _candidate_rows().to_csv(candidates_csv, index=False)
    write_source_calibration_json(_calibration_payload(), calibration_json)

    status = calibration_path_main(
        [
            "--candidates",
            str(candidates_csv),
            "--output-candidates",
            str(ensemble_csv),
            "--mmuad-source-calibration-json",
            str(calibration_json),
            "--calibration-fractions",
            "0,0.25,0.5,1",
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
    ensemble = pd.read_csv(ensemble_csv)
    reservoir = pd.read_csv(reservoir_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert len(ensemble) == 12
    assert len(reservoir) == 12
    assert summary["calibration_path_fractions"] == [0.0, 0.25, 0.5, 1.0]
    assert summary["interpolated_branch_row_count"] == 6
    assert summary["interpolated_branch_count"] == 2
    assert summary["truth_used_for_calibration_path_ensemble"] is False
    assert summary["reservoir_row_count"] == 12
    direct = source_calibration_path_summary(ensemble)
    assert direct["distinct_origin_row_count"] == 3
