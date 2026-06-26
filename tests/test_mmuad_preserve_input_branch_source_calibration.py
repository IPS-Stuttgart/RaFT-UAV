from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.source_calibration import fit_source_calibration, write_source_calibration_json


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mmuad_preserve_input_branch_source_calibration.py"
spec = importlib.util.spec_from_file_location("mmuad_preserve_input_branch_source_calibration", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "x_m": [0.0, 1.0, 2.0, 3.0],
            "y_m": [10.0, 10.0, 10.0, 10.0],
            "z_m": [2.0, 2.0, 2.0, 2.0],
        }
    )


def _candidates() -> pd.DataFrame:
    truth = _truth()
    return pd.DataFrame(
        {
            "sequence_id": truth["sequence_id"],
            "time_s": truth["time_s"],
            "source": ["lidar_360", "lidar_360", "lidar_360", "lidar_360"],
            "track_id": ["a", "b", "c", "d"],
            "candidate_branch": ["static", "dynamic", "static", "dynamic"],
            "x_m": truth["x_m"] + 5.0,
            "y_m": truth["y_m"] - 1.0,
            "z_m": truth["z_m"] + 2.0,
            "confidence": [0.1, 0.9, 0.2, 0.8],
        }
    )


def _calibration_payload() -> dict:
    payload, _pairs, _summary = fit_source_calibration(
        CandidateFrame(_candidates()),
        _truth(),
        mode="source-translation",
        max_truth_time_delta_s=0.1,
        max_pair_distance_m=20.0,
        min_pairs_per_source=2,
    )
    return payload


def test_preserved_branch_union_keeps_static_dynamic_and_raw_calibrated_labels(tmp_path: Path) -> None:
    calibration_json = tmp_path / "calibration.json"
    write_source_calibration_json(_calibration_payload(), calibration_json)

    union = module.build_preserved_branch_union(
        _candidates(),
        source_calibration_json=calibration_json,
        mode="source-translation",
    )

    assert len(union) == 8
    assert set(union["candidate_branch"]) == {
        "static:raw",
        "dynamic:raw",
        "static:source_translation_calibrated",
        "dynamic:source_translation_calibrated",
    }
    assert set(union["mmuad_input_candidate_branch"]) == {"static", "dynamic"}
    assert union["track_id"].astype(str).str.contains("@").all()
    calibrated = union.loc[union["mmuad_candidate_branch_is_calibrated"]]
    assert calibrated["mmuad_calibration_displacement_m"].min() > 0


def test_preserved_branch_cli_writes_reservoir_and_summary(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    calibration_json = tmp_path / "calibration.json"
    union_csv = tmp_path / "union.csv"
    reservoir_csv = tmp_path / "reservoir.csv"
    summary_json = tmp_path / "summary.json"
    _candidates().to_csv(candidates_csv, index=False)
    write_source_calibration_json(_calibration_payload(), calibration_json)

    status = module.main(
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
    assert len(union) == 8
    assert len(reservoir) == 8
    assert summary["candidate_branch_counts"]["static:raw"] == 2
    assert summary["candidate_branch_counts"]["dynamic:source_translation_calibrated"] == 2
    assert summary["reservoir_row_count"] == 8
