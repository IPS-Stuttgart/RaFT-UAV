from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import raft_uav.mmuad.candidate_reservoir as candidate_reservoir
from raft_uav.mmuad.candidate_reservoir import load_candidate_inputs, main


def _write_pose_csv(path: Path) -> None:
    path.write_text(
        "sequence_id,time_s,x_m,y_m,z_m,confidence\n"
        "001,0.0,1.0,2.0,3.0,0.9\n",
        encoding="utf-8",
    )


def test_candidate_reservoir_loader_preserves_zero_padded_sequence(
    tmp_path: Path,
) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    _write_pose_csv(candidate_csv)

    rows = load_candidate_inputs([f"raw={candidate_csv}"])

    assert rows.loc[0, "sequence_id"] == "001"
    assert rows.loc[0, "candidate_branch"] == "raw"


def test_candidate_reservoir_cli_oracle_preserves_zero_padded_sequence(
    tmp_path: Path,
) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_csv = tmp_path / "reservoir.csv"
    oracle_frame_csv = tmp_path / "oracle_frames.csv"
    _write_pose_csv(candidate_csv)
    _write_pose_csv(truth_csv)

    assert (
        main(
            [
                "--candidate",
                f"raw={candidate_csv}",
                "--output-csv",
                str(output_csv),
                "--truth-csv",
                str(truth_csv),
                "--oracle-frame-csv",
                str(oracle_frame_csv),
            ]
        )
        == 0
    )

    reservoir = pd.read_csv(output_csv, dtype=str, keep_default_na=False)
    oracle_frames = pd.read_csv(oracle_frame_csv, dtype=str, keep_default_na=False)
    assert reservoir.loc[0, "sequence_id"] == "001"
    assert oracle_frames.loc[0, "sequence_id"] == "001"


def test_candidate_reservoir_cli_scopes_text_reader_to_legacy_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "typed.csv"
    csv_path.write_text("sequence_id,value\n001,1\n", encoding="utf-8")
    original_read_csv = pd.read_csv
    original_impl_pd = candidate_reservoir._IMPL.pd
    observations: dict[str, object] = {}

    def fake_main(_argv: list[str] | None = None) -> int:
        observations["global_reader_is_original"] = pd.read_csv is original_read_csv
        observations["global_value"] = pd.read_csv(csv_path).loc[0, "value"]
        observations["legacy_sequence_id"] = candidate_reservoir._IMPL.pd.read_csv(
            csv_path
        ).loc[0, "sequence_id"]
        return 0

    monkeypatch.setattr(candidate_reservoir, "_ORIGINAL_MAIN", fake_main)

    assert candidate_reservoir.main([]) == 0
    assert observations["global_reader_is_original"] is True
    assert observations["global_value"] == 1
    assert observations["legacy_sequence_id"] == "001"
    assert pd.read_csv is original_read_csv
    assert candidate_reservoir._IMPL.pd is original_impl_pd
