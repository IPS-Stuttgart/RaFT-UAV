"""Regression coverage for Track 5 speed-limit runtime safeguards."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from raft_uav.mmuad import track5_speed_limit as speed_limit


def _submission(
    sequence_ids: list[str],
    times: list[float],
    x_positions: list[float],
) -> pd.DataFrame:
    row_count = len(sequence_ids)
    return pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            "time_s": times,
            "state_x_m": x_positions,
            "state_y_m": [0.0] * row_count,
            "state_z_m": [0.0] * row_count,
            "Classification": [1] * row_count,
        }
    )


def _limited_rows() -> pd.DataFrame:
    return _submission(
        ["seq-a"] * 4,
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 10.0, 20.0, 30.0],
    )


def _diagnostics(flags: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "speed_limit_applied": flags,
            "speed_limit_correction_m": [0.0] * len(flags),
        }
    )


def test_speed_limit_rejects_duplicate_timestamps_within_sequence() -> None:
    submission = _submission(
        ["seq", "seq"],
        [1.0, 1.0],
        [0.0, 1_000.0],
    )

    with pytest.raises(ValueError, match="duplicate timestamps within a sequence"):
        speed_limit.project_track5_speed_limit(submission, max_speed_mps=1.0)


def test_speed_limit_allows_same_timestamp_in_different_sequences() -> None:
    submission = _submission(
        ["seq-a", "seq-b"],
        [1.0, 1.0],
        [0.0, 1_000.0],
    )

    limited, diagnostics = speed_limit.project_track5_speed_limit(
        submission,
        max_speed_mps=1.0,
    )

    assert limited["sequence_id"].tolist() == ["seq-a", "seq-b"]
    assert diagnostics["sequence_id"].tolist() == ["seq-a", "seq-b"]


def test_output_api_requires_template_before_any_write(tmp_path: Path) -> None:
    output = tmp_path / "out"

    with pytest.raises(ValueError, match="require_leaderboard_ready.*requires.*template"):
        speed_limit.write_track5_speed_limit_outputs(
            limited=_limited_rows(),
            diagnostics=_diagnostics([False, True, False, True]),
            output_dir=output,
            input_submission_path=tmp_path / "input.csv",
            require_leaderboard_ready=True,
        )

    assert not output.exists()


def test_persisted_string_flags_have_exact_manifest_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(speed_limit, "write_official_mmaud_results_csv", Mock())
    monkeypatch.setattr(speed_limit, "write_official_ug2_codabench_zip", Mock())
    diagnostics = _diagnostics(["False", "True", "0", "1"])
    diagnostics.index = [8, 3, 8, 1]
    before = diagnostics.copy(deep=True)

    paths = speed_limit.write_track5_speed_limit_outputs(
        limited=_limited_rows(),
        diagnostics=diagnostics,
        output_dir=tmp_path / "out",
        input_submission_path=tmp_path / "input.csv",
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["changed_row_count"] == 2
    assert manifest["changed_fraction"] == pytest.approx(0.5)
    persisted = pd.read_csv(paths["diagnostics_csv"])
    assert persisted["speed_limit_applied"].tolist() == [False, True, False, True]
    pd.testing.assert_frame_equal(diagnostics, before)


def test_invalid_persisted_flag_fails_before_output_creation(tmp_path: Path) -> None:
    output = tmp_path / "out"

    with pytest.raises(ValueError, match="speed_limit_applied contains invalid Boolean"):
        speed_limit.write_track5_speed_limit_outputs(
            limited=_limited_rows(),
            diagnostics=_diagnostics([False, "maybe", True, False]),
            output_dir=output,
            input_submission_path=tmp_path / "input.csv",
        )

    assert not output.exists()
