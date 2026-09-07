"""Regression coverage for Track 5 speed-limit output safeguards."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad import track5_speed_limit as speed_limit


def _limited_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq-a"] * 4,
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "state_x_m": [0.0, 10.0, 20.0, 30.0],
            "state_y_m": [0.0] * 4,
            "state_z_m": [0.0] * 4,
            "Classification": [1] * 4,
        }
    )


def _diagnostics(flags: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "speed_limit_applied": flags,
            "speed_limit_correction_m": [0.0] * len(flags),
        }
    )


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


def test_legacy_implementation_uses_same_output_preflight(tmp_path: Path) -> None:
    output = tmp_path / "out"

    with pytest.raises(ValueError, match="require_leaderboard_ready.*requires.*template"):
        speed_limit._IMPL.write_track5_speed_limit_outputs(
            limited=_limited_rows(),
            diagnostics=_diagnostics([False, True, False, True]),
            output_dir=output,
            input_submission_path=tmp_path / "input.csv",
            require_leaderboard_ready=True,
        )

    assert not output.exists()


def test_persisted_string_flags_have_exact_manifest_counts(tmp_path: Path) -> None:
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
