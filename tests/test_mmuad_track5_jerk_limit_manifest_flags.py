from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_jerk_limit import write_track5_jerk_limit_outputs


def _repaired_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "state_x_m": [0.0, 1.0, 2.0, 3.0],
            "state_y_m": [0.0] * 4,
            "state_z_m": [1.0] * 4,
            "Classification": [2] * 4,
        }
    )


def _diagnostic_rows(flags: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * len(flags),
            "time_s": list(range(len(flags))),
            "jerk_limit_mps3": [0.0] * len(flags),
            "jerk_limit_applied": flags,
            "jerk_limit_displacement_m": [0.0, 0.0, 1.0, 2.0][: len(flags)],
        }
    )


def test_jerk_manifest_parses_csv_style_applied_flags(tmp_path: Path) -> None:
    paths = write_track5_jerk_limit_outputs(
        repaired=_repaired_rows(),
        diagnostics=_diagnostic_rows(["False", "0", "True", "1"]),
        output_dir=tmp_path / "out",
        input_submission_path=tmp_path / "input.csv",
    )

    manifest = json.loads(
        Path(paths["manifest_json"]).read_text(encoding="utf-8")
    )

    assert manifest["changed_row_count"] == 2
    assert manifest["changed_fraction"] == pytest.approx(0.5)


def test_jerk_manifest_rejects_ambiguous_applied_flags(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="jerk_limit_applied"):
        write_track5_jerk_limit_outputs(
            repaired=_repaired_rows(),
            diagnostics=_diagnostic_rows(["False", "maybe", "True", "1"]),
            output_dir=output_dir,
            input_submission_path=tmp_path / "input.csv",
        )

    assert not output_dir.exists()
