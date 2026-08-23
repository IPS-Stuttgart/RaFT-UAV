from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_acceleration_limit import (
    write_track5_acceleration_limit_outputs,
)


def _repaired_rows(row_count: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * row_count,
            "time_s": np.arange(row_count, dtype=float),
            "state_x_m": np.arange(row_count, dtype=float),
            "state_y_m": np.zeros(row_count, dtype=float),
            "state_z_m": np.ones(row_count, dtype=float),
            "Classification": [2] * row_count,
        }
    )


def _diagnostic_rows(flags: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * len(flags),
            "time_s": np.arange(len(flags), dtype=float),
            "acceleration_limit_applied": flags,
            "acceleration_limit_displacement_m": np.arange(len(flags), dtype=float),
        }
    )


def test_acceleration_limit_manifest_parses_csv_style_applied_flags(
    tmp_path: Path,
) -> None:
    paths = write_track5_acceleration_limit_outputs(
        repaired=_repaired_rows(),
        diagnostics=_diagnostic_rows(["False", "0", "True", "1"]),
        output_dir=tmp_path / "out",
        input_submission_path=tmp_path / "input.csv",
    )

    manifest = json.loads(Path(paths["manifest_json"]).read_text(encoding="utf-8"))
    persisted = pd.read_csv(paths["diagnostics_csv"])

    assert manifest["changed_row_count"] == 2
    assert manifest["changed_fraction"] == pytest.approx(0.5)
    assert persisted["acceleration_limit_applied"].tolist() == [
        False,
        False,
        True,
        True,
    ]


@pytest.mark.parametrize(
    "invalid_value",
    ["maybe", 2, -1.0, np.inf, 1 + 0j, [1]],
    ids=["text", "integer", "negative", "infinite", "complex", "non-scalar"],
)
def test_acceleration_limit_manifest_rejects_ambiguous_flags_before_writing(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    output_dir = tmp_path / "out"

    with pytest.raises(
        ValueError,
        match="acceleration_limit_applied contains invalid Boolean values",
    ):
        write_track5_acceleration_limit_outputs(
            repaired=_repaired_rows(1),
            diagnostics=_diagnostic_rows([invalid_value]),
            output_dir=output_dir,
            input_submission_path=tmp_path / "input.csv",
        )

    assert not output_dir.exists()
