from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_temporal_repair import (
    _boolean_series,
    write_track5_temporal_repair_outputs,
)


def test_temporal_repair_manifest_parses_serialized_repaired_flags(
    tmp_path: Path,
) -> None:
    repaired = pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 6,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "state_x_m": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "state_y_m": [0.0] * 6,
            "state_z_m": [0.0] * 6,
            "Classification": [2] * 6,
        }
    )
    diagnostics = pd.DataFrame(
        {
            "repaired": ["False", "0", "no", "True", "1", "yes"],
            "repair_displacement_m": [0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        }
    )

    paths = write_track5_temporal_repair_outputs(
        repaired=repaired,
        diagnostics=diagnostics,
        output_dir=tmp_path / "out",
        input_submission_path=tmp_path / "input.csv",
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    written_diagnostics = pd.read_csv(paths["diagnostics_csv"])

    assert manifest["repaired_row_count"] == 3
    assert manifest["repaired_fraction"] == 0.5
    assert written_diagnostics["repaired"].tolist() == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_temporal_repair_flags_accept_exact_numeric_values_and_missing() -> None:
    values = pd.Series([0, 1, 0.0, 1.0, np.nan], dtype=object)

    parsed = _boolean_series(values, values.index)

    assert parsed.tolist() == [False, True, False, True, False]


@pytest.mark.parametrize(
    "invalid_value",
    [2, -1, 0.5, np.inf, -np.inf, 1 + 0j, "2", "maybe", [1]],
)
def test_temporal_repair_flags_reject_malformed_values(invalid_value: object) -> None:
    values = pd.Series([invalid_value], dtype=object)

    with pytest.raises(ValueError, match="exact numeric 0/1"):
        _boolean_series(values, values.index)


def test_temporal_repair_outputs_reject_malformed_repaired_flags(
    tmp_path: Path,
) -> None:
    repaired = pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [0.0],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
            "Classification": [2],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "repaired": [2],
            "repair_displacement_m": [0.0],
        }
    )

    with pytest.raises(ValueError, match="invalid Boolean values"):
        write_track5_temporal_repair_outputs(
            repaired=repaired,
            diagnostics=diagnostics,
            output_dir=tmp_path / "out",
            input_submission_path=tmp_path / "input.csv",
        )
