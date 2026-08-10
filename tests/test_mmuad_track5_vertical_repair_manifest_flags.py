from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_vertical_repair import write_track5_vertical_repair_outputs


def _repaired_rows(row_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * row_count,
            "time_s": [float(index) for index in range(row_count)],
            "state_x_m": [float(index) for index in range(row_count)],
            "state_y_m": [0.0] * row_count,
            "state_z_m": [10.0 + float(index) for index in range(row_count)],
            "Classification": [2] * row_count,
        }
    )


def test_vertical_repair_manifest_parses_persisted_boolean_flags(tmp_path: Path) -> None:
    diagnostics = pd.DataFrame(
        {
            "repaired": ["False", "0", "True", "1"],
            "vertical_repair_m": [0.0, 0.0, -4.0, 3.0],
        }
    )

    paths = write_track5_vertical_repair_outputs(
        repaired=_repaired_rows(4),
        diagnostics=diagnostics,
        output_dir=tmp_path / "out",
        input_submission_path=tmp_path / "input.csv",
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["repaired_row_count"] == 2
    assert manifest["repaired_fraction"] == pytest.approx(0.5)

    persisted = pd.read_csv(
        paths["diagnostics_csv"],
        dtype=str,
        keep_default_na=False,
    )
    assert persisted["repaired"].tolist() == ["False", "False", "True", "True"]


@pytest.mark.parametrize(
    "invalid_value",
    [
        "maybe",
        2,
        -1,
        np.inf,
        1 + 0j,
        np.array([1]),
    ],
    ids=["text", "integer", "negative", "infinity", "complex", "non-scalar"],
)
def test_vertical_repair_manifest_rejects_ambiguous_flags_before_writes(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    output_dir = tmp_path / "out"
    diagnostics = pd.DataFrame(
        {
            "repaired": [invalid_value],
            "vertical_repair_m": [0.0],
        }
    )

    with pytest.raises(ValueError, match="repaired contains invalid Boolean values"):
        write_track5_vertical_repair_outputs(
            repaired=_repaired_rows(1),
            diagnostics=diagnostics,
            output_dir=output_dir,
            input_submission_path=tmp_path / "input.csv",
        )

    assert not output_dir.exists()
