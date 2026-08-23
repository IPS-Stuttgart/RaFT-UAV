from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_hampel_repair import write_track5_hampel_repair_outputs
from raft_uav.mmuad.track5_temporal_repair import write_track5_temporal_repair_outputs

Writer = Callable[..., dict[str, Path]]


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


@pytest.mark.parametrize(
    ("writer", "flag_column", "magnitude_column", "count_key", "fraction_key"),
    [
        (
            write_track5_hampel_repair_outputs,
            "hampel_repair_applied",
            "hampel_repair_correction_m",
            "changed_row_count",
            "changed_fraction",
        ),
        (
            write_track5_temporal_repair_outputs,
            "repaired",
            "repair_displacement_m",
            "repaired_row_count",
            "repaired_fraction",
        ),
    ],
    ids=["hampel", "temporal"],
)
def test_repair_manifest_parses_persisted_boolean_flags(
    tmp_path: Path,
    writer: Writer,
    flag_column: str,
    magnitude_column: str,
    count_key: str,
    fraction_key: str,
) -> None:
    diagnostics = pd.DataFrame(
        {
            flag_column: ["False", "0", "True", "1"],
            magnitude_column: [0.0, 0.0, 4.0, 3.0],
        }
    )

    paths = writer(
        repaired=_repaired_rows(4),
        diagnostics=diagnostics,
        output_dir=tmp_path / "out",
        input_submission_path=tmp_path / "input.csv",
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert manifest[count_key] == 2
    assert manifest[fraction_key] == pytest.approx(0.5)

    persisted = pd.read_csv(
        paths["diagnostics_csv"],
        dtype=str,
        keep_default_na=False,
    )
    assert persisted[flag_column].tolist() == ["False", "False", "True", "True"]


@pytest.mark.parametrize(
    ("writer", "flag_column", "magnitude_column"),
    [
        (
            write_track5_hampel_repair_outputs,
            "hampel_repair_applied",
            "hampel_repair_correction_m",
        ),
        (
            write_track5_temporal_repair_outputs,
            "repaired",
            "repair_displacement_m",
        ),
    ],
    ids=["hampel", "temporal"],
)
@pytest.mark.parametrize(
    "invalid_value",
    ["maybe", 2, -1, np.inf, 1 + 0j, np.array([1])],
    ids=["text", "integer", "negative", "infinity", "complex", "non-scalar"],
)
def test_repair_manifest_rejects_ambiguous_flags_before_writes(
    tmp_path: Path,
    writer: Writer,
    flag_column: str,
    magnitude_column: str,
    invalid_value: object,
) -> None:
    output_dir = tmp_path / "out"
    diagnostics = pd.DataFrame(
        {
            flag_column: [invalid_value],
            magnitude_column: [0.0],
        }
    )

    with pytest.raises(ValueError, match=f"{flag_column} contains invalid Boolean values"):
        writer(
            repaired=_repaired_rows(1),
            diagnostics=diagnostics,
            output_dir=output_dir,
            input_submission_path=tmp_path / "input.csv",
        )

    assert not output_dir.exists()
