from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.template_snap_write import write_template_snapped_submission


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001"],
            "Timestamp": [0.0, 10.0],
            "Position": ["(0,0,0)", "(10,20,2)"],
            "Classification": [2, 2],
        }
    )


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001"],
            "Timestamp": [5.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    )


def test_template_snap_writer_serializes_zero_dimensional_gap_threshold(
    tmp_path: Path,
) -> None:
    paths = write_template_snapped_submission(
        results=_results(),
        template=_template(),
        output_dir=tmp_path / "out",
        max_interpolation_gap_s=np.array(10.0),  # type: ignore[arg-type]
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))

    assert manifest["max_interpolation_gap_s"] == 10.0
    assert paths["official_results_csv"].exists()
    assert paths["official_zip"].exists()


@pytest.mark.parametrize("bad_gap", [True, np.array([10.0])])
def test_template_snap_writer_rejects_invalid_gap_before_creating_output(
    tmp_path: Path,
    bad_gap: object,
) -> None:
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="max_interpolation_gap_s"):
        write_template_snapped_submission(
            results=_results(),
            template=_template(),
            output_dir=output_dir,
            max_interpolation_gap_s=bad_gap,  # type: ignore[arg-type]
        )

    assert not output_dir.exists()
