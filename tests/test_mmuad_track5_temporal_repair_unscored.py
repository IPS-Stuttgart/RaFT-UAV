from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.track5_temporal_repair_search import (
    search_track5_temporal_repair_parameters,
    write_temporal_repair_search_outputs,
)


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001"],
            "Timestamp": [0.0, 1.0, 2.0],
            "Position": ["(0, 0, 0)", "(100, 0, 0)", "(2, 0, 0)"],
            "Classification": [2, 2, 2],
        }
    )


def _unmatched_truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["different-sequence"] * 3,
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
        }
    )


def test_temporal_repair_search_rejects_unscored_grid() -> None:
    with pytest.raises(ValueError, match="no scored candidates"):
        search_track5_temporal_repair_parameters(
            _submission_rows(),
            _unmatched_truth_rows(),
            max_speed_grid=(50.0, 200.0),
            interpolation_residual_grid=(5.0,),
            iterations_grid=(1,),
        )


def test_temporal_repair_search_rejects_unscored_before_output_creation(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="no scored candidates"):
        write_temporal_repair_search_outputs(
            submission=_submission_rows(),
            truth=_unmatched_truth_rows(),
            output_dir=output_dir,
            input_submission_path=tmp_path / "input.csv",
            max_speed_grid=(50.0,),
            interpolation_residual_grid=(5.0,),
            iterations_grid=(1,),
        )

    assert not output_dir.exists()
