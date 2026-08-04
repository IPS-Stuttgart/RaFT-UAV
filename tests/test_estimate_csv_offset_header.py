from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import raft_uav.mmuad.estimate_csv  # noqa: F401  # installs guarded pandas reader


def _grid_read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    namespace = {
        "__name__": "raft_uav.mmuad.track5_estimate_ensemble_grid",
        "pd": pd,
    }
    exec(
        "def read(path, **kwargs):\n"
        "    return pd.read_csv(path, **kwargs)\n",
        namespace,
    )
    return namespace["read"](path, **kwargs)


def test_guard_rejects_duplicate_columns_in_nonzero_header_row(tmp_path: Path) -> None:
    path = tmp_path / "estimates.csv"
    path.write_text(
        "metadata,metadata,metadata\n"
        "sequence_id,state_x_m,state_x_m\n"
        "001,1.0,2.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ambiguous"):
        _grid_read_csv(path, header=1)


def test_guard_reads_unique_nonzero_header_row(tmp_path: Path) -> None:
    path = tmp_path / "estimates.csv"
    path.write_text(
        "metadata,metadata,metadata\n"
        "sequence_id,state_x_m,state_y_m\n"
        "001,1.0,2.0\n",
        encoding="utf-8",
    )

    rows = _grid_read_csv(path, header=1)

    assert rows.columns.tolist() == ["sequence_id", "state_x_m", "state_y_m"]
    assert rows.loc[0, "sequence_id"] == "001"
