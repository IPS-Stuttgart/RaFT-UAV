from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.evaluation.golden_artifacts import _check_csv


def test_check_csv_rejects_infinite_numeric_values(tmp_path) -> None:
    path = tmp_path / "estimates.csv"
    pd.DataFrame({"time_s": [0.0, 1.0, 2.0], "east_m": [1.0, np.inf, -np.inf]}).to_csv(
        path,
        index=False,
    )

    results = {row["check"]: row for row in _check_csv(path, max_nan_fraction=0.0)}

    assert results["numeric_nan_fraction"]["passed"] is True
    assert results["numeric_nan_fraction"]["value"] == 0.0
    assert results["numeric_nonfinite_fraction"]["passed"] is False
    assert results["numeric_nonfinite_fraction"]["value"] == 2.0 / 6.0


def test_check_csv_accepts_finite_numeric_values(tmp_path) -> None:
    path = tmp_path / "diagnostics.csv"
    pd.DataFrame({"time_s": [0.0, 1.0], "nis": [1.5, 2.5]}).to_csv(path, index=False)

    results = {row["check"]: row for row in _check_csv(path, max_nan_fraction=0.0)}

    assert results["numeric_nonfinite_fraction"]["passed"] is True
    assert results["numeric_nonfinite_fraction"]["value"] == 0.0
