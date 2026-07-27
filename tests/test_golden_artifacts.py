from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.evaluation.golden_artifacts import _check_csv, check_run_artifacts, main


_ERROR_PATTERN = r"max_nan_fraction must be a finite real scalar in \[0, 1\]"


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


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        1.01,
        np.nan,
        np.inf,
        True,
        0.5 + 0.0j,
        np.array([0.5]),
        np.ma.masked,
        np.array(np.complex64(0.5 + 0.0j), dtype=object),
    ],
)
def test_check_run_artifacts_rejects_invalid_max_nan_fraction(tmp_path, value) -> None:
    with pytest.raises(ValueError, match=_ERROR_PATTERN):
        check_run_artifacts(tmp_path, max_nan_fraction=value)


def test_check_csv_validates_max_nan_fraction_before_reading(tmp_path) -> None:
    with pytest.raises(ValueError, match=_ERROR_PATTERN):
        _check_csv(tmp_path / "missing.csv", max_nan_fraction=-0.01)


@pytest.mark.parametrize(
    "value",
    [0.0, 1.0, np.float64(0.5), np.array(0.5), np.ma.array(0.5, mask=False)],
)
def test_check_csv_accepts_valid_max_nan_fraction_scalars(tmp_path, value) -> None:
    path = tmp_path / "diagnostics.csv"
    pd.DataFrame({"time_s": [0.0, 1.0], "nis": [1.5, 2.5]}).to_csv(path, index=False)

    results = {row["check"]: row for row in _check_csv(path, max_nan_fraction=value)}

    assert results["numeric_nan_fraction"]["passed"] is True
    assert results["numeric_nonfinite_fraction"]["passed"] is True


def test_main_rejects_invalid_max_nan_fraction(tmp_path, capsys) -> None:
    output_path = tmp_path / "checks.json"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                str(tmp_path),
                "--output-json",
                str(output_path),
                "--max-nan-fraction",
                "nan",
            ]
        )

    assert exc_info.value.code == 2
    assert _ERROR_PATTERN.replace("\\", "") in capsys.readouterr().err
    assert not output_path.exists()
