from __future__ import annotations

from collections.abc import Callable

import pytest

from raft_uav.mmuad.cli_types import nonnegative_finite_float
from raft_uav.mmuad.template_snap_cli import main as template_snap_main
from raft_uav.mmuad.track5_template_resample_cli import main as template_resample_main


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-0.1", "not-a-number"])
def test_nonnegative_finite_float_rejects_invalid_values(value: str) -> None:
    with pytest.raises(Exception) as exc_info:
        nonnegative_finite_float(value)

    assert exc_info.type.__name__ == "ArgumentTypeError"


@pytest.mark.parametrize(("value", "expected"), [("0", 0.0), ("1.25", 1.25)])
def test_nonnegative_finite_float_accepts_valid_values(value: str, expected: float) -> None:
    assert nonnegative_finite_float(value) == expected


@pytest.mark.parametrize(
    ("entrypoint", "base_args", "option"),
    [
        (
            template_resample_main,
            [
                "--estimates-csv",
                "missing-estimates.csv",
                "--template",
                "missing-template.csv",
                "--output-dir",
                "out",
            ],
            "--max-nearest-time-delta-s",
        ),
        (
            template_resample_main,
            [
                "--estimates-csv",
                "missing-estimates.csv",
                "--template",
                "missing-template.csv",
                "--output-dir",
                "out",
            ],
            "--max-interpolation-gap-s",
        ),
        (
            template_snap_main,
            [
                "--results",
                "missing-results.csv",
                "--template",
                "missing-template.csv",
                "--output-dir",
                "out",
            ],
            "--max-interpolation-gap-s",
        ),
    ],
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-1"])
def test_template_clis_reject_invalid_time_limits_before_io(
    entrypoint: Callable[[list[str] | None], int],
    base_args: list[str],
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        entrypoint([*base_args, option, value])

    assert exc_info.value.code == 2
