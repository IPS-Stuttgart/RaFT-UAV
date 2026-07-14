from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from raft_uav.multi_uav_lts.coverage_audit import (
    _normalized_min_detection_fraction,
    main as coverage_audit_main,
)


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.bool_(False),
        np.array([0.5]),
        "not-a-fraction",
    ],
)
def test_min_detection_fraction_rejects_malformed_values(value: object) -> None:
    with pytest.raises(ValueError, match="finite number in \\[0, 1\\]"):
        _normalized_min_detection_fraction(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, 0.0),
        (1.0, 1.0),
        (np.float64(0.5), 0.5),
        (np.array(0.25), 0.25),
        ("0.75", 0.75),
    ],
)
def test_min_detection_fraction_accepts_finite_scalars(
    value: object,
    expected: float,
) -> None:
    assert _normalized_min_detection_fraction(value) == pytest.approx(expected)


def test_coverage_audit_cli_rejects_nan_threshold(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    prediction_dir.joinpath("A_00.txt").write_text(
        "1,1,10,20,5,6,0.9,1,1\n",
        encoding="utf-8",
    )
    sequence_root = tmp_path / "TestImages"
    sequence_dir = sequence_root / "A_00"
    sequence_dir.mkdir(parents=True)
    sequence_dir.joinpath("000001.jpg").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="finite number in \\[0, 1\\]"):
        coverage_audit_main(
            [
                str(prediction_dir),
                "--sequence-root",
                str(sequence_root),
                "--min-detection-frame-fraction",
                "nan",
            ]
        )
