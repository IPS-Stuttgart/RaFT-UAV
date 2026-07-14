from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.image_evidence import (
    _sample_nearest_image_rows,
    build_image_evidence,
)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        -1,
        1.5,
        np.nan,
        np.inf,
        -np.inf,
        pd.NA,
        np.array([1]),
    ],
)
def test_build_image_evidence_rejects_invalid_frame_limits(
    tmp_path: Path,
    value,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_frames_per_sequence must be a non-negative integer",
    ):
        build_image_evidence(
            tmp_path / "missing",
            max_frames_per_sequence=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        -0.1,
        np.nan,
        np.inf,
        -np.inf,
        pd.NA,
        np.array([0.5]),
    ],
)
def test_build_image_evidence_rejects_invalid_time_gates(
    tmp_path: Path,
    value,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_image_time_delta_s must be None or a finite non-negative number",
    ):
        build_image_evidence(
            tmp_path / "missing",
            max_image_time_delta_s=value,
        )


def test_direct_sampler_validates_controls_before_iteration() -> None:
    rows = _image_rows()

    with pytest.raises(ValueError, match="max_frames must be a non-negative integer"):
        _sample_nearest_image_rows(
            rows,
            [0.0],
            max_frames=-1,
            max_time_delta_s=0.1,
        )

    with pytest.raises(
        ValueError,
        match="max_time_delta_s must be None or a finite non-negative number",
    ):
        _sample_nearest_image_rows(
            rows,
            [0.0],
            max_frames=1,
            max_time_delta_s=np.nan,
        )


def test_direct_sampler_preserves_valid_integer_equivalents_and_zero_limit() -> None:
    rows = _image_rows()

    limited = list(
        _sample_nearest_image_rows(
            rows,
            [0.0, 1.0, 2.0],
            max_frames=np.array(2.0),
            max_time_delta_s=np.float64(0.0),
        )
    )
    unlimited = list(
        _sample_nearest_image_rows(
            rows,
            [0.0, 1.0, 2.0],
            max_frames=0,
            max_time_delta_s=None,
        )
    )

    assert [target_time for target_time, _row in limited] == [0.0, 2.0]
    assert [target_time for target_time, _row in unlimited] == [0.0, 1.0, 2.0]


def _image_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_path": ["0.png", "1.png", "2.png"],
            "image_time_s": [0.0, 1.0, 2.0],
        }
    )
