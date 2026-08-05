from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.source_calibration import SourceTransform


@pytest.mark.parametrize(
    ("linear", "translation_m", "error_message"),
    [
        (
            [[True, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0, 0.0, 0.0],
            "linear transform must not contain Boolean values",
        ),
        (
            np.eye(3),
            [0.0, np.array(np.bool_(True), dtype=object), 0.0],
            "translation_m must not contain Boolean values",
        ),
        (
            np.array(
                [[1.0 + 2.0j, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=complex,
            ),
            [0.0, 0.0, 0.0],
            "linear transform must not contain complex values",
        ),
        (
            np.eye(3),
            np.array([0.0, 1.0 + 2.0j, 0.0], dtype=complex),
            "translation_m must not contain complex values",
        ),
        (
            np.ma.array(np.eye(3), mask=np.eye(3, dtype=bool)),
            [0.0, 0.0, 0.0],
            "linear transform must not contain masked values",
        ),
    ],
)
def test_source_transform_rejects_lossy_coefficients(
    linear: object,
    translation_m: object,
    error_message: str,
) -> None:
    with pytest.raises(ValueError, match=error_message):
        SourceTransform(linear, translation_m)


def test_source_transform_preserves_valid_real_coefficients() -> None:
    transform = SourceTransform(
        np.eye(3, dtype=np.int64),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )

    assert transform.linear.dtype == float
    assert transform.translation_m.dtype == float
    np.testing.assert_allclose(transform.apply(np.array([4.0, 5.0, 6.0])), [5.0, 7.0, 9.0])
