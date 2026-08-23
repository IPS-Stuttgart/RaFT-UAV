from __future__ import annotations

import numpy as np
import pytest

from raft_uav.diagnostics.paper_offset_sweep import _parse_grid


@pytest.mark.parametrize(
    "spec",
    [
        "0,1,inf",
        "0,1,nan",
        "-inf,1,1",
        "0,inf,1",
    ],
)
def test_parse_grid_rejects_nonfinite_values(spec: str) -> None:
    with pytest.raises(ValueError, match="finite"):
        _parse_grid(spec)


def test_parse_grid_preserves_finite_grid_behavior() -> None:
    np.testing.assert_allclose(
        _parse_grid("-0.1,0.1,0.1"),
        np.array([-0.1, 0.0, 0.1]),
    )


def test_parse_grid_does_not_step_past_stop() -> None:
    grid = _parse_grid("0,1,0.6")

    np.testing.assert_allclose(grid, np.array([0.0, 0.6]))
    assert np.all(grid <= 1.0)


def test_parse_grid_rejects_stop_before_start_even_with_large_step() -> None:
    with pytest.raises(ValueError, match="STOP must be >= START"):
        _parse_grid("1,0.9,1")


def test_parse_grid_keeps_aligned_stop_despite_floating_point_roundoff() -> None:
    np.testing.assert_allclose(
        _parse_grid("0,0.3,0.1"),
        np.array([0.0, 0.1, 0.2, 0.3]),
    )
