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
