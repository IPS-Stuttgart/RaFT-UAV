from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.evaluation.fifth_wave_diagnostics import block_bootstrap_interval
from raft_uav.evaluation.fifth_wave_diagnostics import paired_delta_summary


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("block_size", 0),
        ("block_size", 1.5),
        ("block_size", True),
        ("block_size", np.array([2])),
        ("resamples", 0),
        ("resamples", 2.5),
        ("resamples", np.nan),
        ("resamples", np.ma.masked),
        ("confidence", 0.0),
        ("confidence", 1.0),
        ("confidence", np.nan),
        ("confidence", np.array([0.95])),
    ],
)
def test_block_bootstrap_rejects_invalid_controls_before_empty_return(
    field: str,
    value: object,
) -> None:
    controls: dict[str, object] = {
        "block_size": 2,
        "resamples": 20,
        "confidence": 0.95,
    }
    controls[field] = value

    with pytest.raises(ValueError, match=field):
        block_bootstrap_interval([], **controls)


def test_block_bootstrap_rejects_unknown_metric_before_empty_return() -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        block_bootstrap_interval([], metric="not-a-metric")


@pytest.mark.parametrize("field", ["block_size", "resamples"])
def test_paired_delta_summary_validates_controls_before_empty_return(field: str) -> None:
    controls = {"block_size": 2, "resamples": 20}
    controls[field] = 0

    with pytest.raises(ValueError, match=field):
        paired_delta_summary(pd.DataFrame(), **controls)


def test_bootstrap_normalizes_valid_scalar_like_controls() -> None:
    interval = block_bootstrap_interval(
        np.arange(8.0),
        block_size=np.array(2),
        resamples="20",
        confidence=np.float64(0.9),
        seed=1,
    )

    assert interval.block_size == 2
    assert interval.resamples == 20
    assert interval.confidence == pytest.approx(0.9)
