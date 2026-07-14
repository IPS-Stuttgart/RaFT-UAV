from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import raft_uav.mmuad.track5_segment_source_gate as segment_gate
from raft_uav.mmuad.track5_segment_source_gate import SegmentSourceGateConfig
from raft_uav.mmuad.track5_segment_source_gate import build_track5_segment_source_gate


def _template() -> pd.DataFrame:
    return pd.DataFrame({"Sequence": ["seqA"], "Timestamp": [0.0]})


def _estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("speed_limit_mps", 0.0),
        ("acceleration_limit_mps2", -1.0),
        ("invalid_penalty", 0.0),
        ("switch_penalty", -1.0),
        ("switch_jump_penalty_per_m", -0.1),
        ("weight_log_scale", -1.0),
    ],
)
def test_segment_source_gate_rejects_out_of_range_cost_controls(
    field: str,
    value: float,
) -> None:
    config = replace(SegmentSourceGateConfig(), **{field: value})

    with pytest.raises(ValueError, match=field):
        build_track5_segment_source_gate(
            [("primary", _estimates(), 1.0)],
            _template(),
            config=config,
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, np.array([1.0])])
def test_segment_source_gate_rejects_malformed_cost_controls(value: object) -> None:
    config = replace(SegmentSourceGateConfig(), switch_penalty=value)

    with pytest.raises(ValueError, match="switch_penalty"):
        build_track5_segment_source_gate(
            [("primary", _estimates(), 1.0)],
            _template(),
            config=config,
        )


def test_segment_source_gate_accepts_zero_dimensional_numeric_scalars() -> None:
    config = replace(
        SegmentSourceGateConfig(),
        speed_limit_mps=np.array(85.0),
        acceleration_limit_mps2=np.float64(45.0),
        switch_penalty=np.array(5.0),
    )

    estimates, diagnostics = build_track5_segment_source_gate(
        [("primary", _estimates(), 1.0)],
        _template(),
        config=config,
    )

    assert estimates["selected_source_label"].tolist() == ["primary"]
    assert diagnostics["valid_source_count"].tolist() == [1]


def test_segment_source_gate_cli_resolves_validated_writer() -> None:
    assert (
        segment_gate.main.__globals__["write_track5_segment_source_gate_outputs"]
        is segment_gate.write_track5_segment_source_gate_outputs
    )
