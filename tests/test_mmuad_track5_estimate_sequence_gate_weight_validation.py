from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_sequence_gate import (
    blend_track5_estimate_sequence_gate,
)


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    )


def _estimates(x_m: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [x_m],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
        }
    )


@pytest.mark.parametrize(
    "bad_weight",
    [
        True,
        np.bool_(False),
        np.array(True),
        np.array([0.5]),
        0.5 + 0.0j,
        np.ma.masked,
        pd.NA,
    ],
)
def test_estimate_sequence_gate_rejects_invalid_default_weights(
    bad_weight: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="default_weight must be a finite real scalar",
    ):
        blend_track5_estimate_sequence_gate(
            base_estimates=_estimates(0.0),
            alternate_estimates=_estimates(10.0),
            template=_template(),
            sequence_weights=pd.DataFrame(columns=["sequence_id", "weight"]),
            default_weight=bad_weight,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad_weight", [True, np.bool_(False), np.array([0.5])])
def test_estimate_sequence_gate_rejects_invalid_per_sequence_weights(
    bad_weight: object,
) -> None:
    weights = pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "weight": [bad_weight],
        }
    )

    with pytest.raises(
        ValueError,
        match="sequence_weight must be a finite real scalar",
    ):
        blend_track5_estimate_sequence_gate(
            base_estimates=_estimates(0.0),
            alternate_estimates=_estimates(10.0),
            template=_template(),
            sequence_weights=weights,
        )


def test_estimate_sequence_gate_accepts_scalar_like_real_weights() -> None:
    estimates, diagnostics, weights = blend_track5_estimate_sequence_gate(
        base_estimates=_estimates(0.0),
        alternate_estimates=_estimates(10.0),
        template=_template(),
        sequence_weights=pd.DataFrame(
            {
                "sequence_id": ["seq0001"],
                "weight": ["0.25"],
            }
        ),
        default_weight=np.array(0.5),
    )

    assert estimates["state_x_m"].tolist() == pytest.approx([2.5])
    assert diagnostics["sequence_gate_weight"].tolist() == pytest.approx([0.25])
    assert weights["sequence_gate_weight"].tolist() == pytest.approx([0.25])
