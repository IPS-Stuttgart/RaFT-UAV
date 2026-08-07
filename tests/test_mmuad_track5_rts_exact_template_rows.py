from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_rts_ensemble import build_track5_rts_ensemble


def test_rts_keeps_subnanosecond_template_measurements_independent() -> None:
    template = pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001"],
            "Timestamp": [0.0, 0.5e-9],
        }
    )
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 0.5e-9],
            "state_x_m": [0.0, 100.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
        }
    )

    _, diagnostics = build_track5_rts_ensemble(
        [("estimate", estimates, 1.0)],
        template,
        max_nearest_time_delta_s=0.0,
    )

    assert diagnostics["time_s"].tolist() == pytest.approx([0.0, 0.5e-9], abs=0.0)
    assert diagnostics["valid_input_count"].tolist() == [1, 1]
    assert diagnostics["weighted_x_m"].tolist() == pytest.approx([0.0, 100.0])
    assert diagnostics["input_labels"].tolist() == ["estimate", "estimate"]
