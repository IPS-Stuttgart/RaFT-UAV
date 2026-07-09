from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.track5_segment_source_gate import SegmentSourceGateConfig
from raft_uav.mmuad.track5_segment_source_gate import build_track5_segment_source_gate


def test_segment_source_gate_accepts_padded_template_headers_and_opaque_sequence_ids() -> None:
    template = pd.DataFrame(
        {
            " Sequence ": ["001", "001"],
            " Timestamp ": [0.0, 1.0],
            " Position ": ["(0,0,0)", "(1,0,1)"],
            " Classification ": [2, 2],
        }
    )
    estimates = pd.DataFrame(
        {
            "sequence_id": ["001", "001"],
            "time_s": [0.0, 1.0],
            "state_x_m": [0.0, 1.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [1.0, 1.0],
        }
    )

    gated, diagnostics = build_track5_segment_source_gate(
        [("primary", estimates, 1.0)],
        template,
        config=SegmentSourceGateConfig(
            speed_limit_mps=20.0,
            acceleration_limit_mps2=20.0,
        ),
    )

    assert gated["sequence_id"].tolist() == ["001", "001"]
    assert gated["time_s"].tolist() == [0.0, 1.0]
    assert gated["selected_source_label"].tolist() == ["primary", "primary"]
    assert diagnostics["valid_source_count"].tolist() == [1, 1]
