from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_geometric_median_ensemble import (
    build_track5_geometric_median_ensemble,
)


def test_geomedian_keeps_subnanosecond_template_rows_independent() -> None:
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

    result, diagnostics = build_track5_geometric_median_ensemble(
        [("estimate", estimates, 1.0)],
        template,
    )

    assert result["time_s"].tolist() == pytest.approx([0.0, 0.5e-9], abs=0.0)
    assert result["state_x_m"].tolist() == pytest.approx([0.0, 100.0])
    assert result["geomedian_source_count"].tolist() == [1, 1]
    assert diagnostics["candidate_input_count"].tolist() == [1, 1]
    assert diagnostics["valid_input_count"].tolist() == [1, 1]
