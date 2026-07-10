from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_consensus_ensemble import (
    build_track5_consensus_estimate_ensemble,
)


def test_consensus_ensemble_accepts_padded_template_headers() -> None:
    template = pd.DataFrame(
        {
            " Sequence ": ["seq0001"],
            " Timestamp ": [0.0],
        }
    )
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
        }
    )

    result, diagnostics = build_track5_consensus_estimate_ensemble(
        [("estimate", estimates, 1.0)],
        template,
    )

    assert result["sequence_id"].tolist() == ["seq0001"]
    assert result.loc[0, "state_x_m"] == pytest.approx(1.0)
    assert result.loc[0, "state_y_m"] == pytest.approx(2.0)
    assert result.loc[0, "state_z_m"] == pytest.approx(3.0)
    assert diagnostics.loc[0, "valid_input_count"] == 1
