from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_consensus_ensemble import (
    build_track5_consensus_estimate_ensemble,
)


def _estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [2.0],
            "state_z_m": [3.0],
        }
    )


def _template(*, sequence: object = "seq0001", timestamp: object = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": pd.Series([sequence], dtype=object),
            "Timestamp": pd.Series([timestamp], dtype=object),
        }
    )


def test_consensus_ensemble_rejects_invalid_template_sequence() -> None:
    with pytest.raises(ValueError, match="invalid sequence identifier.*row 0"):
        build_track5_consensus_estimate_ensemble(
            [("estimate", _estimate(), 1.0)],
            _template(sequence=" "),
        )


@pytest.mark.parametrize("timestamp", ["not-a-time", float("nan"), True])
def test_consensus_ensemble_rejects_invalid_template_timestamp(
    timestamp: object,
) -> None:
    with pytest.raises(ValueError, match="invalid timestamp.*row 0"):
        build_track5_consensus_estimate_ensemble(
            [("estimate", _estimate(), 1.0)],
            _template(timestamp=timestamp),
        )


def test_consensus_ensemble_does_not_drop_malformed_template_rows() -> None:
    template = pd.concat(
        [
            _template(sequence="seq0001", timestamp=0.0),
            _template(sequence="", timestamp=1.0),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="invalid sequence identifier.*row 1"):
        build_track5_consensus_estimate_ensemble(
            [("estimate", _estimate(), 1.0)],
            template,
        )
