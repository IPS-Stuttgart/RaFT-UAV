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


@pytest.mark.parametrize(
    ("extra_column", "extra_value"),
    [
        (" sequence ", "seq9999"),
        (" timestamp_s ", 10.0),
    ],
)
def test_consensus_ensemble_rejects_normalized_template_alias_collisions(
    extra_column: str,
    extra_value: object,
) -> None:
    template = pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            extra_column: [extra_value],
        }
    )

    with pytest.raises(ValueError, match="template contains ambiguous columns"):
        build_track5_consensus_estimate_ensemble(
            [("estimate", _estimate(), 1.0)],
            template,
        )


def test_consensus_ensemble_rejects_exact_duplicate_template_columns() -> None:
    template = pd.DataFrame(
        [["seq0001", "seq9999", 0.0]],
        columns=["Sequence", "Sequence", "Timestamp"],
    )

    with pytest.raises(ValueError, match="template contains ambiguous columns"):
        build_track5_consensus_estimate_ensemble(
            [("estimate", _estimate(), 1.0)],
            template,
        )
