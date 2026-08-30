from __future__ import annotations

import numpy as np

from raft_uav.baselines.pyrecest_hypothesis_ranking import _records_from_item


def test_records_from_item_accepts_numpy_replay_arrays() -> None:
    records = _records_from_item(
        {
            "nis_values": np.array([1.0, 2.0], dtype=float),
            "residual_values": np.array([0.1, 0.2], dtype=float),
        }
    )

    assert len(records) == 2
    assert [float(record["nis"]) for record in records] == [1.0, 2.0]
    assert [float(record["residual_norm_m"]) for record in records] == [0.1, 0.2]
