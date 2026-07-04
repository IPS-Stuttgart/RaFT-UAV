from __future__ import annotations

import numpy as np

from raft_uav.mmuad.splits import _manifest_from_payload


def test_split_manifest_accepts_numpy_scalar_sequence_ids() -> None:
    manifest = _manifest_from_payload(
        {
            "splits": {
                "train": [np.int64(101), np.float64(102.0)],
                "val": {np.int64(201): {}},
            }
        }
    )

    assert manifest == {"train": ("101", "102.0"), "val": ("201",)}
