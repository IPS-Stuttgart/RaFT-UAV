from __future__ import annotations

import numpy as np

from raft_uav.mmuad.candidate_identity import canonical_track_id, canonical_track_ids


def test_fractional_track_ids_match_across_numeric_csv_representations() -> None:
    expected = "0.1"

    assert canonical_track_id(0.1) == expected
    assert canonical_track_id(np.float64(0.1)) == expected
    assert canonical_track_id(np.float32(0.1)) == expected
    assert canonical_track_id("0.1") == expected


def test_fractional_track_id_vectorization_preserves_shared_identity() -> None:
    canonical = canonical_track_ids([0.1, np.float32(0.1), "0.1"])

    assert canonical.tolist() == ["0.1", "0.1", "0.1"]
