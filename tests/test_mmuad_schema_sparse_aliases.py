import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.schema import normalize_candidate_columns, normalize_truth_columns


def test_candidate_normalizer_fills_sparse_canonical_identity_columns_from_aliases():
    raw = pd.DataFrame(
        {
            "sequence_id": [None, "seq_keep"],
            "sequence": ["seq_alias", "seq_ignored"],
            "time_s": [1.0, 2.0],
            "source": ["", None],
            "sensor": ["radar", "rf"],
            "track_id": [None, "track_keep"],
            "child_frame_id": ["track_alias", "track_ignored"],
            "x_m": [0.0, 1.0],
            "y_m": [10.0, 11.0],
            "z_m": [20.0, 21.0],
        }
    )

    rows = normalize_candidate_columns(raw)

    assert rows["sequence_id"].tolist() == ["seq_alias", "seq_keep"]
    assert rows["source"].tolist() == ["radar", "rf"]
    assert rows["track_id"].tolist() == ["track_alias", "track_keep"]


def test_truth_normalizer_fills_sparse_canonical_coordinates_from_aliases():
    raw = pd.DataFrame(
        {
            "sequence_id": [None],
            "sequence": ["truth_seq"],
            "time_s": [3.5],
            "x_m": [None],
            "x": [1.5],
            "y_m": [""],
            "y": [2.5],
            "z_m": [np.nan],
            "z": [3.5],
        }
    )

    rows = normalize_truth_columns(raw)

    assert len(rows) == 1
    assert rows.loc[0, "sequence_id"] == "truth_seq"
    assert rows.loc[0, "time_s"] == pytest.approx(3.5)
    assert rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == pytest.approx([1.5, 2.5, 3.5])
