import numpy as np
import pandas as pd

from raft_uav.calibration.empirical_covariance import aligned_residuals


def test_empirical_covariance_normalizes_sequence_ids_before_alignment() -> None:
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq_a", "seq_b"],
            "time_s": [0.0, 0.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 100.0],
        }
    )
    rf = pd.DataFrame(
        {
            "sequence_id": [" seq_b ", "seq_a"],
            "time_s": [0.0, 0.0],
            "east_m": [101.0, 2.0],
            "north_m": [99.0, 3.0],
        }
    )

    residuals = aligned_residuals(
        rf,
        truth,
        source="rf",
        max_time_delta_s=0.25,
    )

    np.testing.assert_allclose(residuals, [[1.0, -1.0], [2.0, 3.0]])


def test_empirical_covariance_does_not_match_missing_sequence_ids() -> None:
    truth = pd.DataFrame(
        {
            "sequence_id": [None, "seq_a"],
            "time_s": [0.0, 0.0],
            "east_m": [1000.0, 10.0],
            "north_m": [1000.0, 20.0],
        }
    )
    rf = pd.DataFrame(
        {
            "sequence_id": ["nan", "seq_a"],
            "time_s": [0.0, 0.0],
            "east_m": [1001.0, 11.0],
            "north_m": [999.0, 19.0],
        }
    )

    residuals = aligned_residuals(
        rf,
        truth,
        source="rf",
        max_time_delta_s=0.25,
    )

    np.testing.assert_allclose(residuals, [[1.0, -1.0]])
