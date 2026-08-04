import pandas as pd

from raft_uav.calibration.empirical_covariance import aligned_residuals


def _frames(*, sequence_scoped: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 0.0, 1.0],
            "east_m": [100.0, 0.0, 1.0],
            "north_m": [100.0, 0.0, 1.0],
        }
    )
    rf = pd.DataFrame(
        {
            "time_s": [0.0, -0.1],
            "east_m": [0.0, 0.0],
            "north_m": [0.0, 0.0],
        }
    )
    if sequence_scoped:
        truth["sequence_id"] = "seq_a"
        rf["sequence_id"] = "seq_a"
    return rf, truth


def test_empirical_covariance_uses_final_duplicate_truth_sample():
    for sequence_scoped in (False, True):
        rf, truth = _frames(sequence_scoped=sequence_scoped)

        residuals = aligned_residuals(
            rf,
            truth,
            source="rf",
            max_time_delta_s=0.2,
        )

        assert residuals.tolist() == [[0.0, 0.0], [0.0, 0.0]]
