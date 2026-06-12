import pandas as pd

from raft_uav.mmuad.submission import estimates_to_mmaud_results_frame


def test_results_frame_applies_default_sequence_mapping_without_sequence_column():
    estimates = pd.DataFrame(
        {
            "time_s": [1.0, 0.5],
            "state_x_m": [10.0, 20.0],
            "state_y_m": [30.0, 40.0],
            "state_z_m": [50.0, 60.0],
        }
    )

    results = estimates_to_mmaud_results_frame(
        estimates,
        class_name="fallback",
        class_map={"default": "mapped"},
    )

    assert list(results["sequence_id"]) == ["default", "default"]
    assert list(results["uav_type"]) == ["mapped", "mapped"]
    assert list(results["timestamp"]) == [0.5, 1.0]
