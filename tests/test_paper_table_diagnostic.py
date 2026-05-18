import numpy as np
import pandas as pd

from raft_uav.diagnostics.paper_table import metric_row, select_radar_for_table


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 10.0, 20.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )


def test_metric_row_reports_paper_style_error_columns():
    row = metric_row(
        method="synthetic",
        modality="radar",
        times_s=np.array([0.0, 1.0]),
        positions_m=np.array([[0.0, 0.0, 0.0], [13.0, 4.0, 0.0]]),
        truth=_truth(),
        candidate_count=2,
        selected_count=2,
        max_time_delta_s=0.5,
        track_ids=[1, 2],
    )

    assert row["matched_count"] == 2
    assert row["coverage"] == 1.0
    assert row["track_switches"] == 1
    assert row["error_2d_mean_m"] == 2.5
    assert row["error_3d_max_m"] == 5.0


def test_select_radar_for_table_oracle_picks_nearest_truth_per_frame():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.5,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 1,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": -100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 1,
                "track_id": 2,
                "time_s": 1.0,
                "east_m": 11.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.4,
            },
        ]
    )

    selected = select_radar_for_table(
        radar=radar,
        truth=_truth(),
        selection="radar-oracle-nearest-truth",
        catprob_threshold=0.4,
        max_time_delta_s=0.5,
    )

    assert selected["track_id"].tolist() == [1, 2]
