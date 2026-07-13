import pandas as pd

import raft_uav.imm_cli as imm_cli


def test_imm_records_to_frame_parses_false_string_acceptance():
    frame = imm_cli._records_to_frame(
        [
            {
                "time_s": 0.0,
                "source": "radar",
                "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "accepted": "False",
            }
        ]
    )

    assert not bool(frame.loc[0, "accepted"])


def test_imm_metrics_counts_csv_like_false_acceptance():
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 1.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    estimate_frame = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "source": ["rf", "radar"],
            "accepted": ["False", "true"],
            "east_m": [0.0, 1.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    metrics = imm_cli._metrics(
        flight_name="synthetic",
        truth=truth,
        rf=pd.DataFrame(index=[0]),
        radar=pd.DataFrame(index=[0]),
        selected_radar=pd.DataFrame(index=[0]),
        estimate_frame=estimate_frame,
        tracker="imm",
        acceleration_std=4.0,
        imm_mode_switch_time_constant=20.0,
        smoother="none",
        smoother_lag_s=20.0,
        radar_selection="catprob",
        radar_catprob_threshold=0.5,
        max_eval_time_delta_s=2.0,
        robust_update="none",
        enable_association_safety_gate=True,
        rf_safety_gate_prob=0.99,
        radar_safety_gate_prob=0.99,
        rf_max_residual_m=750.0,
        radar_max_residual_m=0.0,
    )

    assert metrics["accepted_measurements"] == 1
    assert metrics["rejected_measurements"] == 1
