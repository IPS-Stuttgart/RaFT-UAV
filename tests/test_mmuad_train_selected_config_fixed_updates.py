from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.train_selected_config import build_train_selected_config


def test_viterbi_fixed_mode_overrides_summary_alias(tmp_path: Path) -> None:
    summary = tmp_path / "viterbi.csv"
    pd.DataFrame(
        [
            {
                "selection_mode": "greedy",
                "motion_weight": 2.0,
                "pose_mse_loss_m2": 1.0,
            }
        ]
    ).to_csv(summary, index=False)

    config, records = build_train_selected_config(viterbi_summary_csv=summary)

    assert config["mmuad_selection_mode"] == "viterbi"
    assert config["viterbi_motion_weight"] == 2.0
    assert records[0]["mmuad_selection_mode"] == "viterbi"
