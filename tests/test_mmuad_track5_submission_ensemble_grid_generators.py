from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.track5_submission_ensemble import parse_submission_input
from raft_uav.mmuad.track5_submission_ensemble_grid import (
    write_submission_ensemble_weight_grid_outputs,
)


def _submission_rows(offset: float = 0.0, classification: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": [
                f"({0.0 + offset}, 0.0, 1.0)",
                f"({2.0 + offset}, 0.0, 1.0)",
                f"({10.0 + offset}, 1.0, 2.0)",
            ],
            "Classification": [classification, classification, 2],
        }
    )


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 1.0, 0.0],
            "x_m": [0.0, 2.0, 10.0],
            "y_m": [0.0, 0.0, 1.0],
            "z_m": [1.0, 1.0, 2.0],
            "class_id": [1, 1, 2],
        }
    )


def test_submission_grid_materializes_generator_controls(tmp_path: Path) -> None:
    good = tmp_path / "good.csv"
    bad = tmp_path / "bad.csv"
    output_dir = tmp_path / "out"
    _submission_rows(offset=0.0, classification=1).to_csv(good, index=False)
    _submission_rows(offset=10.0, classification=3).to_csv(bad, index=False)

    paths = write_submission_ensemble_weight_grid_outputs(
        submission_inputs=(
            item
            for item in (
                parse_submission_input(f"good={good}"),
                parse_submission_input(f"bad={bad}"),
            )
        ),
        truth=_truth_rows(),
        weight_grid=(weights for weights in ((1.0, 0.0), (0.0, 1.0))),
        output_dir=output_dir,
        template=_submission_rows(offset=0.0, classification=1),
        class_policies=(policy for policy in ("weighted-vote", "first")),
    )

    summary = pd.read_csv(paths["summary_csv"])
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))

    assert len(summary) == 4
    assert set(summary["class_policy"]) == {"weighted-vote", "first"}
    assert set(
        summary[["weight_good", "weight_bad"]].itertuples(index=False, name=None)
    ) == {(1.0, 0.0), (0.0, 1.0)}
    assert manifest["class_policies"] == ["weighted-vote", "first"]
    assert manifest["grid_row_count"] == 4
