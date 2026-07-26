from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_assignment_action_plan import main as action_plan_main


def test_assignment_action_plan_cli_preserves_numeric_like_sequence_ids(
    tmp_path: Path,
) -> None:
    blocks_csv = tmp_path / "blocks.csv"
    output_dir = tmp_path / "out"
    blocks_csv.write_text(
        "sequence_id,assignment_failure_mode,frame_count,duration_s\n"
        "001,covered,1,0.5\n",
        encoding="utf-8",
    )

    status = action_plan_main(
        [
            "--blocks-csv",
            str(blocks_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    rows = pd.read_csv(
        output_dir / "mmuad_candidate_assignment_action_plan.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert rows.loc[0, "sequence_id"] == "001"

    payload = json.loads(
        (output_dir / "mmuad_candidate_assignment_action_summary.json").read_text(
            encoding="utf-8",
        )
    )
    assert payload["actions"][0]["sequence_id"] == "001"
