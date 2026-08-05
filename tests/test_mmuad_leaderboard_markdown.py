from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.leaderboard import LeaderboardResult, write_leaderboard_artifacts


def test_leaderboard_markdown_escapes_structural_cell_text(tmp_path: Path) -> None:
    result = LeaderboardResult(
        rows=pd.DataFrame.from_records(
            [
                {
                    "rank": 1,
                    "method": "pipe|method",
                    "pose_mse_loss_m2": 1.0,
                    "source_note": "first | path\\segment\r\nsecond",
                }
            ]
        ),
        evaluations={},
    )

    artifacts = write_leaderboard_artifacts(result, output_dir=tmp_path)
    markdown = Path(artifacts["leaderboard_md"]).read_text(encoding="utf-8")
    table_lines = [line for line in markdown.splitlines() if line.startswith("| ")]

    assert len(table_lines) == 3
    assert "pipe\\|method" in table_lines[-1]
    assert "first \\| path\\\\segment<br>second" in table_lines[-1]
