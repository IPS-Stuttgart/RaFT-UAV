from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.track5_submission_ensemble import main as submission_ensemble_main


def _zero_padded_track5_rows(*, offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["001", "001", "002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": [
                f"({0.0 + offset}, 0.0, 0.0)",
                f"({1.0 + offset}, 0.0, 0.0)",
                f"({10.0 + offset}, 0.0, 0.0)",
            ],
            "Classification": [1, 1, 2],
        }
    )


def test_submission_ensemble_cli_preserves_zero_padded_template_ids(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    template = tmp_path / "template.csv"
    output_dir = tmp_path / "ensemble_out"
    _zero_padded_track5_rows(offset=0.0).to_csv(first, index=False)
    _zero_padded_track5_rows(offset=2.0).to_csv(second, index=False)
    _zero_padded_track5_rows(offset=0.0).to_csv(template, index=False)

    status = submission_ensemble_main(
        [
            "--submission",
            f"a=1:{first}",
            "--submission",
            f"b=1:{second}",
            "--template",
            str(template),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    manifest = json.loads((output_dir / "mmuad_track5_ensemble_manifest.json").read_text())
    assert manifest["validation"]["leaderboard_ready"] is True
