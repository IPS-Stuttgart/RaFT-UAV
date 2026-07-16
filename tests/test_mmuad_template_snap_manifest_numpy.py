from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.template_snap_write import write_template_snapped_submission


def test_template_snap_manifest_serializes_zero_dimensional_gap(tmp_path: Path) -> None:
    results = pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001"],
            "Timestamp": [0.0, 10.0],
            "Position": ["(0,0,0)", "(10,20,2)"],
            "Classification": [2, 2],
        }
    )
    template = pd.DataFrame(
        {
            "Sequence": ["seq001"],
            "Timestamp": [5.0],
            "Position": ["(0,0,0)"],
            "Classification": [0],
        }
    )

    paths = write_template_snapped_submission(
        results=results,
        template=template,
        output_dir=tmp_path / "out",
        max_interpolation_gap_s=np.array(10.0),
    )

    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["max_interpolation_gap_s"] == 10.0
    assert paths["official_zip"].exists()
