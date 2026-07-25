from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import raft_uav.mmuad.template_snap_write as template_snap_write


def test_template_snap_manifest_normalizes_serialized_boolean_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapped = pd.DataFrame(
        {
            "Sequence": ["seq001"] * 5,
            "Timestamp": np.arange(5, dtype=float),
            "Position": ["(0,0,0)"] * 5,
            "Classification": [2] * 5,
        }
    )
    diagnostics = pd.DataFrame(
        {
            "valid": [True, False, "true", "false", np.nan],
            "extrapolated": [1.0, 0.0, "1.0", "0.0", None],
            "large_gap_fallback": ["yes", "no", "Y", "N", ""],
        }
    )

    monkeypatch.setattr(
        template_snap_write,
        "snap_official_results_to_template",
        lambda *args, **kwargs: (snapped.copy(), diagnostics.copy()),
    )
    validation = SimpleNamespace(
        summary={"leaderboard_ready": True, "codabench_upload_ready": True},
        rows=pd.DataFrame(),
    )
    monkeypatch.setattr(
        template_snap_write,
        "validate_official_track5_submission",
        lambda *args, **kwargs: validation,
    )

    paths = template_snap_write.write_template_snapped_submission(
        results=snapped,
        template=snapped,
        output_dir=tmp_path / "out",
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))

    assert manifest["valid_snapped_rows"] == 2
    assert manifest["invalid_snapped_rows"] == 3
    assert manifest["extrapolated_rows"] == 2
    assert manifest["large_gap_fallback_rows"] == 2
