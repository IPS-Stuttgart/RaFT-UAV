from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import raft_uav.mmuad.template_snap_write as template_snap_write


def _snapped_rows(row_count: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001"] * row_count,
            "Timestamp": np.arange(row_count, dtype=float),
            "Position": ["(0,0,0)"] * row_count,
            "Classification": [2] * row_count,
        }
    )


def _stub_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    validation = SimpleNamespace(
        summary={"leaderboard_ready": True, "codabench_upload_ready": True},
        rows=pd.DataFrame(),
    )
    monkeypatch.setattr(
        template_snap_write,
        "validate_official_track5_submission",
        lambda *args, **kwargs: validation,
    )


def test_template_snap_manifest_normalizes_serialized_boolean_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapped = _snapped_rows()
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
    _stub_validation(monkeypatch)

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


@pytest.mark.parametrize(
    "invalid_value",
    ["maybe", 2, -1.0, np.inf, 1 + 0j, [1]],
    ids=["text", "integer", "negative", "infinite", "complex", "non-scalar"],
)
def test_template_snap_manifest_rejects_ambiguous_boolean_flags_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_value: object,
) -> None:
    snapped = _snapped_rows(1)
    diagnostics = pd.DataFrame(
        {
            "valid": [invalid_value],
            "extrapolated": [False],
            "large_gap_fallback": [False],
        }
    )
    monkeypatch.setattr(
        template_snap_write,
        "snap_official_results_to_template",
        lambda *args, **kwargs: (snapped.copy(), diagnostics.copy()),
    )
    monkeypatch.setattr(
        template_snap_write,
        "validate_official_track5_submission",
        lambda *args, **kwargs: pytest.fail("validation must not run"),
    )
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="valid contains invalid Boolean values"):
        template_snap_write.write_template_snapped_submission(
            results=snapped,
            template=snapped,
            output_dir=output_dir,
        )

    assert not output_dir.exists()
