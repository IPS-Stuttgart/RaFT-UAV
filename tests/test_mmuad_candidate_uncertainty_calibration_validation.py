from __future__ import annotations

from dataclasses import replace
import json

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_uncertainty_calibration import (
    CandidateSigmaCalibration,
    apply_candidate_sigma_calibration,
    load_candidate_sigma_calibration,
)
from raft_uav.mmuad.schema import CandidateFrame


def _calibration() -> CandidateSigmaCalibration:
    return CandidateSigmaCalibration(
        schema="raft-uav-mmuad-candidate-sigma-calibration-v1",
        input_sigma_column="predicted_sigma_m",
        branch_column="candidate_branch",
        target_quantile=0.5,
        min_group_rows=1,
        shrinkage_rows=0.0,
        scale_min=0.25,
        scale_max=4.0,
        calibration_row_count=1,
        global_scale=1.0,
        source_scales={},
        branch_scales={},
        source_branch_scales={},
        source_counts={},
        branch_counts={},
        source_branch_counts={},
    )


def _candidates() -> CandidateFrame:
    return CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seq001"],
                "time_s": [0.0],
                "source": ["radar"],
                "track_id": ["track"],
                "x_m": [0.0],
                "y_m": [0.0],
                "z_m": [0.0],
                "confidence": [1.0],
                "predicted_sigma_m": [2.0],
            }
        )
    )


def test_loader_rejects_nonpositive_global_scale(tmp_path) -> None:
    payload = _calibration().__dict__ | {"global_scale": -1.0}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="global_scale"):
        load_candidate_sigma_calibration(path)


def test_apply_rejects_nonfinite_group_scale() -> None:
    calibration = replace(_calibration(), source_scales={"radar": float("nan")})

    with pytest.raises(ValueError, match="source_scales"):
        apply_candidate_sigma_calibration(_candidates(), calibration)


def test_replace_covariance_rejects_nonpositive_z_scale() -> None:
    with pytest.raises(ValueError, match="z_scale"):
        apply_candidate_sigma_calibration(
            _candidates(),
            _calibration(),
            replace_covariance=True,
            z_scale=0.0,
        )
