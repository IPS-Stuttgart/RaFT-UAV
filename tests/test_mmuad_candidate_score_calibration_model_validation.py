from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_score_calibration import (
    apply_candidate_score_calibration,
    load_candidate_score_calibration_model,
)


def _model() -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol": "test candidate score calibration",
        "score_column": "ranker_score",
        "fallback_score_column": "confidence",
        "output_score_column": "candidate_class_calibrated_score",
        "score_transform": "probability",
        "class_labels": [0, 1, 2, 3],
        "global_logit_offset": 0.0,
        "branch_class_logit_offsets": {},
        "source_class_logit_offsets": {},
        "branch_source_class_logit_offsets": {},
        "include_branch_source_interactions": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("global_logit_offset", float("nan")),
        ("branch_class_logit_offsets", {"raw": {"0": float("inf")}}),
        ("source_class_logit_offsets", {"radar": {"1": float("-inf")}}),
        (
            "branch_source_class_logit_offsets",
            {"raw||radar": {"2": float("nan")}},
        ),
    ],
)
def test_candidate_score_calibration_loader_rejects_nonfinite_offsets(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    model = _model()
    model[field] = value
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model), encoding="utf-8")

    with pytest.raises(ValueError) as error:
        load_candidate_score_calibration_model(path)

    assert field in str(error.value)
    assert "finite scalar logit offset" in str(error.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("branch_class_logit_offsets", {"raw": 1.0}),
        ("source_class_logit_offsets", []),
        ("branch_source_class_logit_offsets", {"raw||radar": None}),
    ],
)
def test_candidate_score_calibration_loader_rejects_malformed_offset_maps(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    model = _model()
    model[field] = value
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model), encoding="utf-8")

    with pytest.raises(ValueError) as error:
        load_candidate_score_calibration_model(path)

    assert field in str(error.value)
    assert "mapping" in str(error.value)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        np.ma.masked,
        np.ma.array(2.0, mask=True),
        [0.0],
    ],
)
def test_candidate_score_calibration_apply_rejects_non_scalar_global_offset(
    value: object,
) -> None:
    model = _model()
    model["global_logit_offset"] = value

    with pytest.raises(
        ValueError,
        match="global_logit_offset must be a finite scalar logit offset",
    ):
        apply_candidate_score_calibration(
            pd.DataFrame(),
            model,
            class_probabilities=None,
        )
