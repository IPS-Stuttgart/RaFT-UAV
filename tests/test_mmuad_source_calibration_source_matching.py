from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.source_calibration import (
    SourceTransform,
    _match_source_transform,
    apply_source_calibration_payload,
)


def test_source_transform_lookup_is_forward_only_and_prefers_specific_key() -> None:
    generic = SourceTransform.identity()
    specific = SourceTransform.identity()
    transforms = {"sensor": generic, "sensor_detail": specific}

    assert _match_source_transform("sensor", {"sensor_detail": specific}) is None
    assert _match_source_transform("sensor_detail_clusters", transforms) is specific


def test_source_calibration_rejects_case_insensitive_duplicate_transform_keys() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seq"],
                "time_s": [0.0],
                "source": ["radar"],
                "track_id": ["track"],
                "x_m": [1.0],
                "y_m": [0.0],
                "z_m": [0.0],
                "confidence": [1.0],
            }
        )
    )
    identity = {
        "linear": np.eye(3).tolist(),
        "translation_m": [0.0, 0.0, 0.0],
    }
    shifted = {
        "linear": np.eye(3).tolist(),
        "translation_m": [10.0, 0.0, 0.0],
    }

    for transforms in (
        {"radar": identity, "RADAR": shifted},
        {"RADAR": shifted, "radar": identity},
    ):
        payload = {
            "mode": "source-translation",
            "transforms": transforms,
        }
        with pytest.raises(
            ValueError,
            match="ambiguous case-insensitive keys",
        ):
            apply_source_calibration_payload(candidates, payload)


def test_source_calibration_does_not_apply_specific_transform_to_broad_source() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seq", "seq"],
                "time_s": [0.0, 1.0],
                "source": ["sensor", "sensor_detail_clusters"],
                "track_id": ["broad", "specific"],
                "x_m": [1.0, 1.0],
                "y_m": [0.0, 0.0],
                "z_m": [0.0, 0.0],
                "confidence": [1.0, 1.0],
            }
        )
    )
    payload = {
        "mode": "source-translation",
        "transforms": {
            "sensor_detail": {
                "linear": np.eye(3).tolist(),
                "translation_m": [10.0, 0.0, 0.0],
            }
        },
    }

    calibrated = apply_source_calibration_payload(candidates, payload).rows.set_index(
        "track_id"
    )

    assert calibrated.loc["broad", "x_m"] == pytest.approx(1.0)
    assert bool(calibrated.loc["broad", "mmuad_source_calibration_applied"]) is False
    assert calibrated.loc["specific", "x_m"] == pytest.approx(11.0)
    assert bool(calibrated.loc["specific", "mmuad_source_calibration_applied"]) is True
