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


def test_source_transform_prefix_requires_a_token_boundary() -> None:
    transform = SourceTransform.identity()

    assert _match_source_transform("radar2", {"radar": transform}) is None
    assert _match_source_transform("radar_detail", {"radar": transform}) is transform
    assert _match_source_transform("radar-detail", {"radar": transform}) is transform


def test_source_calibration_does_not_leak_to_alphanumeric_source_names() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seq", "seq"],
                "time_s": [0.0, 1.0],
                "source": ["radar2", "radar_detail"],
                "track_id": ["unrelated", "derived"],
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
            "radar": {
                "linear": np.eye(3).tolist(),
                "translation_m": [10.0, 0.0, 0.0],
            }
        },
    }

    calibrated = apply_source_calibration_payload(candidates, payload).rows.set_index(
        "track_id"
    )

    assert calibrated.loc["unrelated", "x_m"] == pytest.approx(1.0)
    assert bool(calibrated.loc["unrelated", "mmuad_source_calibration_applied"]) is False
    assert calibrated.loc["derived", "x_m"] == pytest.approx(11.0)
    assert bool(calibrated.loc["derived", "mmuad_source_calibration_applied"]) is True


def test_source_calibration_rejects_blank_transform_keys() -> None:
    with pytest.raises(ValueError, match="non-blank source keys"):
        _match_source_transform("radar", {"": SourceTransform.identity()})
