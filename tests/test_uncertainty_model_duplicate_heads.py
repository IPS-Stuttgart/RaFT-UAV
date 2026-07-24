from __future__ import annotations

import pytest

from raft_uav.uncertainty import HeteroscedasticUncertaintyModel


def _head_payload(*, coefficient: float) -> dict[str, object]:
    return {
        "source": "rf",
        "dimension": "east",
        "feature_names": ["intercept"],
        "coefficients": [coefficient],
        "min_std_m": 1.0,
        "max_std_m": 100.0,
        "training_rows": 1,
    }


def test_model_rejects_duplicate_source_dimension_heads() -> None:
    payload = {
        "schema_version": 1,
        "metadata": {},
        "heads": [
            _head_payload(coefficient=0.0),
            _head_payload(coefficient=4.0),
        ],
    }

    with pytest.raises(
        ValueError,
        match="duplicate uncertainty head.*source 'rf'.*dimension 'east'",
    ):
        HeteroscedasticUncertaintyModel.from_dict(payload)
