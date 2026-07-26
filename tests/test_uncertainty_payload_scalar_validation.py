from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from raft_uav.uncertainty import VarianceHead, load_uncertainty_model


def _head_payload() -> dict[str, object]:
    return {
        "source": "rf",
        "dimension": "east",
        "feature_names": ["intercept"],
        "coefficients": [1.0],
        "min_std_m": 10.0,
        "max_std_m": 500.0,
        "training_rows": 3,
    }


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("coefficients", [True]),
        ("min_std_m", True),
        ("max_std_m", np.bool_(True)),
        ("training_rows", False),
    ],
)
def test_uncertainty_model_load_rejects_boolean_numeric_fields(
    tmp_path: Path,
    field_name: str,
    value: object,
) -> None:
    head = _head_payload()
    head[field_name] = value
    path = tmp_path / "uncertainty.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": "heteroscedastic-loglinear-variance",
                "metadata": {},
                "heads": [head],
            },
            default=lambda item: bool(item) if isinstance(item, np.bool_) else item,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field_name):
        load_uncertainty_model(path)


@pytest.mark.parametrize("training_rows", [True, np.bool_(False), 1.5, -1])
def test_variance_head_rejects_invalid_training_rows(training_rows: object) -> None:
    with pytest.raises(ValueError, match="training_rows"):
        VarianceHead(
            source="rf",
            dimension="east",
            feature_names=("intercept",),
            coefficients=(1.0,),
            min_std_m=10.0,
            max_std_m=500.0,
            training_rows=training_rows,
        )


@pytest.mark.parametrize("coefficient", [True, np.bool_(False)])
def test_variance_head_rejects_boolean_coefficients(coefficient: object) -> None:
    with pytest.raises(ValueError, match=r"coefficients\[0\]"):
        VarianceHead(
            source="rf",
            dimension="east",
            feature_names=("intercept",),
            coefficients=(coefficient,),
            min_std_m=10.0,
            max_std_m=500.0,
            training_rows=3,
        )


def test_variance_head_keeps_valid_numeric_scalars() -> None:
    head = VarianceHead(
        source="rf",
        dimension="east",
        feature_names=("intercept",),
        coefficients=(np.float64(1.0),),
        min_std_m=np.float64(10.0),
        max_std_m=np.int64(500),
        training_rows=np.int64(3),
    )

    assert head.coefficients == (np.float64(1.0),)
    assert head.training_rows == np.int64(3)
