from __future__ import annotations

import numpy as np
import pytest

from raft_uav._uncertainty_payload_validation_patch import _finite_real_scalar
from raft_uav.uncertainty import VarianceHead


def _boxed(value: object) -> np.ndarray:
    out = np.empty((), dtype=object)
    out[()] = value
    return out


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("coefficients", (_boxed(_boxed(True)),)),
        ("min_std_m", _boxed(_boxed(np.bool_(True)))),
        ("max_std_m", _boxed(np.array([500.0]))),
        ("training_rows", _boxed(_boxed(False))),
    ],
)
def test_variance_head_rejects_nested_lossy_numeric_fields(
    field_name: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {
        "source": "rf",
        "dimension": "east",
        "feature_names": ("intercept",),
        "coefficients": (1.0,),
        "min_std_m": 10.0,
        "max_std_m": 500.0,
        "training_rows": 3,
    }
    arguments[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        VarianceHead(**arguments)


@pytest.mark.parametrize(
    "value",
    [
        _boxed(_boxed(True)),
        _boxed(_boxed(1.0 + 2.0j)),
        _boxed(np.array([1.0])),
    ],
)
def test_uncertainty_scalar_guard_rejects_nested_pseudo_scalars(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="value"):
        _finite_real_scalar(value, name="value")


def test_uncertainty_scalar_guard_rejects_cyclic_object_array() -> None:
    value = np.empty((), dtype=object)
    value[()] = value

    with pytest.raises(ValueError, match="value"):
        _finite_real_scalar(value, name="value")


def test_uncertainty_scalar_guard_keeps_nested_real_scalars() -> None:
    value = _boxed(_boxed(np.float64(2.5)))

    assert _finite_real_scalar(value, name="value") == pytest.approx(2.5)
