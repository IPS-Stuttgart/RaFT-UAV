from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.train_selected_config import validate_train_selected_config


def _nested_zero_dimensional(value: object, *, depth: int = 2) -> np.ndarray:
    nested = value
    for _ in range(depth):
        wrapper = np.empty((), dtype=object)
        wrapper[()] = nested
        nested = wrapper
    assert isinstance(nested, np.ndarray)
    return nested


@pytest.mark.parametrize(
    "invalid_value",
    [
        _nested_zero_dimensional(True),
        _nested_zero_dimensional(np.bool_(False)),
        _nested_zero_dimensional(np.array([0.5])),
        _nested_zero_dimensional(1.0 + 0.0j),
        _nested_zero_dimensional(np.ma.array(0.5, mask=True)),
    ],
)
def test_train_selected_config_rejects_nested_pseudo_scalars(
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError, match="expected finite float"):
        validate_train_selected_config({"smoothing_blend": invalid_value})


def test_train_selected_config_rejects_cyclic_scalar_arrays() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic

    with pytest.raises(ValueError, match="expected finite float"):
        validate_train_selected_config({"smoothing_blend": cyclic})


def test_train_selected_config_accepts_recursively_nested_real_scalars() -> None:
    config = validate_train_selected_config(
        {
            "smoothing_blend": _nested_zero_dimensional("0.5", depth=3),
            "image_nonimage_fusion_weight": _nested_zero_dimensional(0, depth=3),
        }
    )

    assert config["smoothing_blend"] == pytest.approx(0.5)
    assert config["image_nonimage_fusion_weight"] == pytest.approx(0.0)
