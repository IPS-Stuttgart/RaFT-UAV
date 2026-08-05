from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.source_calibration_path_ensemble import (
    _normalize_fractions,
    build_source_calibration_path_ensemble,
)


@pytest.mark.parametrize(
    "value",
    [
        False,
        np.bool_(True),
        0.5 + 0.0j,
        np.array([0.5]),
        np.ma.masked,
    ],
)
def test_path_ensemble_rejects_lossy_fraction_scalars(value: object) -> None:
    with pytest.raises(ValueError, match="finite real scalars"):
        build_source_calibration_path_ensemble(
            pd.DataFrame(),
            {},
            fractions=(value,),
        )


def test_path_ensemble_rejects_recursively_boxed_boolean_fraction() -> None:
    boxed = np.empty((), dtype=object)
    boxed[()] = np.array(True, dtype=object)

    with pytest.raises(ValueError, match="finite real scalars"):
        build_source_calibration_path_ensemble(
            pd.DataFrame(),
            {},
            fractions=(boxed,),
        )


def test_path_ensemble_rejects_cyclic_fraction_container() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic

    with pytest.raises(ValueError, match="finite real scalars"):
        build_source_calibration_path_ensemble(
            pd.DataFrame(),
            {},
            fractions=(cyclic,),
        )


def test_fraction_normalization_preserves_real_scalar_like_inputs() -> None:
    fractions = _normalize_fractions(
        (
            np.array(0.75),
            np.float32(0.25),
            "0.5",
            0,
            1,
        )
    )

    assert fractions == (0.0, 0.25, 0.5, 0.75, 1.0)


@pytest.mark.parametrize("fractions", [True, 0.5, "0.5"])
def test_fraction_normalization_rejects_non_iterable_or_text_grids(
    fractions: object,
) -> None:
    with pytest.raises(ValueError, match="iterable of finite real scalars"):
        _normalize_fractions(fractions)  # type: ignore[arg-type]
