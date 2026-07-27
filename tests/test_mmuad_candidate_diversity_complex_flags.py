from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_diversity import diversify_candidate_reservoir

_COMPLEX_WARNING = getattr(getattr(np, "exceptions", np), "ComplexWarning")


def _rows(flag: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [0.0, 0.0],
            "track_id": ["malformed", "valid"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "candidate_reservoir_score": [1.0, 0.5],
            "candidate_reservoir_protected": pd.Series(
                [flag, False],
                dtype=object,
            ),
        }
    )


@pytest.mark.parametrize(
    "flag",
    [
        np.complex64(1.0 + 2.0j),
        np.complex128(1.0 + 2.0j),
        np.array(np.complex64(1.0 + 2.0j), dtype=object),
        np.ma.array(
            np.array(np.complex64(1.0 + 2.0j), dtype=object),
            mask=False,
        ),
    ],
)
def test_diversity_rejects_complex_protected_flags(flag: object) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", _COMPLEX_WARNING)
        with pytest.raises(ValueError, match="candidate_reservoir_protected"):
            diversify_candidate_reservoir(_rows(flag), radius_m=0.0)


def test_diversity_keeps_object_wrapped_real_protected_flags() -> None:
    output = diversify_candidate_reservoir(
        _rows(np.array(1.0, dtype=object)),
        radius_m=0.0,
    )

    assert output["track_id"].tolist() == ["malformed", "valid"]
