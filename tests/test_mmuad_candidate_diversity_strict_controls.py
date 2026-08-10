from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_diversity import diversify_candidate_reservoir


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [1.0] * 4,
            "track_id": ["best", "duplicate", "protected", "far"],
            "x_m": [0.0, 0.1, 0.2, 5.0],
            "y_m": [0.0] * 4,
            "z_m": [0.0] * 4,
            "candidate_reservoir_score": [1.0, 0.9, 0.1, 0.5],
            "candidate_reservoir_protected": [False, False, True, False],
        }
    )


def _boxed(value: object) -> np.ndarray:
    boxed = np.empty((), dtype=object)
    boxed[()] = value
    return boxed


def test_serialized_false_preserve_protected_disables_override() -> None:
    output = diversify_candidate_reservoir(
        _rows(),
        radius_m=1.0,
        preserve_protected="False",
    )

    assert set(output["track_id"]) == {"best", "far"}


@pytest.mark.parametrize("value", ["sometimes", 2, -1, 0.5, np.asarray([False])])
def test_preserve_protected_rejects_ambiguous_controls(value: object) -> None:
    with pytest.raises(ValueError, match="preserve_protected"):
        diversify_candidate_reservoir(_rows(), preserve_protected=value)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("uncertainty_reference_m", True),
        ("uncertainty_exponent", False),
        ("min_radius_scale", np.bool_(True)),
        ("max_radius_scale", np.asarray([4.0])),
    ],
)
def test_uncertainty_controls_reject_lossy_pseudo_scalars(
    name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=name):
        diversify_candidate_reservoir(_rows(), **{name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("radius_m", _boxed(np.asarray([1.0]))),
        ("max_candidates_per_frame", _boxed(np.asarray([2]))),
    ],
)
def test_existing_controls_reject_nested_non_scalars(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        diversify_candidate_reservoir(_rows(), **{name: value})


def test_zero_dimensional_real_controls_remain_supported() -> None:
    output = diversify_candidate_reservoir(
        _rows(),
        radius_m=np.asarray(1.0),
        max_candidates_per_frame=np.asarray(4),
        preserve_protected=np.asarray(False),
        uncertainty_reference_m=np.asarray(10.0),
        uncertainty_exponent=np.asarray(0.5),
        min_radius_scale=np.asarray(0.25),
        max_radius_scale=np.asarray(4.0),
    )

    assert set(output["track_id"]) == {"best", "far"}
