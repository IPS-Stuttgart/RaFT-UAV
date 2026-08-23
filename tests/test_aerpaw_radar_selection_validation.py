from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.io.aerpaw import select_radar_measurement_rows


def _radar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": [0],
            "time_s": [1.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "up_m": [30.0],
            "cat_prob_uav": [0.8],
            "track_id": [4],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [1.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "up_m": [30.0],
        }
    )


def test_empty_radar_rejects_unknown_selection_mode() -> None:
    with pytest.raises(ValueError, match="unknown radar selection"):
        select_radar_measurement_rows(
            _radar().iloc[0:0],
            selection="cat-prob",
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.nan,
        np.inf,
        -0.01,
        1.01,
        0.5 + 1.0j,
        np.array([0.5]),
        np.ma.masked,
        np.ma.array(0.5, mask=True),
    ],
)
@pytest.mark.parametrize("selection", ["catprob", "catprob-all"])
def test_catprob_selection_rejects_invalid_thresholds(
    selection: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="catprob_threshold"):
        select_radar_measurement_rows(
            _radar(),
            selection=selection,
            catprob_threshold=value,
        )


@pytest.mark.parametrize("selection", ["catprob", "catprob-all"])
@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_catprob_selection_accepts_probability_boundaries(
    selection: str,
    threshold: float,
) -> None:
    selected = select_radar_measurement_rows(
        _radar(),
        selection=selection,
        catprob_threshold=threshold,
    )

    if threshold == 0.0:
        assert selected["track_id"].tolist() == [4]
    else:
        assert selected.empty


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("truth_gate_m", -1.0),
        ("truth_gate_m", True),
        ("truth_gate_m", np.nan),
        ("truth_gate_m", np.inf),
        ("truth_gate_m", 1.0 + 2.0j),
        ("truth_gate_m", np.array([1.0])),
        ("truth_time_gate_s", -1.0),
        ("truth_time_gate_s", True),
        ("truth_time_gate_s", np.nan),
        ("truth_time_gate_s", np.inf),
        ("truth_time_gate_s", 1.0 + 2.0j),
        ("truth_time_gate_s", np.array([1.0])),
    ],
)
def test_truth_gated_selection_rejects_invalid_gates(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        select_radar_measurement_rows(
            _radar(),
            selection="truth-gated",
            truth=_truth(),
            **{field: value},
        )


def test_radar_selection_accepts_valid_numpy_scalar_controls() -> None:
    catprob = select_radar_measurement_rows(
        _radar(),
        selection="catprob",
        catprob_threshold=np.array(0.5),
    )
    truth_gated = select_radar_measurement_rows(
        _radar(),
        selection="truth-gated",
        truth=_truth(),
        truth_gate_m=np.float64(0.0),
        truth_time_gate_s=np.int64(0),
    )

    assert catprob["track_id"].tolist() == [4]
    assert truth_gated["track_id"].tolist() == [4]


def test_inactive_selection_controls_keep_existing_fast_paths() -> None:
    selected_all = select_radar_measurement_rows(
        _radar(),
        selection="all",
        catprob_threshold=np.nan,
        truth_gate_m=-1.0,
        truth_time_gate_s=-1.0,
    )
    selected_none = select_radar_measurement_rows(
        _radar(),
        selection="none",
        catprob_threshold=np.nan,
        truth_gate_m=-1.0,
        truth_time_gate_s=-1.0,
    )

    assert selected_all["track_id"].tolist() == [4]
    assert selected_none.empty


def test_missing_truth_error_precedes_inactive_gate_validation() -> None:
    with pytest.raises(ValueError, match="requires normalized truth"):
        select_radar_measurement_rows(
            _radar(),
            selection="truth-gated",
            truth=None,
            truth_gate_m=np.nan,
            truth_time_gate_s=np.nan,
        )
