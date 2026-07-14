from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble_spread_guard import (
    build_spread_guarded_estimate_ensemble,
)


def _template() -> pd.DataFrame:
    return pd.DataFrame({"Sequence": ["seq0001"], "Timestamp": [0.0]})


def _estimate(x_m: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [x_m],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
        }
    )


@pytest.mark.parametrize(
    "labels",
    [
        ("trusted", "trusted"),
        ("trusted/a", r"trusted\a"),
    ],
)
def test_spread_guard_rejects_ambiguous_normalized_labels(
    labels: tuple[str, str],
) -> None:
    with pytest.raises(ValueError, match="unique after normalization"):
        build_spread_guarded_estimate_ensemble(
            [
                (labels[0], _estimate(0.0), 0.6),
                (labels[1], _estimate(10.0), 0.4),
            ],
            _template(),
            spread_threshold_m=1.0,
        )


def test_spread_guard_rejects_unknown_named_fallback() -> None:
    with pytest.raises(ValueError, match="does not match any estimate input label"):
        build_spread_guarded_estimate_ensemble(
            [
                ("trusted", _estimate(0.0), 0.6),
                ("outlier", _estimate(10.0), 0.4),
            ],
            _template(),
            spread_threshold_m=1.0,
            fallback_policy="label",
            fallback_label="trustd",
        )


def test_spread_guard_accepts_existing_normalized_named_fallback() -> None:
    estimates, _ = build_spread_guarded_estimate_ensemble(
        [
            ("trusted/path", _estimate(0.0), 0.4),
            ("outlier", _estimate(10.0), 0.6),
        ],
        _template(),
        spread_threshold_m=1.0,
        fallback_policy="label",
        fallback_label="trusted/path",
    )

    assert estimates.loc[0, "spread_guard_chosen_label"] == "trusted_path"
    assert estimates.loc[0, "state_x_m"] == pytest.approx(0.0)
