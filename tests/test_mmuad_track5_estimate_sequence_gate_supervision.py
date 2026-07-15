from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import raft_uav.mmuad.track5_estimate_sequence_gate_fit as sequence_gate_fit
from raft_uav.mmuad.track5_estimate_sequence_gate_fit import (
    _nearest_neighbor_predict,
    fit_estimate_sequence_gate_weights,
)


def _template(sequence_ids: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for sequence_id in sequence_ids:
        for time_s in (0.0, 1.0):
            rows.append(
                {
                    "Sequence": sequence_id,
                    "Timestamp": time_s,
                    "Position": "(0,0,0)",
                    "Classification": 0,
                }
            )
    return pd.DataFrame.from_records(rows)


def _estimates(sequence_ids: tuple[str, ...], *, offset_m: float) -> pd.DataFrame:
    rows = []
    for sequence_index, sequence_id in enumerate(sequence_ids):
        for time_s in (0.0, 1.0):
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "time_s": time_s,
                    "state_x_m": 10.0 * sequence_index + time_s + offset_m,
                    "state_y_m": 0.0,
                    "state_z_m": 0.0,
                }
            )
    return pd.DataFrame.from_records(rows)


def _truth(sequence_ids: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for sequence_index, sequence_id in enumerate(sequence_ids):
        for time_s in (0.0, 1.0):
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "time_s": time_s,
                    "x_m": 10.0 * sequence_index + time_s,
                    "y_m": 0.0,
                    "z_m": 0.0,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_nearest_neighbor_ignores_unsupervised_training_sequence() -> None:
    train = pd.DataFrame(
        {
            "sequence_id": ["unsupervised", "supervised"],
            "row_count": [2, 20],
            "sequence_gate_weight": [np.nan, 0.75],
            "matched_rows": [0, 2],
            "pose_mse_m2": [np.nan, 1.0],
        }
    )
    apply = pd.DataFrame({"sequence_id": ["apply"], "row_count": [2]})

    predicted = _nearest_neighbor_predict(train, apply)

    assert predicted.loc[0, "nearest_train_sequence_id"] == "supervised"
    assert predicted.loc[0, "sequence_gate_weight"] == pytest.approx(0.75)


def test_sequence_gate_fit_requires_two_truth_supervised_sequences() -> None:
    sequences = ("seqA", "seqB")

    with pytest.raises(
        ValueError,
        match="at least 2 sequences with finite oracle supervision",
    ):
        fit_estimate_sequence_gate_weights(
            base_estimates=_estimates(sequences, offset_m=0.0),
            alternate_estimates=_estimates(sequences, offset_m=1.0),
            template=_template(sequences),
            truth=_truth(("seqA",)),
        )


def test_sequence_gate_fit_excludes_unsupervised_sequence_from_loso() -> None:
    sequences = ("seqA", "seqB", "seqC")

    result = fit_estimate_sequence_gate_weights(
        base_estimates=_estimates(sequences, offset_m=0.0),
        alternate_estimates=_estimates(sequences, offset_m=1.0),
        template=_template(sequences),
        truth=_truth(("seqA", "seqB")),
    )

    assert set(result["train_features"]["sequence_id"]) == set(sequences)
    assert set(result["loso_weights"]["sequence_id"]) == {"seqA", "seqB"}
    assert np.isfinite(result["loso_weights"]["sequence_gate_weight"]).all()


def test_compatibility_main_forwards_active_pandas_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    original_impl_pd = sequence_gate_fit._IMPL.pd
    observed: dict[str, object] = {}

    def fake_main(_argv: list[str] | None = None) -> int:
        observed["impl_pd"] = sequence_gate_fit._IMPL.pd
        return 0

    monkeypatch.setattr(sequence_gate_fit, "pd", sentinel)
    monkeypatch.setattr(sequence_gate_fit, "_ORIGINAL_MAIN", fake_main)

    assert sequence_gate_fit.main([]) == 0
    assert observed["impl_pd"] is sentinel
    assert sequence_gate_fit._IMPL.pd is original_impl_pd
