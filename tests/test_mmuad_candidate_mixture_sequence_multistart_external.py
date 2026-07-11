from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad import candidate_mixture_map_sequence_multistart as sequence_multistart


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "source": ["lidar", "lidar"],
            "track_id": ["a", "b"],
            "x_m": [0.0, 10.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "ranker_score": [1.0, 1.0],
            "predicted_sigma_m": [1.0, 1.0],
        }
    )


def _external_initialization() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0],
            "state_x_m": [42.0],
            "state_y_m": [1.0],
            "state_z_m": [2.0],
        }
    )


def test_sequence_less_external_initialization_is_reused_for_every_sequence(
    monkeypatch,
) -> None:
    captured: dict[str, pd.DataFrame] = {}
    sentinel = object()

    def fake_run(
        candidates,
        *,
        mixture_config=None,
        multistart_config=None,
        external_initial_estimates=None,
        truth=None,
    ):
        captured["external"] = pd.DataFrame(external_initial_estimates).copy()
        return sentinel

    monkeypatch.setattr(
        sequence_multistart,
        "_ORIGINAL_RUN_SEQUENCE_MULTISTART",
        fake_run,
    )

    result = sequence_multistart.run_sequence_multistart_candidate_mixture_map(
        _candidates(),
        external_initial_estimates=_external_initialization(),
    )

    assert result is sentinel
    external = captured["external"].sort_values("sequence_id").reset_index(drop=True)
    assert external["sequence_id"].tolist() == ["seqA", "seqB"]
    assert external["state_x_m"].tolist() == [42.0, 42.0]
    assert external["time_s"].tolist() == [0.0, 0.0]


def test_external_sequence_alias_is_canonicalized_without_replication() -> None:
    external = _external_initialization().assign(Sequence=["seqB"])

    normalized = sequence_multistart._expand_sequence_less_external_initialization(
        _candidates(),
        external,
    )

    assert normalized is not None
    assert normalized["sequence_id"].tolist() == ["seqB"]
    assert len(normalized) == 1


@pytest.mark.parametrize("alias", ["scene", "scene_id", "clip", "clip_id"])
def test_external_schema_sequence_aliases_are_not_replicated(alias: str) -> None:
    external = _external_initialization()
    external[f" {alias} "] = [" seqB "]

    normalized = sequence_multistart._expand_sequence_less_external_initialization(
        _candidates(),
        external,
    )

    assert normalized is not None
    assert normalized["sequence_id"].tolist() == ["seqB"]
    assert len(normalized) == 1
