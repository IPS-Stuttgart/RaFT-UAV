from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

import raft_uav.evaluation.oracle_candidate_coverage as detailed_coverage
import raft_uav.evaluation.oracle_coverage as compact_coverage
from raft_uav.evaluation import (
    _oracle_coverage_sequence_scope_patch as sequence_scope_patch,
)


def _pooled_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["flight-a", " flight-b "],
            "time_s": [0.0, 0.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )


def test_compact_coverage_restricts_truth_to_radar_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    expected = object()

    def fake_builder(**kwargs: Any) -> object:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        sequence_scope_patch,
        "_ORIGINAL_BUILD_COMPACT_COVERAGE",
        fake_builder,
    )

    result = compact_coverage.build_oracle_candidate_coverage(
        radar=pd.DataFrame({"sequence_id": ["flight-b"]}),
        truth=_pooled_truth(),
    )

    assert result is expected
    assert captured["truth"]["sequence_id"].tolist() == [" flight-b "]


def test_detailed_coverage_restricts_truth_to_flight_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    expected = (pd.DataFrame(), {"ok": True})

    def fake_builder(**kwargs: Any) -> tuple[pd.DataFrame, dict[str, bool]]:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        sequence_scope_patch,
        "_ORIGINAL_BUILD_DETAILED_COVERAGE",
        fake_builder,
    )

    result = detailed_coverage.build_oracle_candidate_coverage_diagnostics(
        radar=pd.DataFrame({"flight_id": ["flight-b"]}),
        truth=_pooled_truth().rename(columns={"sequence_id": "flight_id"}),
    )

    assert result is expected
    assert captured["truth"]["flight_id"].tolist() == [" flight-b "]


@pytest.mark.parametrize(
    "builder",
    [
        compact_coverage.build_oracle_candidate_coverage,
        detailed_coverage.build_oracle_candidate_coverage_diagnostics,
    ],
)
def test_oracle_coverage_rejects_pooled_radar_sequences(
    builder: Callable[..., Any],
) -> None:
    radar = pd.DataFrame({"sequence_id": ["flight-a", "flight-b"]})

    with pytest.raises(ValueError, match="radar rows from one sequence"):
        builder(radar=radar, truth=_pooled_truth())


@pytest.mark.parametrize(
    "builder",
    [
        compact_coverage.build_oracle_candidate_coverage,
        detailed_coverage.build_oracle_candidate_coverage_diagnostics,
    ],
)
def test_oracle_coverage_rejects_one_sided_sequence_metadata(
    builder: Callable[..., Any],
) -> None:
    radar = pd.DataFrame({"sequence_id": ["flight-a"]})
    truth = _pooled_truth().drop(columns="sequence_id")

    with pytest.raises(ValueError, match="on both radar and truth or neither"):
        builder(radar=radar, truth=truth)


@pytest.mark.parametrize(
    "builder",
    [
        compact_coverage.build_oracle_candidate_coverage,
        detailed_coverage.build_oracle_candidate_coverage_diagnostics,
    ],
)
def test_oracle_coverage_rejects_partially_missing_radar_sequence_ids(
    builder: Callable[..., Any],
) -> None:
    radar = pd.DataFrame({"sequence_id": ["flight-a", None]})

    with pytest.raises(ValueError, match="on every radar row"):
        builder(radar=radar, truth=_pooled_truth())
