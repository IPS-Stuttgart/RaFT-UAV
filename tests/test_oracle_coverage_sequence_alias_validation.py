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


_BUILDERS = (
    compact_coverage.build_oracle_candidate_coverage,
    detailed_coverage.build_oracle_candidate_coverage_diagnostics,
)


def _stub_original_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_builder(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    monkeypatch.setattr(
        sequence_scope_patch,
        "_ORIGINAL_BUILD_COMPACT_COVERAGE",
        fake_builder,
    )
    monkeypatch.setattr(
        sequence_scope_patch,
        "_ORIGINAL_BUILD_DETAILED_COVERAGE",
        fake_builder,
    )


@pytest.mark.parametrize("builder", _BUILDERS)
def test_oracle_coverage_accepts_independent_sequence_and_flight_ids(
    builder: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_original_builders(monkeypatch)
    radar = pd.DataFrame(
        {"sequence_id": ["shared"], "flight_id": ["flight-a"]}
    )
    truth = pd.DataFrame(
        {"sequence_id": ["shared"], "flight_id": ["flight-a"]}
    )

    result = builder(radar=radar, truth=truth)

    assert result["radar"].equals(radar)
    assert result["truth"].equals(truth)


@pytest.mark.parametrize("builder", _BUILDERS)
def test_oracle_coverage_filters_truth_by_joint_physical_scope(
    builder: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_original_builders(monkeypatch)
    radar = pd.DataFrame(
        {"sequence_id": ["shared"], "flight_id": ["flight-a"]}
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
        }
    )

    result = builder(radar=radar, truth=truth)

    assert result["truth"]["sequence_id"].tolist() == ["shared"]
    assert result["truth"]["flight_id"].tolist() == ["flight-a"]


@pytest.mark.parametrize("builder", _BUILDERS)
def test_oracle_coverage_rejects_multiple_radar_flights_for_shared_sequence(
    builder: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_original_builders(monkeypatch)
    radar = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
        }
    )
    truth = radar.copy()

    with pytest.raises(ValueError, match="radar rows from one flight_id"):
        builder(radar=radar, truth=truth)


@pytest.mark.parametrize("builder", _BUILDERS)
def test_oracle_coverage_rejects_ambiguous_one_sided_truth_flights(
    builder: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_original_builders(monkeypatch)
    radar = pd.DataFrame({"sequence_id": ["shared"]})
    truth = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
        }
    )

    with pytest.raises(ValueError, match="one-sided flight_id metadata"):
        builder(radar=radar, truth=truth)


@pytest.mark.parametrize("builder", _BUILDERS)
def test_oracle_coverage_rejects_partially_missing_truth_sequence_ids(
    builder: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_original_builders(monkeypatch)
    radar = pd.DataFrame({"sequence_id": ["flight-a"]})
    truth = pd.DataFrame({"sequence_id": ["flight-a", None]})

    with pytest.raises(ValueError, match="on every truth row"):
        builder(radar=radar, truth=truth)


@pytest.mark.parametrize("builder", _BUILDERS)
def test_oracle_coverage_rejects_non_scalar_sequence_ids(
    builder: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_original_builders(monkeypatch)
    radar = pd.DataFrame({"sequence_id": [["flight-a"]]})
    truth = pd.DataFrame({"sequence_id": ["flight-a"]})

    with pytest.raises(ValueError, match="scalar sequence_id values on radar"):
        builder(radar=radar, truth=truth)


@pytest.mark.parametrize("builder", _BUILDERS)
def test_oracle_coverage_accepts_complementary_legacy_identifiers(
    builder: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_original_builders(monkeypatch)
    radar = pd.DataFrame(
        {"sequence_id": [None], "flight_id": ["flight-a"]}
    )
    truth = pd.DataFrame(
        {"sequence_id": ["flight-a"], "flight_id": [None]}
    )

    result = builder(radar=radar, truth=truth)

    assert result["radar"].equals(radar)
    assert result["truth"].equals(truth)
