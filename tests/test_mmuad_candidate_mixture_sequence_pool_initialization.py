from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad import candidate_mixture_map_sequence_pool_selector as selector


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["001", "seqB"],
            "time_s": [0.0, 0.0],
            "source": ["lidar", "lidar"],
            "track_id": ["a", "b"],
            "x_m": [0.0, 10.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )


def _initialization() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0],
            "state_x_m": [42.0],
            "state_y_m": [1.0],
            "state_z_m": [2.0],
        }
    )


@pytest.mark.parametrize(
    "alias",
    ["Sequence", "sequence", "seq", "scene", "scene_id", "clip", "clip_id"],
)
def test_sequence_pool_initialization_alias_is_canonicalized(
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    captured: dict[str, pd.DataFrame] = {}
    sentinel = object()

    def fake_run(
        candidates,
        *,
        mixture_config=None,
        selector_config=None,
        initial_estimates=None,
        truth=None,
    ):
        captured["initial"] = pd.DataFrame(initial_estimates).copy()
        return sentinel

    monkeypatch.setattr(selector, "_ORIGINAL_RUN_SEQUENCE_POOL_SELECTOR", fake_run)
    initial = _initialization()
    initial[f" {alias} "] = [" 001 "]

    result = selector.run_sequence_pool_selector(
        _candidates(),
        initial_estimates=initial,
    )

    assert result is sentinel
    normalized = captured["initial"]
    assert normalized["sequence_id"].tolist() == ["001"]
    assert len(normalized) == 1


def test_sequence_less_pool_initialization_is_reused_for_every_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, pd.DataFrame] = {}
    sentinel = object()

    def fake_run(
        candidates,
        *,
        mixture_config=None,
        selector_config=None,
        initial_estimates=None,
        truth=None,
    ):
        captured["initial"] = pd.DataFrame(initial_estimates).copy()
        return sentinel

    monkeypatch.setattr(selector, "_ORIGINAL_RUN_SEQUENCE_POOL_SELECTOR", fake_run)

    result = selector.run_sequence_pool_selector(
        _candidates(),
        initial_estimates=_initialization(),
    )

    assert result is sentinel
    normalized = captured["initial"].sort_values("sequence_id").reset_index(drop=True)
    assert normalized["sequence_id"].tolist() == ["001", "seqB"]
    assert normalized["state_x_m"].tolist() == [42.0, 42.0]


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("sequence_id", ""),
        (" Sequence ", "   "),
        ("scene_id", None),
    ],
)
def test_blank_sequence_pool_initialization_is_reused_for_every_sequence(
    column: str,
    value: str | None,
) -> None:
    initial = _initialization()
    initial[column] = [value]

    normalized = selector._normalize_sequence_pool_initialization(
        _candidates(),
        initial,
    )

    assert normalized is not None
    normalized = normalized.sort_values("sequence_id").reset_index(drop=True)
    assert normalized["sequence_id"].tolist() == ["001", "seqB"]
    assert normalized["state_x_m"].tolist() == [42.0, 42.0]


def test_aliased_initialization_changes_the_matching_sequence_only() -> None:
    initial = _initialization()
    initial["Sequence"] = ["001"]

    result = selector.run_sequence_pool_selector(
        _candidates().assign(
            candidate_branch="raw",
            ranker_score=1.0,
            predicted_sigma_m=1.0,
        ),
        mixture_config=selector.core.CandidateMixtureMapConfig(
            top_k=0,
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
            score_weight=0.0,
            sigma_log_weight=0.0,
            smoothness_weight=0.0,
            anchor_weight=1.0e6,
            iterations=1,
        ),
        selector_config=selector.CandidatePoolSequenceSelectorConfig(
            include_full_pool=True,
            include_leave_one_out=False,
        ),
        initial_estimates=initial,
    )

    estimates = result.selected_result.estimates.set_index("sequence_id")
    assert estimates.loc["001", "state_x_m"] > 40.0
    assert estimates.loc["seqB", "state_x_m"] == pytest.approx(10.0)
