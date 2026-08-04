from types import ModuleType

import pandas as pd

from raft_uav.baselines._learned_radar_duplicate_index_patch import (
    apply_learned_radar_duplicate_index_patch,
)


def _scorer_module(scored: pd.DataFrame) -> ModuleType:
    module = ModuleType("fake_learned_radar_association")

    def score_radar_candidates_with_learned_likelihood(*args, **kwargs):
        del args, kwargs
        return scored

    module.score_radar_candidates_with_learned_likelihood = (
        score_radar_candidates_with_learned_likelihood
    )
    return module


def test_duplicate_candidate_indices_remain_single_row_selectable() -> None:
    scored = pd.DataFrame(
        {
            "track_id": [1, 2],
            "association_score": [0.2, 0.1],
        },
        index=[7, 7],
    )
    module = _scorer_module(scored)
    apply_learned_radar_duplicate_index_patch(module)

    normalized = module.score_radar_candidates_with_learned_likelihood()
    selected = normalized.loc[normalized["association_score"].idxmin()]

    assert normalized.index.tolist() == [0, 1]
    assert isinstance(selected, pd.Series)
    assert selected["track_id"] == 2


def test_unique_candidate_indices_are_preserved() -> None:
    scored = pd.DataFrame(
        {
            "track_id": [1, 2],
            "association_score": [0.2, 0.1],
        },
        index=[3, 9],
    )
    module = _scorer_module(scored)
    apply_learned_radar_duplicate_index_patch(module)

    normalized = module.score_radar_candidates_with_learned_likelihood()

    assert normalized is scored
    assert normalized.index.tolist() == [3, 9]
