from types import ModuleType

import pandas as pd

from raft_uav.baselines._imm_radar_duplicate_index_patch import (
    apply_imm_radar_duplicate_index_patch,
)
from raft_uav.baselines._learned_radar_duplicate_index_patch import (
    apply_learned_radar_duplicate_index_patch,
)
from raft_uav.baselines._radar_candidate_index_patch import (
    apply_radar_candidate_index_patch,
)


def test_shared_scorer_normalizes_duplicate_indices_and_preserves_attrs() -> None:
    scored = pd.DataFrame({"association_nis": [2.0, 1.0]}, index=[7, 7])
    scored.attrs["source"] = "radar"
    module = ModuleType("fake_radar_association")
    module._nis_scored_candidates = lambda: scored

    apply_radar_candidate_index_patch(module)
    normalized = module._nis_scored_candidates()

    assert normalized.index.tolist() == [0, 1]
    assert normalized.attrs == {"source": "radar"}
    assert isinstance(normalized.loc[normalized["association_nis"].idxmin()], pd.Series)


def test_shared_scorer_preserves_unique_index_object() -> None:
    scored = pd.DataFrame({"association_nis": [2.0, 1.0]}, index=[3, 9])
    module = ModuleType("fake_radar_association")
    module._nis_scored_candidates = lambda: scored

    apply_radar_candidate_index_patch(module)

    assert module._nis_scored_candidates() is scored


def test_learned_scorer_normalizes_duplicate_indices() -> None:
    scored = pd.DataFrame({"association_score": [0.2, 0.1]}, index=[4, 4])
    module = ModuleType("fake_learned_radar_association")
    module.score_radar_candidates_with_learned_likelihood = lambda: scored

    apply_learned_radar_duplicate_index_patch(module)
    normalized = module.score_radar_candidates_with_learned_likelihood()

    assert normalized.index.tolist() == [0, 1]
    assert isinstance(normalized.loc[normalized["association_score"].idxmin()], pd.Series)


def test_imm_selector_reduces_duplicate_label_result_to_best_row() -> None:
    selected = pd.DataFrame(
        {"track_id": [1, 2], "association_score": [2.0, 1.0]},
        index=[7, 7],
    )
    module = ModuleType("fake_imm_radar_association")
    module._select_imm_radar_candidate = lambda: selected

    apply_imm_radar_duplicate_index_patch(module)
    result = module._select_imm_radar_candidate()

    assert isinstance(result, pd.Series)
    assert result["track_id"] == 2


def test_duplicate_index_patches_are_idempotent() -> None:
    radar = ModuleType("fake_radar_association")
    radar._nis_scored_candidates = lambda: pd.DataFrame({"score": [1.0]})
    apply_radar_candidate_index_patch(radar)
    installed = radar._nis_scored_candidates

    apply_radar_candidate_index_patch(radar)

    assert radar._nis_scored_candidates is installed
