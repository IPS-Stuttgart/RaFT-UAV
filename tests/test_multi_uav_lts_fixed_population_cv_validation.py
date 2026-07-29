from __future__ import annotations

import pytest

from raft_uav.multi_uav_lts.fixed_population_cv import build_stratified_folds


def test_stratified_folds_reject_duplicate_sequence_names() -> None:
    sequences = ("C_00", "C_00", "T_00")

    with pytest.raises(ValueError, match="duplicate sequence names: C_00"):
        build_stratified_folds(sequences, fold_count=3, seed=0)
