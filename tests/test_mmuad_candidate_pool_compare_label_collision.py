from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad import candidate_pool_compare as compare


def test_candidate_pool_compare_rejects_normalized_label_collisions(monkeypatch) -> None:
    load_calls: list[list[str]] = []

    def fake_load_candidate_inputs(specs: list[str]) -> pd.DataFrame:
        load_calls.append(list(specs))
        return pd.DataFrame({"time_s": [0.0]})

    monkeypatch.setattr(compare, "load_candidate_inputs", fake_load_candidate_inputs)

    with pytest.raises(ValueError, match="candidate pool labels collide after normalization"):
        compare._load_labeled_candidate_pools(
            ["raw a=first.csv", "raw_a=second.csv"],
        )

    assert load_calls == []


def test_candidate_pool_compare_keeps_repeated_same_label(monkeypatch) -> None:
    load_calls: list[list[str]] = []

    def fake_load_candidate_inputs(specs: list[str]) -> pd.DataFrame:
        load_calls.append(list(specs))
        return pd.DataFrame({"time_s": [0.0]})

    monkeypatch.setattr(compare, "load_candidate_inputs", fake_load_candidate_inputs)

    pools = compare._load_labeled_candidate_pools(
        ["raw a=first.csv", " raw a =second.csv"],
    )

    assert list(pools) == ["raw_a"]
    assert load_calls == [["raw_a=first.csv", "raw_a=second.csv"]]
