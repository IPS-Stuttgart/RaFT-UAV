from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.class_probability_calibration import (
    _load_labels_preserving_ids,
    normalize_label_map,
)


def _conflicting_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["0001", "0001", "0002"],
            "uav_type": ["0", "1", "1"],
        }
    )


@pytest.mark.parametrize("order", ([0, 1, 2], [1, 0, 2]))
def test_conflicting_duplicate_labels_are_rejected_independent_of_row_order(
    order: list[int],
) -> None:
    labels = _conflicting_labels().iloc[order]

    with pytest.raises(ValueError, match=r"conflicting labels.*0001"):
        normalize_label_map(labels)


def test_repeated_identical_labels_remain_valid() -> None:
    labels = pd.DataFrame(
        {
            "sequence_id": ["0001", "0001", "0002"],
            "uav_type": ["0", "0", "1"],
        }
    )

    assert normalize_label_map(labels) == {"0001": "0", "0002": "1"}


def test_csv_label_loader_does_not_swallow_conflicts(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.csv"
    _conflicting_labels().to_csv(labels_path, index=False)

    with pytest.raises(ValueError, match=r"conflicting labels.*0001"):
        _load_labels_preserving_ids(labels_path)
