from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.classification_audit import build_mmuad_classification_audit


def _rows(classifications: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"] * len(classifications),
            "Timestamp": [1000.0 + index for index in range(len(classifications))],
            "Classification": classifications,
        }
    )


@pytest.mark.parametrize(
    "predicted_labels",
    [
        pytest.param([0, None], id="partially-missing"),
        pytest.param([None, None], id="entirely-missing"),
    ],
)
def test_classification_audit_rejects_missing_prediction_labels(
    predicted_labels: list[object],
) -> None:
    audit = build_mmuad_classification_audit(
        truth=_rows([0, 0]),
        results=_rows(predicted_labels),
    )

    assert audit.summary["valid_predicted_class_mapping"] is False
    sequence = audit.classification_audit.set_index("sequence").loc["seq0001"]
    assert bool(sequence["valid_submission_label"]) is False


def test_classification_audit_rejects_missing_truth_labels() -> None:
    audit = build_mmuad_classification_audit(
        truth=_rows([0, None]),
        results=_rows([0, 0]),
    )

    assert audit.summary["valid_truth_class_mapping"] is False
