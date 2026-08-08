from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.classification_audit import (
    _sequence_id_text,
    build_mmuad_classification_audit,
)


def _rows(sequence: object) -> pd.DataFrame:
    return pd.DataFrame(
        [(sequence, 0.0, 0)],
        columns=["Sequence", "Timestamp", "Classification"],
    )


@pytest.mark.parametrize("sequence", [None, np.nan, pd.NA, "", "   "])
def test_classification_audit_rejects_missing_or_blank_sequence_ids(sequence: object) -> None:
    with pytest.raises(ValueError, match="Sequence"):
        build_mmuad_classification_audit(truth=_rows(sequence), results=_rows("seq-a"))


@pytest.mark.parametrize(
    "sequence",
    [np.ma.masked, np.ma.array("seq-a", mask=True)],
)
def test_classification_audit_rejects_masked_sequence_ids(sequence: object) -> None:
    with pytest.raises(ValueError, match="Sequence"):
        _sequence_id_text(sequence, position=0)


def test_classification_audit_normalizes_sequence_whitespace_before_alignment() -> None:
    audit = build_mmuad_classification_audit(
        truth=_rows("seq-a"),
        results=_rows("  seq-a  "),
    )

    assert audit.summary["current_accuracy"] == pytest.approx(1.0)
    assert audit.summary["matched_row_count"] == 1
    assert audit.summary["missing_prediction_row_count"] == 0
    assert audit.summary["extra_prediction_row_count"] == 0
    assert audit.summary["row_key_parity"] is True
    assert audit.sequence_class_summary["sequence"].tolist() == ["seq-a"]


def test_classification_audit_preserves_literal_nan_sequence_id() -> None:
    audit = build_mmuad_classification_audit(
        truth=_rows("nan"),
        results=_rows("nan"),
    )

    assert audit.summary["current_accuracy"] == pytest.approx(1.0)
    assert audit.summary["matched_row_count"] == 1
    assert audit.sequence_class_summary["sequence"].tolist() == ["nan"]
