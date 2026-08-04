from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_sequence_gate_fit import (
    _template_for_apply_estimates,
)


def _estimates(*sequence_ids: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": list(sequence_ids),
            "time_s": [0.0] * len(sequence_ids),
        }
    )


def _template(*sequence_ids: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": list(sequence_ids),
            "Timestamp": [0.0] * len(sequence_ids),
        }
    )


def test_apply_template_rejects_mismatched_estimate_sequence_sets() -> None:
    with pytest.raises(ValueError, match="identical sequence sets") as error:
        _template_for_apply_estimates(
            _template("seqA", "seqB", "seqC"),
            _estimates("seqA", "seqB"),
            _estimates("seqA", "seqC"),
        )

    message = str(error.value)
    assert "missing from alternate: 'seqB'" in message
    assert "missing from base: 'seqC'" in message


def test_apply_template_rejects_sequences_missing_from_template() -> None:
    with pytest.raises(ValueError, match="template is missing") as error:
        _template_for_apply_estimates(
            _template("seqA"),
            _estimates("seqA", "seqB"),
            _estimates("seqA", "seqB"),
        )

    assert "'seqB'" in str(error.value)


def test_apply_template_keeps_every_complete_apply_sequence() -> None:
    rows = _template_for_apply_estimates(
        _template("train", "seqB", "seqA", "seqA"),
        _estimates("seqA", "seqB"),
        _estimates("seqB", "seqA"),
    )

    assert rows["Sequence"].tolist() == ["seqB", "seqA", "seqA"]
