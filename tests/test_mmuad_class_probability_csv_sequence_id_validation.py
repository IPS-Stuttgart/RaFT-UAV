from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.class_probability_csv import read_class_probability_csv


def test_class_probability_csv_compatibility_fallback_preserves_na_like_sequence_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probabilities_csv = tmp_path / "probabilities.csv"
    probabilities_csv.write_text(
        "sequence_id,predicted_probability_0\n"
        "NA,0.9\n"
        "N/A,0.8\n"
        "null,0.7\n",
        encoding="utf-8",
    )
    real_read_csv = pd.read_csv

    def compatibility_read_csv(*args, **kwargs):
        if "keep_default_na" in kwargs:
            raise TypeError("unsupported compatibility keyword")
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(
        "raft_uav.mmuad.class_probability_csv.pd.read_csv",
        compatibility_read_csv,
    )

    probabilities = read_class_probability_csv(probabilities_csv)

    assert probabilities["sequence_id"].tolist() == ["NA", "N/A", "null"]


def test_class_probability_csv_rejects_multiple_sequence_alias_columns(
    tmp_path: Path,
) -> None:
    probabilities_csv = tmp_path / "probabilities.csv"
    probabilities_csv.write_text(
        "scene,clip_id,predicted_probability_0\n"
        "scene-001,clip-001,0.9\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"ambiguous sequence identifier columns: 'scene', 'clip_id'",
    ):
        read_class_probability_csv(probabilities_csv)
