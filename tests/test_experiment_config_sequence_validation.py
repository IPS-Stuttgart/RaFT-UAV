from __future__ import annotations

import json

import pytest

from raft_uav.experiments.config import ExperimentConfig


@pytest.mark.parametrize("field_name", ["flights", "methods", "options"])
@pytest.mark.parametrize(
    "invalid_value",
    [
        pytest.param({"Opt1": True, "Opt2": False}, id="mapping"),
        pytest.param(3, id="non-iterable-scalar"),
    ],
)
def test_from_mapping_rejects_malformed_sequence_fields(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{field_name} must be a string, a non-mapping iterable, or None",
    ):
        ExperimentConfig.from_mapping({field_name: invalid_value})


def test_load_rejects_mapping_valued_flights(tmp_path) -> None:
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps({"flights": {"Opt1": True, "Opt2": False}}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"flights must be a string, a non-mapping iterable, or None",
    ):
        ExperimentConfig.load(path)
