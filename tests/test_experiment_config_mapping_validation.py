from __future__ import annotations

import json

import pytest

from raft_uav.experiments.config import ExperimentConfig


@pytest.mark.parametrize(
    "field_name",
    ["environment", "calibration_artifacts", "metadata"],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        pytest.param(False, id="false"),
        pytest.param(0, id="zero"),
        pytest.param("", id="empty-string"),
        pytest.param([], id="empty-list"),
        pytest.param([("key", "value")], id="pair-list"),
    ],
)
def test_from_mapping_rejects_non_mapping_fields(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError, match=rf"{field_name} must be a mapping or None"):
        ExperimentConfig.from_mapping({field_name: invalid_value})


@pytest.mark.parametrize("field_name", ["environment", "calibration_artifacts", "metadata"])
def test_load_rejects_non_mapping_fields(tmp_path, field_name: str) -> None:
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps({field_name: False}), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{field_name} must be a mapping or None"):
        ExperimentConfig.load(path)


def test_from_mapping_accepts_missing_none_and_empty_mappings() -> None:
    missing = ExperimentConfig.from_mapping({})
    explicit_none = ExperimentConfig.from_mapping(
        {"environment": None, "calibration_artifacts": None, "metadata": None}
    )
    empty = ExperimentConfig.from_mapping(
        {"environment": {}, "calibration_artifacts": {}, "metadata": {}}
    )

    for config in (missing, explicit_none, empty):
        assert config.environment == {}
        assert config.calibration_artifacts == {}
        assert config.metadata == {}
