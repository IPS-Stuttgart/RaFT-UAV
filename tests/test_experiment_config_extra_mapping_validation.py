from __future__ import annotations

import pytest

from raft_uav.experiments.config import write_resolved_experiment_config


@pytest.mark.parametrize(
    "invalid_extra",
    [
        pytest.param(False, id="false"),
        pytest.param(0, id="zero"),
        pytest.param("", id="empty-string"),
        pytest.param([], id="empty-list"),
        pytest.param([("key", "value")], id="pair-list"),
    ],
)
def test_write_resolved_config_rejects_non_mapping_extra(
    tmp_path,
    invalid_extra: object,
) -> None:
    destination = tmp_path / "resolved.json"

    with pytest.raises(ValueError, match="extra must be a mapping or None"):
        write_resolved_experiment_config(
            destination,
            argv=["run-experiment"],
            env_prefixes=(),
            extra=invalid_extra,
        )

    assert not destination.exists()


@pytest.mark.parametrize("extra", [None, {}])
def test_write_resolved_config_accepts_absent_or_empty_extra(tmp_path, extra) -> None:
    destination = tmp_path / "resolved.json"

    resolved = write_resolved_experiment_config(
        destination,
        argv=["run-experiment"],
        env_prefixes=(),
        extra=extra,
    )

    assert destination.exists()
    assert "extra" not in resolved
