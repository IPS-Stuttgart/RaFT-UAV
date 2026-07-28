from __future__ import annotations

import pytest

from raft_uav.uncertainty import HeteroscedasticUncertaintyModel


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "model_type": "heteroscedastic-loglinear-variance",
        "metadata": {},
        "heads": [],
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(None, id="none"),
        pytest.param(False, id="false"),
        pytest.param(0, id="zero"),
        pytest.param("", id="empty-string"),
        pytest.param([], id="empty-list"),
        pytest.param([("source", "fit")], id="pair-list"),
    ],
)
def test_from_dict_rejects_non_mapping_metadata(metadata: object) -> None:
    with pytest.raises(
        ValueError,
        match="uncertainty model metadata must be a mapping",
    ):
        HeteroscedasticUncertaintyModel.from_dict(_payload(metadata=metadata))


@pytest.mark.parametrize(
    "heads",
    [
        pytest.param(None, id="none"),
        pytest.param(False, id="false"),
        pytest.param(0, id="zero"),
        pytest.param("", id="empty-string"),
        pytest.param(b"", id="empty-bytes"),
        pytest.param({}, id="empty-mapping"),
    ],
)
def test_from_dict_rejects_non_iterable_or_mapping_heads(heads: object) -> None:
    with pytest.raises(
        ValueError,
        match="uncertainty model heads must be an iterable of mappings",
    ):
        HeteroscedasticUncertaintyModel.from_dict(_payload(heads=heads))


def test_from_dict_keeps_valid_empty_model_payload() -> None:
    model = HeteroscedasticUncertaintyModel.from_dict(_payload())

    assert model.heads == ()
    assert model.metadata == {}
