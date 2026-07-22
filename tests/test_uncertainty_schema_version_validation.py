import numpy as np
import pytest

from raft_uav.uncertainty import HeteroscedasticUncertaintyModel


@pytest.mark.parametrize(
    "schema_version",
    [
        True,
        1.5,
        np.nan,
        np.inf,
        1.0 + 0.0j,
        np.array([1]),
        np.ma.masked,
    ],
)
def test_from_dict_rejects_non_exact_schema_versions(schema_version):
    with pytest.raises(
        ValueError,
        match="schema_version must be an exact integer scalar",
    ):
        HeteroscedasticUncertaintyModel.from_dict(
            {"schema_version": schema_version, "metadata": {}, "heads": []}
        )


def test_from_dict_preserves_unsupported_exact_schema_error():
    with pytest.raises(ValueError, match="unsupported uncertainty schema 2"):
        HeteroscedasticUncertaintyModel.from_dict(
            {"schema_version": 2, "metadata": {}, "heads": []}
        )


def test_from_dict_accepts_scalar_like_schema_one():
    model = HeteroscedasticUncertaintyModel.from_dict(
        {"schema_version": np.array("1"), "metadata": {}, "heads": []}
    )
    assert model.heads == ()
