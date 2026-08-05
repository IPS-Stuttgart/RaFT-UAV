from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.template_snap_write import _bool_column


def test_template_snap_boolean_diagnostics_keep_supported_serializations() -> None:
    rows = pd.DataFrame(
        {
            "valid": [
                True,
                False,
                "true",
                "false",
                "1.0",
                "0.0",
                "yes",
                "no",
                1,
                0,
                np.nan,
                pd.NA,
                np.ma.masked,
                np.array(1.0),
                np.array(0.0),
            ]
        }
    )

    assert _bool_column(rows, "valid").tolist() == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        False,
    ]


@pytest.mark.parametrize(
    "value",
    [
        2,
        -1,
        0.5,
        np.inf,
        "maybe",
        1 + 0j,
        np.complex64(0 + 0j),
        np.array(1 + 2j),
        np.array([1]),
    ],
)
def test_template_snap_boolean_diagnostics_reject_malformed_values(value: object) -> None:
    rows = pd.DataFrame({"valid": [value]}, index=[17])

    with pytest.raises(ValueError, match=r"valid contains .*Boolean value at row 17"):
        _bool_column(rows, "valid")


def test_template_snap_boolean_diagnostics_reject_cyclic_scalar_containers() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic
    rows = pd.DataFrame({"valid": [cyclic]}, index=[23])

    with pytest.raises(ValueError, match=r"valid contains a cyclic Boolean value at row 23"):
        _bool_column(rows, "valid")
