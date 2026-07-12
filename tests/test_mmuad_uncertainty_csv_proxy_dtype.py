from io import StringIO

import pandas as pd

from raft_uav.mmuad.track5_uncertainty_ensemble import _PandasCsvProxy


def test_uncertainty_csv_proxy_preserves_explicit_selective_dtype_mapping():
    rows = _PandasCsvProxy(pd).read_csv(
        StringIO("Sequence,time_s,state_x_m\n001,1.5,2.5\n"),
        dtype={"Sequence": "string"},
        keep_default_na=False,
    )

    assert rows.loc[0, "Sequence"] == "001"
    assert rows["Sequence"].dtype == object
    assert pd.api.types.is_float_dtype(rows["time_s"])
    assert pd.api.types.is_float_dtype(rows["state_x_m"])
