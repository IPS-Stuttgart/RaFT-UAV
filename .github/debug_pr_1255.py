from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

with TemporaryDirectory() as directory:
    path = Path(directory) / "rows.csv"
    path.write_text(
        "sequence_id,time_s,x_m\n001,0.0,1.0\n",
        encoding="utf-8",
    )
    print("pandas_version=", pd.__version__)
    for label, kwargs in (
        ("default", {}),
        ("dtype_str", {"dtype": str, "keep_default_na": False}),
        ("dtype_object", {"dtype": object, "keep_default_na": False}),
        ("dtype_string", {"dtype": "string", "keep_default_na": False}),
        (
            "converter",
            {
                "converters": {"sequence_id": lambda value: value},
                "keep_default_na": False,
            },
        ),
    ):
        frame = pd.read_csv(path, **kwargs)
        print(label, frame.to_dict("records"), frame.dtypes.astype(str).to_dict())
