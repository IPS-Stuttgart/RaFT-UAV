from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import raft_uav.mmuad.candidate_reservoir as reservoir_module

main = reservoir_module.main
closure = {
    name: cell.cell_contents
    for name, cell in zip(main.__code__.co_freevars, main.__closure__ or (), strict=True)
}
print("main_file=", main.__code__.co_filename, main.__code__.co_firstlineno)
print("freevars=", sorted(closure))
print("pandas_version=", closure["_pd"].__version__)

with TemporaryDirectory() as directory:
    path = Path(directory) / "truth.csv"
    path.write_text(
        "sequence_id,time_s,x_m,y_m,z_m,confidence\n"
        "001,0.0,1.0,2.0,3.0,0.9\n",
        encoding="utf-8",
    )
    raw = closure["_pd"].read_csv(path, dtype=str, keep_default_na=False)
    print("closure_raw=", raw.to_dict("records"), raw.dtypes.astype(str).to_dict())
    normalized = closure["_normalize_truth_columns"](raw)
    print(
        "closure_normalized=",
        normalized.to_dict("records"),
        normalized.dtypes.astype(str).to_dict(),
    )
