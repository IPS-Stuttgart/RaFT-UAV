from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import raft_uav.mmuad.candidate_reservoir as reservoir_module

main = reservoir_module.main
closure = {
    name: cell.cell_contents
    for name, cell in zip(main.__code__.co_freevars, main.__closure__ or (), strict=True)
}
original_read_csv = closure["_pd"].read_csv
original_build_oracle = closure["_candidate_reservoir"].build_oracle_recall_tables


def traced_read_csv(*args, **kwargs):
    result = original_read_csv(*args, **kwargs)
    print("read_csv_call=", args, kwargs)
    print("read_csv_result=", result.to_dict("records"), result.dtypes.astype(str).to_dict())
    return result


def traced_build_oracle(reservoir, truth, **kwargs):
    print("oracle_reservoir=", reservoir[["sequence_id", "time_s"]].to_dict("records"))
    print("oracle_truth=", truth[["sequence_id", "time_s"]].to_dict("records"))
    return original_build_oracle(reservoir, truth, **kwargs)


with TemporaryDirectory() as directory:
    root = Path(directory)
    first_csv = root / "first.csv"
    candidate_csv = root / "candidates.csv"
    truth_csv = root / "truth.csv"
    output_csv = root / "reservoir.csv"
    oracle_csv = root / "oracle.csv"
    text = (
        "sequence_id,time_s,x_m,y_m,z_m,confidence\n"
        "001,0.0,1.0,2.0,3.0,0.9\n"
    )
    first_csv.write_text(text, encoding="utf-8")
    candidate_csv.write_text(text, encoding="utf-8")
    truth_csv.write_text(text, encoding="utf-8")

    first = reservoir_module.load_candidate_inputs([f"raw={first_csv}"])
    print("first_loader=", first[["sequence_id", "time_s"]].to_dict("records"))
    print("direct_after_loader=", original_read_csv(truth_csv, dtype=str, keep_default_na=False).to_dict("records"))

    closure["_pd"].read_csv = traced_read_csv
    closure["_candidate_reservoir"].build_oracle_recall_tables = traced_build_oracle
    try:
        result = main(
            [
                "--candidate",
                f"raw={candidate_csv}",
                "--output-csv",
                str(output_csv),
                "--truth-csv",
                str(truth_csv),
                "--oracle-frame-csv",
                str(oracle_csv),
            ]
        )
        print("result=", result)
        print("oracle_csv_repr=", repr(oracle_csv.read_text(encoding="utf-8")))
        print("global_reader_restored=", pd.read_csv is original_read_csv)
    finally:
        closure["_pd"].read_csv = original_read_csv
        closure["_candidate_reservoir"].build_oracle_recall_tables = original_build_oracle
