from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import raft_uav.mmuad.candidate_reservoir as reservoir_module

main_globals = reservoir_module._ORIGINAL_MAIN.__globals__
original_normalize_truth = main_globals["normalize_truth_columns"]
original_build_oracle = main_globals["build_oracle_recall_tables"]
original_proxy_read_csv = reservoir_module._TextPreservingPandasProxy.read_csv


def debug_proxy_read_csv(self, *args, **kwargs):
    print("proxy_read_called=", args, kwargs)
    result = original_proxy_read_csv(self, *args, **kwargs)
    print("proxy_read_result=", result.to_dict("records"))
    print("proxy_read_dtypes=", result.dtypes.astype(str).to_dict())
    return result


def debug_normalize_truth(frame, *args, **kwargs):
    print("raw_truth=", frame.to_dict("records"))
    print("raw_truth_dtypes=", frame.dtypes.astype(str).to_dict())
    result = original_normalize_truth(frame, *args, **kwargs)
    print("normalized_truth=", result.to_dict("records"))
    print("normalized_truth_dtypes=", result.dtypes.astype(str).to_dict())
    return result


def debug_build_oracle(reservoir, truth, **kwargs):
    print("reservoir_keys=", reservoir[["sequence_id", "time_s"]].to_dict("records"))
    print("truth_keys=", truth[["sequence_id", "time_s"]].to_dict("records"))
    return original_build_oracle(reservoir, truth, **kwargs)


reservoir_module._TextPreservingPandasProxy.read_csv = debug_proxy_read_csv
main_globals["normalize_truth_columns"] = debug_normalize_truth
main_globals["build_oracle_recall_tables"] = debug_build_oracle
try:
    print("globals_are_impl=", main_globals is reservoir_module._IMPL.__dict__)
    print("wrapper_is_original=", reservoir_module.main is reservoir_module._ORIGINAL_MAIN)
    print("wrapper_code=", reservoir_module.main.__code__.co_filename, reservoir_module.main.__code__.co_firstlineno)
    print("normalize_patched=", main_globals["normalize_truth_columns"] is debug_normalize_truth)
    print("builder_patched=", main_globals["build_oracle_recall_tables"] is debug_build_oracle)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        candidate_csv = root / "candidates.csv"
        truth_csv = root / "truth.csv"
        output_csv = root / "reservoir.csv"
        oracle_csv = root / "oracle.csv"
        text = (
            "sequence_id,time_s,x_m,y_m,z_m,confidence\n"
            "001,0.0,1.0,2.0,3.0,0.9\n"
        )
        candidate_csv.write_text(text, encoding="utf-8")
        truth_csv.write_text(text, encoding="utf-8")
        result = reservoir_module.main(
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
        print("global_value=", pd.read_csv(candidate_csv).loc[0, "confidence"])
finally:
    reservoir_module._TextPreservingPandasProxy.read_csv = original_proxy_read_csv
    main_globals["normalize_truth_columns"] = original_normalize_truth
    main_globals["build_oracle_recall_tables"] = original_build_oracle
