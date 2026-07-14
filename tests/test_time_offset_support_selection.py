import pandas as pd

from raft_uav.diagnostics.time_offset import best_offset_row


def test_best_offset_row_prefers_maximum_support_before_objective():
    sweep = pd.DataFrame(
        [
            {"tau_s": 0.0, "matched_count": 4, "mean_error_m": 2.0},
            {"tau_s": 1.0, "matched_count": 4, "mean_error_m": 1.0},
            {"tau_s": 10.0, "matched_count": 1, "mean_error_m": 0.0},
        ]
    )

    best = best_offset_row(sweep, objective="mean")

    assert float(best["tau_s"]) == 1.0
    assert int(best["matched_count"]) == 4
