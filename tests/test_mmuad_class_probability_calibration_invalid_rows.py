from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.class_probability_calibration import (
    MODEL_SCHEMA,
    ClassProbabilityCalibrator,
    apply_temperature_calibrator,
)


def test_apply_marks_only_valid_rows_and_preserves_invalid_source_values() -> None:
    source_columns = ["class_prob_0", "class_prob_1"]
    predictions = pd.DataFrame(
        {
            "sequence_id": ["valid", "zero-sum", "non-finite"],
            "class_prob_0": [0.9, 0.0, np.nan],
            "class_prob_1": [0.1, 0.0, 0.2],
        }
    )
    model = ClassProbabilityCalibrator(
        schema=MODEL_SCHEMA,
        method="temperature",
        temperature=2.0,
        class_labels=["0", "1"],
        source_probability_columns=source_columns,
    )

    calibrated = apply_temperature_calibrator(
        predictions,
        model,
        replace_probabilities=True,
    )

    assert calibrated["class_probability_calibrated"].tolist() == [True, False, False]
    calibrated_columns = ["calibrated_class_prob_0", "calibrated_class_prob_1"]
    assert np.isclose(calibrated.loc[0, calibrated_columns].sum(), 1.0)
    assert calibrated.loc[[1, 2], calibrated_columns].isna().all().all()

    pd.testing.assert_frame_equal(
        calibrated.loc[[1, 2], source_columns],
        predictions.loc[[1, 2], source_columns],
        check_dtype=False,
    )
    raw_columns = [f"raw_{column}" for column in source_columns]
    expected_raw = predictions[source_columns].copy()
    expected_raw.columns = raw_columns
    pd.testing.assert_frame_equal(
        calibrated[raw_columns],
        expected_raw,
        check_dtype=False,
    )
