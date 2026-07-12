from __future__ import annotations

import json

import numpy as np
import pandas as pd

from raft_uav.mmuad.class_probability_calibration import (
    apply_temperature_calibrator,
    fit_temperature_calibrator,
    load_calibrator,
    main as calibration_main,
    save_calibrator,
    temperature_scale_probabilities,
)


_CLASS_PROBABILITIES = {
    "class_prob_0": [
        0.97,
        0.96,
        0.94,
        0.96,
        0.02,
        0.02,
        0.02,
        0.02,
        0.97,
        0.02,
        0.02,
        0.02,
    ],
    "class_prob_1": [
        0.01,
        0.02,
        0.02,
        0.02,
        0.96,
        0.95,
        0.02,
        0.02,
        0.01,
        0.94,
        0.02,
        0.02,
    ],
    "class_prob_2": [
        0.01,
        0.01,
        0.03,
        0.01,
        0.01,
        0.02,
        0.95,
        0.94,
        0.01,
        0.02,
        0.94,
        0.02,
    ],
    "class_prob_3": [
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.02,
        0.01,
        0.02,
        0.02,
        0.94,
    ],
}


def _overconfident_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": [f"{index:04d}" for index in range(12)],
            **_CLASS_PROBABILITIES,
        }
    )


def _labels() -> dict[str, str]:
    return {
        "0000": "0",
        "0001": "0",
        "0002": "0",
        "0003": "1",
        "0004": "1",
        "0005": "1",
        "0006": "2",
        "0007": "3",
        "0008": "0",
        "0009": "1",
        "0010": "2",
        "0011": "3",
    }


def test_temperature_fit_softens_overconfident_probabilities_and_improves_nll() -> None:
    model, summary = fit_temperature_calibrator(
        _overconfident_predictions(),
        _labels(),
        min_temperature=0.1,
        max_temperature=20.0,
    )

    assert model.temperature > 1.0
    assert summary["after"]["nll"] < summary["before"]["nll"]
    assert summary["after"]["accuracy"] == summary["before"]["accuracy"]

    raw = _overconfident_predictions()
    calibrated = apply_temperature_calibrator(raw, model)
    calibrated_columns = [
        f"calibrated_class_prob_{index}" for index in range(4)
    ]
    raw_columns = [f"class_prob_{index}" for index in range(4)]
    values = calibrated[calibrated_columns].to_numpy(float)
    assert np.allclose(values.sum(axis=1), 1.0)
    assert np.array_equal(
        np.argmax(values, axis=1),
        np.argmax(raw[raw_columns].to_numpy(float), axis=1),
    )
    entropy_increased = (
        calibrated["class_probability_entropy_calibrated"]
        > calibrated["class_probability_entropy_raw"]
    )
    assert entropy_increased.mean() > 0.9


def test_apply_can_replace_probabilities_while_preserving_raw_columns() -> None:
    model, _summary = fit_temperature_calibrator(
        _overconfident_predictions(),
        _labels(),
    )

    calibrated = apply_temperature_calibrator(
        _overconfident_predictions(),
        model,
        replace_probabilities=True,
    )

    for index in range(4):
        assert f"raw_class_prob_{index}" in calibrated.columns
    class_columns = [f"class_prob_{index}" for index in range(4)]
    assert np.allclose(calibrated[class_columns].sum(axis=1), 1.0)
    assert not np.allclose(
        calibrated["class_prob_0"],
        calibrated["raw_class_prob_0"],
    )


def test_temperature_scale_rejects_invalid_temperature() -> None:
    probabilities = np.asarray([[0.5, 0.5]], dtype=float)

    for temperature in (0.0, -1.0, float("nan")):
        try:
            temperature_scale_probabilities(
                probabilities,
                temperature=temperature,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid temperature did not raise ValueError")


def test_calibrator_json_roundtrip(tmp_path) -> None:
    model, _summary = fit_temperature_calibrator(
        _overconfident_predictions(),
        _labels(),
    )
    model_path = tmp_path / "calibrator.json"

    save_calibrator(model, model_path)
    loaded = load_calibrator(model_path)

    assert loaded == model


def test_cli_fit_and_apply_preserve_zero_padded_sequence_ids(tmp_path) -> None:
    predictions_path = tmp_path / "oof_predictions.csv"
    labels_path = tmp_path / "labels.csv"
    model_path = tmp_path / "calibrator.json"
    fit_output_path = tmp_path / "train_calibrated.csv"
    apply_output_path = tmp_path / "val_calibrated.csv"
    summary_path = tmp_path / "summary.json"
    _overconfident_predictions().to_csv(predictions_path, index=False)
    pd.DataFrame(
        {
            "sequence_id": list(_labels()),
            "uav_type": list(_labels().values()),
        }
    ).to_csv(labels_path, index=False)

    fit_rc = calibration_main(
        [
            "fit",
            "--predictions-csv",
            str(predictions_path),
            "--labels-csv",
            str(labels_path),
            "--model-json",
            str(model_path),
            "--output-csv",
            str(fit_output_path),
            "--summary-json",
            str(summary_path),
        ]
    )
    apply_rc = calibration_main(
        [
            "apply",
            "--predictions-csv",
            str(predictions_path),
            "--model-json",
            str(model_path),
            "--output-csv",
            str(apply_output_path),
        ]
    )

    assert fit_rc == 0
    assert apply_rc == 0
    output = pd.read_csv(apply_output_path, dtype=str)
    assert output["sequence_id"].tolist()[:2] == ["0000", "0001"]
    assert "calibrated_class_prob_0" in output.columns
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["matched_sequence_count"] == 12
    assert summary["after"]["nll"] < summary["before"]["nll"]
