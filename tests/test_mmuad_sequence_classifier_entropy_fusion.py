from __future__ import annotations

import json

import pandas as pd
import pytest

from raft_uav.mmuad.sequence_classifier_entropy_fusion import (
    EntropyAdaptiveFusionConfig,
    fuse_entropy_adaptive_probabilities,
    main as entropy_fusion_main,
    select_entropy_adaptive_fusion,
)


PROBABILITY_COLUMNS = [f"predicted_probability_{label}" for label in range(4)]


def _probability_rows(records: dict[str, tuple[float, float, float, float]]) -> pd.DataFrame:
    rows = []
    for sequence_id, probabilities in records.items():
        row = {"sequence_id": sequence_id}
        row.update(dict(zip(PROBABILITY_COLUMNS, probabilities, strict=True)))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def test_entropy_adaptation_favors_more_confident_modality() -> None:
    image = _probability_rows({"seq": (0.90, 0.05, 0.03, 0.02)})
    nonimage = _probability_rows({"seq": (0.24, 0.26, 0.25, 0.25)})

    fused = fuse_entropy_adaptive_probabilities(
        image,
        nonimage,
        config=EntropyAdaptiveFusionConfig(
            prior_image_weight=0.2,
            adaptation_strength=1.0,
            entropy_power=1.0,
        ),
    ).iloc[0]

    assert fused["image_weight_effective"] > 0.9
    assert fused["predicted_class"] == "0"
    assert fused["image_entropy"] < fused["nonimage_entropy"]


def test_zero_adaptation_recovers_global_linear_fusion() -> None:
    image = _probability_rows({"seq": (0.8, 0.1, 0.05, 0.05)})
    nonimage = _probability_rows({"seq": (0.1, 0.7, 0.1, 0.1)})

    fused = fuse_entropy_adaptive_probabilities(
        image,
        nonimage,
        config=EntropyAdaptiveFusionConfig(
            prior_image_weight=0.25,
            adaptation_strength=0.0,
        ),
    ).iloc[0]

    assert fused["image_weight_effective"] == pytest.approx(0.25)
    expected = 0.25 * 0.8 + 0.75 * 0.1
    assert fused["predicted_probability_0"] == pytest.approx(expected)


def test_missing_modality_uses_available_probabilities() -> None:
    image = _probability_rows({"image-only": (0.1, 0.8, 0.05, 0.05)})
    nonimage = _probability_rows({"nonimage-only": (0.1, 0.1, 0.7, 0.1)})

    fused = fuse_entropy_adaptive_probabilities(image, nonimage).set_index("sequence_id")

    assert fused.loc["image-only", "image_weight_effective"] == pytest.approx(1.0)
    assert fused.loc["image-only", "predicted_class"] == "1"
    assert fused.loc["nonimage-only", "image_weight_effective"] == pytest.approx(0.0)
    assert fused.loc["nonimage-only", "predicted_class"] == "2"


def test_oof_selection_can_choose_sequence_adaptation() -> None:
    image = _probability_rows(
        {
            "a": (0.95, 0.02, 0.02, 0.01),
            "b": (0.25, 0.25, 0.25, 0.25),
        }
    )
    nonimage = _probability_rows(
        {
            "a": (0.25, 0.25, 0.25, 0.25),
            "b": (0.02, 0.95, 0.02, 0.01),
        }
    )
    labels = {"a": "0", "b": "1"}

    result = select_entropy_adaptive_fusion(
        image_oof_probabilities=image,
        nonimage_oof_probabilities=nonimage,
        train_labels=labels,
        image_predict_probabilities=image,
        nonimage_predict_probabilities=nonimage,
        prior_image_weights=[0.5],
        adaptation_strengths=[0.0, 1.0],
        entropy_powers=[1.0],
        selection_metric="log_loss",
    )

    selected = result.manifest["selected"]
    assert selected["adaptation_strength"] == pytest.approx(1.0)
    assert selected["accuracy"] == pytest.approx(1.0)
    assert result.selected_probabilities["predicted_class"].tolist() == ["0", "1"]


def test_entropy_fusion_cli_writes_artifacts(tmp_path) -> None:
    image = _probability_rows({"a": (0.9, 0.05, 0.03, 0.02), "b": (0.25,) * 4})
    nonimage = _probability_rows({"a": (0.25,) * 4, "b": (0.05, 0.9, 0.03, 0.02)})
    labels = pd.DataFrame({"sequence_id": ["a", "b"], "uav_type": ["0", "1"]})
    paths = {}
    for name, rows in (("image", image), ("nonimage", nonimage)):
        path = tmp_path / f"{name}.csv"
        rows.to_csv(path, index=False)
        paths[name] = path
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)
    output_dir = tmp_path / "output"

    status = entropy_fusion_main(
        [
            "--image-oof-probabilities",
            str(paths["image"]),
            "--nonimage-oof-probabilities",
            str(paths["nonimage"]),
            "--image-predict-probabilities",
            str(paths["image"]),
            "--nonimage-predict-probabilities",
            str(paths["nonimage"]),
            "--train-labels",
            str(labels_path),
            "--output-dir",
            str(output_dir),
            "--prior-image-weight-grid",
            "0.5",
            "--adaptation-strength-grid",
            "0,1",
            "--entropy-power-grid",
            "1",
            "--selection-metric",
            "log_loss",
        ]
    )

    assert status == 0
    manifest = json.loads(
        (output_dir / "mmuad_entropy_adaptive_fusion.json").read_text(encoding="utf-8")
    )
    assert manifest["selected"]["adaptation_strength"] == pytest.approx(1.0)
    assert (output_dir / "mmuad_entropy_adaptive_fusion_cv_summary.csv").exists()
    assert (output_dir / "mmuad_entropy_adaptive_fusion_probabilities.csv").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prior_image_weight", -0.1),
        ("adaptation_strength", 1.1),
        ("entropy_power", 0.0),
        ("probability_floor", float("nan")),
        ("min_image_weight", 0.8),
    ],
)
def test_entropy_fusion_rejects_invalid_controls(field: str, value: float) -> None:
    kwargs = {field: value}
    if field == "min_image_weight":
        kwargs["max_image_weight"] = 0.2
    with pytest.raises(ValueError):
        fuse_entropy_adaptive_probabilities(
            _probability_rows({"seq": (0.25,) * 4}),
            _probability_rows({"seq": (0.25,) * 4}),
            config=EntropyAdaptiveFusionConfig(**kwargs),
        )
