"""Prevent fixed-population sweeps from deleting input data during best-copy output."""

from __future__ import annotations

from pathlib import Path


def reject_best_prediction_copy_aliases(
    target: Path,
    *,
    prediction_path: Path,
    truth_dir: Path,
    first_frame_label_dir: Path,
) -> None:
    """Reject inputs that the destructive best-prediction refresh would remove."""

    target_resolved = Path(target).resolve()
    inputs = (
        ("prediction path", Path(prediction_path)),
        ("truth directory", Path(truth_dir)),
        ("first-frame label directory", Path(first_frame_label_dir)),
    )
    for label, source in inputs:
        source_resolved = source.resolve()
        if source_resolved == target_resolved or source_resolved.is_relative_to(
            target_resolved
        ):
            raise ValueError(
                f"{label} must not be the best_predictions copy target or live inside it"
            )
