from __future__ import annotations

from pathlib import Path
import sys
import zipfile

import pytest

from raft_uav.multi_uav_lts.trackeval_bridge import (
    TrackEvalBridgeError,
    evaluate_lts_with_trackeval,
)


def _row(frame: int, object_id: int, x: float = 0.0) -> str:
    return f"{frame},{object_id},{x},0,10,10,1,1,1\n"


def _write_fake_trackeval(root: Path) -> Path:
    package = root / "trackeval"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        '''from pathlib import Path
import numpy as np


class _Metric:
    def __init__(self, config):
        self.config = config


class HOTA(_Metric):
    pass


class CLEAR(_Metric):
    pass


class Identity(_Metric):
    pass


class _Metrics:
    HOTA = HOTA
    CLEAR = CLEAR
    Identity = Identity


metrics = _Metrics()


class MotChallenge2DBox:
    @staticmethod
    def get_default_dataset_config():
        return {}

    def __init__(self, config):
        self.config = config

    @staticmethod
    def get_name():
        return "MotChallenge2DBox"


class _Datasets:
    MotChallenge2DBox = MotChallenge2DBox


datasets = _Datasets()


def _node(value):
    alpha = np.arange(0.05, 0.99, 0.05)
    return {
        "HOTA": {
            "HOTA": np.full(alpha.shape, value),
            "DetA": np.full(alpha.shape, value - 0.1),
            "AssA": np.full(alpha.shape, value - 0.2),
            "LocA": np.full(alpha.shape, value + 0.1),
        },
        "CLEAR": {"MOTA": value - 0.3},
        "Identity": {"IDF1": value - 0.15},
    }


class Evaluator:
    @staticmethod
    def get_default_eval_config():
        return {}

    def __init__(self, config):
        self.config = config

    def evaluate(self, dataset_list, metrics_list):
        dataset = dataset_list[0]
        tracker = dataset.config["TRACKERS_TO_EVAL"][0]
        tracker_root = Path(dataset.config["TRACKERS_FOLDER"]) / tracker / "data"
        gt_root = Path(dataset.config["GT_FOLDER"])
        raw = {}
        for index, sequence in enumerate(sorted(dataset.config["SEQ_INFO"])):
            assert (tracker_root / f"{sequence}.txt").is_file()
            assert (gt_root / sequence / "gt" / "gt.txt").is_file()
            raw[sequence] = {"pedestrian": _node(0.7 + 0.01 * index)}
        raw["COMBINED_SEQ"] = {"pedestrian": _node(0.8)}
        return (
            {"MotChallenge2DBox": {tracker: raw}},
            {"MotChallenge2DBox": {tracker: "Success"}},
        )
''',
        encoding="utf-8",
    )
    return root


def _clear_fake_trackeval() -> None:
    for name in list(sys.modules):
        if name == "trackeval" or name.startswith("trackeval."):
            del sys.modules[name]


def test_trackeval_bridge_scores_directory_and_materializes_missing_predictions(
    tmp_path: Path,
) -> None:
    truth = tmp_path / "truth"
    predictions = tmp_path / "predictions"
    truth.mkdir()
    predictions.mkdir()
    (truth / "A.txt").write_text(_row(1, 1) + _row(2, 1, 1.0))
    (truth / "B.txt").write_text(_row(1, 5))
    (predictions / "A.txt").write_text(_row(1, 1))
    trackeval_root = _write_fake_trackeval(tmp_path / "fake_trackeval")
    work = tmp_path / "work"

    try:
        report = evaluate_lts_with_trackeval(
            predictions,
            truth,
            trackeval_root=trackeval_root,
            tracker_name="candidate",
            work_dir=work,
        )
    finally:
        _clear_fake_trackeval()

    assert report.sequence_count == 2
    assert report.combined.hota == pytest.approx(0.8)
    assert report.combined.deta == pytest.approx(0.7)
    assert report.combined.assa == pytest.approx(0.6)
    assert report.combined.mota == pytest.approx(0.5)
    assert report.combined.idf1 == pytest.approx(0.65)
    assert set(report.sequences) == {"A", "B"}
    assert (work / "trackers" / "candidate" / "data" / "B.txt").read_text() == ""


def test_trackeval_bridge_accepts_prediction_zip(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    truth.mkdir()
    (truth / "A.txt").write_text(_row(1, 1))
    archive = tmp_path / "submission.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("A.txt", _row(1, 1))
    trackeval_root = _write_fake_trackeval(tmp_path / "fake_trackeval")

    try:
        report = evaluate_lts_with_trackeval(
            archive,
            truth,
            trackeval_root=trackeval_root,
            sequences=["A"],
        )
    finally:
        _clear_fake_trackeval()

    assert report.sequence_count == 1
    assert report.sequences["A"].hota == pytest.approx(0.7)


def test_trackeval_bridge_rejects_prediction_frames_beyond_truth(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    predictions = tmp_path / "predictions"
    truth.mkdir()
    predictions.mkdir()
    (truth / "A.txt").write_text(_row(1, 1))
    (predictions / "A.txt").write_text(_row(2, 1))
    trackeval_root = _write_fake_trackeval(tmp_path / "fake_trackeval")

    with pytest.raises(TrackEvalBridgeError, match="exceeds truth frame"):
        evaluate_lts_with_trackeval(
            predictions,
            truth,
            trackeval_root=trackeval_root,
        )


def test_trackeval_bridge_rejects_duplicate_frame_object_rows(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    predictions = tmp_path / "predictions"
    truth.mkdir()
    predictions.mkdir()
    (truth / "A.txt").write_text(_row(1, 1) + _row(1, 1))
    trackeval_root = _write_fake_trackeval(tmp_path / "fake_trackeval")

    with pytest.raises(TrackEvalBridgeError, match="duplicate frame/object"):
        evaluate_lts_with_trackeval(
            predictions,
            truth,
            trackeval_root=trackeval_root,
        )


def test_trackeval_bridge_uses_image_count_for_trailing_empty_truth_frames(
    tmp_path: Path,
) -> None:
    truth = tmp_path / "truth"
    predictions = tmp_path / "predictions"
    sequences = tmp_path / "images"
    truth.mkdir()
    predictions.mkdir()
    (sequences / "A").mkdir(parents=True)
    (truth / "A.txt").write_text(_row(1, 1))
    (predictions / "A.txt").write_text(_row(3, 1))
    for frame in range(3):
        (sequences / "A" / f"{frame:05d}.jpg").write_bytes(b"frame")
    trackeval_root = _write_fake_trackeval(tmp_path / "fake_trackeval")
    work = tmp_path / "work"

    try:
        report = evaluate_lts_with_trackeval(
            predictions,
            truth,
            trackeval_root=trackeval_root,
            sequence_root=sequences,
            work_dir=work,
        )
    finally:
        _clear_fake_trackeval()

    assert report.sequence_root == str(sequences.resolve())
    assert report.sequence_count == 1


def test_trackeval_bridge_does_not_delete_nonempty_work_directory(
    tmp_path: Path,
) -> None:
    truth = tmp_path / "truth"
    predictions = tmp_path / "predictions"
    work = tmp_path / "work"
    truth.mkdir()
    predictions.mkdir()
    work.mkdir()
    sentinel = work / "keep.txt"
    sentinel.write_text("do not delete")
    (truth / "A.txt").write_text(_row(1, 1))
    trackeval_root = _write_fake_trackeval(tmp_path / "fake_trackeval")

    with pytest.raises(TrackEvalBridgeError, match="work directory must be empty"):
        evaluate_lts_with_trackeval(
            predictions,
            truth,
            trackeval_root=trackeval_root,
            work_dir=work,
        )
    assert sentinel.read_text() == "do not delete"
