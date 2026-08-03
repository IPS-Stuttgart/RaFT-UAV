from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from raft_uav.multi_uav_lts.upstream_fixes import (
    UpstreamPatchError,
    apply_upstream_fixes,
)


INFERENCE = '''import torch
import torch.backends.cudnn as cudnn


def run():
    if True:
        if opt.with_pos and idx == 0:
            prior_box = []

            with open(gt_path, "r") as file:
                for line in file:
                    values = line.strip().split(",")  # Split by comma
                    obj_id = int(values[0])  # Extract ID
                    prior_box.append([x1, y1, x2, y2, 1., 0.])

        if pred[0].numel() != 0:
            for det in pred:
                online_targets, slosts_targets = tracker.update(detections, im0)
                res_list.append([
                    idx, tid, round(tlwh[0], 2), round(tlwh[1], 2),
                    round(tlwh[2], 2), round(tlwh[3], 2), 1, 1, 1
                ])
        else:

            # ===== Detect from Files =====
            pass

if __name__ == '__main__':
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")
    opt.jde = False
    opt.ablation = False
'''

TRACKER = '''import numpy as np


class TrackState:
    Tracked = 1


class BaseTrack:
    _count = 0

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count


class STrack(BaseTrack):
    def __init__(self, tlbr=None, score=1.0, cls=0, feature=None):
        values = tlbr if tlbr is not None else [0.0, 0.0, 1.0, 1.0]
        self._tlwh = np.asarray(values, dtype=float)
        self.score = score
        self.cls = cls
        self.is_activated = False

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        return np.asarray(tlbr, dtype=float)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        return np.asarray(tlwh, dtype=float)

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xywh(self._tlwh)
        )

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id


class Tracker:
    def __init__(self):
        self.frame_id = 0
        self.track_low_thresh = 0.05
        self.tracked_stracks = []
        self.args = type(
            "Args",
            (),
            {
                "track_high_thresh": 0.3,
                "with_reid": False,
                "mot20": False,
            },
        )()

    def update(self, output_results, img):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            bboxes = output_results[:, :4]
            scores = output_results[:, 4]
            classes = output_results[:, 5]
            features = output_results[:, 6:]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]
            features = output_results[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]
            features_keep = features[remain_inds]
        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        if len(dets) > 0:
            if self.args.with_reid:
                detections = [
                    STrack(STrack.tlbr_to_tlwh(tlbr), score, cls, feature)
                    for tlbr, score, cls, feature in zip(
                        dets, scores_keep, classes_keep, features_keep
                    )
                ]
            else:
                detections = [
                    STrack(STrack.tlbr_to_tlwh(tlbr), score, cls)
                    for tlbr, score, cls in zip(dets, scores_keep, classes_keep)
                ]
        else:
            detections = []

        __ADD_TRACKLETS_COMMENT__
        return detections, initial_track_ids_keep

        if not self.args.mot20:
            first()
        if not self.args.mot20:
            second()

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks, []
'''.replace(
    "__ADD_TRACKLETS_COMMENT__",
    "''' Add newly detected tracklets to tracked_stracks'''",
)


class _Kalman:
    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return measurement.copy(), np.eye(len(measurement), dtype=float)


def _checkout(tmp_path: Path) -> Path:
    root = tmp_path / "BoT-SORT"
    (root / "tools").mkdir(parents=True)
    (root / "tracker").mkdir()
    (root / "tools" / "inference.py").write_text(INFERENCE, encoding="utf-8")
    (root / "tracker" / "mc_bot_sort.py").write_text(TRACKER, encoding="utf-8")
    return root


def _patched_tracker_namespace(root: Path) -> dict[str, object]:
    source = (root / "tracker" / "mc_bot_sort.py").read_text(encoding="utf-8")
    namespace: dict[str, object] = {}
    exec(compile(source, "mc_bot_sort.py", "exec"), namespace)
    return namespace


def test_applies_upstream_fixes_idempotently(tmp_path: Path) -> None:
    root = _checkout(tmp_path)

    first = apply_upstream_fixes(root)
    second = apply_upstream_fixes(root)

    assert first.changed_file_count == 2
    assert first.needs_update
    assert second.changed_file_count == 0
    assert not second.needs_update
    inference = (root / "tools" / "inference.py").read_text(encoding="utf-8")
    tracker = (root / "tracker" / "mc_bot_sort.py").read_text(encoding="utf-8")
    compile(inference, "inference.py", "exec")
    compile(tracker, "mc_bot_sort.py", "exec")
    assert "import numpy as np" in inference
    assert "tracker.update(np.empty((0, 6), dtype=np.float32), im0s)" in inference
    assert "obj_id = int(values[1])  # Extract object ID" in inference
    assert "frame_initial_track_ids.append(obj_id)" in inference
    assert "initial_track_ids=frame_initial_track_ids" in inference
    assert 'dest="fuse_score", default=True' in inference
    assert '"--no-fuse-score"' in inference
    assert "round(tlwh[0], 2)" not in inference
    assert "opt.mot20 = not opt.fuse_score" in inference
    assert tracker.count("if not self.args.mot20:") == 2
    assert "initial_track_ids may only be supplied on tracker frame 1" in tracker
    assert "BaseTrack._count = max(BaseTrack._count, forced_track_id)" in tracker
    assert "if track.is_activated" in tracker
    assert (root / "tools" / "inference.py.raft-uav-original").exists()
    assert (root / "tracker" / "mc_bot_sort.py.raft-uav-original").exists()


def test_forced_ids_are_assigned_and_reserve_the_allocator(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    apply_upstream_fixes(root)
    namespace = _patched_tracker_namespace(root)
    base_track = namespace["BaseTrack"]
    strack = namespace["STrack"]
    base_track._count = 0

    seeded = strack()
    seeded.activate(_Kalman(), 1, forced_track_id=np.int64(17))
    ordinary = strack()
    ordinary.activate(_Kalman(), 2)

    assert seeded.track_id == 17
    assert seeded.is_activated
    assert ordinary.track_id == 18


@pytest.mark.parametrize("bad_id", [0, -1, 1.5, True, np.bool_(False)])
def test_forced_track_id_rejects_invalid_values(
    tmp_path: Path, bad_id: object
) -> None:
    root = _checkout(tmp_path)
    apply_upstream_fixes(root)
    strack = _patched_tracker_namespace(root)["STrack"]

    with pytest.raises(ValueError, match="positive integer"):
        strack().activate(_Kalman(), 1, forced_track_id=bad_id)


def test_initial_ids_are_attached_to_corresponding_detections(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    apply_upstream_fixes(root)
    tracker = _patched_tracker_namespace(root)["Tracker"]()
    rows = np.asarray(
        [
            [1.0, 2.0, 4.0, 5.0, 1.0, 0.0],
            [10.0, 20.0, 14.0, 25.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    detections, ids = tracker.update(rows, None, initial_track_ids=[9, np.int64(3)])

    assert ids.tolist() == [9, 3]
    assert [detection.forced_track_id for detection in detections] == [9, 3]


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        ([1], "match the number"),
        ([1, 1], "unique"),
        ([1, 0], "positive integers"),
        ([1, 2.5], "positive integers"),
        ([1, True], "positive integers"),
    ],
)
def test_initial_id_validation(
    tmp_path: Path, ids: list[object], message: str
) -> None:
    root = _checkout(tmp_path)
    apply_upstream_fixes(root)
    tracker = _patched_tracker_namespace(root)["Tracker"]()
    rows = np.asarray(
        [
            [1.0, 2.0, 4.0, 5.0, 1.0, 0.0],
            [10.0, 20.0, 14.0, 25.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match=message):
        tracker.update(rows, None, initial_track_ids=ids)


def test_initial_ids_are_only_accepted_on_first_tracker_frame(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    apply_upstream_fixes(root)
    tracker = _patched_tracker_namespace(root)["Tracker"]()
    rows = np.asarray([[1.0, 2.0, 4.0, 5.0, 1.0, 0.0]], dtype=float)
    tracker.update(rows, None)

    with pytest.raises(ValueError, match="tracker frame 1"):
        tracker.update(rows, None, initial_track_ids=[1])


def test_initialized_detections_may_not_be_thresholded_away(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    apply_upstream_fixes(root)
    tracker = _patched_tracker_namespace(root)["Tracker"]()
    rows = np.asarray([[1.0, 2.0, 4.0, 5.0, 0.2, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="pass the tracking thresholds"):
        tracker.update(rows, None, initial_track_ids=[1])


def test_check_mode_does_not_modify_files(tmp_path: Path) -> None:
    root = _checkout(tmp_path)

    summary = apply_upstream_fixes(root, check_only=True)

    assert summary.needs_update
    assert summary.changed_file_count == 2
    assert (root / "tools" / "inference.py").read_text(encoding="utf-8") == INFERENCE
    assert (root / "tracker" / "mc_bot_sort.py").read_text(encoding="utf-8") == TRACKER


def test_validation_failure_does_not_partially_modify_checkout(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    tracker_path = root / "tracker" / "mc_bot_sort.py"
    malformed_tracker = "class Tracker:\n    pass\n"
    tracker_path.write_text(malformed_tracker, encoding="utf-8")

    with pytest.raises(UpstreamPatchError, match="forced track activation"):
        apply_upstream_fixes(root)

    assert (root / "tools" / "inference.py").read_text(encoding="utf-8") == INFERENCE
    assert tracker_path.read_text(encoding="utf-8") == malformed_tracker
    assert not (root / "tools" / "inference.py.raft-uav-original").exists()
