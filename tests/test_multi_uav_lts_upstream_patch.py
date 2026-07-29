from __future__ import annotations

import ast
from pathlib import Path
import textwrap

import pytest

from raft_uav.multi_uav_lts.upstream_patch import (
    UpstreamPatchError,
    apply_upstream_tracker_patch,
)


MC_BOT_SORT_FIXTURE = textwrap.dedent(
    """\
    import numpy as np
    from tracker import matching
    from tracker.gmc import GMC
    from tracker.basetrack import BaseTrack, TrackState
    from tracker.kalman_filter import KalmanFilter

    from fast_reid.fast_reid_interfece import FastReIDInterface


    class STrack(BaseTrack):
        shared_kalman = KalmanFilter()

        def __init__(self, tlwh, score, cls, feat=None, feat_history=50):

            # wait activate
            self._tlwh = np.asarray(tlwh, dtype=float)
            self.kalman_filter = None
            self.mean, self.covariance = None, None
            self.is_activated = False

            self.cls = -1

        def activate(self, kalman_filter, frame_id):
            \"\"\"Start a new tracklet\"\"\"
            self.kalman_filter = kalman_filter
            self.track_id = self.next_id()

            self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))


    class BoTSORT(object):
        def __init__(self, args, frame_rate=30):
            self.frame_id = 0
            self.args = args

            self.track_high_thresh = args.track_high_thresh

        def update(self, output_results, img):
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

            '''Extract embeddings '''
            if self.args.with_reid:
                features_keep = self.encoder.inference(img, dets)

            if len(dets) > 0:
                '''Detections'''
                if self.args.with_reid:
                    detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, f) for
                                  (tlbr, s, c, f) in zip(dets, scores_keep, classes_keep, features_keep)]
                else:
                    detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                                  (tlbr, s, c) in zip(dets, scores_keep, classes_keep)]
            else:
                detections = []

            # Associate with high score detection boxes
            ious_dists = matching.iou_distance(strack_pool, detections)
            ious_dists_mask = (ious_dists > self.proximity_thresh)

            if not self.args.mot20:
                ious_dists = matching.fuse_score(ious_dists, detections)

            if self.args.with_reid:
                emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
                raw_emb_dists = emb_dists.copy()
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                emb_dists[ious_dists_mask] = 1.0
                dists = np.minimum(ious_dists, emb_dists)

                # Popular ReID method (JDE / FairMOT)
                # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
                # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
                # emb_dists = dists

                # IoU making ReID
                # dists = matching.embedding_distance(strack_pool, detections)
                # dists[ious_dists_mask] = 1.0
            else:
                dists = ious_dists

            dists = matching.iou_distance(r_tracked_stracks, detections_second)
            matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)

            detections = [detections[i] for i in u_detection]
            dists = matching.iou_distance(unconfirmed, detections)
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

            for inew in u_detection:
                track = detections[inew]
                if track.score < self.new_track_thresh:
                    continue

                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(track)

            # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
            output_stracks = [track for track in self.tracked_stracks]
            output_slosts = [track for track in self.lost_stracks]

            return output_stracks, output_slosts
    """
)


INFERENCE_FIXTURE = textwrap.dedent(
    """\
    import os
    import torch
    from pathlib import Path


    def detect():
        for path, img, im0s, vid_cap in dataset:
            ################
            # first frame use gt, no need to detect
            ################
            if opt.with_pos and idx == 0:
                prior_box = []


                # Compute scaling factors
                scale_x = new_width / original_width
                scale_y = new_height / original_height

                prior_box = []

                with open(gt_path, "r") as file:
                    for line in file:
                        values = line.strip().split(",")  # Split by comma
                        obj_id = int(values[0])  # Extract ID
                        # Extract bbox (x, y, width, height)
                        x, y, w, h = map(float, values[2:6])
                        
                        # Apply scaling to convert coordinates
                        x_scaled = x * scale_x
                        y_scaled = y * scale_y
                        w_scaled = w * scale_x
                        h_scaled = h * scale_y
                        
                        # Calculate the new coordinates for top-left and bottom-right corners
                        x1, y1 = x_scaled, y_scaled
                        x2, y2 = x_scaled + w_scaled, y_scaled + h_scaled
                        
                        prior_box.append([x1, y1, x2, y2, 1., 0.])

                prior_box = torch.tensor(prior_box, device="cuda:0")


                # init_loc = [[prior_box[0], prior_box[1], prior_box[0]+prior_box[2], prior_box[1]+prior_box[3], 1., 0.]]
                # init_loc = torch.tensor(init_loc, device="cuda:0")
                pred = prior_box

            pred = [pred]    # [tensor([[], []])]
            # print(pred)

            # if prior is not empty
            if pred[0].numel() != 0:
                for i, det in enumerate(pred):
                    online_targets, slosts_targets = tracker.update(detections, im0)

                    online_tlwhs = []
                    for t in online_targets:
                        pass
            else:
                pass
    """
)


def _write_checkout(root: Path) -> None:
    (root / "tracker").mkdir(parents=True)
    (root / "tools").mkdir(parents=True)
    (root / "tracker" / "mc_bot_sort.py").write_text(MC_BOT_SORT_FIXTURE)
    (root / "tools" / "inference.py").write_text(INFERENCE_FIXTURE)


def test_patch_is_idempotent_and_preserves_first_frame_ids(tmp_path: Path) -> None:
    _write_checkout(tmp_path)

    first = apply_upstream_tracker_patch(tmp_path)
    assert first.changed
    assert {record.action for record in first.files} == {"updated", "created"}

    tracker_text = (tmp_path / "tracker" / "mc_bot_sort.py").read_text()
    inference_text = (tmp_path / "tools" / "inference.py").read_text()
    helper_text = (tmp_path / "tracker" / "raft_uav_association.py").read_text()
    assert "initial_track_ids=None" in tracker_text
    assert "fixed_track_id=track.fixed_track_id" in tracker_text
    assert "RAFT_UAV_LTS_CONFIRMED_OUTPUT" in tracker_text
    assert "self.initial_identity_bank = set()" in tracker_text
    assert "and self.initial_identity_bank" in tracker_text
    assert tracker_text.count("association_distance(") == 3
    assert "obj_id = int(values[1])" in inference_text
    assert "initial_track_ids=initial_track_ids" in inference_text
    assert "for t in reported_targets" in inference_text
    assert "Always advance the tracker" in inference_text
    assert 'device=img.device' in inference_text

    # Syntax regressions are especially costly because this patch touches an
    # external checkout only when a GPU experiment starts.
    ast.parse(tracker_text)
    ast.parse(inference_text)
    ast.parse(helper_text)

    second = apply_upstream_tracker_patch(tmp_path)
    assert not second.changed
    assert all(record.action == "unchanged" for record in second.files)


def test_patch_dry_run_does_not_modify_checkout(tmp_path: Path) -> None:
    _write_checkout(tmp_path)
    report = apply_upstream_tracker_patch(tmp_path, dry_run=True)
    assert report.changed
    assert "RAFT-UAV LTS PATCH" not in (tmp_path / "tracker" / "mc_bot_sort.py").read_text()
    assert not (tmp_path / "tracker" / "raft_uav_association.py").exists()


def test_patch_rejects_unknown_upstream_layout(tmp_path: Path) -> None:
    (tmp_path / "tracker").mkdir(parents=True)
    (tmp_path / "tools").mkdir(parents=True)
    (tmp_path / "tracker" / "mc_bot_sort.py").write_text("unexpected\n")
    (tmp_path / "tools" / "inference.py").write_text("unexpected\n")

    with pytest.raises(UpstreamPatchError, match="upstream anchor"):
        apply_upstream_tracker_patch(tmp_path)
