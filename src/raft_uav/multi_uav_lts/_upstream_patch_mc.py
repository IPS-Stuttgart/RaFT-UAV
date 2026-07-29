"""Guarded source transform for upstream ``tracker/mc_bot_sort.py``."""

from __future__ import annotations

from ._upstream_patch_common import (
    _PATCH_MARKER,
    UpstreamPatchError,
    _replace_once,
)


def _patch_mc_bot_sort(text: str) -> str:
    if _PATCH_MARKER in text:
        required = (
            "initial_track_ids=None",
            "association_distance(",
            "RAFT_UAV_LTS_CONFIRMED_OUTPUT",
            "fixed_track_id",
            "initial_identity_bank",
        )
        missing = [marker for marker in required if marker not in text]
        if missing:
            raise UpstreamPatchError(
                "mc_bot_sort.py contains a partial RaFT-UAV patch; missing "
                + ", ".join(missing)
            )
        return text

    text = _replace_once(
        text,
        "from fast_reid.fast_reid_interfece import FastReIDInterface\n",
        "from fast_reid.fast_reid_interfece import FastReIDInterface\n"
        "from tracker.raft_uav_association import (\n"
        "    association_distance,\n"
        "    env_bool,\n"
        "    env_choice,\n"
        "    env_float,\n"
        "    env_int,\n"
        ")\n\n"
        f"# {_PATCH_MARKER}\n",
        label="mc_bot_sort imports",
    )
    text = _replace_once(
        text,
        "    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):\n",
        "    def __init__(\n"
        "        self, tlwh, score, cls, feat=None, feat_history=50, fixed_track_id=None\n"
        "    ):\n",
        label="STrack constructor signature",
    )
    text = _replace_once(
        text,
        "        self.is_activated = False\n\n        self.cls = -1\n",
        "        self.is_activated = False\n"
        "        self.fixed_track_id = fixed_track_id\n\n"
        "        self.cls = -1\n",
        label="STrack fixed id storage",
    )
    text = _replace_once(
        text,
        "    def activate(self, kalman_filter, frame_id):\n"
        "        \"\"\"Start a new tracklet\"\"\"\n"
        "        self.kalman_filter = kalman_filter\n"
        "        self.track_id = self.next_id()\n",
        "    def activate(self, kalman_filter, frame_id, fixed_track_id=None):\n"
        "        \"\"\"Start a new tracklet, preserving supplied first-frame ids.\"\"\"\n"
        "        self.kalman_filter = kalman_filter\n"
        "        if fixed_track_id is None:\n"
        "            self.track_id = self.next_id()\n"
        "        else:\n"
        "            self.track_id = int(fixed_track_id)\n"
        "            if self.track_id <= 0:\n"
        "                raise ValueError(\"fixed track ids must be positive\")\n"
        "            BaseTrack._count = max(BaseTrack._count, self.track_id)\n",
        label="STrack activation",
    )
    text = _replace_once(
        text,
        "        self.frame_id = 0\n        self.args = args\n\n"
        "        self.track_high_thresh = args.track_high_thresh\n",
        "        self.frame_id = 0\n"
        "        self.args = args\n"
        "        self.confirmed_track_output = env_bool(\n"
        "            \"RAFT_UAV_LTS_CONFIRMED_OUTPUT\", True\n"
        "        )\n"
        "        self.coast_frames = env_int(\"RAFT_UAV_LTS_COAST_FRAMES\", 1, minimum=0)\n"
        "        self.closed_world = env_bool(\"RAFT_UAV_LTS_CLOSED_WORLD\", False)\n"
        "        self.initial_identity_bank = set()\n"
        "        self.association_mode = env_choice(\n"
        "            \"RAFT_UAV_LTS_ASSOCIATION_MODE\",\n"
        "            \"gated-weighted\",\n"
        "            {\"gated-weighted\", \"legacy-min\"},\n"
        "        )\n"
        "        self.nwd_weight = env_float(\n"
        "            \"RAFT_UAV_LTS_NWD_WEIGHT\", 0.5, minimum=0.0, maximum=1.0\n"
        "        )\n"
        "        self.nwd_scale = env_float(\n"
        "            \"RAFT_UAV_LTS_NWD_SCALE\", 20.0, minimum=0.0, strict_minimum=True\n"
        "        )\n"
        "        self.appearance_weight = env_float(\n"
        "            \"RAFT_UAV_LTS_APPEARANCE_WEIGHT\", 0.25, minimum=0.0, maximum=1.0\n"
        "        )\n"
        "        self.appearance_min_side = env_float(\n"
        "            \"RAFT_UAV_LTS_APPEARANCE_MIN_SIDE\",\n"
        "            16.0,\n"
        "            minimum=0.0,\n"
        "            strict_minimum=True,\n"
        "        )\n"
        "        self.motion_gate = env_bool(\"RAFT_UAV_LTS_MOTION_GATE\", True)\n\n"
        "        self.track_high_thresh = args.track_high_thresh\n",
        label="BoTSORT competition configuration",
    )
    text = _replace_once(
        text,
        "    def update(self, output_results, img):\n",
        "    def update(self, output_results, img, initial_track_ids=None):\n",
        label="BoTSORT update signature",
    )
    text = _replace_once(
        text,
        "            features = output_results[:, 6:]\n\n"
        "            # Remove bad detections\n",
        "            features = output_results[:, 6:]\n"
        "            if initial_track_ids is not None:\n"
        "                initial_track_ids = np.asarray(initial_track_ids, dtype=int).reshape(-1)\n"
        "                if len(initial_track_ids) != len(output_results):\n"
        "                    raise ValueError(\n"
        "                        \"initial_track_ids must align with first-frame detections\"\n"
        "                    )\n"
        "                self.initial_identity_bank.update(\n"
        "                    int(track_id) for track_id in initial_track_ids\n"
        "                )\n\n"
        "            # Remove bad detections\n",
        label="initial id validation",
    )
    text = _replace_once(
        text,
        "            features = output_results[lowest_inds]\n\n"
        "            # Find high threshold detections\n",
        "            features = output_results[lowest_inds]\n"
        "            if initial_track_ids is not None:\n"
        "                initial_track_ids = initial_track_ids[lowest_inds]\n\n"
        "            # Find high threshold detections\n",
        label="initial id low-threshold filtering",
    )
    text = _replace_once(
        text,
        "            features_keep = features[remain_inds]\n"
        "        else:\n",
        "            features_keep = features[remain_inds]\n"
        "            initial_ids_keep = (\n"
        "                initial_track_ids[remain_inds]\n"
        "                if initial_track_ids is not None\n"
        "                else None\n"
        "            )\n"
        "        else:\n",
        label="initial id high-threshold filtering",
    )
    text = _replace_once(
        text,
        "            classes_keep = []\n\n        '''Extract embeddings '''\n",
        "            classes_keep = []\n"
        "            initial_ids_keep = None\n\n"
        "        '''Extract embeddings '''\n",
        label="empty initial id handling",
    )
    old_detections = '''        if len(dets) > 0:
            ''' + "'''Detections'''" + '''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, f) for
                              (tlbr, s, c, f) in zip(dets, scores_keep, classes_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                              (tlbr, s, c) in zip(dets, scores_keep, classes_keep)]
        else:
            detections = []
'''
    new_detections = '''        if len(dets) > 0:
            ''' + "'''Detections'''" + '''
            fixed_ids = (
                list(initial_ids_keep)
                if initial_ids_keep is not None
                else [None] * len(dets)
            )
            if self.args.with_reid:
                detections = [
                    STrack(
                        STrack.tlbr_to_tlwh(tlbr),
                        s,
                        c,
                        f,
                        fixed_track_id=fixed_id,
                    )
                    for tlbr, s, c, f, fixed_id in zip(
                        dets,
                        scores_keep,
                        classes_keep,
                        features_keep,
                        fixed_ids,
                    )
                ]
            else:
                detections = [
                    STrack(
                        STrack.tlbr_to_tlwh(tlbr),
                        s,
                        c,
                        fixed_track_id=fixed_id,
                    )
                    for tlbr, s, c, fixed_id in zip(
                        dets, scores_keep, classes_keep, fixed_ids
                    )
                ]
        else:
            detections = []
'''
    text = _replace_once(
        text,
        old_detections,
        new_detections,
        label="detection construction with fixed ids",
    )
    old_association = '''        # Associate with high score detection boxes
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
'''
    new_association = '''        # Associate with high score detections using tiny-object geometry,
        # quality-gated appearance, and a hard Kalman innovation gate.
        dists = association_distance(
            self.kalman_filter,
            strack_pool,
            detections,
            with_reid=self.args.with_reid,
            fuse_score=not self.args.mot20,
            proximity_thresh=self.proximity_thresh,
            appearance_thresh=self.appearance_thresh,
            mode=self.association_mode,
            nwd_weight=self.nwd_weight,
            nwd_scale=self.nwd_scale,
            appearance_weight=self.appearance_weight,
            appearance_min_side=self.appearance_min_side,
            motion_gate=self.motion_gate,
        )
'''
    text = _replace_once(
        text,
        old_association,
        new_association,
        label="primary association",
    )
    text = _replace_once(
        text,
        "        dists = matching.iou_distance(r_tracked_stracks, detections_second)\n"
        "        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)\n",
        "        dists = association_distance(\n"
        "            self.kalman_filter,\n"
        "            r_tracked_stracks,\n"
        "            detections_second,\n"
        "            with_reid=False,\n"
        "            fuse_score=False,\n"
        "            proximity_thresh=self.proximity_thresh,\n"
        "            appearance_thresh=self.appearance_thresh,\n"
        "            mode=self.association_mode,\n"
        "            nwd_weight=self.nwd_weight,\n"
        "            nwd_scale=self.nwd_scale,\n"
        "            appearance_weight=0.0,\n"
        "            appearance_min_side=self.appearance_min_side,\n"
        "            motion_gate=self.motion_gate,\n"
        "        )\n"
        "        matches, u_track, u_detection_second = matching.linear_assignment(\n"
        "            dists, thresh=0.5\n"
        "        )\n",
        label="secondary association",
    )
    old_unconfirmed = '''        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
'''
    new_unconfirmed = '''        detections = [detections[i] for i in u_detection]
        dists = association_distance(
            self.kalman_filter,
            unconfirmed,
            detections,
            with_reid=self.args.with_reid,
            fuse_score=not self.args.mot20,
            proximity_thresh=self.proximity_thresh,
            appearance_thresh=self.appearance_thresh,
            mode=self.association_mode,
            nwd_weight=self.nwd_weight,
            nwd_scale=self.nwd_scale,
            appearance_weight=self.appearance_weight,
            appearance_min_side=self.appearance_min_side,
            motion_gate=self.motion_gate,
        )
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
'''
    text = _replace_once(
        text,
        old_unconfirmed,
        new_unconfirmed,
        label="unconfirmed association",
    )
    text = _replace_once(
        text,
        "        for inew in u_detection:\n"
        "            track = detections[inew]\n"
        "            if track.score < self.new_track_thresh:\n"
        "                continue\n\n"
        "            track.activate(self.kalman_filter, self.frame_id)\n"
        "            activated_starcks.append(track)\n",
        "        for inew in u_detection:\n"
        "            track = detections[inew]\n"
        "            if track.score < self.new_track_thresh:\n"
        "                continue\n"
        "            if (\n"
        "                self.closed_world\n"
        "                and self.initial_identity_bank\n"
        "                and self.frame_id > 1\n"
        "            ):\n"
        "                continue\n\n"
        "            track.activate(\n"
        "                self.kalman_filter,\n"
        "                self.frame_id,\n"
        "                fixed_track_id=track.fixed_track_id,\n"
        "            )\n"
        "            activated_starcks.append(track)\n",
        label="closed-world track activation",
    )
    text = _replace_once(
        text,
        "        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]\n"
        "        output_stracks = [track for track in self.tracked_stracks]\n"
        "        output_slosts = [track for track in self.lost_stracks]\n\n"
        "        return output_stracks, output_slosts\n",
        "        if self.confirmed_track_output:\n"
        "            output_stracks = [\n"
        "                track for track in self.tracked_stracks if track.is_activated\n"
        "            ]\n"
        "        else:\n"
        "            output_stracks = list(self.tracked_stracks)\n"
        "        output_slosts = [\n"
        "            track\n"
        "            for track in self.lost_stracks\n"
        "            if track.is_activated\n"
        "            and 0 < self.frame_id - track.end_frame <= self.coast_frames\n"
        "        ]\n\n"
        "        return output_stracks, output_slosts\n",
        label="confirmed output and one-frame coasting",
    )
    return text
