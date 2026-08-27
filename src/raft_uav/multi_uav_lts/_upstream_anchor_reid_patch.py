"""Second-stage source transform for anchored and multi-crop LTS ReID."""

from __future__ import annotations

from ._upstream_patch_common import UpstreamPatchError, _replace_once


_ANCHOR_PATCH_MARKER = "RAFT-UAV LTS ANCHOR REID PATCH v1"


def _replace_exact_count(
    text: str,
    old: str,
    new: str,
    *,
    expected: int,
    label: str,
) -> str:
    count = text.count(old)
    if count != expected:
        raise UpstreamPatchError(
            f"cannot patch {label}: expected {expected} anchors, found {count}"
        )
    return text.replace(old, new)


def _patch_feature_state(text: str) -> str:
    old = (
        "        self.smooth_feat = None\n"
        "        self.curr_feat = None\n"
        "        if feat is not None:\n"
        "            self.update_features(feat)\n"
        "        self.features = deque([], maxlen=feat_history)\n"
        "        self.alpha = 0.9\n"
    )
    new = (
        "        self.smooth_feat = None\n"
        "        self.curr_feat = None\n"
        "        self.anchor_feat = None\n"
        "        self.features = deque([], maxlen=feat_history)\n"
        "        self.alpha = 0.9\n"
        "        if feat is not None:\n"
        "            self.update_features(\n"
        "                feat, set_anchor=fixed_track_id is not None\n"
        "            )\n"
    )
    if old in text:
        text = _replace_once(
            text,
            old,
            new,
            label="STrack feature initialization",
        )
    elif "        self.anchor_feat = None\n" not in text:
        text = _replace_once(
            text,
            "        self.fixed_track_id = fixed_track_id\n",
            "        self.fixed_track_id = fixed_track_id\n"
            "        self.anchor_feat = None\n",
            label="minimal-fixture anchor feature storage",
        )

    old_update = (
        "    def update_features(self, feat):\n"
        "        feat /= np.linalg.norm(feat)\n"
        "        self.curr_feat = feat\n"
        "        if self.smooth_feat is None:\n"
        "            self.smooth_feat = feat\n"
        "        else:\n"
        "            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat\n"
        "        self.features.append(feat)\n"
        "        self.smooth_feat /= np.linalg.norm(self.smooth_feat)\n"
    )
    new_update = (
        "    def update_features(self, feat, set_anchor=False):\n"
        "        feature = np.asarray(feat, dtype=float).reshape(-1)\n"
        "        norm = np.linalg.norm(feature)\n"
        "        if not np.isfinite(norm) or norm <= 1.0e-12:\n"
        "            return\n"
        "        feature = feature / norm\n"
        "        self.curr_feat = feature\n"
        "        if set_anchor and self.anchor_feat is None:\n"
        "            self.anchor_feat = feature.copy()\n"
        "        if self.smooth_feat is None:\n"
        "            self.smooth_feat = feature.copy()\n"
        "        else:\n"
        "            self.smooth_feat = (\n"
        "                self.alpha * self.smooth_feat + (1 - self.alpha) * feature\n"
        "            )\n"
        "        self.features.append(feature)\n"
        "        smooth_norm = np.linalg.norm(self.smooth_feat)\n"
        "        if np.isfinite(smooth_norm) and smooth_norm > 1.0e-12:\n"
        "            self.smooth_feat /= smooth_norm\n"
    )
    if old_update in text:
        text = _replace_once(
            text,
            old_update,
            new_update,
            label="STrack anchored feature update",
        )
    return text


def _patch_anchor_reid(text: str) -> str:
    """Extend an already supported v1 tracker patch with anchored ReID."""

    if _ANCHOR_PATCH_MARKER in text:
        required = (
            "self.anchor_feat",
            "multiscale_reid_features(",
            "RAFT_UAV_LTS_REID_CROP_SCALES",
            "anchor_weight_late=self.anchor_weight_late",
            "def _sequence_phase(self):",
        )
        missing = [marker for marker in required if marker not in text]
        if missing:
            raise UpstreamPatchError(
                "mc_bot_sort.py contains a partial anchor-ReID patch; missing "
                + ", ".join(missing)
            )
        return text

    text = _replace_once(
        text,
        "    env_float,\n"
        "    env_int,\n"
        ")\n\n"
        "# RAFT-UAV LTS PATCH v1\n",
        "    env_float,\n"
        "    env_float_list,\n"
        "    env_int,\n"
        "    multiscale_reid_features,\n"
        ")\n\n"
        "# RAFT-UAV LTS PATCH v1\n"
        f"# {_ANCHOR_PATCH_MARKER}\n",
        label="anchor-ReID helper imports",
    )
    text = _patch_feature_state(text)

    text = _replace_once(
        text,
        '        self.motion_gate = env_bool("RAFT_UAV_LTS_MOTION_GATE", True)\n\n'
        "        self.track_high_thresh = args.track_high_thresh\n",
        '        self.motion_gate = env_bool("RAFT_UAV_LTS_MOTION_GATE", True)\n'
        "        self.reid_crop_scales = env_float_list(\n"
        '            "RAFT_UAV_LTS_REID_CROP_SCALES",\n'
        '            "1.0",\n'
        "            minimum=0.5,\n"
        "            maximum=3.0,\n"
        "        )\n"
        "        self.anchor_weight = env_float(\n"
        '            "RAFT_UAV_LTS_ANCHOR_WEIGHT",\n'
        "            0.0,\n"
        "            minimum=0.0,\n"
        "            maximum=1.0,\n"
        "        )\n"
        "        self.anchor_weight_late = env_float(\n"
        '            "RAFT_UAV_LTS_ANCHOR_WEIGHT_LATE",\n'
        "            self.anchor_weight,\n"
        "            minimum=0.0,\n"
        "            maximum=1.0,\n"
        "        )\n"
        "        self.appearance_weight_late = env_float(\n"
        '            "RAFT_UAV_LTS_APPEARANCE_WEIGHT_LATE",\n'
        "            self.appearance_weight,\n"
        "            minimum=0.0,\n"
        "            maximum=1.0,\n"
        "        )\n"
        "        self.appearance_thresh_late = env_float(\n"
        '            "RAFT_UAV_LTS_APPEARANCE_THRESH_LATE",\n'
        "            args.appearance_thresh,\n"
        "            minimum=0.0,\n"
        "            maximum=1.0,\n"
        "        )\n"
        "        self.sequence_frame_count = env_int(\n"
        '            "RAFT_UAV_LTS_SEQUENCE_FRAME_COUNT", 0, minimum=0\n'
        "        )\n\n"
        "        self.track_high_thresh = args.track_high_thresh\n",
        label="anchor-ReID tracker configuration",
    )

    text = _replace_once(
        text,
        "    def update(self, output_results, img, initial_track_ids=None):\n",
        "    def _sequence_phase(self):\n"
        "        if self.sequence_frame_count <= 1:\n"
        "            return 0.0\n"
        "        return min(\n"
        "            1.0,\n"
        "            max(0.0, (self.frame_id - 1) / (self.sequence_frame_count - 1)),\n"
        "        )\n\n"
        "    def update(self, output_results, img, initial_track_ids=None):\n",
        label="sequence-phase helper",
    )
    text = _replace_once(
        text,
        "        if self.args.with_reid:\n"
        "            features_keep = self.encoder.inference(img, dets)\n",
        "        if self.args.with_reid:\n"
        "            features_keep = multiscale_reid_features(\n"
        "                self.encoder,\n"
        "                img,\n"
        "                dets,\n"
        "                self.reid_crop_scales,\n"
        "            )\n",
        label="multi-crop ReID extraction",
    )

    old_tail = (
        "            appearance_min_side=self.appearance_min_side,\n"
        "            motion_gate=self.motion_gate,\n"
        "        )\n"
    )
    new_tail = (
        "            appearance_min_side=self.appearance_min_side,\n"
        "            motion_gate=self.motion_gate,\n"
        "            anchor_weight=self.anchor_weight,\n"
        "            anchor_weight_late=self.anchor_weight_late,\n"
        "            appearance_weight_late=self.appearance_weight_late,\n"
        "            appearance_thresh_late=self.appearance_thresh_late,\n"
        "            phase=self._sequence_phase(),\n"
        "        )\n"
    )
    text = _replace_exact_count(
        text,
        old_tail,
        new_tail,
        expected=3,
        label="anchored association calls",
    )
    return text
