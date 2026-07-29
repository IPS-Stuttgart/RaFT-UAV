"""Guarded source transform for upstream ``tools/inference.py``."""

from __future__ import annotations

from ._upstream_patch_common import (
    _PATCH_MARKER,
    UpstreamPatchError,
    _replace_once,
    _replace_span_around_anchor,
)


def _patch_inference(text: str) -> str:
    if _PATCH_MARKER in text:
        required = (
            "initial_track_ids=initial_track_ids",
            "values[1]",
            "reported_targets",
            "RAFT_UAV_LTS_PRESERVE_INITIAL_IDS",
            "Always advance the tracker",
        )
        missing = [marker for marker in required if marker not in text]
        if missing:
            raise UpstreamPatchError(
                "inference.py contains a partial RaFT-UAV patch; missing "
                + ", ".join(missing)
            )
        return text

    text = _replace_once(
        text,
        "from pathlib import Path\n",
        "from pathlib import Path\n\n"
        f"# {_PATCH_MARKER}\n"
        "def _raft_uav_env_flag(name, default=True):\n"
        "    value = os.environ.get(name)\n"
        "    if value is None:\n"
        "        return bool(default)\n"
        "    normalized = value.strip().lower()\n"
        "    if normalized in {\"1\", \"true\", \"yes\", \"on\"}:\n"
        "        return True\n"
        "    if normalized in {\"0\", \"false\", \"no\", \"off\"}:\n"
        "        return False\n"
        "    raise ValueError(f\"{name} must be a boolean, got {value!r}\")\n",
        label="inference environment helper",
    )
    text = _replace_once(
        text,
        "        ################\n"
        "        # first frame use gt, no need to detect\n"
        "        ################\n",
        "        initial_track_ids = None\n\n"
        "        ################\n"
        "        # first frame use gt, no need to detect\n"
        "        ################\n",
        label="per-frame initial id reset",
    )
    text = _replace_once(
        text,
        "        # if prior is not empty\n"
        "        if pred[0].numel() != 0:\n",
        "        # Always advance the tracker.  Lost-track coasting and Kalman time\n"
        "        # propagation must also happen on frames with zero detections.\n"
        "        if True:\n",
        label="zero-detection tracker advancement",
    )
    new_prior = '''            # Compute scaling factors
            scale_x = new_width / original_width
            scale_y = new_height / original_height

            prior_box = []
            prior_track_ids = []
            with open(gt_path, "r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    values = [value.strip() for value in line.split(",")]
                    if len(values) < 6:
                        raise ValueError(
                            f"{gt_path}:{line_number}: expected at least 6 columns"
                        )
                    frame_id = int(values[0])
                    obj_id = int(values[1])
                    if frame_id != 1:
                        raise ValueError(
                            f"{gt_path}:{line_number}: expected frame id 1, got {frame_id}"
                        )
                    if obj_id <= 0:
                        raise ValueError(
                            f"{gt_path}:{line_number}: object ids must be positive"
                        )
                    x, y, w, h = map(float, values[2:6])

                    x_scaled = x * scale_x
                    y_scaled = y * scale_y
                    w_scaled = w * scale_x
                    h_scaled = h * scale_y
                    x1, y1 = x_scaled, y_scaled
                    x2, y2 = x_scaled + w_scaled, y_scaled + h_scaled

                    prior_box.append([x1, y1, x2, y2, 1.0, 0.0])
                    prior_track_ids.append(obj_id)

            if len(prior_track_ids) != len(set(prior_track_ids)):
                raise ValueError(f"{gt_path}: duplicate first-frame object ids")
            prior_box = torch.tensor(
                prior_box,
                device=img.device,
                dtype=img.dtype,
            ).reshape((-1, 6))
            pred = prior_box
            if _raft_uav_env_flag("RAFT_UAV_LTS_PRESERVE_INITIAL_IDS", True):
                initial_track_ids = prior_track_ids
'''
    text = _replace_span_around_anchor(
        text,
        start_marker="            prior_box = []\n",
        anchor_marker="                    obj_id = int(values[0])  # Extract ID\n",
        end_marker="            pred = prior_box\n",
        replacement=new_prior,
        label="first-frame label parsing",
    )
    text = _replace_once(
        text,
        "                online_targets, slosts_targets = tracker.update(detections, im0)\n\n"
        "                online_tlwhs = []\n",
        "                online_targets, slosts_targets = tracker.update(\n"
        "                    detections,\n"
        "                    im0,\n"
        "                    initial_track_ids=initial_track_ids,\n"
        "                )\n"
        "                reported_targets = list(online_targets)\n"
        "                reported_ids = {target.track_id for target in reported_targets}\n"
        "                reported_targets.extend(\n"
        "                    target\n"
        "                    for target in slosts_targets\n"
        "                    if target.track_id not in reported_ids\n"
        "                )\n\n"
        "                online_tlwhs = []\n",
        label="tracker call and coast reporting",
    )
    text = _replace_once(
        text,
        "                for t in online_targets:\n",
        "                for t in reported_targets:\n",
        label="reported target loop",
    )
    return text
