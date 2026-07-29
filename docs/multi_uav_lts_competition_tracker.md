# Multi-UAV LTS competition tracker

This workflow adds competition-specific fixes around the external
`YOLOv12-BoT-SORT-ReID/BoT-SORT` checkout while keeping detector inference and
submission packaging in the existing RaFT-UAV baseline runner.

## What changes

The wrapper applies a guarded, idempotent patch to the external checkout before
inference. The patch:

1. **Preserves first-frame identities.** RaFT-UAV first-frame files use
   `frame_id, object_id, x, y, w, h, ...`. The upstream inference code read
   column 0 as the object ID and then discarded it. The patched path validates
   column 0 as frame 1, reads the ID from column 1, and reserves that exact ID in
   BoT-SORT.
2. **Suppresses one-frame births.** Detector-born tracks are output only after
   confirmation. First-frame initialized tracks remain immediately active.
3. **Advances the tracker on empty frames.** The upstream inference loop skipped
   `tracker.update` when the detector returned no boxes, preventing Kalman
   propagation and any meaningful coasting.
4. **Adds bounded coasting.** Confirmed lost tracks may be reported for a
   configurable number of frames. The competition default is one frame.
5. **Supports a closed identity bank.** In closed-world mode, unmatched
   detections after frame 1 cannot create new output identities. Lost
   first-frame identities can still be reactivated. The restriction activates
   only when the first-frame file contains at least one identity, so sequences
   with an empty initialization file can still start tracks later.
6. **Uses tiny-object-aware association.** The default cost combines IoU and
   normalized Gaussian-Wasserstein box distance, adds appearance only when the
   crop is sufficiently large and geometrically plausible, and rejects matches
   outside the Kalman innovation gate. High-confidence, low-confidence, and
   unconfirmed-track association stages all use the same guarded geometry.

Every patch run records before/after SHA-256 hashes. The patcher fails without
modifying files when the upstream source no longer contains the expected
anchors.

## Recommended run

```bash
source /mnt/lexar4tb/multi_uav_lts/venvs/yolov12-botsort-py312/bin/activate

PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python scripts/run_multi_uav_lts_competition_tracker.py \
  --work-root /mnt/lexar4tb/multi_uav_lts \
  --python python \
  --device 0 \
  --img-size 1920 \
  --normalize \
  --sort-rows
```

Unless `--output-dir` is supplied, this writes to
`outputs/competition_tracker`, so it does not reuse or silently skip the
existing official-baseline predictions.

The competition defaults are:

```text
preserve first-frame IDs  true
confirmed output          true
coast frames              1
closed identity bank      true
association               gated-weighted
NWD weight                0.50
NWD scale                 20 pixels
appearance weight         0.25 maximum
appearance minimum side   16 pixels
Kalman innovation gate    true
```

All detector and ordinary BoT-SORT options not recognized by the wrapper are
forwarded to `run_multi_uav_lts_official_baseline.py`.

## Controlled ablations

Verify patch compatibility without changing the checkout:

```bash
PYTHONPATH=src python scripts/run_multi_uav_lts_competition_tracker.py \
  --verify-upstream-only \
  --work-root /mnt/lexar4tb/multi_uav_lts
```

Apply the patch without starting inference:

```bash
PYTHONPATH=src python scripts/run_multi_uav_lts_competition_tracker.py \
  --patch-only \
  --work-root /mnt/lexar4tb/multi_uav_lts
```

Run confirmed births plus one-frame coasting while retaining the legacy
`min(IoU, ReID)` cost:

```bash
PYTHONPATH=src python scripts/run_multi_uav_lts_competition_tracker.py \
  --association-mode legacy-min \
  --no-closed-world \
  --no-motion-gate \
  --coast-frames 1 \
  --work-root /mnt/lexar4tb/multi_uav_lts \
  --output-dir /mnt/lexar4tb/multi_uav_lts/outputs/ablation_tfps_coast1 \
  --device 0 --img-size 1920 --overwrite
```

Run explicit IDs and the closed identity bank with otherwise legacy
association:

```bash
PYTHONPATH=src python scripts/run_multi_uav_lts_competition_tracker.py \
  --association-mode legacy-min \
  --no-motion-gate \
  --coast-frames 0 \
  --work-root /mnt/lexar4tb/multi_uav_lts \
  --output-dir /mnt/lexar4tb/multi_uav_lts/outputs/ablation_id_bank \
  --device 0 --img-size 1920 --overwrite
```

For a strict untouched-upstream control, invoke the existing official-baseline
runner directly. `--no-upstream-patch` is intended only when the checkout was
already patched or is managed separately; it does not revert a previous patch.

## Local HOTA, MOTA, and IDF1

The existing `score-predictions` command is a fast diagnostic, not the official
metric implementation. Use the TrackEval bridge for experiment selection:

```bash
PYTHONPATH=src python scripts/score_multi_uav_lts_trackeval.py \
  /mnt/lexar4tb/multi_uav_lts/outputs/competition_tracker/predictions \
  /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels \
  --trackeval-root \
    /mnt/lexar4tb/multi_uav_lts/repos/YOLOv12-BoT-SORT-ReID/TrackEval \
  --sequence-root \
    /mnt/lexar4tb/multi_uav_lts/extracted/TrainImages \
  --output-json \
    /mnt/lexar4tb/multi_uav_lts/outputs/competition_tracker/trackeval.json
```

The bridge accepts either a prediction directory or a submission ZIP. It
materializes the MOTChallenge directory layout, validates frame/object keys and
box geometry, uses the image directories to include trailing frames without
annotations, evaluates `HOTA`, `CLEAR`, and `Identity`, and reports combined
and per-sequence values for:

- HOTA averaged across TrackEval's 0.05–0.95 localization thresholds;
- DetA, AssA, and LocA;
- MOTA;
- IDF1;
- the complete HOTA-by-threshold curve.

Codabench remains the final authority because organizers may use a pinned
TrackEval revision or additional preprocessing. Use the same vendored TrackEval
checkout for every local ablation and record its Git commit with the generated
configuration JSON.
