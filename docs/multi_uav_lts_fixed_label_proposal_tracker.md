# Fixed-label Multi-UAV LTS proposal tracker

This experiment turns one or more permissive detector proposal banks into a
valid Multi-UAV LTS prediction set while preserving the identities supplied in
frame one.

It is an intentionally small, falsifiable baseline between the proposal-oracle
audit and a more expensive offline multi-hypothesis smoother. It is not claimed
to improve the hidden leaderboard until held-out training evidence and an
official Codabench upload support that conclusion.

## Model

For every sequence, the tracker:

1. initializes exactly the supplied frame-one object IDs and boxes;
2. reads proposal files one sequence at a time from directories or ZIP files;
3. filters proposals by confidence and removes near-identical cross-source
   duplicates;
4. predicts each seeded identity with a shared frame translation plus an
   identity-specific residual velocity in center and log-size space;
5. solves one global Hungarian assignment per frame over all identities and all
   remaining proposals;
6. gives every identity a private missed-detection alternative, so a bad
   proposal is never forced onto a track;
7. optionally emits a bounded coasted prediction, rejecting coasted boxes that
   conflict with another assigned or coasted identity; and
8. never creates an identity that was absent from the supplied frame-one set.

The assignment cost combines normalized center distance, one minus IoU,
log-width/log-height change, and negative log detector confidence. Center and
scale gates expand after missed detections, but reacquisition stops after the
configured track-memory limit.

The shared translation is updated from the median displacement of continuously
matched identities. This captures some common camera or swarm motion without
requiring image access or a separate optical-flow model.

## Run on one proposal source

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.proposal_tracker \
  --proposal seeded_yolo1920=/mnt/lexar4tb/multi_uav_lts/outputs/proposals \
  --first-frame-label-dir \
    /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels_FirstFrameOnly \
  --sequence-root /mnt/lexar4tb/multi_uav_lts/extracted/TrainImages \
  --truth-dir /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels \
  --output-dir \
    /mnt/lexar4tb/multi_uav_lts/outputs/fixed_label_tracker/default
```

`--sequence-root` is strongly recommended. It supplies the real sequence length
when a proposal bank has empty trailing frames. `--truth-dir` is optional and
is used only to calculate organizer-compatible training metrics.

## Fuse complementary proposal sources

Pass `--proposal` repeatedly. The sources are pooled before global assignment,
but the summary records which source supplied every accepted observation.

```bash
python -m raft_uav.multi_uav_lts.proposal_tracker \
  --proposal global=/path/to/yolo1920/proposals \
  --proposal temporal=/path/to/temporal/proposals \
  --proposal local=/path/to/local_crop/proposals \
  --first-frame-label-dir /path/to/TrainLabels_FirstFrameOnly \
  --sequence-root /path/to/TrainImages \
  --truth-dir /path/to/TrainLabels \
  --output-dir /path/to/fused_tracker
```

The high-IoU duplicate filter prevents the same physical proposal exported by
two sources from being assigned to two different identities. It is not intended
as ordinary detector NMS; proposals with meaningfully different geometry remain
available to the assignment solver.

## Important controls

The safest initial candidate uses no emitted coasting:

```text
--min-confidence 0.003
--duplicate-iou-threshold 0.98
--max-center-distance 6
--max-assignment-cost 8
--missed-cost 2.5
--max-missed-frames 60
--coast-frames 0
```

A one-frame-coasting candidate changes only:

```text
--coast-frames 1
```

Coasting confidence is retained for provenance, but HOTA/CLEAR matching is based
on box presence and geometry. A low confidence does not make a false positive
harmless. Therefore coasting must be selected by held-out HOTA/MOTA evidence,
not intuition.

Useful first sweeps are:

```text
minimum confidence:       0.001, 0.003, 0.01, 0.03
maximum center distance:  3, 6, 10
missed cost:              1.5, 2.5, 4.0
coast frames:             0, 1
confidence weight:        0.0, 0.05, 0.1, 0.2
```

Do not tune all dimensions simultaneously on the complete training set. Use the
existing scenario-stratified folds and retain the raw seeded tracker output as
an explicit control.

## Outputs

The output directory contains valid per-sequence LTS text files plus:

```text
fixed_label_tracker_summary.json
fixed_label_tracker_sequences.csv
```

The summary records:

- input and retained proposal counts;
- accepted observations and unassigned candidates;
- missed states, emitted coasts, and conflict-suppressed coasts;
- mean accepted assignment cost;
- shared-motion update count;
- accepted observations by proposal source; and
- organizer-compatible HOTA, MOTA, and IDF1 when training truth is supplied.

The prediction directory can be passed directly to the existing guarded
tournament as a candidate. It should be selected only when its held-out HOTA
gain has a positive paired-bootstrap lower bound and its MOTA, IDF1, and
worst-scenario floors remain acceptable.

## Current boundary

This implementation is online and keeps one state hypothesis per identity. It
does not yet provide:

- delayed multi-hypothesis decisions at crossings;
- a backward smoothing pass;
- image-derived optical flow or camera compensation;
- per-identity appearance/template likelihoods; or
- local high-resolution crop proposal generation.

If the proposal oracle is strong but this tracker remains below the raw
baseline, the next justified step is delayed beam/min-cost-flow smoothing over
the proposal bank rather than further detector threshold tuning.
