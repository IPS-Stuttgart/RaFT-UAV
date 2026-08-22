# Metric-aware Bayesian Multi-UAV LTS improvements

This layer sits on top of the cross-fitted proposal-graph tracker. It targets the
actual competition asymmetry: the organizer's exported HOTA is evaluated at IoU
0.05, while CLEAR MOTA and IDF1 use IoU 0.5. The implementation therefore keeps
HOTA coverage, accurate localization, and identity continuity as related but
separate signals instead of collapsing them into one generic association score.

## Components

### 1. Metric-aligned edge heads

`raft_uav.multi_uav_lts.proposal_metric_edge_model` fits three calibrated binary
edge heads on training-only truth:

- `identity`: the candidate continues the same labelled UAV;
- `hota_005`: it continues the same UAV and the candidate box reaches IoU 0.05;
- `clear_050`: it continues the same UAV and the candidate box reaches IoU 0.5.

The runtime model combines their negative log probabilities with explicit
weights. The default weights emphasize the official HOTA objective while still
protecting identity and 0.5-IoU behavior:

```text
identity = 0.75
HOTA@0.05 = 1.00
CLEAR@0.5 = 0.25
```

These are candidate parameters, not claims of optimality. They must be selected
by the existing scenario-stratified guarded tournament.

The wrapper
`raft_uav.multi_uav_lts.metric_proposal_graph_tracker` injects a metric model into
the delayed proposal graph without changing the legacy edge-model path.

### 2. RTS trajectory and box calibration

`raft_uav.multi_uav_lts.trajectory_box_calibration` runs a linear-Gaussian
constant-velocity filter over box centres and log box sizes, followed by a full
Rauch-Tung-Striebel backward pass. Measurement noise is confidence- and
box-size-dependent. The output stage can:

- recenter boxes on the smoothed trajectory;
- smooth log width and height;
- enlarge width/height using posterior centre standard deviations;
- add a velocity-direction margin;
- cap the area increase; and
- clip boxes to the native image dimensions.

The transformation preserves frame IDs, object IDs, class, visibility, and
confidence exactly. It does not consume ground truth at application time.

Example:

```bash
python -m raft_uav.multi_uav_lts.trajectory_box_calibration \
  outputs/predictions \
  --output-dir outputs/predictions_rts_u050 \
  --uncertainty-scale-x 0.5 \
  --uncertainty-scale-y 0.5 \
  --max-area-ratio 1.5
```

A prefix policy can override parameters for observable scenario families:

```bash
python -m raft_uav.multi_uav_lts.trajectory_box_calibration \
  outputs/predictions \
  --output-dir outputs/predictions_policy \
  --policy-json config/multi_uav_lts_box_policy.example.json
```

The example policy is intentionally only a starting candidate. Prefix-specific
values must be accepted or rejected by held-out evidence.

### 3. Complementary proposal fusion

`raft_uav.multi_uav_lts.proposal_bank_fusion` unions detector banks without
cross-source NMS. This is intended for combinations such as full-frame YOLO,
tiled inference, a stride-4/P2 tiny-target detector, and temporal residual
proposals. Frame-local proposal IDs are deterministically rekeyed so overlapping
alternatives remain legal inputs to the proposal graph.

```bash
python -m raft_uav.multi_uav_lts.proposal_bank_fusion \
  --proposal full=outputs/yolo1920/proposals \
  --proposal tiled=outputs/yolo_tiled/proposals \
  --proposal temporal=outputs/temporal/proposals \
  --output-dir outputs/fused_proposals \
  --output-json outputs/fused_proposals/summary.json
```

The downstream proposal graph already performs its own canonicalization and
association, so the fusion stage deliberately does not suppress cross-source
hypotheses.

## Cross-fitted evidence runner

The stacked evidence entry point is:

```bash
python scripts/run_multi_uav_lts_metric_aware_evidence.py <same arguments as the cross-fitted runner>
```

It retains every candidate from `run_multi_uav_lts_cross_fitted_edge_evidence.py`
and adds:

```text
graph_metric_edge_swarm_beam_oof
graph_metric_edge_swarm_beam_oof_rts
graph_metric_edge_swarm_beam_oof_rts_u050
graph_metric_edge_swarm_beam_oof_rts_u100
graph_metric_edge_swarm_beam_oof_rts_u100_v015
```

For each of five scenario-stratified folds, the metric-edge model is fitted only
on the complementary four folds. The held-out predictions are assembled exactly
once. RTS/box candidates are deterministic post-processors of that complete
out-of-fold prediction set and use native image dimensions for clipping.

The existing guarded tournament remains the authority for candidate selection.
A metric-aware candidate should not become a test submission unless it clears
the configured mean-HOTA gain, paired-bootstrap lower bound, MOTA/IDF1 floors,
worst-scenario guard, exact-coverage checks, and provenance checks.

## Recommended experiment order

1. Run the existing proposal-oracle audit at IoU 0.05 and 0.5.
2. If IoU-0.05 proposal recall is incomplete, add tiled/P2/temporal sources and
   fuse their proposal banks.
3. Run the cross-fitted metric-edge candidate.
4. Evaluate the pure RTS candidate before any box expansion.
5. Evaluate the bounded uncertainty-expansion ladder.
6. Only after the global ladder is stable, introduce prefix-specific policies
   and require held-out scenario evidence for each override.

This ordering separates detector recall, association, trajectory estimation,
and output-box calibration so leaderboard gains remain attributable rather than
being the result of an opaque all-at-once change.
