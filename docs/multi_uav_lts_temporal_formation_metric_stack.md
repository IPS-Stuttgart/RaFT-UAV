# Temporal, formation, and metric-aware Multi-UAV LTS stack

This experiment is a stacked challenger to the raw seeded BoT-SORT control and
the proposal/association experiments in PR #2540. It does not change the raw
submission path and does not claim a Codabench improvement. A transformed
candidate is usable only if the guarded tournament selects it and a separately
prepared hidden-test submission confirms the gain.

## Added information sources

### Symmetric temporal P2 specialist

`temporal_p2_detector` is a five-channel, stride-4 detector evaluated inside
track-conditioned ROIs. Its middle-frame input contains the current thermal
crop, the median of registered past and future crops, signed and absolute
middle-frame residuals, and temporal MAD. The network is trained on compact
label-centred patches, not full-resolution frames, and retains permissive
low-confidence proposals for downstream association.

The detector uses frames on both sides of the estimated frame. This is valid for
the offline benchmark and supplies observation evidence unavailable to a
strictly causal detector.

### Background camera stabilization

`scene_stabilization` estimates a quality-gated translation from background
phase correlation after masking target boxes. Association and reassociation can
operate in first-frame coordinates, while exported boxes stay in native image
coordinates. Rejected registrations are not promoted to anchors, preventing an
unobserved displacement from contaminating every later cumulative transform.

Two learned transition densities are fitted per fold:

- an image-coordinate motion prior;
- a stabilization-coordinate motion prior.

This makes the stabilization ablation meaningful rather than applying a raw
motion distribution in a different coordinate system.

### Swarm geometry and offline identity revision

`formation_reassociation` treats the supplied frame-one identities as anchors,
assigns detections jointly by Hungarian optimization, and adds robust pairwise
relative-geometry costs. Formation distances evolve by a clipped exponential
update, so the model is soft rather than rigid. Forward and backward solutions
are reconciled with a two-state sequence-level dynamic program.

### HOTA(0)-aware output boxes

The organizer-compatible primary HOTA field uses the 0.05 IoU localization
threshold, while CLEAR MOTA and identity matching use IoU 0.5. Consequently,
unconstrained box inflation can improve HOTA(0) while destroying MOTA and IDF1.
The normal metric-aware candidates therefore cap scale at 1.35, below
`sqrt(2)`, so a perfectly centred scaled box still exceeds IoU 0.5. A separate
2.20-scale stress candidate is retained as a diagnostic and is expected to be
rejected if secondary metrics collapse.

## Candidate matrix

The evidence runner assembles each held-out sequence exactly once for:

- raw controls: IMM, area-adaptive robust smoother, bidirectional formation,
  and guarded metric transforms;
- observation ablations: base proposals, registered residual proposals,
  temporal-P2 proposals, and their union;
- stabilization ablations: no stabilization, stabilized geometry with the raw
  prior, and stabilized geometry with a separately fitted stabilized prior;
- appearance fusion: stabilized learned motion plus the complementary-fold
  thermal hard-negative affinity;
- trajectory variants: IMM, area-adaptive robust smoothing, forward formation,
  backward formation, and bidirectional formation;
- output variants: fixed 1.25/1.35 scaling, uncertainty-adaptive guarded
  scaling, short-gap repair, formation-plus-scaling, and the HOTA-only stress
  candidate;
- a cross-fitted observable expert gate over all complete candidates.

## Leakage contract

Five deterministic scenario-stratified folds are used.

For fold `k`:

1. raw and stabilized motion priors, the thermal affinity model, and the
   temporal-P2 detector are trained only on the other four folds;
2. stabilization caches created from annotations are read only for those
   complementary training sequences;
3. held-out stabilization uses images and raw tracker boxes, never held-out
   annotations;
4. every held-out prediction file is written once and conflicting reassembly is
   an error;
5. the observable expert gate is cross-fitted again from out-of-fold sequence
   scores.

The guarded tournament retains raw as an always-eligible fallback and requires
primary HOTA gain, a nonnegative paired-bootstrap lower bound, MOTA/IDF1 floors,
scenario protection, exact sequence coverage, and content provenance.

## Run

Use the manually dispatched workflow:

```text
Multi-UAV LTS temporal formation evidence
```

The expensive job runs on the data-labelled self-hosted CUDA runner. The pull
request event runs only focused unit/static checks; it does not launch the
experiment.

A direct invocation is:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python scripts/run_multi_uav_lts_temporal_formation_evidence.py \
  --image-root /path/to/Train \
  --truth-dir /path/to/TrainLabels \
  --seed-dir /path/to/TrainLabels_FirstFrameOnly \
  --raw-dir /path/to/raw/predictions \
  --base-proposal-dir /path/to/permissive/proposals \
  --fold-assignments /path/to/folds.csv \
  --run-dir /path/to/output \
  --python /path/to/python \
  --device cuda:0 \
  --temporal-p2-epochs 4 \
  --temporal-p2-max-samples 20000 \
  --require-improvement
```

The result bundle contains the experiment contract, progress state,
per-candidate scores, cross-fit gate evidence, guarded tournament ranking,
selected candidate, and exact held-out prediction sets.
