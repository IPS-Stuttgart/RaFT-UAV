# Multi-UAV LTS proposal-bank oracle audit

This experiment separates detector headroom from association headroom before a
new tracker is trained. It exports a deliberately permissive detector proposal
bank, measures one-to-one proposal recall at the IoU thresholds that matter for
the competition, and materializes identity-oracle predictions for the existing
organizer-compatible evaluator.

The oracle output is a diagnostic upper bound, not a valid hidden-test method:
it assigns each selected proposal the corresponding training truth identity.
It does, however, answer the key implementation question: whether a better
known-label temporal smoother can recover targets from the existing detector
outputs, or whether the detector fails to propose the targets at all.

## 1. Export a low-threshold proposal bank

The proposal runner first applies the maintained upstream identity-seeding fixes
and then adds an idempotent proposal-export patch to the external
`YOLOv12-BoT-SORT-ReID/BoT-SORT/tools/inference.py` checkout.

```bash
source /mnt/lexar4tb/multi_uav_lts/venvs/yolov12-botsort-py312/bin/activate

PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.proposal_baseline \
  --work-root /mnt/lexar4tb/multi_uav_lts \
  --sequence-root /mnt/lexar4tb/multi_uav_lts/extracted/TrainImages \
  --first-frame-label-dir \
    /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels_FirstFrameOnly \
  --output-dir /mnt/lexar4tb/multi_uav_lts/outputs/proposal_baseline \
  --no-template \
  --python python \
  --device 0 \
  --img-size 1920 \
  --proposal-conf-thres 0.001 \
  --proposal-iou-thres 0.95 \
  --overwrite
```

The normal tracker still uses its ordinary `--conf-thres` and `--iou-thres`.
The additional proposal pass uses the lower confidence threshold and relaxed
NMS threshold, converts boxes back to original-image coordinates, and writes
one LTS-shaped proposal file per sequence under:

```text
proposal_baseline/
  predictions/               # ordinary seeded BoT-SORT output
  proposals/                 # low-threshold detector proposal bank
  proposal_export_patch_summary.json
  proposal_run_summary.json
```

Proposal rows use the standard nine-column LTS shape, but column two is only a
frame-local positive proposal identifier. The exporter preserves detector
confidence and converts zero-based detector classes to positive LTS class IDs.
The proposal pass occurs before the first-frame detector result is replaced by
the supplied seed boxes, so detector recall on frame one remains measurable.

This is a permissive **post-decoding proposal bank**, not the undecoded raw
network tensor. Setting the proposal NMS threshold near one retains many
strongly overlapping alternatives without depending on model-internal output
layouts.

## 2. Measure proposal recall and oracle scores

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.proposal_oracle \
  --proposal \
    yolo1920=/mnt/lexar4tb/multi_uav_lts/outputs/proposal_baseline/proposals \
  --truth-dir /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels \
  --output-dir /mnt/lexar4tb/multi_uav_lts/outputs/proposal_oracle \
  --confidence-thresholds 0 0.001 0.003 0.01 0.03 0.09 \
  --iou-thresholds 0.05 0.3 0.5 \
  --oracle-confidence-threshold 0.003 \
  --oracle-iou-threshold 0.05
```

The audit uses maximum-cardinality bipartite matching in every frame and then
maximizes total IoU among equally large matchings. A single proposal therefore
cannot be counted as covering two nearby UAVs.

Generated evidence includes:

```text
proposal_oracle/
  proposal_oracle_summary.json
  coverage.csv
  sequence_coverage.csv
  size_coverage.csv
  oracle_scores.csv
  oracle_predictions/<source>/*.txt
```

`coverage.csv` reports proposal recall, mean matched IoU, and mean best IoU for
every confidence/IoU pair. `size_coverage.csv` separates very small boxes from
larger targets. The oracle prediction set uses exact supplied frame-one truth
rows, because that information is genuinely available to the benchmark
tracker, and uses proposal boxes for later frames. It is scored with the
repository's organizer-compatible `CODABENCH_HOTA`, `CODABENCH_MOTA`, and
`CODABENCH_IDF1` implementation together with canonical HOTA components.

## 3. Audit proposal-source fusion

Repeat `--proposal` to test whether complementary proposal generators close the
coverage gap:

```bash
python -m raft_uav.multi_uav_lts.proposal_oracle \
  --proposal yolo1920=/path/to/yolo1920/proposals \
  --proposal yolo_tiled=/path/to/tiled/proposals \
  --proposal temporal=/path/to/temporal/proposals \
  --truth-dir /path/to/TrainLabels \
  --output-dir /path/to/proposal_oracle_fused
```

The audit keeps each source separately and also evaluates a canonicalized
`fused` union. It does not perform NMS across sources before matching; competing
boxes remain available to a future global known-label smoother.

## Interpretation

The main decision rules are:

- Low recall at IoU 0.05 means the detector proposal generators still discard
  targets; prioritize temporal residuals, local crops, tiling, or a stride-4
  specialist.
- High recall at 0.05 but low recall at 0.5 points to box localization and scale
  instability; prioritize NWD/centre losses and trajectory box calibration.
- High proposal recall and high identity-oracle scores indicate that the next
  high-value implementation is the fixed-label temporal proposal smoother,
  because the observations already contain the missing information.
- Improvements concentrated in one sequence prefix justify a prefix-specific
  proposal generator or parameter policy, selected with the existing
  scenario-stratified evaluation tools.

Do not upload an oracle prediction directory to Codabench. Use its evidence to
choose and falsify the next non-oracle tracker implementation.
