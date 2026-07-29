# Multi-UAV LTS result-improvement workflow

This workflow adds organizer-compatible local scoring and a first-frame-seeded
post-processing stage for the Beyond Strong Baseline Multi-UAV Tracking LTS
benchmark.

The older `score-predictions` command remains useful as a compact IoU-0.5
smoke diagnostic. It is not suitable for selecting leaderboard configurations
because it does not reproduce the competition scorer's HOTA, MOTA, and IDF1
export.

## 1. Audit the fixed-population assumption

The test protocol supplies first-frame boxes. Before suppressing arbitrary
tracker births, verify on all training labels that every identity is present in
frame one:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.population_audit \
  /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels \
  --output-json outputs/multi_uav_lts_population_audit.json \
  --sequence-summary-csv outputs/multi_uav_lts_population_audit.csv \
  --require-no-late-births
```

The audit also reports identities that disappear and later reappear, together
with their maximum annotation gaps. If late births exist, do not use strict
birth suppression without extending the model to handle confirmed late births.

## 2. Evaluate the raw baseline with the organizer export

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.metrics \
  /mnt/lexar4tb/multi_uav_lts/outputs/train_baseline/predictions \
  --truth-dir /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels \
  --output-json /mnt/lexar4tb/multi_uav_lts/outputs/train_baseline/metrics.json \
  --sequence-summary-csv \
    /mnt/lexar4tb/multi_uav_lts/outputs/train_baseline/metrics_by_sequence.csv \
  --alpha-summary-csv \
    /mnt/lexar4tb/multi_uav_lts/outputs/train_baseline/hota_by_alpha.csv
```

The command reports two families of metrics:

- `CODABENCH_HOTA`, `CODABENCH_MOTA`, and `CODABENCH_IDF1` reproduce the
  organizer's published scoring script. Its HOTA field is `HOTA(0)`, the first
  TrackEval localization threshold at IoU 0.05. The script averages every
  sequence row together with the `COMBINED_SEQ` row.
- `HOTA`, `DetA`, `AssA`, and `LocA` are canonical TrackEval diagnostics from
  the detection-weighted combined sequence. Canonical `HOTA` is the mean over
  IoU thresholds 0.05 through 0.95. The reported canonical MOTA and IDF1 are
  likewise computed from combined counts.

Use the `CODABENCH_*` fields as the configuration-selection objective. Retain
canonical HOTA and its components to diagnose whether a change primarily helps
detection, association, or localization. The benchmark text format does not
expose ignored regions, so the local evaluator does not perform MOTChallenge
distractor or ignored-region preprocessing.

## 3. Apply conservative fixed-population post-processing

The safest first experiment only maps tracker identities to the supplied seed
identities and removes tracks that cannot be connected to a first-frame seed:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.fixed_population \
  /mnt/lexar4tb/multi_uav_lts/outputs/train_baseline/predictions \
  --first-frame-label-dir \
    /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels_FirstFrameOnly \
  --output-dir \
    /mnt/lexar4tb/multi_uav_lts/outputs/train_fixed_population/predictions \
  --output-json \
    /mnt/lexar4tb/multi_uav_lts/outputs/train_fixed_population/summary.json
```

Defaults deliberately disable tracklet relinking and interpolation. This makes
the first comparison isolate the effect of preserving the known population and
seed identities.

A bounded relinking experiment can reconnect an unseeded fragment to a seeded
trajectory when the fragment begins shortly after the current trajectory end
and passes a constant-velocity, scale, and IoU compatibility cost:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.fixed_population \
  /mnt/lexar4tb/multi_uav_lts/outputs/train_baseline/predictions \
  --first-frame-label-dir \
    /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels_FirstFrameOnly \
  --output-dir \
    /mnt/lexar4tb/multi_uav_lts/outputs/train_fixed_population_relinked/predictions \
  --relink-max-gap 5 \
  --relink-max-cost 1.5 \
  --output-json \
    /mnt/lexar4tb/multi_uav_lts/outputs/train_fixed_population_relinked/summary.json
```

`--interpolate-single-frame` is optional and should be selected only by
held-out `CODABENCH_HOTA`. Blind interpolation can increase false positives or
reduce localization quality at stricter IoU thresholds.

## 4. Select parameters with scenario-stratified cross-validation

Use the dedicated cross-validation runner for final parameter selection. It
builds deterministic folds stratified by sequence prefix, evaluates every
configuration on each held-out fold, and ranks by mean `CODABENCH_HOTA` with
lower fold-to-fold variance as the first tie-breaker:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.fixed_population_cv \
  /mnt/lexar4tb/multi_uav_lts/outputs/train_baseline/predictions \
  --truth-dir /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels \
  --first-frame-label-dir \
    /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels_FirstFrameOnly \
  --output-dir \
    /mnt/lexar4tb/multi_uav_lts/outputs/fixed_population_cv \
  --fold-count 5 \
  --seed 0
```

Outputs include:

```text
fixed_population_cv/
  cv_ranking.csv
  cv_summary.json
  fold_assignments.csv
  best_predictions/
  configs/<configuration>/predictions/
```

`fold_assignments.csv` makes the split reproducible and shows the scenario
prefix assigned to each fold. The summary retains both fold-averaged Codabench
metrics and canonical TrackEval diagnostics.

For fast exploratory work, `fixed_population_grid` evaluates configurations on
the same selected sequence set and ranks by `CODABENCH_HOTA`. Do not use its
in-sample ranking as the final model-selection result.

## 5. Apply the selected configuration to test predictions

Use only parameter values selected on training folds:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.fixed_population \
  /mnt/lexar4tb/multi_uav_lts/outputs/test_baseline/predictions \
  --first-frame-label-dir \
    /mnt/lexar4tb/multi_uav_lts/extracted/TestLabels_FirstFrameOnly \
  --output-dir \
    /mnt/lexar4tb/multi_uav_lts/outputs/test_fixed_population/predictions \
  --min-seed-iou 0.5 \
  --relink-max-gap 5 \
  --relink-max-cost 1.5 \
  --output-json \
    /mnt/lexar4tb/multi_uav_lts/outputs/test_fixed_population/summary.json
```

Package and validate with the existing RaFT-UAV utilities:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.cli package-submission \
  /mnt/lexar4tb/multi_uav_lts/outputs/test_fixed_population/predictions \
  --template-zip /mnt/lexar4tb/multi_uav_lts/downloads/submission.zip \
  --output-zip \
    /mnt/lexar4tb/multi_uav_lts/outputs/test_fixed_population/submission.zip \
  --normalize \
  --sort-rows \
  --output-json \
    /mnt/lexar4tb/multi_uav_lts/outputs/test_fixed_population/validation.json
```

Run the coverage audit on the final prediction directory before upload. Store
the population audit, cross-validation ranking, fixed-population summary, final
validation, exact command line, git commit, and ZIP checksum together as
submission provenance.
