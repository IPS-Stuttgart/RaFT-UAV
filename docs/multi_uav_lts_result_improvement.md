# Multi-UAV LTS result-improvement workflow

This workflow adds competition-style local metrics and a first-frame-seeded
post-processing stage for the Beyond Strong Baseline Multi-UAV Tracking LTS
benchmark.

The older `score-predictions` command remains useful as a compact IoU-0.5
smoke diagnostic. It is not suitable for selecting leaderboard configurations
because it does not compute HOTA over the standard localization thresholds or
the global identity assignment used by IDF1.

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

## 2. Evaluate the raw baseline with HOTA, MOTA, and IDF1

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

The evaluator follows the TrackEval HOTA, CLEAR, and Identity definitions for
single-class two-dimensional boxes:

- HOTA is averaged over IoU thresholds 0.05 through 0.95.
- Sequence HOTA statistics are combined by detection counts, as in TrackEval.
- CLEAR matching preserves previous-frame identity matches before maximizing
  localization similarity.
- IDF1 uses a global ground-truth-to-predicted-identity assignment.

The benchmark data do not currently expose ignored regions through the LTS text
format, so this evaluator does not perform MOTChallenge distractor or ignored
region preprocessing.

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
held-out HOTA. Blind interpolation can increase false positives or reduce
high-threshold localization accuracy.

## 4. Tune against HOTA on training sequences

The grid runner evaluates every post-processing configuration with the exact
local metrics, ranks primarily by HOTA, and materializes the winning prediction
directory:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.fixed_population_grid \
  /mnt/lexar4tb/multi_uav_lts/outputs/train_baseline/predictions \
  --truth-dir /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels \
  --first-frame-label-dir \
    /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels_FirstFrameOnly \
  --output-dir \
    /mnt/lexar4tb/multi_uav_lts/outputs/fixed_population_grid
```

Outputs include:

```text
fixed_population_grid/
  grid_ranking.csv
  grid_summary.json
  best_predictions/
  configs/<configuration>/predictions/
```

For final tuning, run sequence-wise folds rather than selecting parameters on
all training sequences. Pass the held-out names with `--sequences` and aggregate
fold results outside the command. Scenario prefixes should be distributed
across folds so cloud, tree, building, partially out-of-view, takeoff, landing,
and large-target sequences are represented in every validation round.

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

Run the coverage audit on the final prediction directory before upload. The
fixed-population summary, grid ranking, final validation, exact command line,
git commit, and ZIP checksum should be stored together as submission
provenance.
