# Multi-UAV LTS beam/MHT relinking

The baseline fixed-population postprocessor assigns first-frame tracker IDs to the
supplied benchmark IDs and can reconnect later fragments. Its original relinker is
intentionally conservative: it solves one Hungarian assignment for each tracklet
start frame and commits that assignment immediately.

That local decision can be wrong when two UAVs are close. A fragment may initially
fit two seed trajectories almost equally well, while a later fragment makes only one
of the two assignments globally consistent. The beam/MHT relinker retains several
partial identity hypotheses and delays the hard choice until later tracklets provide
additional motion evidence.

## Run

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.fixed_population_beam \
  /mnt/lexar4tb/multi_uav_lts/outputs/train_baseline/predictions \
  --first-frame-label-dir \
    /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels_FirstFrameOnly \
  --output-dir \
    /mnt/lexar4tb/multi_uav_lts/outputs/train_fixed_population_beam/predictions \
  --relink-max-gap 5 \
  --relink-max-cost 1.5 \
  --relink-beam-width 16 \
  --relink-velocity-weight 0.25 \
  --output-json \
    /mnt/lexar4tb/multi_uav_lts/outputs/train_fixed_population_beam/summary.json
```

`--relink-drop-cost` defaults to `--relink-max-cost`. Dropping a fragment incurs
that cost multiplied by a sublinear tracklet-length factor and clipped mean
confidence. This makes a long, confident fragment harder to discard than a weak
one-frame fragment without forcing a link that fails the hard maximum-cost gate.

The transition cost contains the existing constant-velocity position, scale, IoU,
and gap terms. The beam variant additionally penalizes disagreement between the
recent seed-path velocity and the average candidate-tracklet velocity. Set
`--relink-velocity-weight 0` for an exact ablation of that term.

## Select parameters on training folds

Do not select the beam width or costs from the public leaderboard. Evaluate the
raw, greedy, and beam outputs with the repository's organizer-compatible scorer:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.metrics \
  /mnt/lexar4tb/multi_uav_lts/outputs/train_fixed_population_beam/predictions \
  --truth-dir /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels \
  --output-json \
    /mnt/lexar4tb/multi_uav_lts/outputs/train_fixed_population_beam/metrics.json \
  --sequence-summary-csv \
    /mnt/lexar4tb/multi_uav_lts/outputs/train_fixed_population_beam/metrics_by_sequence.csv
```

Use scenario-stratified held-out folds and rank by `CODABENCH_HOTA`. A compact
initial grid is:

```text
relink_max_gap:          2, 5, 10
relink_max_cost:         1.0, 1.5, 2.0
relink_beam_width:       4, 16, 64
relink_drop_cost:        1.0, 1.5, 2.0
relink_velocity_weight:  0.0, 0.25, 0.5
```

Start with beam widths 4 and 16. Width 1 is a useful local-decision ablation;
larger widths should be retained only when held-out HOTA or IDF1 improves.

## Diagnostics

The JSON summary records, per sequence:

- the number of generated beam hypotheses;
- the best total hypothesis cost;
- the cost margin to the second-best retained hypothesis;
- relinked and dropped track counts.

A small second-best margin identifies an ambiguous sequence. Those sequences are
priority cases for visual inspection, stronger temporal appearance features, or a
larger beam. The diagnostics do not use ground truth and are therefore also valid
for deciding whether a test sequence should fall back to the conservative greedy
postprocessor.

The implementation remains offline and deterministic. It never creates a new
benchmark identity: every retained fragment is attached to a supplied first-frame
seed, and fragments that cannot pass the motion gate are dropped.
