# Multi-UAV LTS closed-world reassociation

The LTS protocol provides every target box and identity in frame one. After the
training population audit confirms that identities do not appear late, this is
stronger information than a generic MOT tracker normally receives: the tracker
can maintain a closed bank of known identities instead of creating unrestricted
new tracks.

The closed-world tool treats upstream tracker rows as detection candidates. It
outputs the exact supplied frame-one rows, globally assigns later candidates to
known identities, suppresses unsupported births, and can optionally emit short
motion predictions during detector gaps.

## Association model

For each seed identity, the tool predicts center, width, and height with a
constant-velocity/log-scale model. Frame-level Hungarian assignment combines:

- normalized center innovation;
- normalized Gaussian-Wasserstein similarity;
- IoU, automatically downweighted for tiny boxes;
- scale change and candidate confidence;
- a soft bonus for preserving the upstream source ID.

The source ID is deliberately not a hard constraint. A geometrically coherent
candidate can therefore absorb an upstream ID switch while retaining the
provided seed identity. Candidates that are not assigned to the seed bank are
removed instead of becoming late births.

Optional coasting is uncertainty-gated. A predicted row is emitted only for a
bounded gap, with decaying confidence, sufficiently stable recent motion, and
no strong overlap conflict with an accepted row. Coasting is disabled by
default.

## Select parameters on training data

First verify the fixed-population premise described in
[`multi_uav_lts_result_improvement.md`](multi_uav_lts_result_improvement.md).
Then run guarded scenario-stratified cross-validation:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.closed_world_cv \
  /mnt/lexar4tb/multi_uav_lts/outputs/train_baseline/predictions \
  --truth-dir /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels \
  --first-frame-label-dir \
    /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels_FirstFrameOnly \
  --output-dir \
    /mnt/lexar4tb/multi_uav_lts/outputs/closed_world_cv \
  --fold-count 5 \
  --seed 0
```

The default grid evaluates maximum gaps `5, 15, 30`, association costs
`1.5, 2.0, 2.5`, source-continuity bonuses `0.0, 0.2, 0.4`, and coasting off/on.
Use a smaller grid for a smoke run:

```bash
  --max-gaps 5 15 \
  --max-costs 1.5 2.0 \
  --source-continuity-bonuses 0.0 0.2 \
  --coast-options off
```

The raw prediction set is an explicit candidate. A transformed configuration is
eligible only when its held-out Codabench MOTA and IDF1 remain within the
configured floors and its HOTA gain reaches the requested minimum. Defaults are:

```text
maximum MOTA drop: 0.005
maximum IDF1 drop: 0.005
minimum HOTA gain: 0.000
```

Consequently, `best_predictions/` falls back to the raw baseline when every
closed-world configuration regresses or violates a secondary-metric guard. The
ranking records HOTA/MOTA/IDF1 deltas against raw, fold variance, dropped
candidates, absorbed source switches, and coasted rows.

Outputs are:

```text
closed_world_cv/
  cv_ranking.csv
  cv_summary.json
  fold_assignments.csv
  best_predictions/
  configs/<configuration>/predictions/
```

## Apply a selected configuration to test predictions

Use only values selected on held-out training folds. For example:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.closed_world \
  /mnt/lexar4tb/multi_uav_lts/outputs/test_baseline/predictions \
  --first-frame-label-dir \
    /mnt/lexar4tb/multi_uav_lts/extracted/TestLabels_FirstFrameOnly \
  --output-dir \
    /mnt/lexar4tb/multi_uav_lts/outputs/test_closed_world/predictions \
  --max-gap 15 \
  --max-cost 2.0 \
  --source-continuity-bonus 0.2 \
  --output-json \
    /mnt/lexar4tb/multi_uav_lts/outputs/test_closed_world/summary.json
```

Add `--emit-coasts --coast-max-gap 2` only when the held-out result selected
coasting. Package, coverage-audit, and validate the resulting directory with the
existing upload workflow.

## Interpretation

This stage can improve identity consistency and remove false births, but it
cannot recover a drone that is absent from all upstream candidate rows. Compare
canonical DetA and AssA in addition to the exported Codabench fields: a low DetA
ceiling indicates that the next experiment should target detector recall or
track-before-detect candidates rather than more association tuning.

No leaderboard improvement is claimed until the complete 102-sequence training
run and an official Codabench upload confirm it.
