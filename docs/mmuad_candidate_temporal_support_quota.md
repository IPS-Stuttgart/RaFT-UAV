# MMUAD temporal-support candidate quota

The branch-preserving candidate reservoir protects source and branch diversity,
but its quotas remain frame-local. A coherent UAV candidate can therefore be
ranked below isolated high-score distractors inside the same source/branch cell
and disappear before pair-state inference or mixture-MAP sees it.

This helper adds a truth-free temporal recall guard. For every candidate it
checks the nearest earlier and later candidate frame in the same sequence.
Motion-compatible neighbours contribute support when the implied speed is below
a configurable limit. The strongest supported candidates are added to the
ordinary branch reservoir before the final per-frame cap.

The helper does not estimate a trajectory and does not use ground truth for
selection. It only preserves hypotheses for the maintained downstream
pair-state and learned-sigma Huber mixture-MAP stages.

## Example

```bash
python scripts/mmuad_candidate_temporal_support_quota.py \
  --candidate-csv raw=outputs/mmuad_raw_candidates.csv \
  --candidate-csv translated=outputs/mmuad_translated_candidates.csv \
  --candidate-csv dynamic=outputs/mmuad_dynamic_candidates.csv \
  --output-csv outputs/mmuad_temporal_reservoir/candidates.csv \
  --summary-json outputs/mmuad_temporal_reservoir/summary.json \
  --global-top-n 20 \
  --per-source-top-n 3 \
  --per-branch-top-n 3 \
  --max-candidates-per-frame 40 \
  --temporal-top-n 2 \
  --max-frame-gap-s 1.0 \
  --max-speed-mps 60 \
  --distance-scale-m 5
```

For a local diagnostic, add train/public-validation truth only for the oracle
tables:

```bash
  --truth-csv path/to/truth.csv \
  --oracle-frame-csv outputs/mmuad_temporal_reservoir/oracle_frames.csv \
  --oracle-summary-csv outputs/mmuad_temporal_reservoir/oracle_summary.csv \
  --oracle-by-sequence-csv outputs/mmuad_temporal_reservoir/oracle_by_sequence.csv
```

Truth is not used to compute temporal support or select the reservoir.

## Row diagnostics

The output preserves the original candidate rows and adds:

- previous/next compatible distance, speed, and time difference;
- `candidate_temporal_support_sides` in `{0, 1, 2}`;
- `candidate_temporal_support_score`, using an exponential distance kernel;
- `candidate_temporal_two_sided`;
- `temporal_support:*side` reservoir provenance for quota-selected rows.

## Recommended train-CV experiment

Keep the existing learned-sigma Huber mixture configuration frozen and select
the temporal quota on training folds:

```text
temporal_top_n:       0, 1, 2, 3
max_frame_gap_s:      0.25, 0.5, 1.0
max_speed_mps:        20, 40, 60, 100
distance_scale_m:     2, 5, 10
min_support_sides:    1, 2
same-source/branch:   off, on
```

Primary diagnostics:

```text
full-reservoir oracle MSE
top-10/top-20 oracle MSE
train-CV mixture-MAP MSE
candidate rows per frame
fraction added only by temporal quota
```

The intended success mode is better top-K recall on hard sequences without
materially increasing the candidate budget. A failure is also informative: it
would show that deeper good candidates are not recoverable through simple local
motion persistence and that the pair-state unary/transition model needs the
next improvement instead.
