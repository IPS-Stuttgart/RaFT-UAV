# Guarded Multi-UAV LTS tournament

This workflow compares complete train-split prediction sets with the repository's
organizer-compatible Multi-UAV LTS evaluator. The unmodified tracker output is
always included as the `raw` control. A transformed candidate is selected only
when it clears every configured evidence gate; otherwise the result is an
explicit raw fallback.

The tournament is intended for model selection and ablation evidence. It does
not turn local training metrics into an official leaderboard result. Only a
Codabench evaluation of a separately prepared test submission can establish a
competition score or rank.

## Selection contract

For each candidate, the tournament evaluates:

- all selected sequences together;
- deterministic folds stratified by sequence-prefix scenario;
- every sequence prefix as a separate scenario group; and
- every sequence individually for paired uncertainty estimation.

The primary objective is mean held-out `CODABENCH_HOTA`. Canonical `HOTA`,
`DetA`, `AssA`, and `LocA`, together with `CODABENCH_MOTA`,
`CODABENCH_IDF1`, identity switches, and prediction counts, remain in the
scorecard for diagnosis.

A non-raw candidate is eligible only when all of the following hold:

1. every required prediction file is present;
2. mean cross-validation `CODABENCH_HOTA` improves by at least the configured
   margin;
3. the lower endpoint of the paired 95% sequence-bootstrap HOTA-gain interval
   clears its configured threshold;
4. mean `CODABENCH_MOTA` and `CODABENCH_IDF1` do not fall beyond their metric
   floors; and
5. no scenario prefix suffers more than the permitted HOTA drop.

Ranking is deterministic. Eligible candidates are ordered by mean CV HOTA,
paired confidence bound, IDF1, MOTA, and stable name tie-breakers. The raw
candidate is always eligible, so an inconclusive or harmful transformation
cannot become the selected output merely because it is the only experiment.

## Command-line use

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.tournament \
  /mnt/lexar4tb/multi_uav_lts/outputs/train_baseline/predictions \
  --candidate \
    fixed_population=/mnt/lexar4tb/multi_uav_lts/outputs/fixed_population_cv/best_predictions \
  --candidate alternative=/path/to/alternative/predictions \
  --truth-dir /mnt/lexar4tb/multi_uav_lts/extracted/TrainLabels \
  --output-dir /mnt/lexar4tb/multi_uav_lts/outputs/guarded_tournament \
  --fold-count 5 \
  --seed 0 \
  --expected-sequence-count 102 \
  --bootstrap-samples 5000 \
  --min-mean-hota-gain 0.001 \
  --min-ci-hota-gain 0.0 \
  --max-mean-mota-drop 0.002 \
  --max-mean-idf1-drop 0.002 \
  --max-worst-scenario-hota-drop 0.01
```

Use `--require-improvement` when a raw fallback should fail an automated job.
The tournament writes its evidence before raising that failure, so the rejected
comparison remains inspectable.

## Workstation2 workflow

The manually dispatched GitHub Actions workflow is:

```text
Multi-UAV LTS guarded tournament
```

It runs on:

```yaml
runs-on: [self-hosted, Linux, X64, nvidia-smi]
```

A safe first invocation uses `probe_only=true`. The probe records the runner,
NVIDIA environment, mounted paths, and observed training-sequence manifest
without installing the experiment environment or running an evaluation.

For a normal run, provide an existing raw prediction path and optional
semicolon- or newline-separated `NAME=PATH` entries through `candidate_specs`.
For example:

```text
fixed_population=/mnt/lexar4tb/multi_uav_lts/outputs/fixed_population_cv/best_predictions;
other_method=/mnt/lexar4tb/multi_uav_lts/outputs/other_method/predictions
```

Two optional expensive stages are available:

- `regenerate_raw_baseline=true` applies the maintained first-frame identity
  seeding patch to the external YOLOv12-BoT-SORT checkout and regenerates all
  raw training predictions at the requested image size;
- `run_fixed_population_cv=true` runs the existing scenario-stratified
  fixed-population grid and adds its `best_predictions` directory as a
  tournament candidate.

Both stages write to run-specific directories. Existing evidence is not deleted
or overwritten. The workflow serializes heavy runs through one concurrency
group and has a 12-hour job limit.

## Evidence bundle

Each completed or failed workflow run keeps an isolated result directory and
uploads it as a GitHub Actions artifact. A completed tournament contains:

```text
tournament/
  tournament_summary.json
  tournament_ranking.csv
  group_scores.csv
  sequence_deltas.csv
  selected_candidate.txt
  provenance.json
  selected_predictions/       # or selected_predictions.<archive suffix>
```

`tournament_summary.json` records the exact folds, scenario groups, guard
thresholds, selected candidate, raw-fallback status, metrics, confidence
intervals, and rejection reasons. `provenance.json` binds every candidate to a
content SHA-256 digest, byte count, file count, source path, Git commit, and
workflow run identifiers.

The workflow also retains the runner probe, NVIDIA inventory, Python package
lock snapshot, console logs, and any raw-regeneration or fixed-population-CV
summaries produced during the run.

## Interpreting an outcome

A transformed selection is evidence that the candidate beat the supplied raw
prediction set under the configured train-split protocol. It is not evidence
that the method beats an external state of the art or improves the hidden test
set.

A raw fallback is a valid and useful result. It means that every supplied
transformation was incomplete, statistically inconclusive, violated a
secondary-metric floor, regressed a scenario group, or failed to achieve the
minimum primary-metric gain. Such methods should remain experimental rather
than being merged into the submission path.
