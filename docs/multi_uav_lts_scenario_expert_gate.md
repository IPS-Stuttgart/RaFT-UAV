# Guarded Multi-UAV LTS scenario expert gate

This experiment selects one complete prediction source per sequence family while
retaining the raw tracker as a deterministic fallback. It is intended for cases
where an association or smoothing change helps one benchmark regime but harms
another.

The gate never consumes test truth. Training labels are used only to fit and
cross-validate a mapping from the public sequence prefix, such as `C`, `T`, or
`BB2P`, to a candidate name. Applying the frozen policy to test predictions
uses only the sequence filename and candidate prediction files.

The selected mixed prediction directory must still pass the repository's
organizer-compatible scorer and guarded tournament. The gate's own held-out
checks are an additional rejection layer, not a replacement for final scoring.

## Inputs

Generate complete training prediction directories for the raw control and each
candidate, then score each directory with the organizer-compatible evaluator:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.metrics \
  /path/to/raw/predictions \
  --truth-dir /path/to/TrainLabels \
  --sequence-summary-csv /path/to/raw_scores.csv

PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.metrics \
  /path/to/graph/predictions \
  --truth-dir /path/to/TrainLabels \
  --sequence-summary-csv /path/to/graph_scores.csv
```

Every score CSV must contain the same sequence set and columns corresponding to
`CODABENCH_HOTA`, `CODABENCH_MOTA`, and `CODABENCH_IDF1`. The loader accepts the
canonical names emitted by RaFT-UAV and a small set of case-insensitive aliases.
The `COMBINED_SEQ` row, when present, is excluded from prefix fitting because it
is not an independent sequence observation.

## Fit and cross-validate a policy

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.scenario_expert_gate fit \
  --score-csv raw=/path/to/raw_scores.csv \
  --score-csv graph=/path/to/graph_scores.csv \
  --score-csv tube=/path/to/tube_scores.csv \
  --candidate raw=/path/to/raw/predictions \
  --candidate graph=/path/to/graph/predictions \
  --candidate tube=/path/to/tube/predictions \
  --raw-candidate raw \
  --output-dir /path/to/scenario_gate \
  --fold-count 5 \
  --min-prefix-samples 3 \
  --min-train-hota-gain 0.001 \
  --prior-strength 3 \
  --max-train-mota-drop 0.002 \
  --max-train-idf1-drop 0.002 \
  --min-cv-hota-gain 0.0005 \
  --max-cv-mota-drop 0.002 \
  --max-cv-idf1-drop 0.002 \
  --max-worst-prefix-hota-drop 0.005 \
  --require-improvement
```

The fitter:

1. creates deterministic folds stratified by sequence prefix;
2. fits a prefix-to-candidate mapping on each training fold;
3. shrinks prefix-specific HOTA gains toward zero for small groups;
4. rejects candidates that violate training MOTA or IDF1 floors;
5. evaluates the fitted policy on held-out sequences;
6. requires a positive held-out HOTA gain, acceptable mean MOTA and IDF1, and
   an acceptable worst-prefix HOTA change; and
7. replaces every mapping entry with `raw` when any global gate fails.

The output includes:

```text
scenario_gate/
  policy.json
  fit_summary.json
  cv_sequence_rows.csv
  cv_fold_rows.csv
  predictions/             # optional mixed training predictions
  materialization.json     # present when --candidate inputs are supplied
```

`policy.json` records the exact thresholds, folds, candidate score sources and
checksums, per-prefix diagnostics, held-out deltas, rejection reasons, and the
final frozen mapping.

## Apply the frozen policy to test candidates

Generate every test candidate independently, then materialize the mixed result:

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.scenario_expert_gate apply \
  --policy-json /path/to/scenario_gate/policy.json \
  --candidate raw=/path/to/test_raw/predictions \
  --candidate graph=/path/to/test_graph/predictions \
  --candidate tube=/path/to/test_tube/predictions \
  --output-dir /path/to/test_scenario_gate/predictions \
  --output-json /path/to/test_scenario_gate/materialization.json
```

Candidate inputs may be directories or root-level prediction ZIP files. All
required candidates must contain the exact same filename set as the raw control.
Files are copied byte-for-byte from the selected source using atomic writes.
The materialization summary records the selected-candidate counts and a content
digest of the resulting directory.

## Interpretation and safeguards

A non-raw policy means only that prefix-specific selection improved the supplied
training score bank under the configured held-out protocol. It does not by
itself establish a hidden-test or leaderboard improvement.

Before packaging, compare the mixed training predictions against the raw
control in the guarded tournament. This final comparison is important because
the competition's `COMBINED_SEQ` contribution is not recoverable by averaging
independent sequence HOTA values, and because the actual mixed output may alter
association counts in ways not represented by a simple score lookup.

Do not fit policies from Codabench test feedback. Candidate definitions,
thresholds, and the final prefix mapping should be frozen from training-side
evidence before test materialization.
