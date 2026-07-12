# MMUAD pair-state group multiplicity correction

The branch-preserving MMUAD reservoir keeps raw, calibrated, dynamic, and
merged representations. Several rows can therefore originate from the same
physical point-cloud cluster. A pair-state forward-backward model sums over
candidate paths, so a physical hypothesis with several sibling rows can obtain
more posterior mass than an equally plausible singleton hypothesis merely
because it has more combinatorial paths.

This experiment corrects that representation multiplicity before temporal
inference. For candidate `i` in origin group `g`, the pair emission is changed
by

```text
log emission_i <- log emission_i - alpha * log(|g|)
```

At `alpha=1`, identical siblings collectively have the same emission mass as a
single candidate. Candidate coordinates and branches are not merged: the
robust mixture-MAP stage can still choose between raw and calibrated coordinate
hypotheses.

## Basic use

```bash
python scripts/mmuad_candidate_pair_group_correction.py \
  --candidate-csv outputs/reservoir/candidates.csv \
  --output-csv outputs/pair_group_corrected/candidates.csv \
  --summary-json outputs/pair_group_corrected/summary.json \
  --correction-strength 1.0
```

The group column is discovered from, in order:

```text
mmuad_calibration_origin_row
candidate_origin_row
origin_row
```

Use `--group-column` to select another column. With the default
`--missing-group-policy unique`, unidentified rows remain singleton groups.
Use `error` for strict experiments that require complete origin metadata.

## Agreement-adaptive pair prior

The corrected emissions can be passed directly through the existing
entropy-and-agreement adaptive pair prior:

```bash
python scripts/mmuad_candidate_pair_group_correction.py \
  --candidate-csv outputs/reservoir/candidates.csv \
  --output-csv outputs/pair_group_corrected/candidates.csv \
  --summary-json outputs/pair_group_corrected/summary.json \
  --correction-strength 1.0 \
  --agreement-adaptive \
  --min-pair-weight 0.0 \
  --max-pair-weight 0.75 \
  --entropy-power 1.0 \
  --agreement-power 1.0
```

## Direct robust mixture-MAP handoff

```bash
python scripts/mmuad_candidate_pair_group_correction.py \
  --candidate-csv outputs/reservoir/candidates.csv \
  --output-csv outputs/pair_group_corrected/candidates.csv \
  --summary-json outputs/pair_group_corrected/summary.json \
  --correction-strength 1.0 \
  --agreement-adaptive \
  --mixture-output-dir outputs/pair_group_corrected/mixture \
  --mixture-top-k 20 \
  --mixture-smoothness-weight 7200 \
  --mixture-huber-delta 1.0
```

Ground truth is never used to construct the corrected pair prior. A truth CSV
is accepted only for downstream mixture-MAP diagnostics.

## Suggested train-only experiment

Select all settings on train folds, then freeze them before public-validation
or hidden-test inference.

```text
correction_strength: 0.0, 0.25, 0.5, 0.75, 1.0
pair mode:            raw pair, entropy adaptive, agreement adaptive
pair max weight:      0.5, 0.75, 1.0
mixture top-K:         10, 20, 40
```

Primary diagnostics:

```text
train-CV pose MSE
candidate top-K oracle recall
posterior mass by physical group
fraction of frames whose top candidate changes
public-validation MSE after frozen train selection
```

The method should be rejected if group correction reduces train-CV top-K recall
or if gains arise only from public-validation tuning.
