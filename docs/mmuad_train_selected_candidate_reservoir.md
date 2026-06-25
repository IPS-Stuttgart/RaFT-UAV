# Train-selected MMUAD candidate reservoirs

The candidate-reservoir train-CV command selects branch/source score offsets and
reservoir sizes using training sequences only. Apply that frozen JSON to
validation or hidden-test candidates with the inference-side command added here.

## 1. Select offsets on train sequences

```bash
raft-uav-mmuad-candidate-reservoir-train-cv \
  --candidate raw=/path/train_raw.csv \
  --candidate dynamic=/path/train_dynamic.csv \
  --candidate translated=/path/train_translated.csv \
  --truth-csv /path/train_truth.csv \
  --branch-score-offset-grid raw=-0.5,0,0.5,1 \
  --branch-score-offset-grid dynamic=-0.5,0,0.5 \
  --branch-score-offset-grid translated=-0.5,0,0.5 \
  --output-dir outputs/mmuad_reservoir_train_cv
```

The reusable artifact is:

```text
outputs/mmuad_reservoir_train_cv/
  mmuad_candidate_reservoir_train_selected_config.json
```

## 2. Apply the frozen config without truth

```bash
raft-uav-mmuad-apply-candidate-reservoir-config \
  --config-json outputs/mmuad_reservoir_train_cv/mmuad_candidate_reservoir_train_selected_config.json \
  --candidate raw=/path/target_raw.csv \
  --candidate dynamic=/path/target_dynamic.csv \
  --candidate translated=/path/target_translated.csv \
  --output-dir outputs/mmuad_reservoir_target
```

The command writes:

- `mmuad_candidate_reservoir_applied.csv`
- `mmuad_candidate_reservoir_apply_summary.json`
- `mmuad_candidate_reservoir_apply_provenance.json`

The provenance records the selected-config SHA-256, candidate inputs, train
selection label/protocol, cap mode, and row counts. No truth/reference file is
read by the apply command.

Pass the resulting CSV to the existing tracker or mixture-MAP experiment:

```bash
raft-uav-mmuad-track \
  --candidate-file outputs/mmuad_reservoir_target/mmuad_candidate_reservoir_applied.csv \
  --output-dir outputs/mmuad_reservoir_tracker
```

## Cap modes

`--cap-mode score` is the default and exactly reproduces the final score cap
used by the current train-CV selector.

`--cap-mode diversity` is an explicit ablation. It first builds the uncapped
branch/source reservoir and then reserves a minimum number of candidates per
source and branch before filling the remaining frame budget by score:

```bash
raft-uav-mmuad-apply-candidate-reservoir-config \
  --config-json /path/mmuad_candidate_reservoir_train_selected_config.json \
  --candidate raw=/path/target_raw.csv \
  --candidate translated=/path/target_translated.csv \
  --output-dir outputs/mmuad_reservoir_target_diverse \
  --cap-mode diversity \
  --diversity-min-per-source 1 \
  --diversity-min-per-branch 1
```

Keep `score` as the paper-valid default unless the diversity policy is also
selected on training data. Treat public-validation comparisons between cap
modes as diagnostics rather than hidden-test selection evidence.
