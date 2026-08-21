# Ambiguity-triggered beam association for Multi-UAV LTS

## Scientific hypothesis

The permissive proposal bank has very high diagnostic recall, but a single
minimum-cost path cover can still commit to the wrong identity at crossings.
Pairwise geometric costs may prefer the spatially nearest continuation even
when the resulting trajectory has an implausible velocity reversal. The
ambiguity beam tests whether retaining several close joint assignments and
using second-order future motion closes part of that association gap.

## Method

The implementation keeps the existing delayed atomic proposal graph and exact
optional matching as its control. For every connected short-link component it:

1. computes the maintained exact pairwise optimum;
2. leaves large components and decisive components unchanged;
3. for a small component, retains a bounded beam of one-to-one link sets with
   explicit unmatched alternatives;
4. always includes the exact pairwise optimum in the candidate set;
5. reranks only when the two best pairwise objectives are within the configured
   ambiguity margin; and
6. adds a clipped acceleration penalty after subtracting the estimated common
   image/swarm translation.

A hypothesis that joins two supplied frame-one seed identities receives
infinite higher-order cost. Consequently, the beam can change an ambiguous
unseeded continuation but cannot merge two known identities.

The pairwise optional-matching objective is preserved exactly. With
`max_link_cost = M`, an unmatched predecessor and unmatched successor each cost
`M/2`; selecting an edge of cost `c` therefore changes the objective by
`c - M`. The beam uses this same adjustment and adds higher-order evidence only
when comparing close complete hypotheses.

The implementation is deliberately bounded. It is not a sequence-wide dense
MHT: easy components use the exact original answer, components larger than the
configured node limit bypass the beam, and partial hypotheses are pruned to a
small deterministic population.

## Command

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.experimental_proposal_graph_tracker \
  /path/to/proposals \
  --first-frame-label-dir /path/to/TestLabels_FirstFrameOnly \
  --output-dir /path/to/beam/predictions \
  --enable-delayed-path-cover \
  --delayed-max-gap 0 \
  --delayed-lookahead-frames 2 \
  --delayed-successors-per-frame 3 \
  --enable-ambiguity-beam \
  --ambiguity-beam-width 8 \
  --ambiguity-beam-max-component-nodes 16 \
  --ambiguity-beam-margin 1.0 \
  --ambiguity-acceleration-weight 1.0 \
  --ambiguity-acceleration-clip 4.0
```

Useful controls are:

- `--ambiguity-beam-width`: number of complete hypotheses retained for final
  reranking;
- `--ambiguity-beam-max-component-nodes`: hard complexity guard;
- `--ambiguity-beam-margin`: maximum pairwise-objective separation that is
  considered ambiguous;
- `--ambiguity-acceleration-weight`: strength of the second-order residual
  motion term;
- `--ambiguity-acceleration-clip`: per-triple robustness cap; and
- `--ambiguity-beam-expansion-factor`: temporary partial-beam width before the
  final complete-hypothesis truncation.

Setting the acceleration weight to zero, using width one, exceeding the
component-size limit, or observing a decisive pairwise margin returns the exact
maintained pairwise result.

## Sequence cache

The experimental CLI now caches complete outputs per sequence by default. The
content key binds:

- exact proposal and seed bytes;
- normalized base-tracker arguments;
- every experimental control;
- the learned edge-model digest, when present; and
- source digests for the tracker, delayed association, beam, common-motion,
  sparse-matching, calibration, and cache modules.

This permits association-only evidence runs to resume across isolated output
directories without reprocessing already completed sequences. Changing one
sequence invalidates only that sequence. Use `--no-sequence-cache` for the
historical single-process path, or `--sequence-cache-dir PATH` to select a
specific persistent cache root.

## Initial regression

The focused crossing case contains two seeded UAVs that approach and cross.
With delayed future weight set to zero, the pairwise path cover selects the
nearest middle-frame edges and swaps both identities. The ambiguity beam retains
the slightly more expensive alternative and selects it because both resulting
trajectories have constant velocity rather than a reversal.

This is a mechanism regression, not dataset evidence.

## Evaluation contract

Compare at least:

```text
raw
pairwise delayed path cover
delayed + ambiguity beam
delayed + common motion + ambiguity beam
delayed + learned edge likelihood + ambiguity beam
```

Use complete out-of-fold predictions for all 102 training sequences and the
existing guarded tournament. A beam candidate is eligible only with:

- positive held-out `CODABENCH_HOTA` gain over the exact raw control;
- a non-negative paired-bootstrap lower confidence bound;
- acceptable MOTA and IDF1 changes;
- no excessive worst-prefix regression;
- complete sequence coverage; and
- immutable source, model, proposal, and prediction provenance.

The expected successful signature is higher AssA/IDF1 and fewer identity
switches. A gain caused primarily by extra output rows is not evidence that the
beam resolved the intended association failure.
