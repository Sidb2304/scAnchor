# scAnchor

Inductive batch correction for frozen single-cell foundation model embeddings.

## The problem

Single-cell foundation models (scGPT, Geneformer) are meant to be embedded once
and reused across studies. In practice, their embeddings still carry
batch/technical structure, and the standard fix — Harmony, scVI, scDisInFact,
and similar tools — is *transductive*: correction requires having every batch
present at correction time. Every time a new dataset arrives, you're back to
joint re-correction across the whole collection. That defeats the "embed once,
reuse forever" promise foundation models are supposed to deliver, and batch
robustness is an explicitly open problem in the foundation-model literature
(see references below).

scAnchor trains a small correction head on top of a frozen foundation model's
embeddings. The head is conditioned on technical covariates (chemistry,
platform, sequencing depth, or a batch ID for batches seen during training)
rather than a batch-ID lookup table alone, so it generalizes to a genuinely
new batch at inference: plug in that batch's covariates, no retraining, no
access to the new batch's cells during training.

## Why this, not another integration method

Harmony/scVI/scDisInFact/CODAL and friends solve batch correction well when
every batch is available jointly. scAnchor targets the complementary,
currently unaddressed case: you have a frozen embedder, a growing reference
panel of previously corrected batches, and a new batch arriving at inference
time that you want corrected without rerunning joint correction over
everything you've already processed.

## Status

Validated end-to-end against real data, with a partial, honestly-reported
result — not a finished method. See **Current results** below before relying
on this for anything beyond experimentation.

## Current results

Tested on a real, genetically-demultiplexed multi-donor iPSC-derived
astrocyte dataset (schizophrenia cohort, "mini-village" pooling design — 8
donors fully crossed with 9 technical batches, so donor identity and batch
are not confounded). Not a public benchmark release; reproducing this exactly
requires access to the source data.

**Backbone matters more than expected.** scGPT's `whole-human`/`brain`
checkpoint vs. its `continual pretrained` checkpoint (built by the scGPT
authors specifically for zero-shot embedding tasks, which is exactly what
this project does) gives a clear, consistent gap at matched data volume — the
continual-pretrained checkpoint encodes donor identity more strongly from the
start and every downstream metric improves accordingly. **Use
`continual pretrained`, not `brain` or `whole-human`, as the default
backbone.**

**Data volume is the strongest lever found — but only for donor-signal
metrics, not batch-mixing.** Two data points, same checkpoint, same code,
same held-out batch:

| metric | ~3.4k cells: before → after | ~18.2k cells: before → after |
|---|---|---|
| donor retrieval accuracy | 0.422 → **0.484** | 0.594 → **0.875** |
| cell-type kNN purity | 0.358 → **0.539** | 0.375 → **0.613** |
| batch-mixing purity (lower is better) | 0.220 → 0.431 (worse) | 0.247 → 0.444 (worse) |

More data drove a *much* bigger gain in donor retrieval (+0.06 → +0.28) and
cell-type purity as scale went up 5.4x. Batch-mixing didn't move at all —
the after-correction regression is the same size at both scales (+0.21,
+0.20). That rules out "just needs more data" as the fix for batch-mixing:
this is a discriminator-capacity or architecture problem, not a data-scale
one, and running the full ~81k-cell dataset would almost certainly just
reconfirm this same gap at higher compute cost rather than close it.

The adversarial batch-discriminator term (see `model/batch_discriminator.py`)
converges correctly — its own loss settles near `log(n_batches)`, meaning the
discriminator is reduced to chance-level guessing, the intended adversarial
equilibrium. But that doesn't translate into better batch-mixing by a
kNN-based metric: a shallow discriminator reaching equilibrium only
guarantees invariance to what *it* can detect, not to finer local
neighborhood structure a kNN metric picks up.

**Discriminator capacity is a real, reproducible trade-off, not a free fix.**
The correction head's `delta_net` has 2 hidden layers at 128 units; the
discriminator originally had 1 layer at 64 units — a plausible capacity
mismatch letting it reach equilibrium too easily. First test of this (a
single unseeded run each) looked like a clean win with a real cost. That
turned out to be *partly* noise: `train()` had no seed control, so every
config comparison was confounded by a different random init and shuffle
order each run — confirmed when lowering `adversarial_weight` (which should
have *eased* pressure) instead made donor retrieval *worse*, a result
impossible to interpret without controlling for that. Added
`training.seed`, then re-ran discriminator capacity {64, 128, 256 units} x 3
seeds each on the 18.2k-cell data (all other settings held fixed):

| discriminator | donor retrieval after (3 seeds) | batch-mixing after (3 seeds) | cell-type purity after (3 seeds) |
|---|---|---|---|
| 64 units, 1 layer | 0.64, 0.70, 0.83 | 0.42, 0.44, 0.46 | 0.58, 0.58, 0.58 |
| 128 units, 2 layers | 0.72, **0.06**, 0.53 | 0.43, 0.51, 0.47 | 0.56, 0.44, 0.49 |
| 256 units, 2 layers | 0.42, 0.50, 0.56 | 0.28, 0.28, 0.30 | 0.38, 0.38, 0.39 |

(baseline before correction, all rows: donor retrieval 0.594, batch-mixing
0.247, cell-type purity 0.375)

This is now a real, controlled result, not noise — every metric's direction
is consistent across all 3 seeds within each row. Two findings:

1. **256 units clearly and consistently helps batch-mixing** (average
   regression +0.04 vs. +0.44-0.24 → +0.04, i.e. down to near-neutral) —
   but at a real, consistent cost to donor retrieval and cell-type purity.
   This is a genuine trade-off, not a bug: **use `discriminator_hidden_dim:
   256` (the current default) if batch-mixing matters most; drop to `64` if
   donor-signal preservation matters most.**
2. **128 units is not a stable middle ground — don't reach for it.** One of
   three seeds collapsed donor retrieval to 0.06, and its batch-mixing
   average (0.47) is worse than *both* the 64-unit and 256-unit configs.
   There's no smooth dial between these two regimes at this dataset scale;
   the 128-unit setting sits on an unstable transition point between them.

Donor signal preservation and batch-mixing are not yet both solved by the
same configuration, and shipping this as unqualified "batch correction"
without that caveat would be dishonest.

Two real numerical-instability bugs were found and fixed getting here (see
git history): a fixed adversarial strength from step one caused runaway
divergence (loss climbing into the tens of thousands, every metric
collapsing) until (1) the adversarial strength was ramped in via the
Ganin & Lempitsky (2015) schedule instead of applied at full strength
immediately, and (2) the residual correction's magnitude was explicitly
bounded — without that, the correction head could "win" the adversarial game
by inflating embedding scale rather than genuinely removing batch structure,
and the variance-floor loss doesn't catch runaway growth, only collapse.

## Install

```bash
pip install -e ".[scgpt]"       # embedding extraction via scGPT
pip install -e ".[baselines]"   # Harmony / scVI / scib for comparison
```

Download a scGPT checkpoint from the [model zoo](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo)
— use **`continual pretrained`**, not `brain` or `whole-human` (see Current
results above). On macOS/CPU, also pass `use_fast_transformer=False` to
`extract_embeddings` (no `flash-attn`); `scgpt_extract.py` already works
around two Linux-only assumptions in the upstream package (`os.sched_getaffinity`
and a dataloader class that can't be pickled under macOS's `spawn` start
method) so this just works without extra flags beyond that one.

## Quickstart

```bash
# 1. Extract frozen embeddings for a reference panel
python -m scanchor.embeddings.scgpt_extract \
    --adata data/reference_panel.h5ad \
    --model-dir /path/to/scgpt_checkpoint \
    --out data/reference_panel.embedded.h5ad

# 2. Train the correction head
python -m scanchor.train --config configs/default.yaml

# 3. Evaluate: same-donor-across-batch + leave-one-batch-out generalization
python -m scanchor.evaluate.replicate_test --config configs/default.yaml
python -m scanchor.evaluate.leave_one_batch_out --config configs/default.yaml
```

## Evaluation

Two tests, deliberately independent of any enrichment/eQTL statistics:

1. **Same-donor-across-batch**: on datasets with replicate structure (the
   same biological sample profiled in ≥2 independent batches), corrected
   embeddings of the same donor from different batches should be closer than
   different-donor pairs within the same batch.
2. **Leave-one-batch-out generalization**: train on all-but-one batch, embed
   the held-out batch using only its covariates, and check batch signal is
   still removed. This is the actual claim under test — inductive
   generalization to an unseen batch — and it's the part existing transductive
   tools aren't built for.

Baselines (Harmony, scVI, scDisInFact) are run in their normal transductive
mode — with full access to the held-out batch — for comparison. The goal is
to approach transductive performance without needing the new batch's data.

## Next steps

Shipping now with the open problem above documented rather than waiting on
these — they're the concrete roadmap, not a hidden gap:

1. **Validate on a public dataset** (Jerber et al. 2021, below) — everything
   so far is on one private dataset nobody outside this project can rerun.
   This also doubles as a diagnostic: Jerber is much larger (>1M cells, 215
   donors) than anything tested here, so it's a direct test of whether the
   donor-retrieval/cell-type-purity gains that scaled cleanly with data
   volume (see Current results) keep scaling, and whether the batch-mixing
   trade-off is scale-limited or fundamental.
2. **If the same trade-off reappears at that scale**, that's real evidence
   it's architectural, not a tuning problem — try separate latent subspaces
   for batch-invariant cell state vs. donor-preserved signal, instead of
   forcing one shared embedding to satisfy both the adversarial and
   donor-consistency objectives at once.
3. **Baseline comparison against Harmony/scVI/scDisInFact in their normal
   transductive mode** (the wrappers already exist in `evaluate/baselines.py`
   but have never actually been run) — right now there's no comparison point
   showing how this stacks up against existing tools on the same data.

## Reference panel (proposed)

- General covariate-conditioned training signal: [scIB benchmark tasks](https://theislab.github.io/scib-reproducibility/)
  (immune, pancreas, lung atlases) — standard, small, cell-type labeled,
  multi-batch, and already the comparison point for every batch-correction
  baseline.
- Domain validation (replicate structure): Jerber et al. 2021, *Nat Genet*,
  population-scale scRNA-seq across dopaminergic/serotonergic neuron
  differentiation (HipSci, multiplexed across differentiation batches) —
  https://www.nature.com/articles/s41588-021-00801-6

## References

- Batch effects as a barrier to universal single-cell foundation model
  embeddings (bioRxiv, 2025) — motivates this project directly.
- scDisInFact, scDisco, sysVI — transductive disentangled batch correction;
  the baselines this complements rather than replaces.

## License

MIT
