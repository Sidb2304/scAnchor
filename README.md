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

**Data volume is the strongest lever found.** Going from ~1.8k to ~3.4k real
cells (same checkpoint, same code) took donor-retrieval accuracy after
correction from flat/negative to a clear improvement. This wasn't tuned for —
it was the single biggest change across every experiment run.

**Where the method stands today** (continual-pretrained checkpoint, 3405
real cells, held-out batch never seen during training):

| metric | before correction | after correction |
|---|---|---|
| donor retrieval accuracy | 0.422 | **0.484** (improved) |
| cell-type kNN purity | 0.358 | **0.539** (improved) |
| batch-mixing purity (lower is better) | 0.220 | 0.431 (**worse**) |

The adversarial batch-discriminator term (see `model/batch_discriminator.py`)
converges correctly — its own loss settles near `log(n_batches)`, meaning the
discriminator is reduced to chance-level guessing, the intended adversarial
equilibrium. But that doesn't translate into better batch-mixing by a
kNN-based metric: a shallow discriminator reaching equilibrium only
guarantees invariance to what *it* can detect, not to finer local
neighborhood structure a kNN metric picks up. **This is the open problem** —
donor signal preservation is real and improving, batch-mixing is not yet
solved, and shipping this as "batch correction" without that caveat would be
dishonest. Candidate next steps: a stronger/deeper discriminator, reweighting
the loss terms, running on the full dataset rather than a subsample (the
strongest lever so far), or reconsidering whether kNN purity is the right
metric for what an adversarial-linear approach can realistically achieve.

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
