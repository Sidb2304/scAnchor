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

Early scaffold. Architecture, training loop, and evaluation protocol are
implemented; embedding extraction currently targets scGPT. Not yet validated —
see `docs/validation_plan.md` (Evaluation, below) before relying on results.

## Install

```bash
pip install -e ".[scgpt]"       # embedding extraction via scGPT
pip install -e ".[baselines]"   # Harmony / scVI / scib for comparison
```

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
