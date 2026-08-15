# Changelog

All notable changes to this project are documented here. A version bump
happens when a real, validated finding lands (or, for 1.0.0, when the
public interface is declared stable) — not for infrastructure-only
commits.

## [1.0.0] - 2026-08-15

### Added
- Config schema (`configs/default.yaml`'s four top-level sections and
  their keys) declared the stable public interface — see README's new
  Configuration section for what "stable" commits to.
- GitHub Actions CI running the test suite on every push/PR.
- An integration test exercising `train()` end-to-end through
  `leave_one_batch_out`/`replicate_test` on synthetic data, not just the
  existing unit-level coverage of individual losses/model pieces.
- This CHANGELOG.

### Changed
- Tightened the README's Status section into one firm recommendation
  (`mmd_weight=20` as the shipped default and balanced middle ground,
  `mmd_weight=100` if batch-mixing matters most for a given use case)
  instead of leaving the trade-off for a reader to synthesize themselves.
- Reframed the batch-mixing-vs-cell-type-purity trade-off as a structural
  property of the problem this project has consistently found — every
  mechanism tried lands somewhere on the same curve, never off of it —
  rather than an open problem still being solved.
- Closed out the cross-study transfer asymmetry's exact driver (Next
  steps) as a documented, known limitation: isolating it needs a third
  dataset or an artificially-sparsified source (real new data-collection
  effort, not more analysis of what's in hand), so it's not blocking 1.0.

## [0.9.1] - 2026-08-15
Explains Stephenson's outsized batch-mixing regression: the held-out site
(Sanger) is 100% Covid patients, a disease-status population never
represented that way during training, and `Status` isn't fed to the model
as a covariate. Also fixes a real cache-bloat bug — `build_subsample()`
wasn't clearing `.uns` before writing, so unrelated full-647k-cell
structures (CITE-seq antibody data, a UMAP neighbor index) rode along on
every write/read, bloating the ~21k-cell subsample cache to 7.4GB.

## [0.9.0] - 2026-08-15
First real three-way comparison (scAnchor vs. Harmony vs. scDisInFact) on
identical cells from a genuinely independent public dataset (Stephenson
et al. 2021's COVID PBMC atlas). No method wins cleanly — each fails
differently.

## [0.8.2] - 2026-08-15
Corrects README framing: scDisInFact/CODAL/sysVI are raw-count generative
models, not like-for-like baselines to scAnchor's frozen-embedding
correction layer. Documents a real per-method feasibility verdict for
each (scDisInFact worth attempting, CODAL/sysVI not).

## [0.8.1] - 2026-08-15
Seed-checks the scIB benchmark results (3 seeds x 3 datasets): the
cell-type-purity win over Harmony is robust across all nine runs; the
batch-mixing result is dataset- and seed-dependent, not a clean win.

## [0.8.0] - 2026-08-15
First validation on the standard scIB atlas-level integration benchmarks
(immune, pancreas, lung) — the field's standard comparison point. Beats
Harmony on cell-type purity across all three.

## [0.7.2] - 2026-08-11
Rules out source-dataset size as the driver of the cross-study transfer
asymmetry: a Levy source matched to Jerber's exact cell count still
transfers in the same direction, just weaker.

## [0.7.1] - 2026-08-11
Tests the reverse cross-study transfer direction (Jerber → Levy): the
effect is asymmetric, not a general property of inductive transfer.

## [0.7.0] - 2026-08-11
Tests true cross-study zero-shot transfer for the first time: a
Levy-trained head applied, with no retraining, to Jerber — a completely
different study.

## [0.6.0] - 2026-08-11
Tests Jerber's day-30 timepoint; falsifies the day-11 "cell homogeneity"
hypothesis for why donor-retrieval didn't replicate on Jerber — the real
cause is sparse donor crossing, structural rather than biological.

## [0.5.0] - 2026-08-11
Adds a multi-scale MMD kernel; shifts the batch-mixing/cell-type-purity
trade-off curve to a different shape but doesn't dominate the
single-bandwidth version on all three metrics at once.

## [0.4.0] - 2026-08-11
Fixes a real noisy-bandwidth bug in class-conditional MMD (one shared
minibatch bandwidth instead of a separate one per cell type). Verdict
unchanged: still doesn't beat global MMD.

## [0.3.0] - 2026-08-11
Adds class-conditional MMD to test whether restricting MMD comparisons to
same-cell-type pairs fixes global MMD's cell-type-purity cost. Real data
refutes the hypothesis; not adopted.

## [0.2.0] - 2026-08-11
Finds the MMD batch-mixing ceiling (~`mmd_weight=100-200`); combining the
adversarial discriminator with MMD doesn't beat MMD alone.

## [0.1] - 2026-08-10
Documents two split-latent architecture attempts and the first Harmony
baseline calibration run.

## [0.0] - 2026-08-10
Initial scaffold: inductive batch correction head for frozen single-cell
foundation model embeddings.
