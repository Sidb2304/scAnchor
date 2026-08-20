# Changelog

All notable changes to this project are documented here. A version bump
happens when a real, validated finding lands (or, for 1.0.0, when the
public interface is declared stable) — not for infrastructure-only
commits.

## [1.6.0] - 2026-08-20

Completes Sinkhorn's validation against all three of MMD's original
validation axes, and closes out a broader architecture-exploration
thread with a real, convergent negative result.

**scIB atlas check (immune/pancreas/lung): the Levy pattern replicates,
not the Stephenson one.** sinkhorn_weight=0.5 vs mmd_weight=20,
seed-checked (3 seeds), same cached embeddings as the original MMD-only
scIB validation: on all three datasets, Sinkhorn loses to MMD on
batch-mixing but wins on cell-type purity, every time. Combined with
Levy: four independent real datasets show this consistent trade-off;
only Stephenson/Geneformer (the same underlying cells, two backbones)
showed a clean win on both axes. Honest conclusion: Sinkhorn's "beats
MMD on everything" result looks specific to Stephenson's particular
structure, not Sinkhorn's general behavior -- its real signature is a
systematic trade-off relative to MMD, not a strict improvement. This
completes all three of MMD's original validation axes for Sinkhorn.

**Three further architecture changes tried (isolated experiment,
scanchor-architecture-experiment/, not part of this package), on top of
the loss-mechanism exploration -- none escaped the curve:**
- Neighbor-attention (cross-attend to nearest neighbors in other
  batches): neutral/negative, essentially a wash vs. the published
  baseline.
- Mixture-of-experts by cell type: first attempt collapsed to one
  expert handling 20,411 of 21,336 cells (a real load-balancing
  failure, fixed with a standard entropy-based auxiliary loss). The
  fixed version's gate learned real cell-type structure (verified via
  mutual information, not assumed), but the corrected-embedding outcome
  was nearly identical to the broken version -- a properly-working
  implementation of cell-type-specialized capacity, still no
  improvement.
- Batch-statistic conditioning (condition on each batch's own raw-space
  mean/std instead of a fixed-vocabulary ID embedding, targeting the
  gap that new batches otherwise get an "unknown batch" fallback):
  worse than the plain baseline on both axes, not another point on the
  curve.

Six genuinely different interventions across both the loss-mechanism
axis and the architecture axis, including one independently verified to
be working exactly as designed, all converging on the same trade-off
surface -- real evidence this is a structural property of single-pass,
per-cell correction of a frozen embedding, not a fixable limitation of
any one mechanism. Documented as the honest current ceiling for this
paradigm rather than continuing to search for a mechanism that escapes
it; further progress likely requires changing what information the
correction step has access to (amortized OT, or a semi-transductive
design), not another loss or architecture variant.

## [1.5.0] - 2026-08-19

Tested whether combining mmd_weight and sinkhorn_weight in one
correction head escapes the batch-vs-bio trade-off curve, motivated by
their complementary weaknesses on Levy (MMD weak on donor
retrieval/cell-type purity, Sinkhorn weak on batch-mixing). A
single-seed grid initially looked like a real win -- every combined
config beat both individual mechanisms on donor retrieval -- but a
proper 3-seed check on the two most promising points showed this didn't
replicate (donor retrieval has real seed variance, std up to 0.043, that
one seed doesn't reveal). Honest result: mmd_weight=20+sinkhorn_weight=0.5
does show a real, seed-robust batch-mixing improvement over either
mechanism alone, but at a genuine cost to donor retrieval and cell-type
purity -- a different point on the same trade-off curve, not an escape
from it.

Getting the 3-seed check done at all required a real performance fix:
correction_loss loops over every batch-pair in Python, which at Levy's
8 batches is dozens of small sequential GPU kernel launches per
minibatch and made the single-seed sweep take ~50-55 min/seed for
Sinkhorn-containing configs. scripts/_vectorized_batch_losses.py batches
every pair into one op instead -- numerically verified equivalent to
losses.py (including a gradient check), a ~16x speedup. Kept separate
from losses.py rather than modifying those validated, tested functions
in place. Verified its divergence from the sequential version's exact
trained-model output (after enough SGD steps) is ordinary
floating-point summation-order non-associativity, not a bug -- loss
values matched to ~1e-6 at step 1, drifting to ~1e-3 by step 3.

## [1.4.0] - 2026-08-19

Sinkhorn checked against two of the three validation axes MMD's default
status originally rested on (cross-backbone, replicate-structure) --
findings honestly nuance the "beats MMD on both axes" result 1.3.0
shipped, not just confirm it further.

**Cross-backbone (Geneformer): the Pareto improvement replicates.** Same
Geneformer-embedded Stephenson cells as the earlier MMD cross-backbone
check, `sinkhorn_weight=0.5`, seed-checked (3 seeds): batch-mixing
regression +0.017 vs. MMD's +0.116 (~7x smaller), cell-type-purity gain
+0.083 vs. +0.092 (comparable). One new wrinkle: batch-mixing has much
higher seed variance on this backbone (std 0.029 vs. 0.008 on scGPT) --
one seed even improves batch-mixing past the raw baseline.

**Replicate-structure (private Levy dataset): a genuinely mixed result,
not a repeat win.** Located the real Levy astrocyte mini-village files
(previously unlocated -- required a real filesystem investigation across
the lab's data drive, documented for future reference) and ran the full
~81k-cell panel (no subsampling) through a real scGPT + GPU cluster
pipeline, seed-checked (3 seeds), sinkhorn_weight=0.5 vs. mmd_weight=20:
Sinkhorn wins clearly on donor retrieval (0.741 vs. 0.722) and cell-type
purity (+0.269 vs. +0.150 -- nearly double), but *loses* on batch-mixing
(+0.089 vs. +0.016) -- the opposite of its pattern on the other two
datasets. Also ~12x slower per seed (8 batches means up to 28 batch-pairs,
each needing Sinkhorn's 50-iteration solve vs. MMD's single kernel call).

Net: Sinkhorn is a real, validated *alternative* to MMD with a genuinely
different profile (better biological-signal preservation, worse
batch-mixing, much slower) -- not a strict improvement or a replacement
for MMD's default status. Still ships off by default
(`sinkhorn_weight: 0.0`).

Also fixed a real, separate performance issue surfaced by running at
Levy's true 81k-cell/8-batch scale (the first dataset in this project's
history run at this scale): `correction_loss` always computes
`mmd_loss`/`class_conditional_mmd_loss`/`sinkhorn_ot_loss` regardless of
their weight (by design, for diagnostic visibility), which is cheap at
Stephenson's 3-batch scale but genuinely expensive at 8 batches x 14 cell
types (up to 392 pairwise kernel/Sinkhorn computations per minibatch for
terms half of any given run doesn't even use). `scripts/run_levy_sinkhorn_comparison.py`
composes the loss by hand instead of using the shared wrapper --
100% behavior-preserving (a term at weight=0 already contributed exactly
0 to the total), purely a wall-clock fix for this comparison script, not
a change to `correction_loss`'s public behavior or its tests.

## [1.3.0] - 2026-08-18

New batch-mixing mechanism, `sinkhorn_weight` (entropic-regularized
optimal transport), added and shipped off by default. Every mechanism
validated before this one was a *moment-matching* loss (adversarial
discriminator, every MMD variant) and all of them landed on the same
batch-mixing-vs-cell-type-purity trade-off curve regardless of the
specific loss used. Sinkhorn is a matching-based mechanism from a
genuinely different class, motivated by the same evidence that motivated
trying neighbor-attention: since every moment-matching variant traced the
same curve, the limitation looked like it could be about mechanism class,
not the specific loss.

Real result on the Stephenson/scGPT reference panel (same dataset/split
as the published MMD numbers), seed-checked (3 seeds) across a weight
sweep from 0.1 to 20: at `sinkhorn_weight=0.5`, both batch-mixing
regression (+0.030 vs. MMD's +0.120) and cell-type-purity improvement
(+0.091 vs. MMD's +0.085) beat the published `mmd_weight=20` result
simultaneously — a genuine Pareto improvement on this dataset, not
another point on the same curve. Also found: the mechanism is
non-monotonic outside `sinkhorn_weight`≲2 (both metrics degrade together
at weight≥10, a real instability), so the validated range is narrower
than MMD's.

Shipped off by default (`sinkhorn_weight: 0.0`) despite the positive
result: it has only been validated on one dataset/split so far, not yet
checked against the donor-retrieval, replicate-structure, or
cross-backbone axes that earned MMD its default status. A real,
seed-checked bug was also found and fixed during development (the
Sinkhorn dual-update iteration accumulated instead of replacing f/g each
step, diverging to NaN regardless of epsilon) — covered by a regression
test (`test_sinkhorn_ot_loss_finite_not_nan_over_many_iterations`).

Ported from an isolated, ungitted architecture-experiment copy (kept
around separately, outside this repo, for further exploration — not part
of this package) that also tested a neighbor-attention correction head;
that idea did not beat MMD and was not ported.

## [1.2.0] - 2026-08-18

Cross-backbone validation: does scAnchor's characterized batch-vs-bio
trade-off hold with a structurally different foundation model, or is it a
scGPT-specific artifact? Re-embedded the identical Stephenson cells (same
seed, same per-donor cap, same 21,000 cells already used for the
published scGPT results) with Geneformer (V1-10M) instead, trained the
same validated config, seed-checked across 3 seeds.

Real, seed-robust result: the magnitude of scAnchor's effect is nearly
identical across backbones -- batch-mixing regression +0.116 (Geneformer)
vs. +0.120 (scGPT), cell-type-purity improvement +0.092 vs. +0.085, with
Harmony's near-flat pattern also replicating. This is evidence the
trade-off is a property of scAnchor's correction mechanism, not an
artifact of scGPT's embedding space specifically.

Getting this result required resolving several real, separate
Geneformer/cluster-infrastructure issues unrelated to scAnchor's own
code: a dependency chain needing a compiler-standard override and version
pins, a CUDA-driver/toolkit mismatch, a home-directory disk quota
repeatedly exhausted by CUDA-heavy conda environments (fixed by building
the environment on the lab's shared storage instead), and a scheduler
default silently requesting the wrong OS on GPU nodes (jobs sat in `qw`
indefinitely with no error, regardless of actual GPU availability, until
`-l operating_system=RedHat8` was set explicitly). Documented in
`geneformer_feasibility/` for anyone reproducing this.

## [1.1.2] - 2026-08-16

Real, verified finding, not just documentation: a plain `pip install
scanchor` fails on older HPC clusters (RHEL7-era, old GCC/glibc) because
`scanpy`/`anndata` transitively pull in unpinned `pandas`/`h5py`, which
stop shipping prebuilt wheels for old glibc past certain versions and fail
to build from source -- the exact same class of failure this project's
own cluster scripts had to work around repeatedly (pandas 2.3.3's
meson/cython build needing C99 support this cluster's GCC doesn't have by
default). Reproduced directly on this project's own cluster, then
verified the fix (`pip install "pandas<2.3" "h5py==3.14.0"` before
`pip install scanchor`) actually resolves it end-to-end, including a
successful `import scanchor`.

Documented in the README's Install section rather than pinned in
`pyproject.toml`'s own dependencies -- pinning globally would needlessly
hold back `pandas`/`h5py` for the majority of users on modern systems
where newer versions install from wheels without any issue.

## [1.1.1] - 2026-08-16

Packaging/distribution milestone, not a new finding -- same treatment as
1.0.0's API-stability milestone. First version published to PyPI
(`pip install scanchor`), via trusted publishing (OIDC, no stored token)
triggered on this release. Added PyPI-facing metadata to `pyproject.toml`
(classifiers, project URLs) and confirmed the package builds correctly
(`python -m build`) and the name `scanchor` was unclaimed before wiring up
the automation.

## [1.1.0] - 2026-08-16

### Added
- General technique, not just a Stephenson-specific fix: feeding a known
  condition/clinical covariate (anything correlated with which cells land
  in which batch, but not already captured by `batch` itself) as an
  additional `categorical_covariate_cols` entry. Validated via a seed-
  checked (0/1/2) ablation on Stephenson et al. 2021, reusing the already-
  embedded cells (no re-run of the scGPT step needed): adding `Status`
  alongside `batch` shrank the batch-mixing regression documented in
  v0.9.1 by about 23% on average, consistent direction at every seed, at
  no cost to cell-type purity. Documented as a general recommendation in
  the README's Configuration section, not only in Current results.

### Changed
- `scripts/run_stephenson_benchmark.py` now ships `categorical_covariate_cols:
  ["batch", "Status"]` as its default (previously `["batch"]` alone) and
  saves `Status` through the scGPT embedding step accordingly. The
  README's Stephenson comparison table now reports scAnchor's result
  under this default; the original `batch`-only numbers are kept visible
  alongside the ablation for comparison.

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
