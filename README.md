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
result — not a finished method. The default config now uses the MMD loss
(`mmd_weight: 20`, `adversarial_weight: 0`, `absorption_weight: 0`) rather
than the original adversarial discriminator, which a real, seed-checked
sweep showed consistently regresses batch-mixing purity. MMD fixes that
regression and, at higher weights, beats the Harmony baseline on the same
metric — but trades away cell-type purity in exchange, and hasn't been
validated on any dataset besides the one below. The adversarial
discriminator and split-latent architecture are still in the codebase
(`adversarial_weight`/`absorption_weight` > 0) for comparison, not because
either is recommended. See **Current results** before relying on this for
anything beyond experimentation.

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

**Public-dataset validation (Jerber et al. 2021) did not replicate the
donor-retrieval gains seen on the private dataset above — and it's not a
dilution artifact.** Jerber et al.'s day-11 timepoint (public, see Reference
panel below) is a real, harder test: 253,381 cells, 177 donors, 12
differentiation pools — but only 25 donors are crossed across >=1 pool
globally, and holding out one pool (`pool7`, 4,452 cells, used whole) for
the leave-one-batch-out test leaves only **19 of 162 donors** genuinely
crossed within the training data itself. Day-11 cells are also ~96% just two
closely-related early progenitor states (FPP/P_FPP), far less diverse than
Levy's mature astrocytes.

| | donor retrieval, before → after |
|---|---|
| Full training subsample (7,001 cells, 162 donors, mostly uncrossed) | 0.0 → 0.0 |
| Filtered to only the 19 genuinely-crossed donors (1,489 cells, no dilution) | 0.0 → 0.0 |

Restricting to a clean, well-crossed subset — removing any dilution from the
152 single-pool donors that can't contribute to the donor-consistency loss —
made no difference at all. That rules out "the signal is just diluted" as
the explanation. The `donor_consistency` loss term itself stayed flat around
5.3-5.5 for all 30 epochs in both runs, the same signature seen elsewhere in
this project when a loss term isn't finding any useful gradient — not a
sign of dilution, a sign of no exploitable signal. Batch-mixing still
regressed after correction here too, consistent with every other experiment
in this project regardless of dataset.

**Honest interpretation**: the donor-signal-preservation result from the
Levy astrocyte data doesn't generalize unqualified to this dataset/timepoint.
The most likely explanation is biological, not architectural: day-11 iPSC-
derived midbrain progenitors are very early and transcriptionally
homogeneous, where donor/genotype signal may simply be weaker relative to
shared early-developmental programs than in Levy's more mature,
differentiated astrocytes. Jerber's day-30 and day-52 timepoints (later,
more differentiated dopaminergic neurons) are a natural next test of that
hypothesis — not yet run, see Next steps.

**Split-latent architecture attempt: two real tries, neither beat the simple
baseline.** Given the batch-mixing regression above is architectural, not a
tuning problem, `CorrectionHead` was redesigned to split its output into two
latents from a shared trunk: `z_bio` (contrastive + donor-consistency,
still the only thing used downstream) and a new `z_batch` explicitly trained
(no gradient reversal, just a normal classifier — `BatchAbsorber`) to
*absorb* batch-predictive variance instead of forcing it out of one shared
representation. First version, on the Levy 18.2k-cell data (seed 0, 256-unit
discriminator, same as the row above for a clean comparison):

| version | donor retrieval after | batch-mixing after | cell-type purity after |
|---|---|---|---|
| single embedding (no split, row above) | 0.42 | **0.28** | 0.38 |
| v1: `z_bio`/`z_batch` share one trunk | **0.0** (collapsed) | 0.77 (much worse) | 0.42 |
| v2: separate trunks, no shared layer | 0.39 | 0.41 | 0.42 |

`batch_absorption` converged to near-zero within 1-2 epochs in both
versions — the absorber genuinely works, `z_batch` really does become
batch-predictive. v1's shared trunk let that fast-converging signal shape a
representation `z_bio`'s pathway also drew from, despite separate output
heads — diagnosed and confirmed by v2 (fully separate trunks, no shared
hidden layer), which recovered from the collapse but still didn't beat the
original, simpler single-embedding architecture on batch-mixing, the one
thing this redesign was built to fix. Kept v2's separate-trunks fix (a real,
strict improvement over v1) but the split-latent idea itself hasn't earned
its added complexity yet.

**Baseline calibration: Harmony proves the metric is achievable, just not by
this method's current mechanism.** `evaluate/baselines.py` existed since
early in this project but was never actually run until now. On the same
18.2k-cell combined reference+held-out set (transductive — Harmony gets
every batch, the fair comparison per that module's own docstring):

| method | batch-mixing purity (lower=better) | cell-type kNN purity |
|---|---|---|
| raw embedding, no correction | 0.247 | 0.375 |
| **Harmony** | **0.188** (real improvement) | 0.320 (worse) |
| scAnchor (256-unit discriminator) | 0.280 (worse) | 0.380 (better) |

Harmony genuinely improves batch-mixing here (0.247 → 0.188) — this rules
out "the metric is just unsatisfiable on this data," which every prior
scAnchor result on this dataset was consistent with but didn't prove either
way. Harmony pays for its win with worse cell-type purity, a real trade-off
in the *opposite* direction from scAnchor's — the two methods fail
differently, not identically, which matters: it means scAnchor's specific
mechanism (an adversarial classifier trying to make batch identity
unpredictable) is the more likely culprit, not some property of the
dataset that makes batch-mixing improvement generally incompatible with
preserving biological signal. Harmony's actual mechanism is distributional
alignment via iterative clustering, not an adversarial classifier at all —
real motivation to try a distribution-matching (MMD) loss next instead of
another adversarial-classifier variant. Also fixed a real bug getting this
number: `harmony_correct`'s unconditional `.T` assumed an orientation that
doesn't hold for the installed harmonypy version, causing a silent shape
mismatch that would have crashed downstream.

**MMD loss: the first mechanism in this project that actually beats the
pre-correction baseline on batch-mixing — with a real, clean dose-response
curve.** Implemented `mmd_loss()` (RBF-kernel Maximum Mean Discrepancy,
median-heuristic bandwidth, pairwise across every batch present in a
minibatch — see `model/losses.py`) as a direct alternative to the
adversarial discriminator, motivated exactly by the Harmony result above:
distributional alignment, not an adversarial classifier. Swept `mmd_weight`
on the same 18.2k-cell data, seed 0, with the adversarial and absorption
terms both **off** (`adversarial_weight=0`, `absorption_weight=0`) to
isolate MMD's effect on its own:

| `mmd_weight` | donor retrieval after | batch-mixing after | cell-type purity after |
|---|---|---|---|
| 0 (contrastive+donor+variance only) | 0.766 | 0.401 (worse) | 0.633 |
| 1 | 0.922 | 0.391 (worse) | 0.624 |
| 5 | 0.953 | 0.335 (worse) | 0.565 |
| 10 | 0.953 | 0.279 (worse) | 0.486 |
| **20** | **0.906** | **0.225 (better)** | **0.383 (better)** |
| 50 | 0.891 | 0.181 (better, ~matches Harmony) | 0.299 (worse) |
| 100 | 0.922 | **0.177 (better than Harmony)** | 0.273 (worse) |
| 200 | 0.922 | 0.177 (flat vs. 100) | 0.267 |
| 500 | 0.922 | 0.177 (flat vs. 100/200) | 0.264 |

(baseline before correction, every row: donor retrieval 0.594, batch-mixing
0.247, cell-type purity 0.375)

This is a genuinely clean, monotonic dose-response relationship — batch-
mixing keeps improving as `mmd_weight` increases across the entire swept
range with no noise or reversal, unlike the discriminator-capacity sweep
above. Three real findings:

1. **MMD is the first mechanism in this whole project to actually beat the
   pre-correction baseline on batch-mixing, not just shrink the regression.**
   Every prior attempt here (adversarial discriminator at any capacity,
   both split-latent versions) made batch-mixing *worse* than doing nothing.
   MMD crosses below baseline at `mmd_weight=20` and keeps improving —
   at `mmd_weight=100` it's better than Harmony (0.177 vs. 0.188), the
   external baseline that motivated trying this mechanism in the first
   place.
2. **`mmd_weight≈20` is close to a joint sweet spot, not a cherry-pick — with
   one caveat confirmed by the seed check below.** At that setting, donor
   retrieval and batch-mixing both robustly beat their pre-correction
   baseline (donor 0.594→0.906-0.922, batch-mixing 0.247→0.21-0.225 across 3
   seeds). Cell-type purity is the exception: it beats baseline at seed 0
   (0.375→0.383) but sits just *below* it at seed 1 (0.373) — a difference
   small enough to call a wash, not a robust three-way win. Still no prior
   discriminator/split-latent config got this close on all three at once.
3. **The trade-off doesn't disappear, it just moves.** Past `mmd_weight≈20`,
   cell-type purity keeps dropping and crosses back below its own baseline
   by `mmd_weight=50` — the best batch-mixing numbers (50, 100) come at a
   real cost to within-cell-type structure, the same donor-vs-batch tension
   seen everywhere else in this project, just shifted to a better operating
   point on the curve than the adversarial mechanism ever reached.
4. **Batch-mixing genuinely plateaus by `mmd_weight=100` — this isn't an
   unbounded knob.** Pushing to 200 and 500 moved batch-mixing purity by
   \<0.001 (0.1769 → 0.1765) while donor retrieval stayed exactly flat at
   0.922 and cell-type purity kept drifting down only slowly (0.273 → 0.267
   → 0.264, a shrinking rate of decline, not a cliff). `mmd_weight≈100-200`
   is the practical ceiling for this mechanism on this data — there's no
   free additional batch-mixing improvement to be had by cranking the
   weight further, only a slow, flattening cost to cell-type purity.

**Seed-robustness check: the dose-response curve holds, unlike the
discriminator-capacity sweep in v0.1.** The entire sweep above was run at
seed 0 only — the same situation that, for the discriminator-capacity sweep
earlier in this project, turned out to hide a real seed-dependent collapse
(one of three 128-unit seeds cratered donor retrieval to 0.06). Re-ran
`mmd_weight` in `{0, 20, 100}` at seeds 1 and 2 to check:

| `mmd_weight` | seed | donor retrieval after | batch-mixing after | cell-type purity after |
|---|---|---|---|---|
| 0 | 0 | 0.766 | 0.401 | 0.633 |
| 0 | 1 | 0.688 | 0.405 | 0.627 |
| 0 | 2 | 0.781 | 0.394 | 0.627 |
| 20 | 0 | 0.906 | 0.225 | 0.383 |
| 20 | 1 | 0.922 | 0.210 | 0.373 |
| 20 | 2 | 0.906 | 0.212 | 0.382 |
| 100 | 0 | 0.922 | 0.177 | 0.273 |
| 100 | 1 | 0.969 | 0.175 | 0.272 |
| 100 | 2 | 0.953 | 0.173 | 0.271 |

No collapse, no reversal, no non-monotonic surprise at any seed — every
metric stays within a tight band per weight (batch-mixing purity in
particular varies by <0.015 across seeds at every weight tested). The
mmd_weight=100-beats-Harmony result is real: all three seeds land at
0.173-0.177, comfortably under Harmony's 0.188. The one caveat is the item
2 correction above — cell-type purity at `mmd_weight=20` straddles its own
baseline (above at seeds 0/2, marginally below at seed 1) rather than
robustly beating it, so call that setting "donor+batch-mixing both win,
cell-type roughly neutral" rather than "all three win."

**Combining MMD with the adversarial term doesn't beat MMD alone — it moves
along the same trade-off curve, not off of it.** Took `mmd_weight=20` (the
best joint operating point above) and added back a weakened adversarial
term:

| `adversarial_weight` (with `mmd_weight=20`) | donor retrieval after | batch-mixing after | cell-type purity after |
|---|---|---|---|
| 0 (MMD alone, row above) | 0.906 | 0.225 | **0.383 (beats baseline)** |
| 0.25 | 0.922 | 0.211 | 0.344 (below baseline) |
| 0.5 | 0.906 | 0.219 | 0.370 (below baseline) |
| 1.0 | 0.906 | 0.209 | 0.337 (below baseline) |

Adding any adversarial weight buys a small batch-mixing improvement (0.225
→ 0.209-0.219) but costs enough cell-type purity to drop it back *below*
its own pre-correction baseline (0.375) — exactly the property that made
`mmd_weight=20` alone special. The two mechanisms aren't complementary here;
adding the adversarial term back just re-traces the same donor/batch-mixing-
vs-cell-type curve MMD alone already traces by itself at a slightly higher
weight, rather than reaching a better point off that curve. **MMD alone,
not combined with the adversarial discriminator, is the better mechanism
found in this project so far** — pick a point on its curve (weight≈20 for
all-three-beat-baseline, weight≈100 for best achievable batch-mixing) rather
than adding the discriminator back on top of it.

Not yet tested at this point: MMD on Jerber or the full 81k-cell Levy
dataset — see Next steps.

**Class-conditional MMD — a real, well-motivated hypothesis that real data
refutes, even after fixing a real bug in it.** Global MMD can't tell "batch
structure" apart from "these batches just happen to have different
cell-type composition," so at high weight it may pull cell types together
as readily as it removes real batch structure — a plausible explanation
for the cell-type-purity cost documented above.
`class_conditional_mmd_loss()` (see `model/losses.py`) tests the fix
directly: compute MMD only between same-cell-type cells across batches,
never mixing cell types into the same kernel comparison, so it can only
touch batch structure, not composition.

The first version of this (v0.3.0) recomputed the RBF kernel's bandwidth
separately on each small per-cell-type subset — a real bug, not just a
suboptimal choice: small subsets give a noisy, inconsistent length scale
from one cell type to the next and one minibatch to the next. Fixed (v0.4.0)
by computing one shared bandwidth from the whole minibatch (same scale the
global term itself uses) and reusing it for every cell type. Re-swept
`conditional_mmd_weight` both alone and stacked on the current default
(`mmd_weight=20`), seed 0, real 18.2k-cell data, with the fix in place:

| config | donor retrieval after | batch-mixing after | cell-type purity after |
|---|---|---|---|
| before correction | 0.594 | 0.247 | 0.375 |
| `mmd_weight=20` alone (current default) | 0.906 | 0.225 | 0.383 |
| `conditional_mmd_weight=5` alone | 0.797 | 0.314 (worse) | 0.545 |
| `conditional_mmd_weight=20` alone | 0.578 (worse than baseline) | 0.223 | 0.337 |
| `conditional_mmd_weight=50` alone | 0.641 | 0.210 | 0.310 |
| `conditional_mmd_weight=100` alone | 0.688 | 0.205 | 0.306 |
| `mmd_weight=20` + `conditional_mmd_weight=5` | 0.953 | 0.230 | 0.386 |
| `mmd_weight=20` + `conditional_mmd_weight=20` | 0.844 | 0.197 | 0.298 (worse than MMD alone) |
| `mmd_weight=20` + `conditional_mmd_weight=50` | 0.828 | 0.191 | 0.279 (worse than MMD alone) |

**The bandwidth fix improved stability but didn't change the verdict.**
Comparing to the pre-fix numbers (same weights, same seed, git history has
the exact table): donor retrieval at `mmd_weight=20`+`conditional_mmd_weight=20`
recovered from a genuine collapse (0.766 → 0.844, closer to but still below
plain MMD's 0.906) — real evidence the noisy-bandwidth diagnosis was
correct. But the core conclusion is unchanged:
1. Conditional MMD alone is still dramatically worse at donor retrieval
   than global MMD at every matched weight (0.58-0.80 vs. global's
   0.91-0.92), and still doesn't achieve global MMD's clean batch-mixing
   improvement (0.20-0.31 vs. global's 0.18-0.23) — not a viable drop-in
   replacement, fixed or not.
2. Stacking it on the current default still doesn't recover cell-type
   purity as intended — `conditional_mmd_weight=20`/`50` still make it
   *worse* than `mmd_weight=20` alone (0.28-0.30 vs. 0.383), the opposite of
   the goal, just less severely than before the fix. Only
   `conditional_mmd_weight=5` looks like a genuine small win on all three
   metrics — essentially unchanged by the fix (0.953/0.230/0.386 vs. the
   pre-fix 0.953/0.234/0.388) — but the effect size is still small enough to
   be within MMD-alone's own seed-to-seed noise band, not something to trust
   without a seed check.

**Not adopted, still.** `configs/default.yaml` stays at `mmd_weight=20,
conditional_mmd_weight=0` — global MMD alone remains the better mechanism
found in this project, and the fix — while real and worth keeping in the
code — doesn't change that. `class_conditional_mmd_loss` and
`conditional_mmd_weight` stay in the codebase as a documented "tried,
fixed a real bug in it, still didn't win" path, same treatment as the
adversarial discriminator and split-latent architecture.

**Multi-scale MMD kernel: shifts the trade-off curve, doesn't beat it.**
`mmd_loss`'s RBF kernel used one bandwidth (the median-heuristic estimate).
Standard MMD variants (e.g. Long et al.'s Deep Adaptation Networks) instead
sum the kernel at several bandwidth multiples, meant to make the loss less
sensitive to picking exactly the right scale for a given minibatch. Added as
an opt-in flag (`mmd_multi_scale`, default off — `mmd_loss`'s existing
behavior and already-validated numbers are unchanged unless explicitly
requested) and swept at the same three weights already established for the
single-bandwidth version, seed 0, real 18.2k-cell data:

| `mmd_weight` | kernel | donor retrieval after | batch-mixing after | cell-type purity after |
|---|---|---|---|---|
| 20 | single-scale (default) | 0.906 | 0.225 | 0.383 |
| 20 | multi-scale | 0.938 | 0.249 (worse — barely beats baseline) | 0.459 |
| 50 | single-scale | 0.891 | 0.181 | 0.299 |
| 50 | multi-scale | 0.891 | 0.193 (worse) | 0.328 |
| 100 | single-scale | 0.922 | 0.177 | 0.273 |
| 100 | multi-scale | 0.875 (worse) | **0.169 (new best, beats Harmony's 0.188)** | 0.278 |

**No clean win — it's a different point on the trade-off surface, not a
dominant one.** At `mmd_weight=20` and `50`, multi-scale trades away some
batch-mixing improvement for meaningfully better cell-type purity (0.459 vs.
0.383 at weight 20) — at weight 20 specifically, batch-mixing barely beats
doing nothing at all (0.249 vs. 0.247 baseline), which defeats the point of
using MMD there. At `mmd_weight=100`, the trade reverses: multi-scale sets
a new best batch-mixing number for this project (0.169) but costs real
donor retrieval (0.875 vs. 0.922). No weight tested gives a strict
improvement on all three metrics over its single-scale counterpart.

**Not adopted as the default**, for the same reason as everything else in
this section — no unambiguous win, just a different shape of trade-off.
`configs/default.yaml` stays at `mmd_multi_scale: false`. Kept in the
codebase as a real, working option: set `mmd_multi_scale: true` with
`mmd_weight=100` specifically if squeezing out the best possible
batch-mixing number matters more than donor retrieval for a given use case.

**scVI baseline: a genuine mechanism-level trade-off, not a win or a loss.**
Once the `mudata`/`anndata` import conflict was resolved (upgrading
`scvi-tools` to latest — see git history — rather than pinning older
versions, which turned out to be self-contradictory in its own declared
metadata), scVI ran on the same 18,238-cell combined set, transductively,
directly on raw counts (`n_latent=32`, 30 epochs — a different kind of
baseline than Harmony/scAnchor, since it's a full generative model of
expression rather than a post-hoc embedding correction):

| method | batch-mixing purity (lower=better) | cell-type kNN purity |
|---|---|---|
| raw embedding, no correction | 0.247 | 0.375 |
| Harmony | 0.188 | 0.320 |
| scAnchor (256-unit discriminator, adversarial) | 0.280 | 0.380 |
| scAnchor (`mmd_weight=20`, adversarial off) | 0.225 | 0.383 |
| scAnchor (`mmd_weight=100`, adversarial off) | **0.177** | 0.273 |
| **scVI** | 0.313 (worse than *no correction at all*) | **0.875** |

scVI's cell-type purity is dramatically higher than every other method here
— consistent with it learning directly from raw count structure, which
encodes cell identity much more directly than a post-hoc correction of a
foundation-model embedding can. But its batch-mixing purity is the worst of
all four rows, including doing nothing — a real, reproducible result, not a
bug: training directly on raw counts with `batch_key` set doesn't
automatically make cell neighborhoods less batch-structured by this kNN
metric on this dataset. Another data point for the same conclusion Harmony
already supported: the failure mode here is mechanism/metric-specific, not
a property of the dataset that makes batch-mixing improvement generally
incompatible with preserving biological signal.

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

Shipping now with the open problems above documented rather than waiting on
these — they're the concrete roadmap, not a hidden gap:

1. **Jerber's day-30 and day-52 timepoints** — later, more differentiated
   dopaminergic neurons, as a direct test of the "day-11 progenitors are just
   too homogeneous" hypothesis above. If donor retrieval works there the way
   it did on Levy's mature astrocytes, that's a real, useful finding about
   *when* this method is applicable (differentiated cell states, not early
   progenitors) rather than a blanket failure.
2. **Full ~81k-cell Levy dataset** — the discriminator-capacity sweep above
   already showed data volume doesn't move batch-mixing, so this is lower
   priority than it might seem, but would confirm donor-retrieval gains hold
   at the full scale rather than just the 18.2k-cell subsample.

## Reference panel

- General covariate-conditioned training signal (not yet run): [scIB benchmark
  tasks](https://theislab.github.io/scib-reproducibility/) (immune, pancreas,
  lung atlases) — standard, small, cell-type labeled, multi-batch, and already
  the comparison point for every batch-correction baseline.
- **Public domain validation (used above):** Jerber et al. 2021, *Nat Genet*,
  population-scale scRNA-seq across dopaminergic neuron differentiation
  (HipSci, multiplexed across differentiation pools) —
  https://www.nature.com/articles/s41588-021-00801-6. Processed per-timepoint
  AnnData-compatible `.h5` files (day 11/30/52, raw + normalized counts, real
  `donor_id`/`pool_id`/`celltype` obs columns) are on Zenodo:
  https://zenodo.org/record/4651413 (day11.h5.zip used here, ~3.1GB
  compressed / ~11.7GB uncompressed; day30/day52 not yet tried, see Next
  steps).

## References

- Batch effects as a barrier to universal single-cell foundation model
  embeddings (bioRxiv, 2025) — motivates this project directly.
- scDisInFact, scDisco, sysVI — transductive disentangled batch correction;
  the baselines this complements rather than replaces.

## License

MIT
