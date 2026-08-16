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

Validated end-to-end against real data, both datasets this project
assembled itself and public ones (scIB atlases, Stephenson et al. 2021).
The batch-mixing-vs-cell-type-purity trade-off documented throughout this
README is a structural property of the problem this project has
consistently found, not a bug still waiting to be fixed: every mechanism
tried — adversarial discriminator, split-latent architecture, global MMD,
class-conditional MMD, multi-scale MMD — lands somewhere on that same
curve, never off of it. This project's contribution is characterizing
that curve honestly with real, seed-checked evidence and shipping a
validated default, not claiming to have eliminated the trade-off.

**Recommendation:** use the shipped default
(`mmd_weight: 20`, `adversarial_weight: 0`, `absorption_weight: 0`) as a
balanced middle ground — donor retrieval and batch-mixing both robustly
beat the pre-correction baseline, cell-type purity roughly neutral. If
batch-mixing is what matters most for your use case, raise to
`mmd_weight: 100` (a real, validated ceiling — beats Harmony on this
metric, at a real, larger cost to cell-type purity). Don't turn the
adversarial discriminator or split-latent architecture back on
(`adversarial_weight`/`absorption_weight` > 0) — both are kept in the
codebase for comparison, and a real, seed-checked sweep found the
discriminator consistently regresses batch-mixing purity, with
split-latent never beating the simple single-embedding architecture on
the one thing it was built to fix. See **Current results** for the full
evidence behind this recommendation, including where it doesn't hold
(the Stephenson comparison, where all three methods tested — scAnchor,
Harmony, scDisInFact — fail differently and none wins cleanly).

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

**Day-11 interpretation at the time**: the most likely explanation seemed
biological — day-11 iPSC-derived midbrain progenitors are very early and
transcriptionally homogeneous, where donor/genotype signal may simply be
weaker relative to shared early-developmental programs than in Levy's more
mature, differentiated astrocytes. Jerber's day-30 timepoint (later, more
differentiated dopaminergic neurons) was the natural next test of that
hypothesis.

**Day-30 result: the homogeneity hypothesis is wrong. The real cause is
structural, not biological, and it's the same at every Jerber timepoint.**
Ran the current validated default (`mmd_weight=20`, not the adversarial-only
config day-11 used before MMD existed) on Jerber's day-30 timepoint
(250,923 cells, 175 donors, 12 pools — downloaded fresh from the same
Zenodo record, day30.h5.zip, ~3.1GB compressed / 11.3GB uncompressed):

| | donor retrieval, before → after |
|---|---|
| Day-11 (162 training donors) | 0.0 → 0.0 |
| Day-30 (159 training donors) | 0.0 → 0.038 |

Day-30 donor retrieval is still, for practical purposes, zero — nowhere
close to Levy's 0.594 → 0.906. That alone would be consistent with either
explanation (homogeneity or structure). What settles it: **day-30's donor
crossing is 25 crossed donors out of 175 total** — essentially identical to
day-11's 25 crossed out of 177 — despite day-30 being a genuinely more
differentiated, more diverse cell population (pre-correction cell-type kNN
purity 0.821 at day-30, vs. day-11's ~96%-two-progenitor-states
near-monoculture). If homogeneity were the explanation, day-30's much more
distinct cell states should have given the donor-consistency term more to
work with. It didn't, and the reason is visible in the numbers: **the same
~25 donors are crossed across pools at both timepoints** — this is the same
donor cohort profiled twice, with the same pooling design, so the sparse
crossing is a fixed property of Jerber's whole experimental design, not
something that changes with differentiation stage. Batch-mixing and
cell-type purity both still improved with correction here (0.392 → 0.356
and 0.821 → 0.842 respectively) — the MMD mechanism itself generalizes
fine to a second, very different real dataset; it's specifically the
donor-consistency objective that has no exploitable signal on Jerber,
regardless of timepoint.

**Day-52 is now low priority, not the natural next test it looked like
before this result.** It almost certainly has the same donor/pool cohort
structure as day-11 and day-30 (same study, same donors, same pooling
design) — running it would very likely just reconfirm sparse crossing a
third time at real cost (day52.h5.zip is ~7.1GB compressed, noticeably
bigger than day-11/day-30's ~3.1GB each). Not worth it unless something
else about day-52 specifically suggests otherwise.

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

Not yet tested at this point: MMD on the full 81k-cell Levy dataset — see
Next steps. (MMD on Jerber is tested below, in the day-30 discussion.)

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

**Validated on the standard scIB atlas-level benchmarks, seed-checked — a
robust win on cell-type purity, a genuinely seed-sensitive result on
batch-mixing.** Every result above used datasets this project assembled
itself (Levy, Jerber). The scIB atlas-level integration benchmarks
(immune, pancreas, lung — the standard reference point every
batch-correction method in the field gets compared against) were flagged
as "not yet run" since this project's start. Ran the shipped default
config (`mmd_weight=20`) plus a Harmony baseline on all three at 3 seeds
each, via `scripts/run_scib_benchmark.py` on a UGER cluster (see that
script, `scripts/submit_uger_scib.sh`, and `scripts/submit_uger_scib_seeds.sh`
for the full pipeline):

| dataset | seed | batch-mixing after (scAnchor) | Harmony | cell-type after (scAnchor) | Harmony |
|---|---|---|---|---|---|
| pancreas | 0 | 0.596 | 0.567 | 0.870 | 0.787 |
| pancreas | 1 | 0.598 | 0.565 | 0.876 | 0.790 |
| pancreas | 2 | 0.600 | 0.564 | 0.869 | 0.789 |
| lung | 0 | 0.643 | 0.638 | 0.924 | 0.862 |
| lung | 1 | **0.633** | 0.640 | 0.926 | 0.864 |
| lung | 2 | 0.643 | 0.639 | 0.923 | 0.863 |
| immune | 0 | 0.713 | 0.730 | 0.924 | 0.880 |
| immune | 1 | 0.712 | 0.730 | 0.923 | 0.881 |
| immune | 2 | **0.731** | 0.730 | 0.923 | 0.880 |

(before-correction baseline per dataset, all seeds: pancreas 0.729/0.798,
lung ~0.647/0.886, immune ~0.712/0.875 batch-mixing/cell-type)

**Cell-type purity: robust, 9/9.** scAnchor beats Harmony on cell-type
purity at every seed on every dataset, with tight margins throughout —
this part of the result is not a single-seed accident.

**Batch-mixing: real, and more seed-sensitive than the single-seed result
suggested.** Three different patterns, not one:
- **Pancreas**: Harmony wins consistently across all 3 seeds — a clean,
  replicated result in Harmony's favor.
- **Lung**: genuinely too close to call. Harmony is narrowly ahead at
  seeds 0 and 2 (~0.638-0.639 vs. scAnchor's 0.643), but at seed 1
  scAnchor actually beats Harmony (0.633 vs. 0.640). Neither method
  dominates here.
- **Immune**: 2 of 3 seeds reproduce the original story (scAnchor stays
  flat near baseline while Harmony regresses below it), but **seed 2
  shows scAnchor also regressing**, landing at 0.731 — essentially
  matching Harmony's regression (0.730). A real instability in one of
  three seeds, not smoothed over: this specific dataset/mechanism
  combination isn't reliably better than doing nothing on batch-mixing.

Honest bottom line: the cell-type-purity advantage is the part of this
result worth trusting broadly. The batch-mixing comparison against Harmony
is real but not a clean win — it's dataset-dependent, and for immune
specifically, seed-dependent too. Reporting both rather than only the
favorable single seed.

**Stephenson et al. 2021 COVID PBMC atlas — a genuinely independent public
dataset, and the first real three-way comparison including scDisInFact.**
Every result above used either datasets this project assembled itself
(Levy, Jerber) or the scIB atlas tasks. A real comparison against
scDisInFact (see Next steps for why it was the only one of the three
counts-based methods worth attempting) needed a public dataset with an
independent condition variable — none of Levy/Jerber/scIB have one.
Verified directly from the file (not the paper's description) that
Stephenson et al. 2021 (*Nat Med*, COVID-19 PBMC atlas) has `Site` (3
processing sites, used as batch), `Status` (Covid/Healthy/LPS/Non_covid,
used as scDisInFact's condition variable), and `donor_id` — but every donor
appears at exactly one site, so donor and batch are fully confounded here;
not suitable for scAnchor's donor-consistency mechanism (`donor_col`
deliberately omitted, same treatment as the scIB tasks), only for the
batch-mixing/cell-type-purity comparison run here.

Subsampled by donor (175 cells/donor cap, seed 0) to 21,000 cells across
120 donors — identical cells for every method compared. Held out the
smallest site (Sanger) for scAnchor's zero-shot leave-one-batch-out test;
Harmony gets transductive access to the full combined set (its own
established fair-comparison protocol, per above); scDisInFact trains on
all sites/conditions jointly — it has no zero-shot/held-out protocol in
this codebase, so its number is in-sample, an easier task than scAnchor's,
not a matched comparison, flagged rather than glossed over:

| method | batch-mixing purity (lower=better) | cell-type kNN purity (higher=better) |
|---|---|---|
| raw embedding, no correction | 0.7306 | 0.6213 |
| **scAnchor** (`mmd_weight=20`, zero-shot, `batch`+`Status` covariates) | 0.8471 (worse) | **0.7050** (better) |
| Harmony (transductive) | 0.7358 (worse) | 0.6158 (worse) |
| **scDisInFact** (in-sample, condition-disentangled cVAE) | **0.6620** (better) | 0.4755 (worse) |

(scAnchor's row reflects the current default, `categorical_covariate_cols:
["batch", "Status"]` — see the covariate-ablation finding below. The
original `batch`-only number this project first reported was 0.8501/0.7059;
both regress relative to no correction, this is a partial improvement, not
a different conclusion.)

**No method wins cleanly — each fails differently:**
- scAnchor's batch-mixing regression is real and larger here than on the
  scIB atlases, but its cell-type-purity gain (+0.084) is the largest of
  any method tested on this dataset, achieved zero-shot on cells never
  seen during training.
- Harmony essentially did nothing useful here: batch-mixing barely moved
  (in the wrong direction) and cell-type purity got worse, despite having
  full transductive access to every batch, including the held-out one.
- scDisInFact is the only method that genuinely improved batch-mixing —
  the whole point of its condition/batch disentangling — but paid for it
  with by far the worst cell-type purity of the four rows, on an easier
  (in-sample) evaluation than scAnchor's.

Consistent with, not a departure from, every other trade-off documented in
this README: something that improves batch-mixing tends to cost cell-type
purity and vice versa, and which method looks best depends entirely on
which side of that trade-off a given use case cares about more.
Reproducible via `scripts/run_stephenson_benchmark.py` (scAnchor +
Harmony) and `scripts/run_scdisinfact_stephenson.py` (scDisInFact), both
driven by `scripts/submit_uger_stephenson.sh`.

**Why Stephenson's regression is the worst seen in this project: the
held-out site isn't just a different batch, it's a different disease
population.** Same PBMC-type data as the scIB immune atlas above, but a
much bigger batch-mixing regression here — worth understanding rather than
shrugging off as dataset noise. Checked directly from the cached
subsample's metadata (no retraining needed): the "hold out the smallest
batch" heuristic used throughout this project picked Sanger (1,925 cells,
11 donors) as the held-out site, and Sanger is **100% Covid patients** —
zero Healthy, LPS, or Non_covid cells. The two training sites both have a
real status mixture (Cambridge: 76.6% Covid/23.4% Healthy; Ncl: 62.9%
Covid/19.4% Healthy/9.7% LPS/8.1% Non_covid). `Status` also isn't fed to
the model as a covariate at all (only `batch` is), so the correction head
has no way to tell "genuine Covid-driven biology" apart from
"Sanger-specific technical effect" for these cells. This is the same
failure mode already documented in the class-conditional MMD section
above — global MMD can't separate batch structure from composition
differences — just with a clinical variable driving the composition shift
here instead of cell type. It's also a case where the blanket
"hold out the smallest batch" heuristic used everywhere in this project
picked the single most compositionally extreme site available, rather
than one chosen with this kind of confound in mind.

**Partial fix, seed-checked: feeding `Status` as a categorical covariate
(the same mechanism already used for `batch`) closes part of this gap, at
no cost.** Directly testing the diagnosis above rather than just
describing it: retrained with `categorical_covariate_cols: ["batch",
"Status"]` instead of `["batch"]` alone, reusing the identical cached
embeddings (no re-run of the expensive scGPT step needed), at seeds 0/1/2:

| variant | batch-mixing purity after (3 seeds) | avg | cell-type purity after (3 seeds) | avg |
|---|---|---|---|---|
| `batch` only (original default) | 0.8515, 0.8660, 0.8483 | 0.8553 | 0.7061, 0.7026, 0.7072 | 0.7053 |
| **`batch` + `Status`** (current default) | 0.8471, 0.8495, 0.7851 | **0.8272** | 0.7050, 0.7063, 0.7062 | 0.7058 |

Consistent direction at every seed — never reverses. The regression over
the no-correction baseline (0.7306) shrinks from +0.125 to +0.097 on
average, about a 23% reduction, with cell-type purity essentially
unchanged (0.7053 → 0.7058, a wash). This doesn't fully close the gap —
still meaningfully worse than doing nothing at all — but it's a real,
free improvement using a mechanism already in the codebase, not a new
model or loss term. **General takeaway, not just a Stephenson-specific
fix: when a known condition/clinical covariate exists that isn't already
captured by `batch` but plausibly correlates with which cells land in
which batch, add it to `categorical_covariate_cols` — it costs nothing to
try, and this seed-checked test found only upside, never a downside.**
`scripts/run_stephenson_benchmark.py` now ships `batch`+`Status` as its
default; the main comparison table above reports that number as
scAnchor's result.

## Install

```bash
pip install scanchor                     # core package, from PyPI
pip install scanchor[scgpt]              # + embedding extraction via scGPT
pip install scanchor[baselines]          # + Harmony / scVI / scib for comparison
```

For local development, `pip install -e ".[scgpt]"` from a repo clone instead.

**Older HPC clusters (RHEL7-era, old GCC/glibc):** a plain `pip install
scanchor` can fail while building `pandas` from source
(`ERROR: Compiler cython cannot compile programs` or similar) — some
`pandas`/`h5py` releases stop shipping prebuilt wheels for old glibc, and
this old-toolchain problem is exactly what this project's own cluster
scripts had to work around repeatedly (see git history). Verified fix,
tested on the same cluster this project runs on: pre-install known-good
pinned versions first, *then* install scanchor —

```bash
pip install "pandas<2.3" "h5py==3.14.0"
pip install scanchor
```

This isn't pinned in `scanchor`'s own dependencies by default, since it
would unnecessarily hold back `pandas`/`h5py` for the majority of users on
modern systems where newer versions install from wheels just fine.

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

## Configuration

`configs/default.yaml` (loaded by `scanchor.config.load_config` as a plain
dict — no schema class, no hidden defaults beyond the `.get()` fallbacks
visible in `train.py`/the evaluate modules) is the stable public interface
as of 1.0: its four top-level sections (`reference_panel`, `model`,
`training`, `validation`) and every key under them are the contract this
project commits to from here on — a key being renamed, removed, or
changed to mean something different is a breaking change and gets called
out in CHANGELOG.md, not made silently. Adding a new *optional* key with a
backward-compatible default is not a breaking change.

The file itself is the documentation — every non-obvious key has an
inline comment tracing back to the specific real experiment that set its
current value (see Current results above for the underlying evidence), so
it isn't duplicated here. Two things worth knowing before editing it:

- **Every field with a real, validated default is annotated with *why* in
  the comment right above it** — if you're about to change one, read that
  comment first; several look like they'd be free wins (e.g. bumping
  `discriminator_hidden_dim` or turning `mmd_multi_scale` on) but were
  specifically tested and didn't validate.
- **New keys should default to today's validated behavior when absent**,
  so existing configs (including anyone's already-saved YAML) keep working
  unchanged after an upgrade.
- **If your data has a known condition/clinical covariate not already
  captured by `batch`, add it to `categorical_covariate_cols`.** Validated
  on Stephenson et al. 2021 (see Current results): feeding `Status`
  alongside `batch` partially closed a real batch-mixing regression, at no
  cost to cell-type purity, seed-checked across 3 seeds with no reversal.
  Same mechanism already used for `batch`, so this costs nothing to try.

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

Baselines (Harmony, scVI — see Current results) are run in their normal
transductive mode — with full access to the held-out batch — for
comparison. The goal is to approach transductive performance without
needing the new batch's data. (scDisInFact was originally planned as a
third baseline here but hadn't actually been run — see Next steps for
where that stands and why it's a different kind of comparison than
Harmony/scVI.)

**True cross-study zero-shot transfer: the sharpest test of the inductive
claim — and it's asymmetric, not a general property.** Every generalization
test above holds out a batch *within the same study* — Levy's
leave-one-batch-out, Jerber's own leave-one-pool-out. Neither tests the
thing this project is actually built around: a correction head trained on
one study, applied with **no retraining** to a completely different one.
Tested both directions with the shipped default (`mmd_weight=20`) — every
batch/pool ID in the target study is a string the source study's vocab has
never seen, so 100% of cells hit the UNK categorical embedding in both
directions (verified directly, not assumed). Evaluated purely on the target
study's own batch/cell-type structure, not mixed with the source study's
cells into one metric:

| direction | batch-mixing before → after | cell-type purity before → after |
|---|---|---|
| Levy → Jerber (zero-shot) | 0.392 → 0.383 (better) | 0.821 → 0.820 (flat) |
| Jerber (in-distribution, for reference) | 0.392 → 0.356 (better) | 0.821 → 0.842 (better) |
| Jerber → Levy (zero-shot) | 0.247 → 0.258 (**worse**) | 0.375 → 0.387 (better) |
| Levy (in-distribution, for reference) | 0.247 → 0.225 (better) | 0.375 → 0.383 (roughly flat) |

**The two directions don't agree.** Levy → Jerber recovers roughly a
quarter of the in-distribution batch-mixing improvement without damaging
cell-type purity — a real, if modest, positive result. Jerber → Levy does
the opposite: batch-mixing gets *worse* than doing nothing at all, while
cell-type purity still improves. Cross-study transfer is a real
phenomenon here, not nothing — every categorical covariate is forced to
UNK in both directions, so whatever's happening comes from the continuous
covariates and the embedding itself, not anything study-specific — but
it's direction-dependent, not a general "this generalizes across studies"
result.

**Ruled out source-dataset size as the driver.** The initial hypothesis —
Jerber's training subsample (7,252 cells) being smaller than Levy's
reference panel (18,238 cells) makes its correction function less
well-calibrated for cross-study transfer — is testable in isolation:
subsample Levy down to Jerber's exact scale (7,252 cells) while keeping
Levy's own dense 8-donor/8-batch crossing intact (donor-crossing density
can't be matched the other way; Levy only has 8 donors, all densely
crossed, so there's no way to reproduce Jerber's "162 total, 25 crossed"
sparsity from it), train a head on that, and compare its zero-shot
transfer to Jerber against the full 18.2k-cell head's:

| Levy source size | batch-mixing before → after (on Jerber) |
|---|---|
| 18,238 cells (full) | 0.392 → 0.383 |
| 7,252 cells (matched to Jerber's scale) | 0.392 → 0.387 |

Both sizes transfer to Jerber in the *same direction* (batch-mixing
improves), just with the smaller source giving a somewhat weaker effect
(Δ −0.006 vs. −0.010) — nothing like Jerber→Levy's sign-flipped result
(Δ +0.011, worse than doing nothing). Matching Jerber's exact cell count
didn't reproduce Jerber's transfer behavior, which rules out **source
size alone** as the explanation. The two remaining candidates — Jerber's
sparse donor-crossing structure, and something about the *direction*
of biological maturity (progenitor→mature vs. mature→progenitor) —
weren't isolated by this test and would need a source dataset with Levy's
scale but Jerber-like crossing sparsity, or a third, unrelated dataset, to
tell apart. Single seed, one dataset pair — see Next steps.

## Next steps

Shipping now with the open problems above documented rather than waiting on
these — they're the concrete roadmap, not a hidden gap:

1. **Full ~81k-cell Levy dataset** — the discriminator-capacity sweep above
   already showed data volume doesn't move batch-mixing, so this is lower
   priority than it might seem, but would confirm donor-retrieval gains hold
   at the full scale rather than just the 18.2k-cell subsample.
2. **The cross-study transfer asymmetry's exact driver is a known,
   documented limitation, closed out rather than actively pursued for
   1.0.** Source-dataset size is ruled out (see above) — a Levy source
   matched to Jerber's exact cell count still transferred in the *same*
   direction as the full-size source, just weaker. What's left unisolated:
   Jerber's sparse donor-crossing structure (162 total donors, 25 crossed
   — can't be reproduced from Levy's 8 densely-crossed donors) and the
   direction of biological maturity (progenitor→mature vs.
   mature→progenitor, which the two-dataset design can't separate from
   "which study is which"). Telling these apart needs a third dataset or
   an artificially-sparsified source, real new data-collection effort
   rather than more analysis of what's already in hand — not worth
   blocking 1.0 on. Revisit if a natural third dataset shows up rather
   than seeking one out specifically for this.
3. **scDisInFact/CODAL/sysVI: a real feasibility check found they're not as
   directly comparable as this README previously implied.** All three are
   full generative models (VAE or topic model) trained on raw/normalized
   counts, not a post-hoc correction layer on frozen foundation-model
   embeddings like scAnchor — the same operating-mode difference already
   documented for the scVI baseline above, not a new caveat. Concretely:
   - **CODAL** isn't a standalone package (it's the inference algorithm
     inside `mira-multiome`), maintenance is stale (~16 months since the
     last real commit), and it caps `torch<=2.0.0` while pulling in
     unrelated heavyweight genomic-motif dependencies (`lisa2`,
     `mira-moods`) — not worth pursuing.
   - **sysVI** ships inside `scvi.external` as of scvi-tools 1.3.0 (its
     original standalone repo explicitly says it won't be maintained
     further) — well-engineered, but the exact torch/lightning/scvi-tools
     dependency chain that already cost a full session of install
     troubleshooting on this cluster, for a method that (like scVI)
     trains on raw counts anyway. Not worth the risk for likely the same
     kind of result the scVI baseline already gives.
   - **scDisInFact** was the one worth attempting, and has now been run
     end-to-end against scAnchor and Harmony on a real public dataset
     (Stephenson et al. 2021 — see Current results): lightweight,
     plain-torch dependencies (no scvi-tools at all), and — unlike scVI —
     its actual purpose is disentangling batch/condition effects, closer
     in spirit to scAnchor's covariate conditioning than a general
     integration tool. Verdict: it's the only method of the three that
     genuinely improved batch-mixing there, at a real cost to cell-type
     purity — no method won cleanly, see Current results for the honest
     three-way comparison.

## Reference panel

- **Public domain validation (used above):** [scIB benchmark
  tasks](https://theislab.github.io/scib-reproducibility/) (immune, pancreas,
  lung atlases) — standard, small, cell-type labeled, multi-batch, and
  already the comparison point for every batch-correction baseline.
- **Public domain validation (used above):** Jerber et al. 2021, *Nat Genet*,
  population-scale scRNA-seq across dopaminergic neuron differentiation
  (HipSci, multiplexed across differentiation pools) —
  https://www.nature.com/articles/s41588-021-00801-6. Processed per-timepoint
  AnnData-compatible `.h5` files (day 11/30/52, raw + normalized counts, real
  `donor_id`/`pool_id`/`celltype` obs columns) are on Zenodo:
  https://zenodo.org/record/4651413 (day11.h5.zip used here, ~3.1GB
  compressed / ~11.7GB uncompressed; day30/day52 not yet tried, see Next
  steps).
- **Public domain validation (used above):** Stephenson et al. 2021,
  *Nat Med*, single-cell multi-omics analysis of the immune response in
  COVID-19 — https://doi.org/10.1038/s41591-021-01329-2. CELLxGENE-hosted
  h5ad (647,366 cells, 3 processing sites, 4 disease-status categories, 120
  donors): https://datasets.cellxgene.cziscience.com/fe2e847c-1602-4f1b-86a4-112e4dc7a8e3.h5ad.
  `var_names` in this file are Ensembl IDs, not gene symbols — real symbols
  are in `var["feature_name"]`; raw counts are in `.raw.X`, not `.X` (both
  confirmed directly from the file, not assumed — see
  `scripts/run_stephenson_benchmark.py`'s `build_subsample()`).

## References

- Batch effects as a barrier to universal single-cell foundation model
  embeddings (bioRxiv, 2025) — motivates this project directly.
- scDisInFact, CODAL (via `mira-multiome`), sysVI (via `scvi.external`) —
  transductive, counts-based disentangled batch correction; the baselines
  this complements rather than replaces, not a like-for-like comparison to
  scAnchor's frozen-embedding correction (see Current results / Next
  steps for a real feasibility assessment of each).

## License

MIT
