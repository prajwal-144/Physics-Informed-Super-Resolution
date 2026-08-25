# Project assessment (v2) — scores, reasoning, and what moves each number

Re-scored 22 August 2026 after B4, B5, the injection experiment, and the refinement at scale.
Scores are calibrated against *what a NeurIPS-workshop reviewer or an astro-ML referee would
think*, not against effort. **↑/↓ marks a change from the 21 August version.**

---

## Summary

| # | metric | v1 | **v2** | one-line reason |
|---|---|---|---|---|
| 1 | **Novelty** | 6.5 | **7.5** ↑ | The four-representation comparison and the injection measurement are things nobody in this line has. |
| 2 | **Chance of acceptance at R/PS** | 7 | **8** ↑ | Better ML content, a novel measurement, and a cleaner single claim. |
| 3 | **Complexity** | 9 | **9** | Unchanged — already high. |
| 4 | **"Reads as AI-written"** | work 2 / prose 8 | **work 2 / prose 8** | The experimental record is unmistakably real; the prose still has strong tells. |
| 5 | **Fit to the GSoC project statement** | 8 | **8.5** ↑ | B4 is literally the original project's architecture, revived. Real galaxy sources still untouched. |
| 6 | **Performance on simulation data** | 9 | **9** | Already near the ceiling for this dataset. |
| 7 | **Performance on real data** | 1.5 | **1.5** | Still none. |
| 8 | **Physics** | 9.5 | **9.5** | Unchanged, still the strongest dimension. |
| 9 | **vs other ML4SCI DeepLense work** | 8 | **8.5** ↑ | You now beat the predecessor architecture *using* the predecessor architecture. |
| 10 | **ML / DL content** | 5.5 | **7** ↑ | Fully convolutional SR with sub-pixel upsampling, and three genuine architectural ablations. |
| 11 | **Evaluation honesty / integrity** | 9.5 | **9.5** | Four nulls reported as results. Two verification lapses, both caught before publication. |
| 12 | **Reproducibility** | 8.5 | **8.5** | Same strengths, one new stale-numbers risk. |
| 13 | **Statistical power** | 7.5 | **8.5** ↑ | n = 800 and n = 6000 for the refinement; 2,400 injections. |
| 14 | **Generalisation evidence** | 3 | **3** | Still one class, one instrument, one band. |
| 15 | **Code quality & documentation** | 9 | **9** | Same standard maintained across six new files. |
| 16 | **Astronomical impact (today / potential)** | 5 / 8.5 | **5.5 / 8.5** ↑ | The injection experiment is the first result with direct observational meaning. |
| 17 | **GSoC deliverable readiness** | 8.5 | **9** ↑ | A complete arc with a working final model and a measured limit. |
| 18 | **Risk of a reviewer finding a hole** | medium | **medium** | Different holes; see §18. |
| — | **Composite, research quality** | ≈7.5 | **≈8.0** ↑ | Physics-deep, now genuinely ML-competent, still simulation-only. |

---

## 1. Novelty — 7.5 / 10 ↑

**Still not new:** physics-informed unsupervised SR of lensing images (Shankar, Toomey & Gleyzer,
ML4PS 2024); CNNs regressing lens parameters (Hezaveh 2017, Perreault Levasseur 2017); neural and
pixellated source reconstruction (Morningstar 2019, Adam 2022, Karchev 2022); semilinear inversion
(Warren & Dye 2003, Suyu 2006); EPL deflection (Tessore & Metcalf 2015); SRResNet/ESPCN.

**What is new, and what B4/B5 added:**

1. **The four-representation comparison** — 64,516 free pixels, 4,096 free pixels, 14 + 1,024 from
   a globally-pooled vector, 7 + 2,304 from a fully convolutional decoder — all on one dataset, one
   forward model, one scorer. **New in v2:** the fourth point separates *conditioning* from
   *decoder architecture*, which the three-point version could not do. B3 and B4 are both
   comfortably overdetermined and B4 still halves the error.
2. **The coverage-versus-resolution trade-off**, now with a demonstration rather than only
   arithmetic: `--base sersic` at H = 1.2 shows no truncation artefact while free-form at the
   *larger* H = 1.6 still does.
3. **The injection measurement.** 2,400 injections of a 0.76-detector-pixel clump through the full
   instrument model at matched noise, measuring what comes back as a function of local
   magnification and SNR. **This is the single most novel thing in the project** and it directly
   answers the question every reviewer of an unsupervised-SR paper asks: *is the extra detail
   real?* Answer: 17–23% of injected contrast returns.
4. **Four independent nulls on magnification-adaptive regularisation, plus a measured
   explanation.** Tried as a linear-inversion prior, twice as a loss weight, and finally as a hard
   architectural gate. The explanation — ρ(recovery, SNR) = +0.152 against ρ(recovery, μ) = +0.056
   — turns four failures into one finding.
5. **The estimator-versus-initialiser decomposition**, now verified at n = 800 and *used* at
   n = 6000.

**Why not 8.5:** no new algorithm, no new objective, no theory. The novelty is in the experimental
design and the measurements.
**What moves it to 8.5:** a real galaxy image (Galaxy10 DECaLS) as the source, showing the
parametric fit smooths away structure that B4 recovers. That converts the model-family caveat from
a stated limitation into a demonstrated result, and it is about a day.

## 2. Chance of acceptance at R/PS — 8 / 10 ↑

**Improved because:** (a) the ML content objection is much weaker — you now have a fully
convolutional super-resolution model with sub-pixel upsampling and three real ablations;
(b) the paper has a *single* claim with four supporting data points instead of a list of results;
(c) the injection experiment is a novel measurement, not just a better number; (d) B4 being the
precedent's own architecture makes the comparison a controlled single-component swap, which reads
as rigour rather than rivalry.

**Remaining risks:** seven days; the stale `mu_resolution` numbers in the markdown (§18); axion-only;
and the framing decision — written as a super-resolution paper this is a mid-fit astro submission,
written as a representation paper it is a strong fit. That decision is entirely yours and it is
still the highest-leverage thing in the process.

**Estimate:** 8/10 executing the strategy document's plan; ~6/10 submitted as-is with the current
framing and the stale numbers.

## 3. Complexity — 9 / 10

Unchanged and still high: an EPL deflection re-implemented as a truncated hypergeometric series
with an adaptive term count, differentiable and matching `lenstronomy` to 2.7e−15; two
magnification routes cross-checked to 2e−9; a removable-singularity fix that is invisible in the
forward pass; a NumPy/PyTorch parity harness; a bounded staged least-squares fit; three network
architectures; and eight test suites asserting *measured* tolerances.

B4/B5 add: an adjoint back-projection with the "discard, don't clamp" rule, a documented half-pixel
`grid_sample` alignment convention (getting it wrong is 25% of the claimed super-resolution), a
flux-conserving magnification gate, and an injection harness that reuses the same noise realisation
for the with- and without-clump reconstructions.

## 4. Chance someone concludes "this was done by AI" — split, unchanged

**(a) The work: ~2/10 — no.** Git history with ordinary commit messages, checkpoints and figures
whose timestamps trace real working days, a retraction on 17 August that was itself withdrawn on
18 August after a decisive test, failed runs preserved (`Option2_failed/`, the v1 figures), four
null results kept and reported. B4 and B5 add to this: a documented pre-registration of thresholds
in `B4.md` *before* the run, then `B4_RESULTS.md` scoring against them. That is a research
practice, and it is hard to fake.

**(b) The prose: ~8/10 — yes, and the new documents are no different.** Em-dash rhythm, "not X —
Y", heavy bold lead-ins, headers as full declarative clauses, claim-then-parenthetical-number,
tables with a "why" column, phrases repeated verbatim across files.

**What to do, unchanged and still the honest version:** follow the venue's policy (check the R/PS
page and the NeurIPS LLM policy this cycle; authors are responsible for every claim regardless of
tools used, and drafting assistance is normal and permitted at essentially every venue). Then
**draft the four pages yourself from the tables** — not to disguise anything, but because a
uniform assistant register is worse writing than yours. Vary the surface features. And remember
that the real protection is that you can defend every number: if someone asks why the
regularisation power is 0.5, you have a measurement (source correlation 0.96 → 0.74 under raw 1/μ).

## 5. Fit to the GSoC project statement — 8.5 / 10 ↑

Stated project: *"developing an unsupervised super-resolution architecture to upscale the quality
of lensing images constructed using real galaxy sources, and to obtain insight about the lenses
themselves."*

| deliverable | status |
|---|---|
| unsupervised super-resolution architecture | **✔✔✔** now three of them, and the newest is the project's own original architecture running on a corrected lens |
| insight about the lenses | **✔✔✔** over-delivered: seven parameters, the magnification field, the critical curve to 0.63 px, total magnification to ~1% |
| **"lensing images constructed using real galaxy sources"** | **✘** still Model_A only, whose sources are analytic profiles |
| "bridge the gap to real images" | **✘** designed, not run |

**Why it went up:** B4 revives and vindicates the grid-based line the project descends from, which
is a better outcome for the programme than abandoning it. **Why it is not 10:** the real-galaxy
half of the title is untouched, and it is now a *one-day* experiment rather than a research
project — render lensed images from Galaxy10 DECaLS sources and show the free-form residual
recovers what the smooth fit cannot. This is the highest-value remaining item on every axis.

## 6. Simulation data — 9 / 10

Path A: ρ = +0.57 to +0.96 on seven parameters at n = 2000; source size ratio 10.65 → 1.081;
super-resolution fidelity flat 1×→8×; critical-curve swing to 1.5%; total magnification to ~1%.
B4: best free-form source in the project, nmse 0.0408, size ratio 1.043, peak ratio 0.927 — the
last two *better calibrated than the parametric fit*.

**Deductions:** axion-only; residual biases (θ_E +3.5%, R_sersic +10%) understood but not fixed;
χ²/dof ≈ 3000 explained but not reduced.

## 7. Real data — 1.5 / 10

No change. No HSC/HST/DES cutouts, and the controlled domain-gap step — degrade Model_A to
ground-based sampling and seeing, check recovery survives, which is fully validatable because
truth exists on both sides — is designed and not run. **The biggest single gap between the project
and its own title.**

## 8. Physics — 9.5 / 10

Unchanged and still the strongest dimension. B4/B5 preserve the standard: the back-projection is
the exact adjoint of the applied lensing operator and is validated by a round trip against the
true source (correlation 0.9989, median value error 3.3%) — **the very check that three of five of
the original stored operator sets would have failed**. The gate conserves flux block by block.
Surface brightness is still never multiplied by magnification anywhere.

## 9. Comparison with other ML4SCI DeepLense work — 8.5 / 10 ↑

| project | where you sit |
|---|---|
| **Shankar — Physics-Informed SR (2024) + Grid-based lensing** | The direct predecessor. **You now run his architecture on your lens (B4) and measure the difference** — the cleanest possible comparison, and it strengthens rather than displaces his contribution. |
| **Atal Gupta — diffusion SR (2024)** | Supervised, needs HR/LR pairs, heavier ML, no physics in the loop, no parameter recovery. They win on ML sophistication and real-source data; you win on physical validity, interpretability and validation. |
| **Pranath Reddy — SR (2023)** | Same axis. |
| **Lucas Jose / Ashutosh Ojha — physics-informed transformers** | Physics as architectural inductive bias, for classification. Yours is a hard forward-model constraint in the loss — the stronger sense of "physics-informed". |
| **Sreehari Iyer / Yashwardhan Deshmukh — SSL** | Self-supervised in the *contrastive* sense; yours in the *physics* sense. A nice one-sentence contrast for a representations workshop. |
| **Marcos Tidball — Domain Adaptation (ApJ)** | The gap you have not crossed; the template for your §7. |

**Ranking:** first in the repository on evaluation rigour and physical correctness; now roughly
mid-pack on ML sophistication (was bottom third); still bottom third on real-data engagement.

## 10. ML / DL content — 7 / 10 ↑

**What B4/B5 added:**
- A **fully convolutional super-resolution network** — residual blocks, long skip, sub-pixel
  (PixelShuffle) upsampling — and a stated architectural hypothesis (the global average pool is
  the bottleneck) that was then **tested and confirmed**: same conditioning, same physics, same
  loss family, error halved.
- **Three genuine ablations**: B3 decoder vs B4 decoder; `--base none` vs `--base sersic`; gate on
  vs off vs gate + weighted smoothing. v1 had only one.
- **A structural constraint rather than a penalty** (the magnification gate) — a real architectural
  idea, honestly reported as not helping.
- Deliberate, justified departures from the reference architecture: GroupNorm over BatchNorm
  (arc brightness varies 260× between images), softplus over ReLU (zero gradient on the negative
  side means a pixel that starts negative can never recover).
- The residual mode's `− ln 2` offset so that a zero network output is exactly the parametric
  solution — a small, correct piece of design.

**What is still missing:**
- **No uncertainty quantification.** Still the biggest omission, and now more glaring because you
  have a *measured, explained* bias (the direction-quantity shrinkage) that would become a reported
  error bar under a density head or a normalising flow.
- No equivariance, despite a direction-valued target on a rotationally symmetric problem.
- No width/depth ablation.
- The networks remain small and conventional (~0.85 M and ~0.64 M parameters).
- On the parameters, the classical optimiser still wins — though you now have the right answer to
  that (use the network as the initialiser) with strong numbers behind it.

## 11. Evaluation honesty and integrity — 9.5 / 10

Still exceptional, and B4/B5 strengthen it:

- **Thresholds were written down before the run.** `B4.md` sets pass/acceptable/fail bands for
  eight metrics; `B4_RESULTS.md` scores against them and reports five good, three acceptable.
  That is pre-registration in spirit.
- **Four null results reported as results**, including on the project's own signature idea.
- `B4_RESULTS.md` states plainly which knob was accidentally left off (`lambda_l2: 0.0`) rather
  than quietly rerunning.
- The "free-form fits the data better and recovers the source worse" observation is volunteered,
  not hidden.
- `eval_b4.py` prints the *flux-in-box ceiling* next to the measured value, so the metric cannot
  be read as better than physically possible.
- The "why the true source looks pixelated" section explains an apparent embarrassment correctly
  rather than cropping around it.

**Deductions (verification, not honesty):** two lapses now, both caught before publication —
the axion-only class truncation, and `MAGNIFICATION_SR.md` quoting an older, smaller
`mu_resolution` run than the JSON on disk, with a materially different low-SNR conclusion (3.1×
in the markdown, 1.25× in the data). **Fix the second one today**; it is the kind of discrepancy
that, found by a reviewer, costs you the benefit of the doubt everywhere else.

## 12. Reproducibility — 8.5 / 10

Unchanged strengths: fixed seeds, full config in every results JSON, figures that recompute
nothing, exact commands with expected outputs and explicit red flags, tests asserting measured
tolerances, and startup banners that print the conditioning ratio and warn if the requested
super-resolution exceeds the measured lens stretch.

**New risks:** results files now number in the dozens with similar names (`b4_free_*`,
`b4_sersic_*`, `b5_gate_*`, `b5_gate_curv_*`) and no manifest saying which figure belongs to which
run; and a markdown file now disagrees with a JSON. Both are ten-minute fixes.

## 13. Statistical power — 8.5 / 10 ↑

- The refinement result moved from **n = 25 to n = 800**, and was then *used* at **n = 6000**.
- The injection experiment is **2,400 injections over 60 images**, with a non-parametric test
  (Mann–Whitney, p = 5.3e−5) rather than an eyeballed difference.
- All source metrics are n = 800 or n = 1000.

**Remaining weaknesses:** still a single seed per training run, so the ~1% differences between B4
and B5 have no variance estimate — and those differences are exactly what the fourth null rests
on. **Two more seeds is the single highest-value statistics fix**, and it is a few GPU-hours. Also,
two Experiment A bins have n ≤ 3 and should simply be dropped from any published table.

## 14. Generalisation evidence — 3 / 10

Unchanged. One dataset, one instrument, **one band of two**, one dark-matter class, one source
family, one noise regime. The two-band point is newly documented and is a genuinely large lever:
both bands exist in every file, they correlate at 0.75, and **the lens is achromatic** — so a
two-band fit constrains one lens with twice the data while letting the source differ. Cheapest
gains, in order: cdm/wdm (30 min) → Galaxy10 sources (a day) → second band (2–3 days) → degrade to
ground-based conditions (a day).

## 15. Code quality and documentation — 9 / 10

The standard held across six new files. Docstrings still state *why*, with a measurement attached,
and still name the specific earlier failure being avoided ("that bug was caught by
tests/test_pathb.py check 3 and is not repeated here"). Same two deductions: doc drift (the B3
docstring still says ~0.5 M weights; it is 849,519) and deliberate duplication between
`sie_pipeline/` and `superres/`.

## 16. Astronomical impact — 5.5 today ↑ / 8.5 potential

**Why today moved up:** the injection experiment is the first result with a directly observational
statement attached — *structure smaller than a detector pixel is recoverable at 17–23% contrast,
and magnification helps but photon count helps three times more*. That is a sentence an
observational astronomer can act on when planning exposure times or target selection.

**Potential unchanged:** millisecond inference is survey-scale; the magnification field and
critical curve are what lensing cosmology needs; and the residual after a validated smooth model
is the substructure signal the whole DeepLense programme is about. **The highest-impact unexecuted
step is still the substructure one** — you have `image` − `image_nss` as the exact signal and
`kappa_sub` as the truth, and your residual figure already shows arc-shaped structure rather than a
smooth halo.

## 17. GSoC deliverable readiness — 9 / 10 ↑

You now have a complete arc: a diagnosis, a rewrite, three working models, a measured limit, four
honest negative results, a dataset bug report to the mentors, and a documentation set that a
successor could pick up cold. Missing for a top mark: a merged PR to `ML4SCI/DeepLense` with a
clean folder and README, a blog post in the style of the other projects, and the cross-class fix.

## 18. The holes a reviewer will find, ranked

1. **`MAGNIFICATION_SR.md` disagrees with `mu_resolution.json`** on both experiments, and the
   markdown's headline "3.1× at low SNR" is 1.25× in the data. Re-run once, quote the JSON, delete
   the stale tables.
2. **"All three DM classes" is false** — everything is axion.
3. **Single seed** behind the ~1% B4-vs-B5 differences that the fourth null rests on.
4. **Three χ²/dof conventions** across `evaluate.py`, `eval_pathb.py`, `eval_b4.py`.
5. **"source nmse" means two different things** in B3's tables (0.0845 Sérsic-only vs 0.0799 with
   the correction).
6. **"Levenberg–Marquardt" is loose** — the code calls `scipy.optimize.least_squares(method="trf")`,
   trust-region reflective. Say that.
7. **Two Experiment A bins have n ≤ 3** and should be dropped from published tables.

All seven are fixable in under two days; five of them in under two hours.

---

## 19. Three more metrics worth tracking

**Falsifiability — 9.5 / 10 ↑.** Already high, and B4/B5 raise it: thresholds written before the
run, a spatial-correspondence test that pokes one input pixel and asserts where the response lands,
a flux-in-box ceiling printed next to the measurement, a gate whose mean value is reported so a
no-op would be visible, and an injection experiment whose null hypothesis is explicit. Very few
projects build this many ways to be proven wrong.

**Time-to-insight per compute — 9.5 / 10 ↑.** The most valuable result of the last two days —
the injection experiment — required **no training at all**; it ran on an existing checkpoint. So
did the lens-model ladder, the PSF sweep, the `skill` null baselines and the gradient-share
analysis. That instinct, that the decisive experiment is usually a measurement rather than a run,
is the most transferable thing in this project and is worth naming explicitly in the GSoC report.

**Bus factor / inheritability — 9 / 10 ↑.** Six new files, each with a header that explains the
design decision and cites the measurement behind it, plus `B4.md` / `B4_RESULTS.md` /
`MAGNIFICATION_SR.md` written as if for a successor. Someone else could take this over from the
documents alone.

---

## 20. Bottom line

**Composite ≈ 8.0 / 10 as research; ≈ 9 / 10 as engineering and scientific practice; ≈ 7 / 10 as
machine learning.**

> This is unusually careful physics-informed inverse-problem work with an exceptional evaluation
> culture, now backed by a competent modern super-resolution model and a controlled four-way
> comparison of source representations. Its central measurement — that structure below the detector
> pixel scale is recoverable, and that magnification predicts where far less strongly than photon
> count does — is novel, honest, and explains four earlier null results at once. Its weaknesses are
> that all evidence comes from a single dark-matter class of a single simulated instrument using
> one of its two bands, that no uncertainty is attached to any prediction, and that the "real galaxy
> sources" half of the project title remains untouched — which is now a one-day experiment rather
> than an open problem.
