# Assessment v3 — the project and the R/PS draft

23 August 2026. Supersedes `PROJECT_ASSESSMENT.md` (v2, 22 Aug) where the two disagree, and says
why in each case. Scores are calibrated against what a NeurIPS-workshop reviewer or an astro-ML
referee would think, not against effort. The DeepLense comparison in §9 is based on the repository
listing and the project READMEs read today, not on recollection.

---

## Summary

### The project

| # | metric | v2 | **v3** | one-line reason |
|---|---|---|---|---|
| 1 | Novelty | 7.5 | **7.0** ↓ | Real experimental novelty, but the null count is three not four, and there is still no new method. |
| 2 | Chance of acceptance at R/PS | 8 | **7.0** ↓ | Good paper, audience mismatch on two of the call's four axes. |
| 3 | Complexity | 9 | **9.0** | Unchanged. |
| 4 | Reads as AI-written (work) | 2 | **2.0** | The experimental record cannot be faked. |
| 4b | Reads as AI-written (the new draft) | 8 (old docs) | **5.5** ↓ | Better than the repo docs, still has a rhythm. |
| 5 | Fit to the GSoC project statement | 8.5 | **7.0** ↓ | Three of five stated deliverables are untouched, two of them named in "Expected results". |
| 6 | Simulation data | 9 | **9.0** | Near the ceiling for this dataset. |
| 7 | Real data | 1.5 | **1.5** | Still none. |
| 8 | Physics | 9.5 | **9.5** | Strongest dimension by a distance. |
| 9 | vs other ML4SCI DeepLense work | 8.5 | **8.5** | Confirmed and now better grounded: no SR project in that repo scores against source truth. |
| 10 | ML / DL content | 7 | **6.5** ↓ | No uncertainty, no equivariance, and no neural-field row in a representations paper. |
| 11 | Evaluation honesty | 9.5 | **9.0** ↓ | Culture is exemplary; the documentation layer has now drifted from the data three times. |
| 12 | Statistical power | 8.5 | **8.0** ↓ | Single seed is load-bearing for the null results. |
| 13 | Generalisation evidence | 3 | **3.0** | One class, one instrument, one band, one source family. |
| — | **Composite as research** | ≈8.0 | **≈7.5** | Physics-deep, ML-competent, simulation-only. |
| — | **Composite as engineering and practice** | ≈9 | **≈9.0** | Unchanged and unusual. |
| — | **Composite as machine learning** | ≈7 | **≈6.5** | Conventional models, careful ablations, no uncertainty. |

### The draft

| # | metric | **score** | one-line reason |
|---|---|---|---|
| D1 | Fit to the R/PS call | **7.0** | Nails self-supervision, touches sampling, misses transfer and tokenisation. |
| D2 | Clarity of the central claim | **8.5** | One question, one table, one figure per claim. |
| D3 | Evidence density per page | **9.0** | Almost every sentence carries a number that traces to a file. |
| D4 | Reviewer attack surface | **6.0** | Five predictable attacks, one of them currently unanswered in the text. |
| D5 | Length discipline | **6.0** | 300 to 400 words over, before figures are placed. |
| D6 | Figure readiness | **5.0** | Figure 1 does not exist; 2 and 3 need regenerating at publication size. |
| D7 | Anonymity compliance | **9.0** | Clean, with one attribution to tighten. |
| D8 | Feasibility before 29 August | **7.5** | Six days, the writing is done, the remaining work is mechanical plus one experiment. |
| — | **Composite, the draft as a submission** | **≈7.5** | A solid, honest, submittable workshop paper that is not yet defended against its most obvious criticism. |

---

## 1. Novelty — 7.0 / 10 (down from 7.5)

**Not new, and all cited:** physics-informed unsupervised super-resolution of lensing images
(Shankar, Toomey and Gleyzer, ML4PS 2024); CNNs regressing lens parameters (Hezaveh 2017;
Perreault Levasseur 2017); pixellated and neural source reconstruction (Warren and Dye 2003;
Suyu 2006; Morningstar 2019; Adam 2022; Karchev 2022); EPL deflection (Tessore and Metcalf 2015);
SRResNet and ESPCN.

**Genuinely new, in order of strength:**

1. **The injection measurement.** 2,400 injections of a 0.76-detector-pixel clump through the full
   instrument model at matched noise, measuring what returns as a function of local magnification
   and photon count. This directly answers the question every reviewer of an unsupervised
   super-resolution paper asks, which is whether the extra detail is real, and it answers it with a
   number and a p-value rather than a figure. Nobody in this line has this.
2. **The decoder-versus-conditioning separation.** Two comfortably overdetermined representations
   under identical physics differing by a factor of two in source error, with the cause isolated to
   a global average pool. This is the most ML-flavoured claim in the paper and the one an ML
   audience will find interesting.
3. **The coverage-versus-resolution statement**, now demonstrated rather than only argued.
4. **The estimator-versus-initialiser decomposition**, with a mechanism (conditional-mean shrinkage
   of a spin-2 quantity) rather than a hyperparameter story.

**Why it went down 0.5.** The v2 score partly rested on "four independent nulls on
magnification-adaptive regularisation". There are three with evidence on disk. `B5_RESULTS.md`
lists a B4 `--reg-mode mu` versus `uniform` ablation quoting nmse 0.0799 against 0.0789, but those
are B3's numbers verbatim, and `results/` contains no `b4_uniform` metrics file. Both B4 v2 runs
used `--reg-mode mu` with no uniform control. Three nulls with a mechanism is still a good result.
Four would have been better, and the difference matters because the claim's strength is exactly
"we tried this every way we could think of".

**Why not 8.5.** No new algorithm, no new objective, no theory. This is a measurement paper and a
comparison paper. Those are undervalued relative to their scientific worth, but they are scored
lower on novelty at every venue, and pretending otherwise sets you up for disappointment.

**What moves it to 8.** A real galaxy source. Render lensed images from Galaxy10 DECaLS and show
the parametric fit smooths away structure that the hybrid recovers. That converts the model-family
caveat from a stated limitation into a demonstrated result and simultaneously fixes the largest
gap in metric 5. It is roughly a day, and it is the highest-value unexecuted experiment on every
axis in this document.

## 2. Chance of acceptance at R/PS — 7.0 / 10 (down from 8)

The base rate matters here. Non-archival NeurIPS workshops typically accept somewhere between 40
and 60 per cent of submissions, and this paper is comfortably above the median of what gets sent
to one: complete, honest, numerically dense, with a real measurement and a clean claim.

The reason for marking down from v2 is the audience. The organisers are a representation-learning,
operator-learning and climate-ML group rather than astronomers, and the call names four axes:

| axis | fit | comment |
|---|---|---|
| Self-supervision on unlabelled scientific data while preserving physical constraints | **strong** | This is the paper. Put it in the first sentence of the abstract, which the draft does. |
| Adaptive sampling and simulation-driven data generation | **moderate** | The injection experiment is a simulator-in-the-loop measurement. It is not adaptive sampling. A reviewer looking for that axis will not find it. |
| Transfer learning and out-of-distribution generalisation | **absent** | One class, one instrument, one band. This is a named topic of the workshop and the paper scores zero on it. |
| Tokenisation and discrete representations for continuous multi-scale physical data | **weak** | The paper is about representations, but none of them are discrete or tokenised. |

Two of four axes unaddressed is survivable at a workshop. It is not free.

**The single largest swing factor** is not the science, it is whether a reviewer reads this as an
ML paper with an astronomy testbed or as an astronomy paper submitted to an ML workshop. The
decoder-versus-conditioning finding is the lever: it is a statement about architectures that
happens to be measured on lenses. Foreground it and the paper is a good fit. Bury it under lensing
formalism and the paper is a mid-fit astronomy submission.

**Estimate:** 7/10 with the draft as written; 7.5 if the framing is pushed further toward the
architecture claim and a sentence is added acknowledging the transfer axis honestly rather than
leaving its absence to be noticed.

## 3. Complexity — 9.0 / 10

Unchanged and still high. The differentiable EPL deflection re-derived as a truncated
hypergeometric series with an adaptive term count and validated to 2.7e−15; two independent
magnification routes agreeing to 2e−9; a removable-singularity fix that is invisible in the
forward pass and fatal in the backward one; a numpy and torch parity harness; a bounded staged
least-squares fit; three network architectures; an adjoint back-projection validated by round trip;
a half-pixel `grid_sample` alignment convention that is worth 25 per cent of the claimed
super-resolution if you get it wrong; and an injection harness that reuses the same noise
realisation on both sides of the comparison.

The deduction is on the other side of the ledger: the ML machinery is deliberately simple, and
simplicity there is a virtue, but it does mean the complexity is concentrated in one half of the
project.

## 4. Chance someone concludes it was written by AI

**(a) The work: 2.0 / 10, meaning no.** The experimental record is not fakeable. Git history with
ordinary commit messages; checkpoints and figures whose timestamps trace real working days; a
result retracted on 17 August and un-retracted on 18 August after a decisive phase test; failed
runs preserved rather than deleted; thresholds written down in `B4.md` *before* the run and scored
against in `B4_RESULTS.md`; null results kept. That is research practice, and nobody produces it by
prompting.

**(b) The new draft: 5.5 / 10.** Better than the repo documents, which sat around 8. What was
removed: em dashes entirely, the "not X, it's Y" construction, the tables with a "why" column, the
American spelling drift. What remains, and what a suspicious reader would notice:

- Bold run-in paragraph headers used uniformly through Sections 2 and 3. Common in ML papers, also
  a strong LLM signature when applied that consistently.
- Section headings written as full declarative sentences or questions.
- The antithesis pattern, used twice in adjacent paragraphs ("Conditioning explains the failures" /
  "Conditioning does not explain the rest").
- A repeated rhythm of long sentence, then a short punchy one.
- Paragraph lengths that are unusually even.

**What to do about it.** Nothing evasive. NeurIPS-family policy permits LLM assistance and holds
authors responsible for every claim, which you can discharge because every number in the draft
traces to a file you can open. The practical reason to rewrite is that a uniform assistant register
is worse writing than yours, not that it is against the rules. Concretely: rewrite the Introduction
and the Limitations paragraph from scratch in your own hand, delete about half the bold lead-ins,
and let three or four paragraphs run long and one run to two sentences. That alone would take this
under 3.

**Also relevant:** automated AI-text detectors are unreliable at this length and on text this
number-dense, and no NeurIPS workshop runs them as a gate. The realistic risk is a reader's
impression, not a procedural one.

## 5. Fit to the GSoC project statement — 7.0 / 10 (down from 8.5)

The v2 score was generous and I am marking it down on re-reading the statement rather than on any
new information. The project text reads:

> "developing an unsupervised super-resolution architecture to upscale the quality of lensing
> images constructed using **real galaxy sources**, and to obtain insight about the lenses
> themselves"

with expected results "a more capable architecture that can operate on **a wider variety** of
lensing images, including lensing images created with **real galaxy datasets**" and "insight into
the lensing systems, and **their sub-structures**".

| deliverable | status |
|---|---|
| unsupervised super-resolution architecture | **done, three times over** |
| insight about the lenses | **over-delivered**: seven parameters, the magnification field, the critical curve to 0.63 px, total magnification to 1 per cent |
| lensing images from **real galaxy sources** | **not attempted** |
| bridging the gap to real images | **designed, not run** |
| insight into **sub-structure** | **not attempted**; the residual figure is qualitative |
| "a wider variety of lensing images" | **contradicted**: one class, one instrument, one band |

Two of the three items in "Expected results" are untouched. That is what a 7 looks like, not an
8.5. The two halves that were delivered were delivered better than the statement asked for, which
is why it is not lower.

The redeeming fact is that both gaps are now short experiments rather than open problems. Galaxy10
sources are about a day. Substructure recovery has the exact truth available already, since
`image` minus `image_nss` is the signal and `kappa_sub` is the ground truth, and the residual
figure already shows arc-shaped rather than smooth structure.

## 6. Performance on simulation data — 9.0 / 10

Seven parameters at Spearman +0.54 to +0.96; source size ratio from 10.65 to 1.081 on the same
data; super-resolution fidelity flat from 1× to 8× (correlation moves 0.0017); critical curve to
0.63 px with the azimuthal swing recovered to 1.5 per cent; total magnification to 1 per cent; and
the best free-form source in the project at nmse 0.0408 with size and peak ratios *better
calibrated* than the parametric fit.

Deductions: axion-only; three biases understood but not fixed (θ_E +3.5 per cent, R_sersic
+10 per cent, n_sersic +17 per cent, all consistent with the response wings); χ²/dof near 3,200
explained but not reduced.

## 7. Performance on real data — 1.5 / 10

No real cutouts, and the controlled domain-gap step is designed and not run. The 1.5 rather than
1.0 is for two habits that real data actually requires and that most simulation-only work skips:
the instrument response was *measured* from the data rather than assumed, and the noise level is
estimated per image from a source-free annulus rather than taken from a header. Those are the
right reflexes. They have simply not been exercised on anything real.

This remains the largest single gap between the project and its own title.

## 8. Physics — 9.5 / 10

Still the strongest dimension, and the reason the whole thing holds together.

- Surface brightness is never multiplied by magnification anywhere in the codebase, which is the
  single most common error in lensing ML code.
- Magnification is computed by differentiating the deflection field that is actually applied, so
  μ and the ray-tracer are structurally incapable of describing different lenses. That was exactly
  the defect in the inherited pipeline, where a hard-coded circular formula had no link to the
  sparse operators in use.
- The back-projection is the exact adjoint of the applied operator and is validated by a round trip
  against the true source at correlation 0.9989 and median value error 3.3 per cent, which is the
  check three of five of the original stored operator sets would have failed.
- The lens is validated on quantities it was never fitted to. The critical curve's azimuthal swing
  recovered at 0.540″ against a true 0.548″ is the good one, because a circular lens gives exactly
  zero and no amount of overfitting the arc produces it.

Deducting 0.5, not 0: μ is derived from the fitted lens rather than measured, the mass-sheet
degeneracy is sidestepped by parameterisation rather than addressed, and the substructure that is
the entire scientific point of DeepLense is left in the residual and never modelled.

## 9. Comparison with other ML4SCI DeepLense work — 8.5 / 10

Grounded in the repository listing and the project READMEs, read today. There are four
super-resolution efforts in `ML4SCI/DeepLense`:

| project | what it does | how it is evaluated |
|---|---|---|
| **Super_Resolution_Atal_Gupta** | Supervised. 2,834 LR/HR pairs, LR synthesised from HR by adding Gaussian noise and blur. SRCNN, RCAN, SRGAN, VAESR, iterative autoencoder, plus diffusion variants. | PSNR 26.0 to 33.6, SSIM 0.566 to 0.890. No lens model in the loop. |
| **Super_Resolution_Pranath_Reddy** / **DiffLense** | Conditional and residual diffusion models for lensing SR. | Image-domain metrics. |
| **DeepLense_Physics_Informed_SR_Anirudh_Shankar** | Unsupervised. A residual CNN produces deflection angles, applied to a Sérsic source and re-lensed; multi-scale physics loss with intensity conservation and a variation-density term. | MSE 0.0015 to 0.0046, SSIM 0.215 to 0.819, PSNR 24.1 to 29.2 dB. |
| **Grid_based_..._Anirudh_Shankar** | The direct ancestor of this codebase. Precomputed sparse forward and backward operators on a fixed circular lens. | Image-domain. |

**The single sharpest thing you can say, and it is defensible:** every super-resolution project in
that repository is evaluated with PSNR, SSIM or MSE in the image plane. None of them scores against
source-plane ground truth, and none of them reports physical parameter recovery. This project does
both, and additionally *measures* that the image-plane metric is misleading on this data, since a
two-pixel blur of the noisy input scores 0.951 on the conventional skill metric while the exact
physical model with the true source, true lens and true response scores 0.630, and skill is
anti-correlated with signal-to-noise at −0.85.

That is not a rhetorical point. It means the numbers in those READMEs cannot be compared with each
other or with yours, and you are the only person in the repository who can say why.

Where others are ahead: Gupta and Reddy on ML sophistication (diffusion, larger models); Tidball
(domain adaptation, published in ApJ) and Sreehari Iyer (self-supervised learning on a real
dataset) on real-data engagement, which is the axis you score 1.5 on.

Where you sit: first in the repository on evaluation rigour and physical correctness, mid-pack on
ML sophistication, bottom third on real-data engagement.

**One accuracy warning for the draft.** The published ML4PS 2024 abstract describes a singular
isothermal sphere model, while the physics-informed SR README describes a CNN that predicts
deflection angles. Those are different characterisations, and the draft currently describes its
Table 1 row 1 baseline as "a fixed circular isothermal lens in the manner of [11]". Check which is
true of the published version before submitting, or, more safely, describe the row purely as the
baseline you inherited on this dataset and drop the attribution. A reviewer who knows that paper
will notice a mischaracterisation faster than they will notice anything else in the table.

## 10. ML / DL content — 6.5 / 10 (down from 7)

**Present:** a fully convolutional super-resolution decoder with residual blocks, a long skip and
sub-pixel upsampling; a stated architectural hypothesis that was tested and confirmed; three
genuine ablations; a structural constraint rather than a penalty, honestly reported as not working
and with the failure mechanism identified; GroupNorm over BatchNorm and softplus over ReLU with
measured reasons; arcsinh input stretching that compresses a 260× across-image range to 1.9×; a
warm-up curriculum; loss weights expressed as a fraction of χ² so they cannot silently be six
orders of magnitude out again; dihedral augmentation that is exactly valid because the objective is
self-supervised.

**Absent, and the first three will be asked about:**

1. **No uncertainty quantification.** This is the biggest omission and it is now conspicuous,
   because you have a *measured and explained* bias (the flat 0.22 shrinkage of a spin-2 quantity)
   that under a density head or a normalising flow would become a reported error bar instead of an
   error.
2. **No neural-field or implicit-representation row in Table 1.** For a workshop called
   Representations for the Physical Sciences, the table omits the representation the ML audience is
   most likely to reach for, and there is directly relevant lensing prior work (Karchev 2022;
   ML4Astro 2022 on continuous neural fields). Expect this as the first reviewer question. You do
   not need to run it to survive; you need a paragraph saying why the comparison would or would not
   change the conclusion.
3. **No equivariance**, despite a direction-valued target on a rotationally symmetric problem, and
   despite two equivariant-network projects existing in the same repository.
4. No width or depth ablation, and a single seed.
5. The models are small and conventional (0.85 M and 0.63 M parameters), which is a defensible
   choice and still a fact a reviewer will register.

## 11. Evaluation honesty and integrity — 9.0 / 10 (down from 9.5)

The culture remains exceptional in absolute terms: thresholds pre-registered before the B4 run;
three null results reported as results, including on the project's own signature idea; the
"free-form fits the data better and recovers the source worse" observation volunteered rather than
buried; `eval_b4.py` printing the physical flux-in-box ceiling next to the measured value so the
metric cannot be read as better than possible; a knob that was accidentally left off stated plainly
rather than quietly rerun.

The 0.5 comes off for a pattern rather than an incident. There have now been three cases where the
*markdown* drifted from the *data*: the axion-only class truncation described as three classes, the
stale `mu_resolution` numbers with a materially different conclusion (3.1× against 1.25×), and the
null count. In each case the code and the JSON were correct and the prose was wrong. The code layer
is more trustworthy than the documentation layer, and since the paper is written from the
documentation layer, that is exactly the wrong way round. A single pass regenerating every quoted
number from its JSON would close it.

## 12. Statistical power — 8.0 / 10 (down from 8.5)

Good where it counts: n = 800 for the refinement and then n = 6,000 in use; 2,400 injections with a
non-parametric test rather than an eyeballed difference; source metrics at n = 800 or 2,000.

The binding weakness is that **a single seed per training run is load-bearing**. The three nulls
rest on differences of 0.6 to 1.3 per cent between runs, and there is no variance estimate for
those differences. If a reviewer asks "is 0.0414 against 0.0408 distinguishable from seed noise?",
the honest answer today is that you do not know. Two more seeds on the control and the gate is a
few GPU-hours and it converts the weakest part of the argument into the strongest kind of negative
result. This is the highest-value statistics fix available before the deadline.

Also: two Experiment A bins have n ≤ 3 and must be dropped from any published table. The draft
already reports "five populated bins" and quotes the slope over those five, which handles it.

## 13. Generalisation evidence — 3.0 / 10

One dataset, one instrument, one of two available bands, one dark-matter class, one source family,
one noise regime. Cheapest gains in order: cdm and wdm (30 minutes), Galaxy10 sources (a day),
second band (2 to 3 days), degrade to ground-based sampling and seeing (a day). The second band is
the largest lever nobody has pulled: both bands exist in every file, they correlate at 0.75, and
the lens is achromatic, so a two-band fit constrains one lens with twice the data while letting the
source differ.

---

# The draft, scored

## D1. Fit to the call — 7.0 / 10

See §2. Strong on self-supervision, moderate on simulator-in-the-loop, absent on transfer and
out-of-distribution generalisation, weak on tokenisation. The abstract's opening sentence is right.

## D2. Clarity of the central claim — 8.5 / 10

The paper asks one question, answers it with one table, and supports it with one measurement per
section. A reviewer can state the contribution after reading the abstract, which is the practical
test. The four contribution bullets are specific enough to be checked and none of them overclaims.

## D3. Evidence density — 9.0 / 10

Very high. Nearly every claim carries a number, and every number traces to a file on disk. This is
the draft's real strength and it is what makes the AI question moot in practice.

## D4. Reviewer attack surface — 6.0 / 10 (moderate risk)

Five predictable attacks. Four are pre-empted in the text. One is not, and it is the most likely.

1. **"Your best result is the seven-parameter parametric fit. So representation learning did not
   help, and this is a paper about a curve fit."** Currently unanswered where it matters. The
   defence exists and is in Limitations: on this dataset the true sources genuinely are Sérsics, so
   the parametric row is an oracle-family upper bound and not a competitor, and the deployable
   question is which *free-form* representation gets closest to it, which the hybrid decoder answers
   by halving the error of every previous free-form attempt. **Move that sentence up into
   Section 3, immediately after Table 1.** This is my single most important recommendation on the
   draft, and it costs about 40 words.
2. "Why no implicit neural representation row?" See §10. Needs a paragraph, not an experiment.
3. "One dark-matter class." Handled honestly in Limitations. Consider surfacing it in the abstract,
   because a reviewer who meets it only in Section 6 reads it as a concession rather than a
   disclosure.
4. "No uncertainty on any prediction." Handled in Limitations only. Acceptable at a workshop.
5. "Is 0.0414 against 0.0408 seed noise?" See §12. Two more seeds fixes it.

## D5. Length discipline — 6.0 / 10

Sections 1 to 6 run about 2,620 words with three figures and two tables. Four NeurIPS pages holds
roughly 2,300 in that configuration. The cut order is in the draft's own notes block.

## D6. Figure readiness — 5.0 / 10

Figure 1 does not exist and must be drawn. Figures 2 and 3 exist but were produced at diagnostic
size and will be illegible at column width. This is the largest remaining block of mechanical work
and it is the thing most likely to eat the last two days. Do it before the prose polish, not after.

## D7. Anonymity compliance — 9.0 / 10

Nothing names a person, institution or programme; the prior-work citation is third person; the
inherited baseline is described without identifying whose it was. The one item to tighten is the
attribution discussed at the end of §9.

## D8. Feasibility before 29 August — 7.5 / 10

Six days. The writing exists, which was the hard part. Remaining: three figures, LaTeX assembly,
the three fixes in the notes block, two extra seeds if you want §12 closed, and a mentor read.
That fits, provided the Galaxy10 experiment is treated as optional rather than assumed.

---

## Other metrics worth having

**Defensibility under questioning — 9.5 / 10.** Every number traces to a JSON or a test. If someone
asks why the regularisation power is 0.5, there is a measurement behind it (source correlation
falling 0.96 to 0.74 under raw 1/μ). This is the property that makes the difference between a paper
that survives a hostile review and one that does not, and it is the property this project has most
of.

**Falsifiability — 9.5 / 10.** Thresholds written before runs; a spatial-correspondence test that
pokes one input pixel and asserts where the response lands; a flux-in-box ceiling printed beside the
measurement; a gate whose mean value is reported so that a no-op would be visible; an injection
experiment with an explicit null. Very few projects build this many ways to be proven wrong.

**Time-to-insight per unit compute — 9.5 / 10.** The most valuable result in the project, the
injection experiment, required no training at all. So did the lens-model ladder, the response
calibration, the skill null baselines and the gradient-share diagnosis. The instinct that the
decisive experiment is usually a measurement rather than a run is the most transferable thing here
and is worth naming explicitly in the GSoC report.

**Inheritability — 9.0 / 10.** A successor could take this over from the documents alone. That is
rare and it is worth something to the programme independent of the paper.

**Would a lensing astronomer cite this today — 5.0 / 10.** Simulation-only, one class, no
uncertainties. The injection result is the one an observer could act on when planning exposures or
selecting targets, and the dataset bug report is the one that saves other people days.

---

## Bottom line

**Project: ≈7.5 / 10 as research, ≈9 / 10 as engineering and scientific practice, ≈6.5 / 10 as
machine learning.**

**Draft: ≈7.5 / 10 as a workshop submission**, and roughly 7/10 likely to be accepted.

The project's shape is unusual and worth naming plainly. The physics and the evaluation culture are
at or near the level you would expect from a good methods paper in a refereed astronomy journal.
The machine learning is competent, deliberately conventional, and has one real architectural finding
in it. The empirical base is narrow in a way that no amount of care compensates for: one dark-matter
class, one instrument, one band, one source family, and no real data at all. The draft is honest,
dense and well organised, and its main structural weakness is that it lets the parametric fit win
Table 1 without immediately explaining why that is the expected and non-damaging outcome.

**If you do only three things before 29 August, in this order:** move the oracle-family defence up
under Table 1; regenerate the three figures at publication size; and run two extra seeds on the B5
control and gate. If a fourth fits, render one lensed image from a Galaxy10 source and put it in the
appendix, because it is the only thing that touches both the largest gap in the paper and the
largest gap in the GSoC project statement.
