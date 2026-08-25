# Submission strategy v3 — Sim2Science, `superres/` only

23 August 2026. Supersedes `WORKSHOP_SUBMISSION_STRATEGY.md` (v2, R/PS). Two things changed:
the venue, and the decision that the paper reports only work inside `superres/`.

---

## 1. What survives from v2, and what does not

| v2 section | status |
|---|---|
| §0 decision 1: write the `superres/` paper, not the 2.5-month paper | **holds**, and your scoping decision confirms it |
| §0 decision 3: non-archival means low risk | **holds** |
| §2 the representation table as the spine | **holds**, now six rows |
| §6 novelty and the delta from the precedent | **holds** |
| §8 things that cost you a review | **holds**, plus two new ones (null count, `--n-pos`) |
| §0 decision 2: frame it as a *representation* paper | **superseded**. Frame it as identifiability plus misspecification. |
| §3 the four-page budget | **superseded**. Five pages. |
| §5 what to skip | **partly invalid**: most of its "skip to two sentences" items are pre-`superres/` and are now out of scope entirely, not merely compressed. |
| §7 seven-day plan | **superseded**, dates passed |
| §9 abstract skeleton | **superseded**, rewritten in §5 below |
| §12 "three pieces of the arc that must survive" | **conflicts with your scoping**. All three are pre-`superres/`. See §3. |
| Part 2 (§10 to §14, the whole-arc paper) | **moot for this submission** |

Roughly 60 per cent of v2 holds. The framing, the budget, the timeline and the "what to carry over
from the earlier work" sections do not.

## 2. Does Sim2Science still win under `superres/`-only scoping?

**Yes, and by almost the same margin, but the emphasis flips.**

The restriction hurts one of the two Sim2Science pillars and leaves the other untouched.

| pillar | before | with `superres/` only |
|---|---|---|
| Simulator structure, degeneracy, identifiability | 9 | **9, unchanged.** Every piece of it is in `superres/`: the conditioning table, Option 2's speckle at 1.44 rays per source pixel, the coverage-versus-resolution arithmetic, and "free-form fits the data better and recovers the source worse". |
| Model misspecification, diagnostics, discrepancy modelling | 9 | **7.5.** The PSF story survives intact (`calibrate_psf.py`), so does χ² rising with SNR as the systematic signature, the σ-floor as an explicit discrepancy term with its measured effect, the null baselines, and the three co-directional biases. What is lost is the lens-model ladder and the dataset phase test, both pre-`superres/`. |
| Emulator and surrogate, hybrid, physics-informed | strong | **unchanged** (`refine_pathb.py`) |
| Differentiable frameworks | strong | **unchanged** (`lens_models.py`, validated to 2.7e−15) |
| Closed-loop, experiment-in-the-loop | moderate | **unchanged** (`mu_resolution.py`) |

R/PS is unaffected by the scoping, and it was already behind: one strong axis, two of its four named
topics scoring zero. So the comparison holds. **Sim2Science, 8.0 against 6.5.**

The change is in what leads. Under the old scoping I would have opened with three misspecifications.
Now I would open with **identifiability**, which is entirely yours and entirely inside `superres/`,
and use misspecification as the second pillar. That is arguably the better paper anyway: every
Sim2Science submission will bring a misspecification story, and very few will bring a controlled
six-representation conditioning study with a measured "better fit, worse recovery" result.

## 3. The scoping audit: what you lose, and the one thing to re-run

**Out of scope, cannot be claimed as your result:**

| item | where it lives | handling |
|---|---|---|
| Lens-model ladder, 0.869 → 0.997 | `diagnostics_2026_08_12/PROJECT_REPORT.md` §7.1 | **re-run it, see below** |
| 10.6 to 12.8× across the 100× TV sweep | old SIS pipeline | keep as a *baseline*, not a result. The draft already words it as "the free-field baseline we inherited on this dataset", which is correct under this scoping and also correct under double-blind. |
| `image_nss` circular-lens phase test (1.6° vs 44.3°) | `sie_pipeline/audit_dataset.py` | cut, or one footnote saying you fit `image` and why, with no numbers |
| `skill` metric study (2-px blur 0.951 vs exact model 0.630, ρ = −0.85) | `PROJECT_REPORT.md` §7.3 | cut. `superres/evaluate.py` already prints χ²-based null baselines (constant 37,511; blur2 2,796; blur3 4,091), which carries most of the same argument and **is** in scope. |

**The one thing worth re-running: the lens-model ladder.** It is the cleanest misspecification
measurement you have, it justifies six lens parameters in one table, and it is currently the only
load-bearing number in the draft that is out of scope. Everything needed is already in `superres/`:
`raytrace.render`, `lens_models.deflection_epl`, `sources.SersicSource`, and the manifest truth.

> Render the true Sérsic through each candidate deflection model at 3× supersample, convolve, bin,
> no noise; reference is the same source through the true EPL+shear lens; arc-weighted correlation,
> n = 60; report split by lens |e|.

A script of roughly 50 lines, no training, no GPU, about two hours including the split table. Do it.
It converts the draft's best justification paragraph from a citation into a result and it repairs
the misspecification pillar from 7.5 back to about 8.5.

## 4. Revised section map (5 pages)

| section | pages | content |
|---|---|---|
| **1. Introduction** | 0.6 | Fitting a physical forward model to data requires the model to be right and the unknowns to be identifiable. Neither is free. What we measure about both, on one dataset with full generative truth. |
| **2. The forward model and the objective** | 0.9 | β = θ − α, the differentiable EPL+shear series validated to 2.7e−15, the observation operator, χ² over 6,361 pixels, the unsupervised ring initialisation in two sentences, the lens-model ladder (once re-run). |
| **3. Identifiability: what the data can support** | 1.3 | **Table 1**, six rows. Conditioning explains the failures. Conditioning does not explain the rest, and the decoder does. Coverage versus resolution. "Lower validation χ² (329 vs 343), worse source (0.0527 vs 0.0408)." This is the lead now. |
| **4. Residual misspecification, diagnosed and absorbed** | 0.9 | PSF assumed 0.10″, measured 0.18 to 0.20″ with wings 29× a Gaussian; χ² rising with SNR (ρ = +0.449) as the signature that separates systematic from noise; the three co-directional biases; the σ-floor as an explicit discrepancy term, flattening χ² across SNR from 1411→8455 to 329→229 and the p90/p10 spread from 95× to 8.7×; null baselines. |
| **5. Where the model is right, and what it buys** | 0.9 | Magnification validated on quantities never fitted (critical-curve swing 0.540″ vs 0.548″, zero for a circular lens). Then the injection measurement: 17 to 22 per cent returns, μ ≥ 2 beats μ < 2 at p = 5.3e−5, but ρ(contrast, SNR) = +0.152 against ρ(contrast, μ) = +0.056. |
| **6. Constraints the likelihood undoes** | 0.5 | The gate pre-compensation result (1.61 × 0.624 = 1.005) and the three nulls. General lesson, not a lensing lesson. |
| **7. Limitations** | 0.4 | Axion-only; model-family ceiling; μ derived not measured; no posterior, with the measured shrinkage named as what a density head would report; simulation only, one of two bands, single seed. |

Three figures as before. Table 1 and Table 2 unchanged. The extra page over R/PS goes to Section 4,
which is currently a parenthetical and becomes a result.

## 5. Abstract skeleton (~200 words)

> Fitting a physical forward model to data requires two things that are rarely measured: that the
> model is right, and that its unknowns are identifiable from the data available. We measure both,
> on strong gravitational lensing, where the only supervision is physics. An analytic differentiable
> elliptical-power-law-plus-shear deflection field, a measured instrument response and detector
> binning close a χ² directly against the observed pixels, with no high-resolution target and no
> labels. Comparing six source representations under this one forward model, we find that
> conditioning predicts which representations fail and that decoder architecture accounts for a
> further factor of two among those that do not, and we measure a case where the representation with
> the lower validation χ² recovers the source worse. On the misspecification side, the instrument
> response assumed by every prior pipeline on this dataset is wrong by a factor of two in width; the
> signature is that χ² rises with signal-to-noise, and absorbing it as an explicit discrepancy term
> flattens the residual across a 260× brightness range. Injecting structure smaller than a detector
> pixel, we recover 17 to 22 per cent of it, significantly more where the lens magnifies more, though
> photon count predicts recovery three times more strongly. Three attempts to turn magnification into
> a prior all came out neutral, and we identify why the structural one cannot work.

## 6. Contribution bullets, revised

1. **A controlled comparison of six source representations** under one forward model, separating
   conditioning from decoder architecture, with a measured instance of better data fit and worse
   recovery.
2. **A measured sub-detector-pixel recoverability result**: 2,400 injections at matched noise,
   17 to 22 per cent returns, with the magnification and photon-count contributions separated.
3. **Misspecification diagnosed by signature and absorbed by discrepancy modelling**, with the
   effect measured rather than assumed.
4. **Physics-only training makes a one-pass network an excellent initialiser and a poor point
   estimator**, mechanism identified, 480 → 151 model evaluations at a slightly better χ².

Plus three nulls on magnification-adaptive regularisation, one with a general explanation:
a differentiable constraint placed before the likelihood is not a constraint if the network is
trained through it.

## 7. Six-day plan

| day | do |
|---|---|
| **Sun 24** | Tell the mentors, ask for a read on the 27th. Compile a stub with `\usepackage[dblblindworkshop]{neurips_2026}` and `\workshoptitle{Sim2Science}`; wrong template is a stated desk-reject. Write the ladder script and launch it. |
| **Mon 25** | Rewrite abstract, introduction and section order to §4 above. Write Section 4 from `DEEPLENSE_TECHNICAL_BRIEF.md` §3.3, §3.4 and §5.6. |
| **Tue 26** | Figures at publication size. Draw Figure 1. Launch two extra seeds on the B5 gate and the B4+Sérsic control in the background. |
| **Wed 27** | Assemble at five pages. Send to mentors. Fix the χ²/dof convention, the null count and the [11] attribution. |
| **Thu 28** | Revisions. Anonymise. Fold in the seed results. |
| **Fri 29** | Submit early. |

**Cut list, in order:** the two seeds → the ladder re-run (then cite the baseline instead of
claiming it) → Figure 3. **Never cut:** Table 1, the injection figure, the σ-floor result.
