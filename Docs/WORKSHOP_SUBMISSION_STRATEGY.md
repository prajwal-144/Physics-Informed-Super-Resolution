# R/PS @ NeurIPS 2026 — submission strategy (v2, updated for B4 / B5)

**Target:** *Representations for the Physical Sciences (R/PS)* — a NeurIPS 2026 workshop on
self-supervision, transfer, sampling, and tokenization. Paris, 12 or 13 December 2026.
**Deadline: 29 August 2026 (AoE) — 7 days from today (22 Aug).**
Submission: OpenReview, `NeurIPS.cc/2026/Workshop/RPS`. Workshop LaTeX template, NeurIPS style, no
checklist. **4 pages of main content**; references and appendices unlimited. **Double-blind**,
NeurIPS main-track standards. **Non-archival.** Posters, with selected contributed talks.

> **What changed since v1.** You built B4 and B5 and ran the refinement at scale. That moves three
> things: the representation table gains a fourth and decisive data point; the magnification story
> becomes four nulls *plus a measurement that explains them*; and the experiment I called "the
> single most convincing missing experiment" — recovering injected sub-detector-pixel structure —
> **now exists**. The paper is materially stronger than it was a day ago.

---

## 0. The three decisions, unchanged and now better supported

1. **Write the `superres/` paper, not the 2.5-month paper.** §5 and Part 2.
2. **Frame it as a representation paper, not a super-resolution paper.** You now have *four*
   source representations measured on identical data, physics, and scoring — that is an unusually
   complete experiment for a 4-page workshop paper.
3. **Non-archival means low risk.** Submit this, send a longer version elsewhere later.

**Still the most urgent non-technical item:** mentor co-authorship and a review slot. Ask today
if you have not. Michael Toomey and Sergei Gleyzer are co-authors on the direct precedent
(Shankar, Toomey & Gleyzer, ML4PS @ NeurIPS 2024), and B4 is explicitly their architecture on your
lens — which is an advantage for framing and a reason to be scrupulous about credit.

---

# PART 1 — the paper built on `superres/`

## 1. Fit to the call

| workshop axis | fit | how to make it explicit |
|---|---|---|
| **Self-supervision** — "extracting structure from abundant but unlabeled scientific data **while preserving physical constraints**" | **Very strong.** No labels, no high-resolution target; the supervision is χ² through an analytic differentiable lensing operator. | First sentence of the abstract. |
| **Tokenization / structured representations** — "discrete representations for inherently continuous, multi-scale physical data" | **Now very strong.** Four source representations, one comparison table, same forward model. | §2 — this is the spine. |
| **Sampling** — "leverage simulators and experimental loops to adaptively generate training data" | **Now moderate-to-good.** `mu_resolution.py` Experiment B is simulator-in-the-loop: inject a known sub-pixel structure, re-render through the full instrument model with the *same noise realisation*, and measure what comes back. 2,400 injections. | One subsection; it is also your best figure. |
| **Transfer** — "reliable extrapolation into physically meaningful regimes" | **Still the weakest.** Axion-only, simulation-only. | Limitations, honestly. Do not build on it. |

## 2. The spine: four representations, one table

This is now the strongest thing in the paper and it is entirely yours. Every row is the *same*
data, the *same* forward model, the *same* frozen lens where applicable, and the *same* scoring
function.

| source representation | unknowns / image | per informative datum | source nmse | outcome |
|---|---|---|---|---|
| free field, 254² (prior work) | 64,516 | ~15 : 1 under | — | **size ratio 10.6–12.8**, flat across a 100× regulariser sweep |
| free field, 64², linear inversion | 4,096 | 0.64 : 1 | 0.082 | speckle; CG failed on ~30% |
| physical code + bounded residual, decoded from a **global vector** (B3) | 14 + 1,024 | 1 : 4.1 over | 0.080 | works, but the residual never exceeds 3–4% of the source |
| **physical code + residual field from a fully convolutional decoder on the back-projection (B4)** | **7 + 2,304** | **1 : 1.87 over** | **0.0408** | **best free-form result in the project** |
| parametric only, 7 numbers (Path A) | 7 | 1 : 900 over | **0.0252** | the ceiling on this data, because the true sources *are* this family |

**The two claims this table supports, in one sentence each:**

1. **Conditioning is necessary but not sufficient.** B3 and B4 are both comfortably
   overdetermined, yet B4 halves the error. **The bottleneck was the decoder, not the
   conditioning** — B3 pushes the image through a global average pool, discarding position, and
   64% of its 849,519 weights sit in the single dense layer that then has to regenerate a spatial
   map from a 132-dimensional summary. B4 removes the pool and hands the network data already in
   the source plane and already in spatial register.
2. **Free-form must trade coverage against resolution; a hybrid need not.** A free-form box wide
   enough to hold the source (±2.5″, since 28% of the flux lies outside ±0.8″) at the same
   0.05″/px needs ~9,600 unknowns against ~4,300 data pixels. The hybrid keeps the wings in 7
   parameters and spends its pixels only on the residual — measured: the `--base sersic` run at
   H = 1.2 shows **no truncation artefact** while the free-form run at the *larger* H = 1.6 still
   does.

Add the corroborating number: **free-form has the lower training χ² (329) and the worse source
(nmse 0.0527 vs 0.0408)**, with more data and more epochs. *Fitting the data better is not the
same as recovering the source* — ill-posedness, measured rather than argued.

## 3. The four-page budget

| section | pages | content |
|---|---|---|
| **1. Introduction** | 0.55 | Scarcity of high-resolution lensing data; the lens as a physical magnifier; the representation question; four contribution bullets (§4). |
| **2. Method** | 1.0 | **Figure 1** (pipeline, both decoders). β = θ − α(θ; 6 params); EPL series in one line (cite Tessore & Metcalf); observation operator lens→PSF→bin; the σ-floor; λ as a fraction of χ²; the μ-weighted prior; unsupervised ring initialisation in two sentences. |
| **3. Representations** | 1.0 | **Table 1** (§2). The conditioning argument, the decoder argument, the coverage/resolution arithmetic, and the "lower χ², worse source" observation. |
| **4. Does the physics predict where it works?** | 0.85 | **Figure 2**: the injection experiment. Sub-detector-pixel structure recovered at 17–23% of injected contrast; μ ≥ 2 beats μ < 2 (0.208 vs 0.175, Mann–Whitney p = 5.3e−5); but ρ(contrast, SNR) = +0.152 vs ρ(contrast, μ) = +0.056. Then the four magnification-adaptive nulls, explained by that last line. |
| **5. Cost** | 0.4 | **Figure 3**: one-pass network as an initialiser — 480 → 151 model evaluations, four stages → one, χ² 3302 → 3228, recovery better on 6/7. Verified n = 800; used for real at n = 6000. |
| **6. Limitations** | 0.2 | Model-family ceiling; axion-only; μ derived not measured; simulation-only. |
| references / appendix | — | unlimited |

**Three figures, and only three.**

1. **Pipeline**, showing both decoders and, explicitly, that χ² closes against the *observation*
   with no high-resolution target anywhere. Reviewers scan for this.
2. **The injection experiment** — `figMU1_resolution_vs_mu.png` plus one panel of
   `figS5_sersic_stages` (cropped). This is the paper's most novel content.
3. **Cost/accuracy** — `figR1_cost.png` or `figR4_summary.png`, with the ellipticity inset.

`fig1_superres`, `fig2_recovery`, `fig3_sizeratio`, `figM*`, `figS1/S4/S6`, `figB*` → appendix.

## 4. The four contribution bullets

1. **An unsupervised per-image joint lens+source inversion validated against full generative
   ground truth** — seven physical parameters at ρ = +0.57 to +0.96 from the pixels alone, and a
   reconstructed-source size error reduced from 10.6× to 1.08× on the same data.
2. **A controlled comparison of four source representations** under one forward model, showing
   that conditioning explains the two failures but **decoder architecture** explains the
   remaining factor of two, and that a hybrid parametric-plus-residual field escapes the
   coverage-versus-resolution trade-off that binds any purely free-form grid.
3. **A direct measurement that sub-detector-pixel structure is recoverable**, via 2,400 injections
   of a 0.76-pixel clump through the full instrument model at matched noise: 17–23% of injected
   contrast returns, and recovery is significantly better where the lens magnifies more
   (p = 5.3e−5) — but photon count predicts it about three times more strongly.
4. **Physics-only training makes a one-pass network an excellent initialiser and a poor point
   estimator**, with the mechanism identified (shrinkage of a direction-valued quantity under a
   squared loss, a flat 0.22 factor across every orientation-error bin) and the consequence
   measured at n = 800 and used at n = 6000.

**Plus, as results and not asides:** four independent null results on magnification-adaptive
regularisation — a linear-inversion prior, two loss weights, and a hard architectural gate —
together with the measurement in bullet 3 that explains why.

## 5. What to skip

| skip | why |
|---|---|
| Model_4 / operator bank | Two sentences of motivation at most. |
| Option 2 and B2 as separate stories | One row each in Table 1. |
| The `image_nss` dataset defect | Appendix + footnote. Consider a separate short data note with the mentors. |
| `skill` metric critique | Two sentences: "a 3-px blur of the input scores 0.951 on the conventional image-plane skill metric while the exact physical model scores 0.630, and skill is anti-correlated with SNR at ρ = −0.85; we therefore report χ²/dof with null baselines and source-plane truth." Full study to the appendix. It is a *strong* two sentences. |
| PSF calibration | Two sentences + appendix. |
| The B5 gate as a headline | It is one row of the magnification section — a fourth null, honestly reported. |
| Parameter-recovery table in full | Two rows in the text (θ_E, |e|), full table to the appendix — the representation story needs the space more. |
| All numerics/validation engineering | One sentence: "all physics is validated against `lenstronomy` to 1e−15 and the NumPy and PyTorch paths agree to 1e−6." That sentence buys real credibility for one line. |

## 6. Novelty, stated plainly

**Not new** (cite all): physics-informed unsupervised SR of lensing images — **Shankar, Toomey &
Gleyzer, ML4PS @ NeurIPS 2024**; CNNs predicting lens parameters — Hezaveh et al. 2017, Perreault
Levasseur et al. 2017; neural/pixellated source reconstruction — Morningstar et al. 2019, Adam et
al. 2022 (recurrent inference machines), Karchev, Coogan & Weniger 2022 (continuous neural
fields); regularised semilinear inversion — Warren & Dye 2003, Suyu et al. 2006, Vegetti &
Koopmans 2009; EPL deflection — Tessore & Metcalf 2015; `lenstronomy` — Birrer & Amara. Also cite
SRResNet/ESPCN for the sub-pixel upsampler.

**Your delta from the precedent — put this in the introduction, generously:**

1. **The lens is fitted per image, not fixed.** They use a fixed circular SIS with θ_E from the
   simulation metadata. Measured cost of that choice on this data: SIS explains the arcs at
   correlation 0.869 (0.658 for |e| > 0.3) where SIE+shear reaches 0.997; refitting θ_E buys
   +0.02, adding ellipticity buys +0.10.
2. **B4 is their architecture on our lens**, so the comparison is a controlled swap of exactly one
   component. Say this explicitly — it is a *stronger* framing than "we did something different".
3. **Evaluation against source-plane truth and physical parameters**, not image-plane
   MSE/SSIM/PSNR — and a demonstration that image-plane metrics on noisy targets are actively
   misleading here.
4. **The resolution claim is calibrated by a measured magnification field and then tested by
   injection**, rather than asserted by a chosen upscaling factor.

**Genuinely new:** the four-representation comparison with the decoder-versus-conditioning
separation; the coverage/resolution trade-off statement; the injection measurement of sub-pixel
recoverability with its SNR-versus-μ decomposition; and the estimator-versus-initialiser result.

## 7. Seven-day plan

| day | do |
|---|---|
| **Fri 22 (today)** | Mentors: abstract + claim + co-authorship + a review slot on the 27th. Fix the class sampling (`limit` per class) and launch `fit_per_image.py --n 600` across all three classes in the background. Freeze Table 1. |
| **Sat 23** | Write Method + Representations. Regenerate the three figures at publication size. **Re-run `mu_resolution.py` once more at the same settings** so the numbers in the paper come from a single, current JSON (see §8, item 1). |
| **Sun 24** | Write the magnification section around the injection experiment. If the cross-class run finished, fold it in; if not, state axion-only. |
| **Mon 25** | Write Cost, Limitations, Introduction (last). |
| **Tue 26** | Appendix: full recovery table, PSF calibration, `skill` study, validation ladder, the four nulls in detail, the dataset note. |
| **Wed 27** | Send to mentors. |
| **Thu 28** | Revisions. Anonymise: strip GSoC/ML4SCI/institution from the body, anonymised repo link, acknowledgements held for camera-ready. |
| **Fri 29** | Submit early in the day. |

**Cut list, in order:** the cross-class re-run (then state axion-only prominently) → the B5 gate
row → the full parameter table. **Never cut:** Table 1, the injection figure, the initialiser
result, the four nulls.

## 8. Things that will cost you a review if left in

1. **`MAGNIFICATION_SR.md` quotes an older, smaller run of `mu_resolution.py` than the JSON now on
   disk, and the difference is material.** The markdown reports the low-SNR μ effect as **3.1×**;
   the 2,400-injection JSON gives **1.25×**, with the SNR-stratified ratios running 1.25 → 1.22 →
   1.08. The Experiment A error-vs-μ table also differs. **Quote the JSON, delete the stale
   tables, and re-run once so there is exactly one authoritative version.** This is the single
   highest-risk item in the paper right now.
2. **"All three DM classes" is false** — everything is axion. Fix or state it.
3. **Three different χ²/dof denominators** across `evaluate.py`, `eval_pathb.py` and `eval_b4.py`.
   State one convention.
4. **"source nmse" means Sérsic-only in one table and Sérsic+correction in another** (0.0845 vs
   0.0799 for B3). It is the super-resolution number; be explicit.
5. **Do not describe the fit as Levenberg–Marquardt without qualification.** You call
   `scipy.optimize.least_squares(method="trf")` — trust-region reflective, LM's bounded relative.
   Write "bounded nonlinear least squares (trust-region reflective)".
6. **Do not claim B4 beats the parametric fit.** It does not on corr/nmse; it does on size and
   peak ratio. Say exactly that.
7. **Do not present the B5 gate as working.** nmse 0.0414 vs the control's 0.0408.
8. **Keep the model-family caveat in the abstract**, not only in the discussion.

## 9. Abstract skeleton (~160 words, cut to fit)

> High-resolution images of strong gravitational lenses are scarce, and upcoming surveys will make
> the scarcity worse rather than better. We ask **what representation a source galaxy should have
> when the only available supervision is physics**. We fit a lens and a source jointly to each
> observation with no high-resolution target and no labels: an analytic, differentiable
> elliptical-power-law-plus-shear deflection field, a measured instrument response and detector
> binning close a χ² directly against the observed pixels. On Roman-like simulations with full
> generative ground truth we recover seven physical parameters from the pixels alone
> (ρ = +0.57 to +0.96) and reduce the reconstructed-source size error from 10.6× to 1.08×
> relative to a free-field baseline on the same data. Comparing four source representations under
> one forward model, we find that conditioning explains which representations fail and **decoder
> architecture** explains the rest, and that a hybrid of a low-dimensional physical code with a
> convolutional residual field escapes the coverage-versus-resolution trade-off that binds any
> purely free-form grid. Injecting structure smaller than a detector pixel, we measure that 17–23%
> of it is recovered, significantly more where the lens magnifies more — though photon count
> predicts recovery three times more strongly, which is why four attempts at
> magnification-adaptive regularisation all came out neutral.

---

# PART 2 — the whole 2.5-month project

Unchanged in substance from v1: **it is a real paper, but not this paper and not this deadline.**

## 10. What the arc contains

| act | content | strength |
|---|---|---|
| **1 — the bank** | Operator sign audit (3 of 5 directories had forward and backward with the same sign, invisible in the loss); a 13-bin Einstein-radius operator bank that improved every Model_4 source metric at matched settings | Real engineering. Fatal caveat: all *plausibility* metrics, no source truth, so correctness is unknowable. |
| **2 — the diagnosis** | SIS 0.869 vs SIE+shear 0.997; the error is **angular not radial** (mean radius right to 0.7%, peak-to-peak swing 5.4 px on a 12 px ring); ≥3 arcs in 98% of images refutes a circular lens with no reference to the manifest; PSF is 0.18″ not 0.10″; `skill` anti-correlated with SNR at −0.85; the θ_E manifest leak | **The strongest part of the arc** — a negative result with a quantitative cause, obtained with ground truth, directly actionable for anyone building fixed-lens SR pipelines. |
| **3 — the rewrite** | everything in `superres/` | the Part 1 paper |
| **plus** | the `_nss` circular-lens dataset defect, settled by a phase test (1.6° vs 44.3°) | a short data note of its own |

## 11. Why it is not the R/PS submission

Four pages cannot hold three acts; Act 1's headline is unverifiable by construction; it reads as a
project report rather than a paper with one claim; and Acts 1–2 are *more* persuasive as three
tables of evidence inside a positive paper than as a narrative.

## 12. Three pieces of the arc that must survive into the 4-page paper

1. **The lens-model ladder** (SIS 0.869 / SIE 0.973 / EPL 0.983 / SIE+shear 0.997, split by |e|) —
   two lines in Method. It is the entire justification for six lens parameters, and it is measured
   against truth, which almost no comparable paper does.
2. **The 100× TV sweep flat at 10.6–12.8** — one row of Table 1. It forecloses the obvious
   objection ("did you tune the regulariser?").
3. **The `skill` null-baseline study** — two sentences. It justifies your evaluation protocol and
   pre-empts "why not PSNR/SSIM like everyone else?".

## 13. Where the whole arc should go

| venue / form | why | when |
|---|---|---|
| **arXiv technical report / GSoC final report** (8–15 pp) | 90% written already (`PROJECT_REPORT.md`, `DEEPLENSE_TECHNICAL_BRIEF.md`, `CONCEPTS_AND_METHODS.md`). Genuinely useful to whoever inherits this line. | after 29 Aug, before the GSoC deadline |
| **ML4PS @ NeurIPS** or a similar astro-ML venue | Astro-ML audiences are receptive to diagnosis-plus-repair narratives; the precedent went there. | next cycle |
| **A short data note (RNAAS or a dataset-doc PR)** for the `_nss` defect | Separable, self-contained, saves other people days. Do it with the mentors — it is their dataset. | anytime; do not let it block the paper |
| **MNRAS / ApJ methods paper** | The full thing, once you have HSC-degraded results and two-band fits. | 3–6 months |

## 14. If you write the whole-arc version anyway

Only one framing survives four pages:

> **"What a fixed lens costs you: a measured account."** Hold everything constant and vary only the
> lens model. With a circular lens the source inflates 10–13× at *every* regularisation strength;
> this is the optimal response to a misspecified forward model, not a tuning failure; replacing the
> lens with a fitted EPL+shear collapses the error to 1.08× — **and the conventional image-plane
> metric reports improvement throughout the failure.**

Good paper, different paper, and it needs the |e| < 0.08 control from `PROJECT_REPORT.md` §8.3 to
nail causality (four confounds still outstanding). Not in seven days.

---

## 15. Include / skip summary

| item | 4-page paper | appendix | omit |
|---|---|---|---|
| Unsupervised χ² objective through analytic EPL+shear | **core** | | |
| **Four-representation table** | **core** | | |
| Decoder-vs-conditioning separation | **core** | | |
| Coverage-vs-resolution arithmetic | **core** | | |
| "Lower χ², worse source" | **core** (one sentence) | | |
| **Injection experiment (2,400 clumps)** | **core** | full stratification | |
| Four magnification nulls | **core** (short) | detail | |
| One-pass network as initialiser | **core** | full table | |
| Spin-2 shrinkage mechanism | **core** (compressed) | phase-bin table | |
| Parameter recovery, 7 rows | 2 rows | full | |
| size_ratio 10.65 → 1.08 | **core** | | |
| Flatness 1×–8× + measured 3.07× stretch | 2 sentences | full table | |
| Lens-model ladder | 2 lines | full | |
| 100× TV sweep | 1 row | | |
| `skill` null baselines | 2 sentences | full | |
| Ring-harmonic initialisation | 2 sentences | detail | |
| PSF calibration | 2 sentences | full | |
| B5 gate | 1 row | detail | |
| B2 / Option 2 | 1 row each | brief | |
| `image_nss` dataset defect | footnote | full | (or separate note) |
| Operator bank / Model_4 | 1 sentence | | mostly |
| Numerics, backend shim, test ladder | 1 sentence | brief | |
| Critical curve / μ_tot validation | 1 sentence | full | |
