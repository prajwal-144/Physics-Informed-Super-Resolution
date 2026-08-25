# Venue decision: Sim2Science or R/PS

23 August 2026. Both deadlines are 29 August (AoE), both notify 29 September, both are in Paris on
12 or 13 December. Six days.

**Recommendation: Sim2Science, reframed. Not R/PS.** Reasoning below, then the six-day plan.

---

## 1. The two calls side by side

| | **R/PS** (Representations for the Physical Sciences) | **Sim2Science** (ML with Imperfect Scientific Models) |
|---|---|---|
| page limit | **4** excl. references | **5** excl. references (plus a 2-page "tiny paper" track) |
| appendices | unlimited | unlimited, "reviewers are not obligated to read it" |
| template | workshop zip, NeurIPS style | `\usepackage[dblblindworkshop]{neurips_2026}` + `\workshoptitle{Sim2Science}`; wrong template may be **desk-rejected** |
| review | double-blind | double-blind, and non-anonymised submissions "may lead to desk rejection" |
| archival | non-archival | non-archival, accepted papers linked from the site |
| concurrent submission | **permitted** ("check other venue policies") | **discouraged**: "We discourage parallel submission of the same paper to multiple NeurIPS 2026 workshops" |
| portal | `NeurIPS.cc/2026/Workshop/RPS` | `NeurIPS.cc/2026/Workshop/Sim2Sci` |
| what they want | not stated beyond topics | "**contributions of any kind — new methods, applications, analyses, benchmarks, or position pieces**" |
| organisers | Novelli, Pontil, d'Alché-Buc, Gentine, Lamb, Niepert | Channing (HF), Éltetö (DeepMind), Gao (Frankfurt), Gedon (Tübingen), Lederbauer (MIT); advisors Clementi, **Macke**, **Welling** |
| speakers | not listed on the page I read | **Shirley Ho** (Polymathic/NYU), Köhler, Castro, Skreta, Cadena, Koumoutsakos |
| example domains named | none | includes "**cosmological simulations**" |

The concurrent-submission asymmetry settles one thing immediately: **you pick one.** R/PS would
allow both, Sim2Science asks you not to, and the deadlines are the same day so you cannot sequence
them either.

## 2. Topic fit, mapped to evidence you already have

### R/PS: one strong axis out of four

| axis | fit | evidence |
|---|---|---|
| Self-supervision on unlabelled scientific data with physical constraints | **strong** | The whole objective. No labels, no HR target, χ² through an analytic forward model. |
| Adaptive sampling / simulation-driven data generation | moderate | The 2,400-injection experiment is simulator-in-the-loop. It is not adaptive sampling. |
| Transfer learning and OOD generalisation | **absent** | One class, one instrument, one band. Scores zero. |
| Tokenisation / discrete representations | weak | The paper is about representations, none of them discrete. |

### Sim2Science: two of its named topics are your two best assets

| topic | fit | evidence you already have |
|---|---|---|
| **Model misspecification, simulator diagnostics, discrepancy modelling** | **very strong** | The instrument response was assumed 0.10″ and measured at 0.18 to 0.20″ with wings 29× a Gaussian at 0.25″; a 1 per cent kernel-shape error is a 50σ per-pixel residual; χ² *rises* with SNR (ρ = +0.449), which is the signature that separates a systematic from noise. The σ-floor $\sigma_{\rm eff}^2=\sigma_{\rm bg}^2+(f\cdot\text{model})^2$ is textbook discrepancy modelling and it flattens χ² across SNR from 1411→8455 down to 329→229. The dataset itself was diagnosed: the substructure-free arrays were rendered with a circular deflector, caught by a phase test (1.6° against 44.3°) rather than an amplitude test. |
| **Simulator structure, degeneracy, simplifications, identifiability** | **very strong** | The six-representation table *is* an identifiability study. "Free-form has the lower validation χ² (329 against 343) and the worse source (nmse 0.0527 against 0.0408)" is a pure identifiability result: better data fit, worse recovery. The spin-2 shrinkage is an estimator-identifiability finding with a mechanism. Mass-sheet and ellipticity-shear degeneracies are handled explicitly. |
| Emulator / surrogate, hybrid and physics-informed | **strong** | The amortised network as an initialiser: 480 → 151 model evaluations, four stages → one, at a slightly better χ². That is surrogate-accelerated inference with a measured speedup. |
| Differentiable frameworks | **strong** | A differentiable EPL+shear lens validated against `lenstronomy` to 2.7e−15, with the hypergeometric series and the singularity handling. This is a contribution here; at R/PS it is a method detail. |
| Closed-loop / experiment-in-the-loop | moderate | The injection experiment: inject a known 0.76-pixel clump, re-render through the full instrument model at matched noise, measure what returns. |
| Simulation-based inference and parameter inference | **partial, and the risk** | You do maximum likelihood per image and amortised point estimation. No posterior, no uncertainty. See §4. |

Four strong or very strong against R/PS's one. That is not a close call.

## 3. The sentence that matters most

> "We welcome contributions of any kind — new methods, applications, **analyses**, benchmarks, or
> position pieces."

The lowest score in the v3 assessment is novelty (7.0), and the reason is that this is a
measurement-and-comparison paper with no new algorithm. R/PS does not say it wants analyses.
Sim2Science says it explicitly. **The call neutralises your single biggest scoring weakness**, which
is worth more than any amount of rewriting.

Two secondary practical wins: five pages instead of four, when the draft is already 300 to 400
words over four; and Shirley Ho on the speaker list plus "cosmological simulations" among the named
example domains, which means a lensing paper is expected rather than tolerated.

## 4. The risks of choosing Sim2Science, honestly

**1. The reviewer pool is simulation-based-inference heavy.** Jakob Macke is an advisor, Richard Gao
is an organiser, Max Welling is an advisor. That community's first question about a point estimate
with a known bias is "where is the posterior?". You have no uncertainty quantification anywhere,
which the v3 assessment already flagged as the biggest ML gap. It becomes more conspicuous here
than at R/PS.

*Mitigation, and it is a good one:* you have a **measured and mechanistically explained** bias, the
flat 0.22 shrinkage of a spin-2 quantity across every phase-error bin. State plainly that this is
what a conditional-mean estimator does under a squared loss, that a density head or normalising
flow over the fourteen parameters would report it as a width instead of committing it as an error,
and that you demonstrate the practical workaround (use the estimator as an initialiser and let the
likelihood finish). That converts "no posterior" from an omission into an identified, motivated
next step, which is exactly the register this workshop is in.

**2. The reframing costs about a day of your six.** Nothing needs re-running. What changes is the
abstract, the introduction, the section headings and roughly one page of emphasis, promoting the
σ-floor from a parenthetical to a result. Section map in §6.

**3. One causal claim becomes load-bearing and it is not yet clean.** In the R/PS draft, "the
free-field baseline inflated the source 10.6 to 12.8× across a 100× regulariser sweep" is one row of
Table 1. In a misspecification paper it moves near the headline, and your own
`diagnostics_2026_08_12/PROJECT_REPORT.md` §8.3 says the decisive control has not been run: training
the unchanged pipeline on the |e| < 0.08 subset, with four confounds still outstanding (grid null
space, response misspecification, training budget, and whether the bin routing was used at all).

*How to keep it honest and still strong.* Split the claim in two.
- The **clean** half: the deflection-family ladder. True source held fixed, only the deflection
  varied, n = 60, arc-weighted correlation 0.869 → 0.973 → 0.997, and the failure is angular rather
  than radial (mean critical radius right to 0.7 per cent while the omitted swing is 5.4 px on a
  12 px ring against a 2 px arc width). No confounds. This is a genuinely nice misspecification
  measurement because the error is invisible to any azimuthally averaged metric.
- The **observational** half: the pipeline built on that misspecified family inflated sources 10.6
  to 12.8× at every regularisation strength, which rules out regulariser tuning specifically and
  leaves four candidate causes, one control away from separation.

Saying "here are four candidate causes and here is the single experiment that separates them, which
we have not yet run" is not a weakness at this workshop. It is the genre.

**4. Unknown selectivity.** Neither workshop publishes an acceptance rate and I could not find one
for either. NeurIPS workshops broadly sit in the 40 to 65 per cent range. Treat both estimates below
as informed guesses, not data.

## 5. Acceptance estimates

| branch | estimate | reasoning |
|---|---|---|
| **Sim2Science, reframed** | **~8 / 10** | Four topic axes hit, the call explicitly invites analyses, five pages, a cosmology speaker on the programme, and no new experiments needed. |
| Sim2Science, current draft submitted as-is at 5 pages | ~7 / 10 | The evidence is still on topic even when the framing is not, and the extra page removes the length problem. |
| **R/PS, current draft** | **~7 / 10** | Good paper, one strong axis, two named topics scoring zero, and 300 to 400 words to cut. |

Sim2Science is better or equal in every branch and strictly better if you spend the day. That, plus
the extra page, plus a call that names your weakest dimension as a welcome contribution type, is the
whole argument.

## 6. The reframe, concretely

Working title: **"What a misspecified forward model costs: a measured account from strong
gravitational lensing."**

Everything below reuses experiments you already have. Nothing new runs.

| new section | pages | built from |
|---|---|---|
| **1. Introduction** | 0.6 | The simulator here is a forward model of an observation: deflection, instrument response, detector. Fitting it to data requires it to be right, and it usually is not. Three misspecifications found by measurement rather than assumption, their cost quantified, and three mitigations, one of which fails instructively. |
| **2. The forward model and how we interrogate it** | 0.9 | Current Section 2, plus the differentiable-lens validation promoted from a footnote (it is a named topic here). |
| **3. Three misspecifications, found by measurement** | 1.2 | (a) the deflection family, via the ladder, with the angular-not-radial diagnosis; (b) the instrument response, 0.10″ assumed against 0.18 to 0.20″ measured, with χ² rising with SNR as the diagnostic signature; (c) the dataset, via the phase test. Currently these are scattered across Section 2 and Appendices B and D. |
| **4. What misspecification costs, and how to absorb it** | 1.0 | The 10.6 to 12.8× inflation with the mechanism and the four-confound caveat; then the σ-floor as an explicit discrepancy model with its measured effect (1411→8455 becomes 329→229, p90/p10 spread 95× becomes 8.7×). This is the section that is currently a parenthetical and becomes a result. |
| **5. Identifiability: which representation the data can support** | 1.0 | Current Section 3 and Table 1 essentially unchanged, reframed as an identifiability study. Keep "lower χ², worse source" and give it its own emphasis, because it is the cleanest identifiability statement in the paper. |
| **6. Where the model is right, and what that buys** | 0.8 | Magnification recovered on quantities never fitted (critical-curve swing 0.540″ against 0.548″, where a circular lens gives exactly zero), then the injection measurement and the three nulls. |
| **7. Limitations** | 0.3 | Current Section 6, plus the posterior paragraph from §4 above. |

Three figures as before. Table 1 unchanged. Table 2 unchanged.

**What gets promoted:** the σ-floor, the response calibration, the phase test, the differentiable
lens.
**What gets demoted:** the "representation" framing of the introduction, and the ring initialiser
(one sentence, appendix).
**What is unchanged:** every number, every table, every figure, every appendix.

## 7. Six-day plan

| day | do |
|---|---|
| **Sun 24** | Decide. Tell the mentors which venue and why, and ask for a read on the 27th. Download the NeurIPS 2026 template and confirm `\usepackage[dblblindworkshop]{neurips_2026}` compiles with `\workshoptitle{Sim2Science}`, because the wrong template is a desk reject. |
| **Mon 25** | Rewrite the abstract, introduction and section headings to the §6 map. Write the new Section 3 and the σ-floor half of Section 4 from `DEEPLENSE_TECHNICAL_BRIEF.md` §3.3, §3.4 and §5.7. |
| **Tue 26** | Figures at publication size. Draw Figure 1. This is the block most likely to overrun, so do it before polish. Launch two extra seeds on the B5 control and gate in the background. |
| **Wed 27** | Assemble in LaTeX at five pages. Send to mentors. Fix the three items in the draft's notes block: the χ²/dof convention, the null count, and the [11] attribution. |
| **Thu 28** | Revisions. Anonymise: no programme, institution or personal names; anonymised repository link; acknowledgements held for camera-ready. Check the seed results and either tighten or soften the null-result wording accordingly. |
| **Fri 29** | Submit early in the day. |

**Cut list if you fall behind, in order:** the two extra seeds → Figure 3 → the dataset phase test
(moves to appendix). **Never cut:** Table 1, the injection figure, the response-calibration
measurement, the σ-floor result.

## 8. Decision table

### Scored side by side

| criterion | weight | **Sim2Science** | **R/PS** | why |
|---|---|---|---|---|
| Topic fit to evidence you already have | high | **9** | 6 | Four of its topics hit, two of them very strongly, against one strong axis at R/PS with two named topics scoring zero. |
| Call wants the kind of paper this is | high | **9** | 5 | "analyses" named as a welcome contribution type. Your lowest score is novelty precisely because this is an analysis paper. |
| Room for the content | medium | **8** | 5 | 5 pages against 4; the draft is already 300 to 400 words over 4. |
| Audience expects a lensing testbed | medium | **8** | 6 | "Cosmological simulations" named as an example domain, Shirley Ho on the speaker list. |
| Reviewers likely to reward your strengths | high | **8** | 6 | Misspecification diagnostics and identifiability are their core; representation-learning theorists will not weight them as highly. |
| Reviewers likely to punish your gaps | high | 5 | **7** | Macke, Gao and Welling means "where is the posterior?" is the first question. R/PS will ask it less hard. |
| Work needed before 29 August | medium | 6 | **9** | Sim2Science costs a reframing day. The R/PS draft is written. |
| Value for the GSoC report | low | 7 | 7 | Equivalent. Both are NeurIPS 2026 workshops in Paris, both non-archival. |

### Advantages and disadvantages

| | **Sim2Science** | **R/PS** |
|---|---|---|
| **Advantages** | Two of its named topics are your two most distinctive assets. The call explicitly invites analyses, which neutralises your weakest score. Five pages. A cosmology speaker and cosmological simulations named as in-scope. Your differentiable lens and your σ-floor become contributions rather than method details. No new experiments needed. | The draft is already written for it and fits the format. Self-supervision is a clean, unambiguous match. Concurrent submission is permitted, so it constrains you less. The decoder-versus-conditioning finding lands well with a representation-learning audience. |
| **Disadvantages** | Costs about one of your six days to reframe. The reviewer pool will press hard on the absence of uncertainty quantification. The "misspecified lens caused the 10.6 to 12.8× inflation" claim moves closer to the headline and still has four unresolved confounds. Wrong LaTeX template is a stated desk-reject. | Two of four named topics score zero, transfer and OOD most visibly. Four pages when you are already over. The paper reads as an astronomy application unless the architecture claim is pushed to the front. Nothing in the call signals that a comparison study is welcome. |
| **Chance of acceptance** | **≈8 / 10** reframed, ≈7 / 10 submitted as-is | **≈7 / 10** |
| **Overall score as your venue** | **8.0 / 10** | **6.5 / 10** |

**Verdict: Sim2Science.** It is better or equal on every branch, the extra page solves a problem you
already have, and the one day it costs buys the largest single improvement available to the
submission. Only choose R/PS if you would rather spend that day on the Galaxy10 experiment.

## 9. If you disagree and want R/PS anyway

It is a defensible choice and it costs you almost nothing today, because the draft is already
written for it. Take R/PS if either of these is true: you would rather spend the six days on the
Galaxy10 experiment than on a reframe, or you judge that a representations audience will reward the
decoder-versus-conditioning finding more than a simulation audience will. Both are reasonable. What
is not reasonable is submitting the current draft to Sim2Science without the reframe and expecting
the topical fit to do the work by itself.
