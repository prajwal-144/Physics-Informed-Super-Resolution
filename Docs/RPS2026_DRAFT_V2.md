# What Should the Source Be Made Of? A Label-Free Lens and Source Inversion

> **Format note.** Drafted for *Representations for the Physical Sciences (R/PS)*, NeurIPS 2026.
> Four pages of main content; references and appendices excluded. Double-blind, non-archival.
> Workshop template (`rps_2026_template.zip`, NeurIPS style). Sections 1 to 6 are the four pages.
>
> **Scope.** Everything reported here comes from the `superres/` codebase. One number is flagged in
> the notes at the end.
>
> **What changed from the previous R/PS draft.** The subject is now the system rather than the
> comparison. Section 2 describes a thing that works and shows evidence that it works; Table 1 is a
> design decision inside it rather than the paper's spine; the magnification validation moved into
> Section 2 as evidence rather than standing alone; the σ-floor and the response calibration went
> back to being method detail. The evidence is identical.
>
> **Alternative titles.** (a) *Physics as the only label: an unsupervised joint lens and source
> inversion.* (b) *A label-free strong-lens inversion, and the representation its source should
> have.*

---

## Abstract

High-resolution imaging of strong gravitational lenses is scarce, and the surveys that will find the
next hundred thousand lenses will not make it less scarce, so the paired data that supervised
super-resolution assumes will not exist at survey scale. We present a system that needs none. For
each observation it fits a lens and a source jointly with no labels and no high-resolution target:
an analytic, differentiable elliptical-power-law-plus-shear deflection field, a measured instrument
response and detector binning close a χ² directly against the observed pixels. On Roman-like
simulations with full generative ground truth it recovers seven physical parameters from the pixels
alone (Spearman +0.54 to +0.96), reconstructs a source whose size error is 1.08× against a free-field
baseline's 10.6×, and reproduces magnification quantities it was never fitted to, including the
tangential critical curve to 0.63 px. Inference costs 12 ms amortised or 0.46 s refined. The one
design choice inside the system that is not settled by physics is what the source is made of, so we
measured it: five representations under one forward model, where conditioning predicts which ones fail
and decoder architecture accounts for a further factor of two among those that do not. We then test
whether the recovered sub-pixel detail is real by injecting structure smaller than a detector pixel
and measuring what returns. All results are on one simulated instrument and one dark-matter class.

---

## 1. Introduction

Strong gravitational lensing turns a foreground mass into a telescope. The lens spreads a patch of the
background galaxy over many detector pixels, so the sky has already performed some of the
oversampling that a super-resolution method would otherwise have to invent. Recovering the source
from the image, though, means inverting a map that is many-to-one by construction, since that is what
makes arcs and rings in the first place.

Euclid, LSST and Roman will find lenses in numbers that put higher-resolution follow-up out of reach
for all but a handful of systems. Paired low- and high-resolution data will therefore not exist for
the population anyone wants to study, which rules out the supervised route. What remains is the
forward model: whatever lens and source we propose must, after being lensed, blurred by the
instrument and integrated by the detector, reproduce the pixels we observed. That is the entire
supervision signal in this paper.

Building a system on that principle forces one choice that the physics does not make for you. The
lens is a small, identifiable, well-motivated family and nobody argues much about it. The source is
not. Published choices range from seven analytic parameters to tens of thousands of free pixels, and
the choice is usually settled by convention. We describe the system, show what it recovers, and then
treat the source representation as the open design question it is, measuring five options under one
fixed forward model.

Our contributions.

1. **A working label-free joint lens and source inversion**, validated against full generative truth:
   seven physical parameters at Spearman +0.54 to +0.96, a reconstructed source at 1.08× the true
   size where a free-field baseline on the same data sits at 10.6×, and a magnification field
   validated on quantities never entering the fit.
2. **A controlled comparison of five source representations** inside that system, separating
   conditioning from decoder architecture, including a case where the representation reaching the
   lower validation χ² recovers the source substantially worse.
3. **A direct test that the recovered sub-pixel detail is real**, by injection through the full
   instrument model at matched noise.
4. **An amortised front end that is a poor estimator and an excellent initialiser**, with the
   mechanism identified and the cost measured.

## 2. The system

**What goes in and what comes out.** One observed image goes in. Out come six lens parameters, seven
source parameters, a sky level, a source rendered on any grid the user asks for, and a magnification
field. Nothing else is required: no catalogue, no simulation metadata, no high-resolution target.

**The forward model.** Deflection is
$\boldsymbol\beta = \boldsymbol\theta - \boldsymbol\alpha(\boldsymbol\theta)$ with an elliptical power
law plus external shear. We need it differentiable and available in torch, which rules out evaluating
the Tessore and Metcalf [1] closed form through `scipy.special.hyp2f1`, so we evaluate the underlying
power series, whose argument satisfies $|w| = (1-q)/(1+q) < 1$ for every physical axis ratio and
therefore converges geometrically with an adaptive term count. The result agrees with `lenstronomy`
[2] to $2.7\times10^{-15}$. Rays are then shot, convolved on the sky, and area-averaged onto detector
pixels, in that order, because the optics act before the detector integrates.

The instrument response is measured rather than assumed. The stored unlensed array is the analytic
source already convolved with the true kernel, which over-determines it: a parametric sweep gives a
clean minimum at 0.18 to 0.20″ where prior work on these data used 0.10″, and a Wiener-regularised
extraction shows wings 29 times higher than a Gaussian at $r = 0.25''$. Noise is estimated per image
from a source-free annulus.

**The objective** is $\chi^2 = \sum_i ((\text{model}_i - d_i)/\sigma)^2$ over a 45-pixel disc, 6,361
pixels. For the amortised runs we replace σ with
$\sigma_{\rm eff}^2 = \sigma_{\rm bg}^2 + (0.02\,\text{model})^2$, admitting the measured
one-to-three-per-cent kernel-shape systematic; without it four images in every four hundred carry half
the batch gradient.

**Starting without labels.** Where a starting value is needed we measure it from the image, fitting
harmonics to the radius of peak brightness along 72 azimuths. The $m = 0$ term estimates the Einstein
radius to a median 0.48 px on a ring of median radius 12.5 px, and the $m = 1$ term estimates the
source offset at Spearman +0.875. Six lens parameters are justified by measurement rather than by
convention: holding the true source fixed and varying only the deflection, arc-weighted correlation
with the truth runs 0.869 for a circular isothermal sphere, 0.973 for an ellipsoid and 0.997 with
external shear.

**Evidence that it works.** On 800 validation images the seven physical parameters come back at
Spearman +0.960 ($\theta_E$), +0.954 (half-light radius), +0.945 (Sérsic index), +0.915 (source
offset), +0.839 (lens ellipticity), +0.650 (external shear) and +0.538 (radial slope), with none of
these available to the fit. The reconstructed source matches the stored truth at correlation 0.987
with a size ratio of 1.081, against 10.6 to 12.8 for a free-field baseline previously run on this
dataset.

The check we find most convincing is on quantities the fit never saw. The magnification field derived
from the fitted lens matches the truth at median log-correlation 0.929, total magnification comes out
at 5.996 against 5.941, and the tangential critical curve is recovered to 0.63 px r.m.s. Its azimuthal
swing is 0.540″ against a true 0.548″, and a circular lens gives exactly zero swing, so recovering it
means the four shape parameters were recovered jointly rather than in magnitude alone.

Reduced χ² settles near 3,200 rather than 1, for a reason we can name. The model does not include
substructure and the data contain it: inside $r < 30$ px the r.m.s. difference between images with and
without subhaloes is 4.33 against a background σ of 0.032. What remains is instrument-response shape,
where a one per cent error on arcs reaching peak-to-background ratios of $10^3$ to $10^4$ is a 50σ
per-pixel residual. The signature separating both from an underestimated noise level is that the
residual grows with brightness: median χ²/dof runs 1,321, 2,706 and 7,140 across three
signal-to-noise bins. We report χ²/dof with null baselines beside it (37,511 for a constant image,
2,796 for a two-pixel blur of the observation) and score everything else against source-plane truth.

> **Figure 1.** The system. Observation in; a fitted lens, a source and a magnification field out.
> Both inference paths, per-image least squares and the amortised front end, re-lens, convolve and
> bin, and close χ² against the same observed pixels. No high-resolution target enters anywhere.
> *[To draw; component panels exist in `figS5_sersic_stages.png`.]*

## 3. The design question: what the source is made of

Everything in Section 2 is fixed by physics or by measurement except one thing. The source has to be
parameterised somehow, and the choice changes the answer. Table 1 varies only that, holding the data,
the deflection model, the instrument model and the scorer constant.

> **Table 1.** Five source representations inside the same system, scored on the same 800 validation
> images against the stored truth, convolution-matched. "Data per value" uses the 4,312 pixels above
> 3σ. Row 1 is a free-field baseline previously run on this dataset with the lens fixed to a circular
> isothermal sphere; its size ratio is the range across a hundredfold sweep of the regulariser.
>
> | source representation | free source values | data per value | corr | size ratio | nmse |
> |---|---|---|---|---|---|
> | free field, 254², lens fixed and circular | 64,516 | 0.07 | – | 10.6 – 12.8 | – |
> | free field, 62², convolutional decoder on the back-projection | 3,844 | 1.12 | 0.973 | 1.130 | 0.0527 |
> | Sérsic + 48² residual, same decoder | 7 + 2,304 | 1.87 | 0.979 | 1.043 | **0.0408** |
> | Sérsic + 32² bounded correction from a 132-vector | 7 + 1,024 | 4.18 | 0.959 | 0.937 | 0.0799 |
> | Sérsic only, 7 values | 7 | 616 | **0.987** | 1.081 | **0.0252** |

Read the last row first, because it sets how to read the rest. The sources in this dataset are
genuinely Sérsic profiles, so seven parameters is the exactly correct functional form and 0.0252 is a
ceiling rather than a competitor. Nothing free-form should beat it here. The useful question is which
free-form representation gets closest, because that is the one that transfers to real galaxies, where
no analytic family applies.

Conditioning explains the two failures at the top. The 254² field is underdetermined fifteen to one
and inflates the source tenfold to thirteenfold at every regularisation strength across a
hundredfold sweep. That is not a tuning failure: the unconstrained minimiser of total variation is a
flat field, so raising the weight moves toward the wash, and the most compact result in the sweep was
the least regularised one. The mechanism generalises. When the forward model cannot place flux at the
right azimuth whatever source is proposed, the best strategy under a pixel loss is to spread the
source until the lensed ring is fat enough to overlap the true arc everywhere, so an
underdetermination problem and a misspecification problem produce the same visible symptom.

Among the representations that do not fail, conditioning predicts very little. The two hybrid rows are
both overdetermined, at 4.18 and 1.87 data per free value, they differ by a factor of two in source
error, and the better-conditioned one is the worse. What separates them is where spatial information
is allowed to go. The 32² correction is decoded from a 132-dimensional global summary produced by an
average pool, and 64 per cent of that model's 849,519 weights sit in the single dense layer that then
has to regenerate a spatial map from it; measurably, the correction it produces never exceeds 3 to 4
per cent of the Sérsic amplitude. The 48² residual comes from a fully convolutional decoder acting on
the observation already back-projected into the source plane, so it starts in spatial register and
only has to deblur and sharpen. It carries fewer weights, about 0.63 M against 0.85 M, so capacity is
not the explanation, though the two runs also differ in box size, so the attribution rests on the
mechanism and on the measured saturation rather than on a single-variable swap.

One comparison in the table is worth isolating. The free-form decoder run had more training images
(8,000 against 6,000), more epochs and a wider box than the hybrid, and it reached the lower
validation χ² of the two on the objective both were trained on, 329 against 343. Its source is the
worse one, at nmse 0.0527 against 0.0408. Every one of those advantages should have helped recovery as
well as fit. Selecting a representation on held-out data fidelity would have chosen wrongly here,
which is worth knowing for any system where the loss is the only signal available.

## 4. Is the recovered detail real?

The system renders its source on grids finer than the detector, and the obvious objection is that fine
detail could be invention rather than recovery. The physical licence is the magnification field:
median tangential stretch is 3.075× against a radial 0.994×, so the data support a source grid roughly
three times finer than the detector, and both decoders ask for about 2.05×, inside that ceiling.

Licence is not evidence, so we test it directly. We add a Gaussian clump of FWHM 0.08″, which is 0.76
detector pixels, at 15 per cent of the source peak; re-render the whole system through the lens, the
instrument response and the detector; and add noise at that image's measured σ. The identical system
without the clump is then reconstructed under the same noise realisation, so the noise cancels in the
difference, and we measure what survives in a small aperture at the clump's position. 2,400 injections
across 60 images, on a trained checkpoint, with no retraining.

Between 17 and 22 per cent of the injected contrast returns, depending on local magnification. Not
zero, so sub-detector-pixel information is genuinely being recovered; not all of it, so most is lost.
Recovery is better where the lens magnifies more, at mean contrast 0.208 for $\mu \geq 2$ against
0.175 for $\mu < 2$, Mann-Whitney $p = 5.3\times10^{-5}$, and fractional reconstruction error over the
whole source falls from 0.39 in the $\mu = 1$ to 2 bin to 0.038 in the $\mu = 16$ to 32 bin.

Magnification is nonetheless not the binding constraint. Rank correlation of recovered contrast is
+0.056 with local magnification and +0.152 with signal-to-noise, and stratifying by the latter the
magnification advantage runs 1.25, then 1.22, then 1.08 as photons become plentiful. It decides
recoverability when photons are scarce, and this dataset is photon-rich almost everywhere. That
explains two separate attempts we made to build magnification into the prior, as a penalty on the
correction map and as a structural gate on the decoder output, both of which changed source error by
about one per cent. The gate is the instructive one: it
sits before the likelihood, the network is trained through it, and the network learns its inverse,
emitting 1.61× the sub-pixel variance of the control before a gate that attenuates by 0.624.

> **Figure 2.** Recovered contrast against local magnification, 2,400 injections, with the
> signal-to-noise stratification showing the effect closing from 1.25× to 1.08×.
> *[Source: `figMU1_resolution_vs_mu.png`, to regenerate at publication size.]*

## 5. Cost

One forward pass of the amortised front end costs 12 ms against 815 ms for the per-image fit, and it
matches the fit on the Einstein radius (+0.960) and half-light radius (+0.953). It fails on
ellipticity, at +0.269 against +0.762, and not because it gets the orientation wrong: the median phase
error is 19° while the modulus is shrunk by a flat factor near 0.22 in every phase-error bin,
including the bin where orientation is right to within 5°. A network trained on a squared loss is a
conditional-mean estimator, and for a spin-2 quantity under orientation uncertainty a large
ellipticity pointed the wrong way costs more than none at all. More epochs cannot fix that.

So the network starts the fit rather than replacing it. On 800 images with identical bounds and
tolerances, the cold fit takes 480 model evaluations and 0.815 s across four staged parameter releases
and reaches χ²/dof 3,302; from the network's start it takes 151 evaluations and 0.460 s in a single
stage and reaches 3,228, with recovery better on six of seven parameters and the shrinkage undone
entirely.

## 6. Scope

Every result here comes from one dark-matter class. The dataset loader sorted a concatenated list of
per-class paths before truncating, so all runs drew from the axion class; training and validation
remain disjoint splits and this class carries the heaviest subhalo population of the three, so the
smooth-model results are conservative, but cross-class generalisation is untested. Everything is
simulated, on one instrument, using one of two available bands.

The last row of Table 1 is a ceiling specific to a dataset whose sources are Sérsic profiles by
construction, and the system has not been shown on a galaxy that is not one. The magnification field
is derived from the fitted lens rather than measured independently. We report point estimates and no
posterior, which is the largest methodological gap and the one our own results point at: the flat 0.22
shrinkage in Section 5 is a bias we can measure and explain, and a density head over the fourteen
parameters would report it as a width rather than commit it as an error.

---

## References

[1] N. Tessore and R. B. Metcalf. The elliptical power law profile lens. *Astronomy & Astrophysics*, 580:A79, 2015.

[2] S. Birrer and A. Amara. lenstronomy: multi-purpose gravitational lens modelling software package. *Physics of the Dark Universe*, 22:189, 2018.

[3] S. Warren and S. Dye. Semilinear gravitational lens inversion. *The Astrophysical Journal*, 590:673, 2003.  *(now uncited in the body -- either cite it beside [4] and [5] as prior art on pixellated source inversion, or drop it and renumber)*

[4] S. H. Suyu, P. J. Marshall, M. P. Hobson and R. D. Blandford. A Bayesian analysis of regularized source inversions in gravitational lensing. *MNRAS*, 371:983, 2006.

[5] S. Vegetti and L. V. E. Koopmans. Bayesian strong gravitational-lens modelling on adaptive grids. *MNRAS*, 392:945, 2009.

[6] Y. D. Hezaveh, L. Perreault Levasseur and P. J. Marshall. Fast automated analysis of strong gravitational lenses with convolutional neural networks. *Nature*, 548:555, 2017.

[7] L. Perreault Levasseur, Y. D. Hezaveh and R. H. Wechsler. Uncertainties in parameters estimated with neural networks: application to strong gravitational lensing. *ApJL*, 850:L7, 2017.

[8] W. R. Morningstar et al. Data-driven reconstruction of gravitationally lensed galaxies using recurrent inference machines. *The Astrophysical Journal*, 883:14, 2019.

[9] A. Adam, L. Perreault Levasseur, Y. Hezaveh and M. Welling. Pixelated reconstruction of foreground density and background surface brightness in gravitational lensing systems using recurrent inference machines. *The Astrophysical Journal*, 925:124, 2022.

[10] K. Karchev, A. Coogan and C. Weniger. Strong-lensing source reconstruction with variationally optimized Gaussian processes. *MNRAS*, 512:661, 2022.

[11] A. Shankar, M. W. Toomey and S. Gleyzer. Unsupervised physics-informed super-resolution of strong lensing images for sparse datasets. *Machine Learning and the Physical Sciences workshop, NeurIPS*, 2024.

[12] C. Ledig et al. Photo-realistic single image super-resolution using a generative adversarial network. *CVPR*, 2017.

[13] W. Shi et al. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. *CVPR*, 2016.

[14] T. E. Collett. The population of galaxy-galaxy strong lenses in forthcoming optical imaging surveys. *The Astrophysical Journal*, 811:20, 2015.

*Still to add: Roman, Euclid and LSST mission references, and a citation for the simulation dataset.*

---

# Appendix

Unlimited length. A to E are drafted; F to I are outlined.

## A. Full parameter recovery

Per-image bounded nonlinear least squares (trust-region reflective, not plain Levenberg-Marquardt,
since several parameters carry hard bounds), warm-started from the amortised front end, $n = 800$.

| parameter | fit median | true median | median &#124;err&#124; | p90 &#124;err&#124; | Spearman |
|---|---|---|---|---|---|
| $\theta_E$ (″) | 1.370 | 1.320 | 0.047 | 0.179 | +0.960 |
| β (″) | 0.292 | 0.305 | 0.033 | 0.091 | +0.915 |
| $R_{\rm sersic}$ (″) | 0.505 | 0.484 | 0.041 | 0.159 | +0.954 |
| $n_{\rm sersic}$ | 1.140 | 1.005 | 0.138 | 0.786 | +0.945 |
| slope γ | 2.003 | 2.058 | 0.078 | 0.247 | +0.538 |
| lens &#124;e&#124; | 0.200 | 0.222 | 0.032 | 0.098 | +0.839 |
| shear &#124;g&#124; | 0.049 | 0.032 | 0.015 | 0.050 | +0.650 |

Source plane: correlation 0.9873, size ratio 1.0808, centroid error 0.506 px, peak ratio 0.834,
nmse 0.0252. χ²/dof median 3,227.9. The slope is the least constrained parameter, sitting close to
isothermal; the lens-model ladder puts the free slope at about +0.010 in arc correlation and the
external shear at about +0.014, so both matter and shear matters more.

## B. Instrument response calibration

Parametric Gaussian sweep, residual nmse against the stored convolved source, 25 compact sources:
0.01062, 0.00707, 0.00435, 0.00300, 0.00288, 0.00312, 0.00855 at FWHM 0.00, 0.10, 0.14, 0.18, 0.20,
0.22 and 0.30″. Wiener-regularised Fourier extraction stacked over 40 sources gives the wing
measurement quoted in Section 2. The generator was verified rather than assumed: an analytic
power-law-plus-shear convergence matches the stored map at log-correlation 0.936, and an analytic
elliptical Sérsic matches the stored source at correlation 0.967, rising to 0.9989 with the response
included.

## C. Resolution is not a rendering choice

The parametric source is continuous, so it can be sampled on any grid. Scored against the true source
rendered on the same grid, $n = 2{,}000$:

| factor | scale (″/px) | corr | nmse | size ratio |
|---|---|---|---|---|
| 1× | 0.1059 | 0.9787 | 0.0420 | 0.979 |
| 2× | 0.0530 | 0.9773 | 0.0446 | 0.955 |
| 4× | 0.0265 | 0.9770 | 0.0452 | 0.942 |
| 8× | 0.0132 | 0.9769 | 0.0453 | 0.933 |

Correlation moves by 0.0017 across an eightfold range. The sampling does not limit the reconstruction,
the parameters do.

## D. Validation ladder

| test | asserts | measured |
|---|---|---|
| deflection | power law + shear against `lenstronomy` 1.14.2 | ≤ 3e−15 |
| source profile | elliptical Sérsic against `lenstronomy` | 0.00e+00 |
| hypergeometric series | truncation against `scipy.special.hyp2f1` | ≤ 3e−15 |
| magnification | autograd Jacobian against the analytic Hessian | ≤ 2e−9 |
| supersampling | error against an $S = 9$ reference | 5.5e−3 at $S = 3$ |
| backend parity | identical physics under numpy and torch | 1e−6 |
| torch forward model | against the numpy renderer | 5.5e−16 |
| back-projection | round trip reproduces the source's value, not only its shape | 3.3% median |

Two gates run before any training: one pokes a single input pixel and requires the peak output
response to land on the corresponding output pixel; the other requires no non-finite gradient on step
one, which matters because the ellipticity parameterisation is singular at $e_1 = e_2 = 0$ under
reverse-mode autodiff and that is a legitimate place for the optimum to sit.

## E. The two magnification nulls

| mechanism | μ-adaptive | uniform |
|---|---|---|
| correction penalty, 32² decoder | 0.0799 | 0.0789 |
| structural resolution gate, 48² residual decoder | 0.0414 | 0.0408 |

For the gate, sub-block variance before the gate is 0.0427 against the control's 0.0266, the gate
attenuates by 0.624, and $1.61 \times 0.624 = 1.005$. Where it does survive is the faint outskirts,
buying 14 to 18 per cent less speckle below 5 per cent of peak surface brightness while being
invisible to every flux-weighted metric.

## F to I. Outlined

- **F. Box size as a measured choice.** Median flux inside the box against half-extent: 0.715 at 0.8″,
  0.891 at 1.2″, 0.958 at 1.6″, with the data-per-unknown ratios.
- **G. Spin-2 shrinkage.** Modulus ratio by phase-error bin: 0.21, 0.23, 0.23, 0.22 across 0 to 5°,
  5 to 15°, 15 to 30° and 30 to 90°, counts 120, 221, 200, 259.
- **H. Injection details.** Aperture definition, per-bin counts (253, 1215, 572, 295, 65), the full
  signal-to-noise stratification, and the fixed-amplitude caveat: the clump is 15 per cent of the
  global peak wherever it lands, so Spearman(distance from centre, contrast) = +0.40.
- **I. Reproduction.** Anonymised repository, exact commands, χ²/dof convention.

---

## Notes for the author (delete before submission)

**What this draft is.** The system is the subject. Section 2 describes a thing that works and offers
evidence that it works, including the out-of-sample magnification check, which is the single most
persuasive paragraph you have and was buried before. Table 1 is now a design decision inside the
system rather than the paper's spine, which is what you asked for and, I think, the better paper. The
σ-floor and the response calibration went back to being method detail; if you ever want them
foregrounded again, that version exists as `SIM2SCIENCE_DRAFT.md`.

**One number still out of scope.** The lens-model ladder in Section 2 (0.869 / 0.973 / 0.997) comes
from the earlier diagnostics work, not from `superres/`. It is two hours to re-run in scope: render
the true Sérsic through each candidate deflection model at 3× supersample, convolve, bin, no noise,
reference the same source through the true lens, arc-weighted correlation, $n = 60$, split by lens
$|e|$. Everything needed is in `raytrace.py`, `lens_models.py` and `sources.py`. If you cannot, delete
the numbers and write that clause as motivation rather than measurement. The response sweep in
Appendix B just needs `superres/calibrate_psf.py` re-run for three minutes.

**Three things to fix.**

- **χ²/dof convention.** `evaluate.py` divides by $6361 - 14$; `eval_pathb.py` divides by
  $6361 - 14 - 1024$, a factor 1.192. Confirm what `eval_b4.py` uses and state the convention once.
  Raw χ²/dof comparisons between the parametric fit and the decoders are deliberately kept out of the
  main text; Section 3 uses the floored validation χ² both decoders were trained on.
- **Two nulls now, not three or four.** Option 2 was removed from the paper at the author's
  request, which also removes the linear-inversion null. `B5_RESULTS.md` separately lists a B4
  `--reg-mode mu` versus `uniform` ablation quoting B3's numbers, and there is no `b4_uniform`
  metrics file, so that one never existed either. Appendix E reports two. Running the B4 uniform
  control would take it back to three honestly.
- **The stale comment in `lens_models.py`.** Lines 253 to 258 say "audit_dataset.py shows Model_A's
  macro lens IS circular", which is the 17 August conclusion you retracted on the 18th. If you link an
  anonymised repository, that line contradicts Section 2.

**Length.** Sections 1 to 6 run about 2,580 words including the abstract, with two figures and one
table. Four NeurIPS pages in that configuration holds roughly 2,500, so you are close for once.
Expect to shed 100 to 200 words once typeset. Trim in this order: the
mechanism sentence in Section 3's conditioning paragraph (about 55 words), the instrument-response
paragraph in Section 2 down to one sentence pointing at Appendix B (about 60), the resolution-licence
sentences opening Section 4 (about 55). **Do not cut** the out-of-sample magnification paragraph, the
"read the last row first" paragraph, or Figure 2.

**Anonymity.** Nothing names a person, institution or programme; [11] is third person; Table 1's first
row is described as "previously run on this dataset" without claiming or disclaiming authorship, which
is the right handling under double-blind.

**Style.** Bold run-in headers appear only in Section 2, where they are doing real navigational work
in a section that covers five things. The Introduction and Section 6 are the two places most worth
rewriting in your own hand before submission.
