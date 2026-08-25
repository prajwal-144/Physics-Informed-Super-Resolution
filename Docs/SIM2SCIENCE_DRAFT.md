# What the Data Can Support: Identifiability and Misspecification When Physics Is the Only Supervision

> **Format note.** Drafted for *Sim2Science: ML with Imperfect Scientific Models*, NeurIPS 2026.
> Five pages of main content; references and appendices excluded. Double-blind, non-archival.
> Template: NeurIPS 2026 style with `\usepackage[dblblindworkshop]{neurips_2026}` and
> `\workshoptitle{Sim2Science}`. Using anything else is a stated desk-reject, so compile a stub
> first. Sections 1 to 7 are the five pages. Everything under Appendix is unlimited.
>
> **Scope.** Every result reported here comes from the `superres/` codebase. Two numbers currently
> sourced from earlier work are flagged in the notes at the end.
>
> **Alternative titles.** (a) *Better fit, worse recovery: identifiability and misspecification in a
> physics-supervised inverse problem.* (b) *Six source representations under one forward model.*
> (c) *Conditioning is not enough: what a physics-only objective can and cannot identify.*

---

## Abstract

Fitting a physical forward model to data asks two things that are rarely measured: that the model is
close enough to right, and that its unknowns are identifiable from the data actually available. We
measure both on strong gravitational lensing, an inverse problem where the only supervision available
at survey scale is physics. An analytic differentiable elliptical-power-law-plus-shear deflection
field, a measured instrument response and detector binning close a χ² against the observed pixels,
with no high-resolution target and no labels anywhere. Comparing six source representations under this
one forward model, we find that conditioning predicts which representations fail and predicts almost
nothing among those that do not, and we report a case where the representation reaching the lower
validation χ² recovers the true source substantially worse. The instrument response assumed by prior
work on this dataset is wrong by a factor of two in width; the signature is that χ² rises with
signal-to-noise, and absorbing what remains as an explicit discrepancy term flattens the residual
across a 260-fold brightness range. Injecting structure smaller than a detector pixel, we recover 17
to 22 per cent of it, significantly more where the lens magnifies more, though photon count predicts
recovery three times more strongly. Three attempts to turn magnification into a prior came out
neutral, and for the structural one we can say exactly why.

---

## 1. Introduction

Strong gravitational lensing is an inverse problem with an unusually good forward model. Given a mass
distribution and a background galaxy, the observed image is determined by geometry, an optical
convolution and a detector integration, all of which can be written down. The catch is the direction
of travel. We observe the image and want the mass and the galaxy, and the map from one to the other
is many-to-one by construction, since that is what produces arcs and rings in the first place.

Supervised super-resolution is not available here at scale. Euclid, LSST and Roman will find lenses
in numbers that put higher-resolution follow-up out of reach for all but a handful of systems, so
paired low- and high-resolution training data will not exist for the population we care about. The
supervision has to be the forward model itself: whatever we propose must, after being lensed, blurred
and binned, reproduce the pixels we observed.

Committing to that objective is the easy part. What it leaves open is which unknowns to expose. The
lens is a small, identifiable, physically motivated family and nobody argues about it much. The
source is not, and the field's choices range from seven analytic parameters to tens of thousands of
free pixels, usually settled by convention rather than by measurement. We hold the data, the
deflection model, the instrument model and the scoring function fixed, vary only the source
representation, and report what changes. We then ask the complementary question, which is what the
residual misspecification in the forward model costs and whether it can be absorbed rather than
removed.

Our contributions.

1. **A controlled comparison of six source representations** under one forward model, separating
   conditioning from decoder architecture, and including a measured instance of better data fit with
   worse source recovery.
2. **Misspecification diagnosed by signature and absorbed by discrepancy modelling.** The instrument
   response is measured rather than assumed; the residual rises with signal-to-noise, which is what
   distinguishes a systematic from noise; an explicit multiplicative model-error term flattens it.
3. **A direct measurement of sub-detector-pixel recoverability.** 2,400 injections of a 0.76-pixel
   clump through the full instrument model at matched noise, with the magnification and photon-count
   contributions separated.
4. **Two results about what a physics-only objective does to things attached to it**: a squared loss
   shrinks a direction-valued estimate by a measured constant factor, and a differentiable structural
   constraint placed before the likelihood is learned around rather than obeyed.

## 2. The forward model, and how we interrogate it

The data are Roman-like `lenstronomy` [2] simulations at 0.10593″ per pixel on a 127² grid: an
elliptical-power-law-plus-shear deflector that contributes no light of its own, an elliptical Sérsic
source, and a subhalo population. Substructure here is not a small perturbation. Inside $r < 30$ px
the r.m.s. difference between images rendered with and without subhaloes is 4.33 against a background
σ of 0.032, so a smooth model cannot reach χ²/dof ≈ 1 against these data and should not be tuned
until it does.

The deflection is $\boldsymbol\beta = \boldsymbol\theta - \boldsymbol\alpha(\boldsymbol\theta)$ with
six parameters. We need it differentiable and we need it in torch, which rules out evaluating the
Tessore and Metcalf [1] closed form through `scipy.special.hyp2f1`, so we evaluate the underlying
power series instead. Because $|w| = (1-q)/(1+q) < 1$ for every physical axis ratio the series
converges geometrically, and the term count can be chosen per image from the target tolerance: thirty
terms for the most elliptical lens in the dataset, eight for a near-circular one. The result agrees
with `lenstronomy` to $2.7\times10^{-15}$ and the source profile agrees exactly, because we reproduce
that package's own $b(n)$ rather than a more accurate expansion. The goal is to invert the simulator,
not to out-model it.

Ray-shooting is followed by convolution on the sky and then area-averaging onto detector pixels, in
that order, since the optics act before the detector integrates. A free additive background absorbs
the sky level. The objective is
$\chi^2 = \sum_i \left((\text{model}_i - d_i)/\sigma\right)^2$ over a 45-pixel disc, which is 6,361
pixels, with σ measured per image from a source-free annulus at $r > 50$ px rather than taken from a
header.

Nothing in the loss touches a label, a high-resolution target or a simulation parameter. Where a
starting value is needed we measure it from the image, fitting harmonics to the radius of peak
brightness along 72 azimuths. The $m=0$ term estimates the Einstein radius to a median 0.48 px on a
ring of median radius 12.5 px, and the $m=1$ term estimates the source offset at Spearman +0.875,
which matters because the brightest annulus sits at $\theta_E + \beta$ rather than at $\theta_E$.

Why the lens is fitted rather than fixed is worth one measurement. Holding the true source fixed and
varying only the deflection model, arc-weighted correlation with the truth runs 0.869 for a circular
isothermal sphere, 0.888 if its Einstein radius is refitted per image, 0.973 for an ellipsoid and
0.997 once external shear is added. Refitting the radius buys 0.02; adding ellipticity buys 0.10. The
failure is angular rather than radial, and that is the part an azimuthally averaged metric cannot
see: a circular lens recovers the mean critical radius to 0.7 per cent while omitting a peak-to-peak
azimuthal swing of 5.4 px on a 12 px ring, against a response-limited arc width near 2 px.

We score against source-plane truth, with the model source convolved by the instrument response first
because the stored truth is post-convolution, and against the seven physical parameters. Null
baselines are reported beside every χ²/dof so that the absolute number means something: on these data
a constant image scores 37,511, a two-pixel blur of the observation 2,796, and a three-pixel blur
4,091.

> **Figure 1.** The two inference paths. Observation → (a) per-image bounded nonlinear least squares
> over fourteen parameters, or (b) back-projection through a frozen fitted lens into a fully
> convolutional decoder. Both re-lens, convolve and bin, and close χ² against the same observed
> pixels. No high-resolution target enters anywhere. *[To draw; component panels exist in
> `figS5_sersic_stages.png`.]*

## 3. Identifiability: which source the data can support

Table 1 holds the data, the deflection model, the instrument model and the scorer fixed and varies
only how the source is represented. Where a lens is frozen it is the same fitted lens.

> **Table 1.** Six source representations under one forward model, scored on the same 800 validation
> images against the stored truth, convolution-matched. "Data per value" uses the 4,312 pixels above
> 3σ as the informative count. Row 1 is a free-field baseline previously run on this dataset with the
> lens fixed to a circular isothermal sphere; its size ratio is the range across a hundredfold sweep
> of the total-variation weight.
>
> | source representation | free source values | data per value | corr | size ratio | nmse |
> |---|---|---|---|---|---|
> | free field, 254², lens fixed and circular | 64,516 | 0.07 | – | 10.6 – 12.8 | – |
> | free field, 64², linear inversion on the fitted lens | 4,096 | 1.05 | 0.958 | 1.064 | 0.082 |
> | free field, 62², convolutional decoder on the back-projection | 3,844 | 1.12 | 0.973 | 1.130 | 0.0527 |
> | Sérsic + 48² residual, same decoder | 7 + 2,304 | 1.87 | 0.979 | 1.043 | **0.0408** |
> | Sérsic + 32² bounded correction from a 132-vector | 7 + 1,024 | 4.18 | 0.959 | 0.937 | 0.0799 |
> | Sérsic only, 7 values, per-image fit | 7 | 616 | **0.987** | 1.081 | **0.0252** |

Read the last row first, because it decides how to read the rest. The sources in this dataset are
genuinely Sérsic profiles, so the seven-parameter fit has exactly the right functional form and its
0.0252 is an oracle-family ceiling rather than a competitor. Nothing free-form should be expected to
beat it here. The question the table answers is which free-form representation gets closest, and that
question is the one that transfers to real galaxies, where no analytic family applies.

Conditioning explains the two failures at the top. The 254² field is underdetermined fifteen to one
and inflates the reconstructed source by a factor of ten to thirteen at every regularisation strength
tried across a hundredfold sweep. That is not a tuning failure. The unconstrained minimiser of total
variation is a flat field, so raising the weight moves toward the wash rather than away from it, and
the most compact result in the sweep was the least regularised one. The mechanism is worth naming
because it is general: when the forward model cannot place flux at the right azimuth whatever source
is proposed, the best available strategy under a pixel loss is to spread the source until the lensed
ring is fat enough to overlap the true arc everywhere. Underdetermination and misspecification
compound, and the visible symptom belongs to the source while the cause belongs to the lens. One row
down, the 64² linear inversion [3] on a correct frozen lens is close to critically determined and
produces speckle, at a median ray coverage of 1.44 rays per source pixel.

Among the representations that do not fail, conditioning predicts very little. The two hybrid rows
are both overdetermined, at 4.18 and 1.87 data per free value, and they differ by a factor of two in
source error, with the better-conditioned one worse. What separates them is where spatial information
is allowed to go. The 32² correction is decoded from a 132-dimensional global summary produced by an
average pool, and 64 per cent of that model's 849,519 weights sit in the single dense layer that then
has to regenerate a spatial map from it; measurably, the correction it produces never exceeds 3 to 4
per cent of the Sérsic amplitude. The 48² residual comes from a fully convolutional decoder acting on
the observation already back-projected into the source plane, so it starts in spatial register and
only has to deblur and sharpen. It carries fewer weights, about 0.63 M against 0.85 M, so capacity is
not the explanation. We should be careful about how much this isolates: the two runs also differ in
box half-extent (1.2″ against 0.8″) and therefore in unknown count, so the attribution to the decoder
rests on the mechanism and on the measured saturation of the smaller correction rather than on a
single-variable swap.

A boxed free-form source has to trade coverage against resolution, and a hybrid does not. At the
median, 28 per cent of the true source flux falls outside ±0.8″, and a box wide enough to hold the
wings at ±2.5″ with the same 0.05″ sampling would need about 9,600 unknowns against 4,300 informative
data, which is the regime of the first row. Keeping the wings in seven parameters removes the trade:
the Sérsic-based run at half-extent 1.2″ shows no truncation artefact, while the free-form run at the
larger 1.6″ still clips the lensed model to a rounded square.

The most useful row comparison is between the two free-form-decoder entries. The free-form run had
more training images (8,000 against 6,000), more epochs (50 against 30) and a wider box, and it
reached the lower validation χ² of the two on the objective both were trained on, 329 against 343.
Its source is the worse one, at nmse 0.0527 against 0.0408 and peak ratio 0.748 against 0.927. Every
one of those training advantages should have helped recovery as well as fit, and none of them did.
Ill-posedness is usually asserted in this literature. Here it is a measurement with a direction:
selecting a representation on held-out data fidelity would have selected the wrong one.

## 4. Residual misspecification, diagnosed and absorbed

Two things are still wrong with the forward model after the lens family is fixed, and only one of
them can be repaired.

The instrument response is the repairable one. Every prior pipeline on this dataset assumed a 0.10″
Gaussian, a number that was never measured. It can be, because the stored unlensed array is the
analytic source already convolved with the true kernel, which over-determines the kernel. A
parametric sweep gives residual nmse of 0.00707 at 0.10″ against a clean minimum of 0.00288 at 0.20″,
so the assumed width was a factor of two too narrow and 2.4 times worse in residual. A
Wiener-regularised Fourier extraction, stacked over 40 sources of differing size and assuming nothing
about the shape, then shows why the sweep does not bottom out at zero: at $r = 0.25''$ the empirical
profile sits at 0.139 of its peak against 0.005 for the best-fit Gaussian, a factor of 29. The kernel
is not Gaussian, and swapping in the empirical one is the repair.

What remains is a shape error at the one to three per cent level, and it is not negligible. Arcs here
reach peak-to-background ratios of $10^3$ to $10^4$, so a one per cent error in kernel shape is a
50σ per-pixel residual. This is the entire reason χ²/dof settles near 3,200 in a fit that recovers the
Einstein radius to 0.047″, and the diagnostic that establishes it is the direction of the trend rather
than its size. Noise-limited residuals fall with brightness; systematics grow with it. Stratified by
the dataset's own signal-to-noise variable, median χ²/dof runs 1,321, then 2,706, then 7,140 across
the three bins. The recovered source size ratio drifts the same way, 1.050 to 1.076 to 1.119, and the
three parameter biases all point one direction, with the Einstein radius high by 3.5 per cent, the
half-light radius by 10 per cent and the Sérsic index by 17 per cent, exactly as a too-narrow wing
would force.

Since the remaining error scales with model flux rather than with background, the honest response is
to say so in the likelihood. We use
$$\sigma_{\rm eff}^2 = \sigma_{\rm bg}^2 + (f\cdot\text{model})^2, \qquad f = 0.02,$$
an explicit multiplicative discrepancy term. Its effect is measured, not assumed: across the
signal-to-noise range the per-image χ² goes from 1,411 to 8,455 without it and from 329 to 229 with
it, and the p90 to p10 spread across images falls from 95-fold to 8.7-fold. That matters
operationally rather than cosmetically. Without the term, four images in every four hundred carry
half of the total gradient and the effective batch size for a network trained on this objective
collapses to about two, which is enough to make a regularisation ablation return two numerically
identical models.

We do not claim the discrepancy term is the right model of the error. It is the standard first-order
one, its scale was set by the measured wing mismatch rather than tuned, and what we report is that it
converts a systematic that dominated the gradient into one that does not.

## 5. Where the model is right, and what that buys

A forward model can be interrogated for correctness on quantities it was never fitted to. The
magnification field is a good choice here because it depends on the full lens Jacobian rather than on
the Einstein radius alone. From the fitted lens on 2,000 images, the per-pixel log-magnification map
over the arc annulus matches the truth at median correlation 0.929, total magnification comes out at
5.996 against 5.941, and the tangential critical curve is recovered to an r.m.s. radial error of 0.63
px. The discriminating number is that curve's azimuthal swing, 0.540″ recovered against 0.548″ true,
because a circular lens gives exactly zero swing and no amount of fitting the arc brightness produces
it. Since magnification is a derived function of the six lens parameters, this is a consistency check
rather than an independent measurement, and we label it as one.

The same field sets a physical ceiling on resolution. Median tangential stretch is 3.075× against a
radial 0.994×, so the data support a source grid roughly three times finer than the detector and no
more; both decoders in Table 1 ask for about 2.05×, inside that ceiling.

Whether the extra resolution is real is a separate question from whether it is licensed, and it can
be measured. We add a Gaussian clump of FWHM 0.08″, which is 0.76 detector pixels, at 15 per cent of
the source peak; re-render the whole system through the lens, the instrument response and the
detector; and add noise at that image's measured σ. The identical system without the clump is
reconstructed under the same noise realisation, and we measure the difference in a small aperture at
the clump position. 2,400 injections across 60 images, on a trained checkpoint, with no retraining
anywhere.

Structure below the detector pixel scale does come back, at 17 to 22 per cent of the injected contrast
depending on local magnification. Recovery is better where the lens magnifies more: mean contrast
0.208 for $\mu \geq 2$ against 0.175 for $\mu < 2$, Mann-Whitney $p = 5.3\times10^{-5}$. Fractional
reconstruction error over the whole source falls from 0.39 in the $\mu = 1$ to 2 bin to 0.038 in the
$\mu = 16$ to 32 bin, a power-law slope of $-0.78$ across the five populated bins.

Magnification is nonetheless not what limits the reconstruction. Rank correlation of recovered
contrast is +0.056 with local magnification and +0.152 with signal-to-noise, and stratifying by the
latter, the $\mu \geq 2$ advantage runs 1.25, then 1.22, then 1.08 as photons become plentiful.
Magnification decides whether sub-pixel structure survives when photons are scarce, and this dataset
is photon-rich almost everywhere.

> **Figure 2.** Recovered contrast against local magnification, 2,400 injections, with the
> signal-to-noise stratification showing the effect closing from 1.25× to 1.08×.
> *[Source: `figMU1_resolution_vs_mu.png`, to regenerate at publication size.]*

## 6. What the objective does to what you attach to it

That measurement retro-explains three separate attempts to make the reconstruction
magnification-aware, none of which paid.

> **Table 2.** Three implementations of magnification-adaptive regularisation. Source nmse, medians,
> same 800 validation images; rows 2 and 3 are matched ablations against a control identical in every
> respect but the magnification term.
>
> | mechanism | μ-adaptive | uniform |
> |---|---|---|
> | regularisation weight, 64² linear inversion | ~4% worse at matched data fidelity | – |
> | correction penalty, 32² decoder | 0.0799 | 0.0789 |
> | structural resolution gate, 48² residual decoder | 0.0414 | 0.0408 |

The third is the interesting one, because it was designed to be immune to the objection that a
penalty can be traded away. The gate is a sigmoid in $\log\mu$ applied to the decoder's output before
the forward model, so that sub-pixel detail is structurally removed wherever the lens delivered no
resolution. It sits before the likelihood and the network is trained through it, and the network
learns its inverse. Measured on twelve images, the gated model emits 1.61 times the sub-pixel variance
of the control before the gate; the gate attenuates by 0.624; the product is 1.005. The two models
finish with the same sub-pixel content to within 0.3 per cent. Where the gate does survive is where
χ² is indifferent to it, which is the faint outskirts, and there it buys 14 to 18 per cent less
speckle below 5 per cent of peak surface brightness while being invisible to every flux-weighted
metric. A separate diagnostic makes the same point from the other side: the penalty term changed by a
factor of 35 between control and gate while validation χ² moved from 343 to 344.

A differentiable constraint placed before the likelihood is not a constraint if the network is trained
through it. That statement has nothing to do with lensing.

The objective shapes the estimator in a second way, and here the fix is practical. One forward pass of
the amortised decoder costs 12 ms against 815 ms for the per-image fit, and it matches the fit on the
Einstein radius (Spearman +0.960) and the half-light radius (+0.953). It fails on ellipticity, at
+0.269 against +0.762, and the failure is not orientation: the median phase error is 19° and the
modulus is shrunk by a flat factor near 0.22 in every phase-error bin, including the bin where
orientation is right to within 5°. A network trained on a squared loss is a conditional-mean
estimator, and for a spin-2 quantity under orientation uncertainty a large ellipticity pointed the
wrong way costs more than none at all, so the expected-error-minimising answer shrinks toward zero.
More epochs cannot fix that.

So we stop asking the network to be the estimator and use it to start one. On 800 common validation
images with identical bounds, tolerances and residual function, the cold fit takes 480 model
evaluations and 0.815 s across four staged parameter releases and reaches χ²/dof 3,302; started from
the network it takes 151 evaluations and 0.460 s in a single stage and reaches 3,228. Recovery
improves on six of seven parameters and the shrinkage is undone entirely, from +0.269 through +0.776
to +0.839. The staged schedule exists only because a neutral start falls into poor minima, and from
the network's start it is unnecessary.

> **Figure 3.** Cost against accuracy for the network, the cold fit and the warm fit, with the
> ellipticity scatter inset. *[Source: `figR1_cost.png`, `figR3_ellipticity.png`.]*

## 7. Limitations

Every number here comes from one dark-matter class. The dataset loader sorted a concatenated list of
per-class paths before truncating, so all runs drew from the axion class; training and validation
remain disjoint splits and this class carries the heaviest subhalo population of the three, so the
smooth-model results are conservative, but cross-class generalisation is untested.

We report point estimates and no posterior anywhere, which is the largest gap and the one most
directly indicated by our own results: the flat 0.22 shrinkage of Section 6 is a bias we can measure
and explain, and a density head or normalising flow over the fourteen parameters would report it as a
width instead of committing it as an error. That is the next thing we would build.

The remaining caveats are shorter. Magnification is derived from the fitted lens rather than measured.
The injected clump has a fixed amplitude wherever it lands, so it is a larger local perturbation far
from the source centre, which shows up as Spearman +0.40 between radius and recovered contrast. Table
1's last row is an oracle-family ceiling specific to a dataset whose sources are Sérsic profiles by
construction. Everything is simulated, one of the two available bands is used, and the roughly one per
cent differences in Table 2 rest on a single seed per training run.

---

## References

[1] N. Tessore and R. B. Metcalf. The elliptical power law profile lens. *Astronomy & Astrophysics*, 580:A79, 2015.

[2] S. Birrer and A. Amara. lenstronomy: multi-purpose gravitational lens modelling software package. *Physics of the Dark Universe*, 22:189, 2018.

[3] S. Warren and S. Dye. Semilinear gravitational lens inversion. *The Astrophysical Journal*, 590:673, 2003.

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

*Still to add: the Roman, Euclid and LSST mission references you prefer, a citation for the simulation dataset, and one or two model-misspecification references from outside astronomy so the framing is visibly connected to the workshop's literature. Candidates worth checking: work on simulation-based inference under misspecification, and on discrepancy or model-error terms in likelihoods.*

---

# Appendix

## A. Full parameter recovery

Per-image bounded nonlinear least squares (trust-region reflective, not plain Levenberg-Marquardt,
since several parameters carry hard bounds), warm-started from the amortised decoder, $n = 800$.
Nothing in this table was available to the fit.

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
nmse 0.0252. χ²/dof median 3,227.9.

## B. Instrument response calibration

Parametric Gaussian sweep, residual nmse against the stored convolved source, 25 compact sources:
0.01062, 0.00707, 0.00435, 0.00300, 0.00288, 0.00312, 0.00855 at FWHM 0.00, 0.10, 0.14, 0.18, 0.20,
0.22 and 0.30″. Wiener-regularised extraction stacked over 40 sources gives the wing measurement
quoted in Section 4. The generator itself was verified rather than assumed: an analytic
power-law-plus-shear convergence matches the stored map at log-correlation 0.936, and an analytic
elliptical Sérsic matches the stored source at correlation 0.967, rising to 0.9989 with the response
included.

## C. The discrepancy term, in full

Per-image χ² percentiles with and without the σ-floor, the gradient-share calculation showing 54.1
per cent of a batch of 16 taken by its largest member against 6.2 per cent for uniform, and the
resulting collapse of an ablation into two numerically identical models
(Pearson +0.9992 on $\theta_E$ between the two runs).

## D. Resolution is not a rendering choice

The parametric source is continuous, so it can be sampled on any grid. Scored against the true source
rendered on the same grid, $n = 2{,}000$:

| factor | scale (″/px) | corr | nmse | size ratio |
|---|---|---|---|---|
| 1× | 0.1059 | 0.9787 | 0.0420 | 0.979 |
| 2× | 0.0530 | 0.9773 | 0.0446 | 0.955 |
| 4× | 0.0265 | 0.9770 | 0.0452 | 0.942 |
| 8× | 0.0132 | 0.9769 | 0.0453 | 0.933 |

Correlation moves by 0.0017 across an eightfold range, which is the point: the sampling does not limit
the reconstruction, the parameters do.

## E. Validation ladder

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
response to land on the corresponding output pixel, and one requires no non-finite gradient on step
one. The second earns its keep repeatedly, because the ellipticity parameterisation is singular at
$e_1 = e_2 = 0$ under reverse-mode autodiff and that is a legitimate place for the optimum to sit
rather than merely a place to pass through.

## F to I. Outlined

- **F. Box size as a measured choice.** Median flux inside the box against half-extent: 0.715 at
  0.8″, 0.891 at 1.2″, 0.958 at 1.6″, with the corresponding data-per-unknown ratios.
- **G. The three nulls in detail**, including the pre-compensation measurement and the speckle
  statistics the gate does buy.
- **H. Spin-2 shrinkage.** Modulus ratio by phase-error bin: 0.21, 0.23, 0.23, 0.22 across 0 to 5°,
  5 to 15°, 15 to 30° and 30 to 90°, with counts 120, 221, 200, 259.
- **I. Injection details and reproduction.** Aperture definition, per-bin counts (253, 1215, 572, 295,
  65), full signal-to-noise stratification, anonymised repository, exact commands, χ²/dof convention.

---

## Notes for the author (delete before submission)

**Two numbers are not yet in scope, and one of them is load-bearing.**

1. **The lens-model ladder in Section 2** (0.869 / 0.888 / 0.973 / 0.997, plus the 5.4 px swing) comes
   from the earlier diagnostics work, not from `superres/`. It is the cleanest misspecification
   measurement in the paper and it justifies six lens parameters in one paragraph. Re-run it inside
   `superres/`: render the true Sérsic through each candidate deflection model at 3× supersample,
   convolve, bin, no noise, reference the same source through the true lens, arc-weighted correlation,
   $n = 60$, split by lens $|e|$. Everything needed is in `raytrace.py`, `lens_models.py` and
   `sources.py`. Roughly 50 lines, no GPU, about two hours. **If you cannot, delete the numbers and
   write the paragraph as motivation rather than measurement.**
2. **The response sweep in Section 4 and Appendix B** was produced by the earlier
   `calibrate_psf.py`. `superres/calibrate_psf.py` is effectively the same script, so just re-run it
   (three minutes) and quote the fresh output.

**Three things to fix.**

- **χ²/dof convention.** `evaluate.py` divides by $6361 - 14$; `eval_pathb.py` divides by
  $6361 - 14 - 1024$, a factor 1.192. Confirm what `eval_b4.py` uses, then state the convention in a
  footnote. I have deliberately kept raw χ²/dof comparisons between the parametric fit and the
  decoders out of the main text and used the floored validation χ² (329 against 343) instead, which
  both decoders were trained on.
- **Three nulls, not four.** `B5_RESULTS.md` lists a B4 `--reg-mode mu` versus `uniform` ablation
  quoting 0.0799 against 0.0789, but those are B3's numbers and there is no `b4_uniform` metrics file.
  Table 2 reports three.
- **Anonymity.** Nothing above names a person, institution or programme, and [11] is third person.
  Row 1 of Table 1 is described as "previously run on this dataset" without claiming or disclaiming
  authorship, which is the right handling under double-blind. Keep it.

**Length.** Sections 1 to 7 run **about 3,560 words** including the abstract, with three figures and
two tables. NeurIPS style at 10pt over a 5.5in measure holds roughly 800 words to a full text page, so
five pages minus the title block, three figures at about 1.8in each and two tables leaves room for
**roughly 3,050**. Budget on cutting 400 to 500 words once it is typeset and the page break is
visible. Named cuts, in order, totalling about 520 words:

1. The final paragraph of Section 4, the "we do not claim the discrepancy term is the right model"
   hedge (about 60 words). Compress to one clause inside the paragraph above it.
2. The coverage-versus-resolution paragraph in Section 3 (about 95 words), down to its first and last
   sentences. Appendix F carries the arithmetic.
3. The resolution-ceiling sentences in Section 5 (about 60 words). Appendix D covers it.
4. The hypergeometric-series detail in Section 2 (about 70 words), down to "we evaluate the series
   form because the closed form is neither differentiable nor available in torch, and it agrees with
   `lenstronomy` to 2.7e−15". Appendix E has the rest.
5. The ring-initialiser sentences in Section 2 (about 65 words). One sentence saying no manifest value
   is used and the starting point is measured from the ring is enough; Appendix can carry the numbers.
6. Figure 3, replaced by the two numbers it carries (about 170 word-equivalents of space). Section 6
   already states them.

**Do not cut** Table 1, Figure 2, the "read the last row first" paragraph in Section 3, or the σ-floor
measurement in Section 4. Those are the paper.

**What changed from the R/PS draft.** The spine flipped. Identifiability now leads and
misspecification is the second pillar, because that ordering is what the Sim2Science call asks for and
because the identifiability material is entirely inside `superres/`. Section 4 is new and is the
biggest single addition: the σ-floor was a parenthetical in the R/PS version and is a result here.
Section 6 merges the amortised-initialiser result with the gate result under one idea, which is what
a physics-only objective does to estimators and constraints attached to it. Dropped for scope: the
image-plane skill study and the dataset phase test.

**One thing worth doing if a day appears.** Two extra seeds on the B4 + Sérsic control and the B5
gate. The Table 2 differences are 0.6 to 1.3 per cent and currently rest on one seed each, and this
audience will ask.

**Style.** I have kept the bold run-in headers to a minimum and varied paragraph length deliberately,
since you flagged both. The Introduction and Section 7 are the two places most worth rewriting in your
own hand before submission.
