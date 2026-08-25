# `SourceForm`: Label-Free Lens and Source Inversion for Strong Gravitational Lensing

> **Notes before you paste this into LaTeX.**
>
> 1. **Restyled to match `FlowLensing` (ML4PS 2025) and Shankar et al. (ML4PS 2024).** Plain noun
>    section headings, four-paragraph introduction with a textbook opening, contribution bullets
>    converted to prose, bold run-in labels in Methods and Results, method name in `\texttt{}`
>    throughout, deliberate repetition across abstract, introduction and conclusion. The metaphor
>    opening is gone.
> 2. **Spelling switched to American** to match both reference papers and the NeurIPS default. Your
>    own documents are British, so if you would rather stay British it is a find-and-replace on
>    "modeled", "normalized", "regularization", "analyze", "neighboring".
> 3. **`SourceForm` is a placeholder name.** It came back clean on arXiv, GitHub, ASCL and ADS.
>    Replace it everywhere or drop it, but the house style of both reference papers wants a named
>    method, so decide before submission.
> 4. **No Acknowledgements section.** Both reference papers have one naming GSoC and funding. R/PS is
>    double-blind, so it must wait for camera-ready.
> 5. Scope is `superres/` only. One number is still out of scope; see the end.

---

## Abstract

Strong gravitational lensing is among the most direct probes of dark matter substructure, yet the
high-resolution imaging required to exploit it is scarce, and upcoming surveys will widen rather
than close that gap. Supervised super-resolution depends on paired low- and high-resolution images
that will not exist for the population those surveys find. We present `SourceForm`, a label-free
system that fits a lens and a source jointly to each observation using the forward model as its only
supervision: an analytic, differentiable elliptical-power-law-plus-shear deflection field, a measured
instrument response and detector binning close a χ² directly against the observed pixels, with no
high-resolution target at any stage. On Roman-like simulations with full generative ground truth we
recover seven physical parameters from the pixels alone (Spearman +0.54 to +0.96), reduce the
reconstructed-source size error from 10.6× to 1.08× against a free-field baseline on the same data,
and reproduce magnification quantities that never entered the fit. Comparing five source
representations under one forward model, we find that conditioning predicts which representations
fail while decoder architecture accounts for a further factor of two among those that do not.
Injecting structure smaller than a detector pixel, we recover 17 to 22 per cent of it, significantly
more where the lens magnifies more, although photon count predicts recovery three times more
strongly. All results are on simulated data from a single instrument.

---

## 1 Introduction

Strong gravitational lensing occurs when the gravity of a foreground mass bends light from a
background galaxy, producing arcs, rings and multiple images [1]. Because the deflection depends on
the total projected mass rather than on the light, lensed images carry information about dark matter
that is difficult to obtain by other means, in particular about the small-scale substructure that
distinguishes competing dark matter models [2, 3]. Lensing also magnifies: the lens spreads a small
patch of the background galaxy over many detector pixels, so a lensed observation samples that
galaxy more finely than the same instrument would sample it unlensed.

Recovering the source from such an image requires inverting a mapping that is many-to-one by
construction, which is what produces the arcs in the first place. The standard machine learning
approach is super-resolution trained on paired low- and high-resolution images, but those pairs do
not exist at the scale required. Euclid, the Legacy Survey of Space and Time and the Roman Space
Telescope are expected to discover of order 10⁵ galaxy-galaxy lenses [4], while higher-resolution
follow-up will remain available for only a small fraction of them. Supervised methods therefore
cannot be trained on the population they are intended to analyze.

An alternative is to use the forward model itself as the supervision. A candidate lens and source
are proposed, ray-traced, convolved with the instrument response and integrated onto detector
pixels, and the prediction is compared with the observed image, so no high-resolution target is
required at any stage [5]. This removes the need for labels, but it does not settle what the source
should be. The deflection field is a small and identifiable parametric family, whereas published
choices for the source range from a handful of analytic parameters [6, 7] to tens of thousands of
free pixels [8, 9], and the choice is usually made by convention rather than by measurement.

In this work we present `SourceForm`, a label-free system that fits a lens and a source jointly to
each observation, and we use it to measure that choice: holding the data, the deflection model, the
instrument model and the scoring function fixed, we vary only the source representation. We then test
whether the recovered sub-pixel detail is genuine by injecting structure smaller than a detector
pixel and measuring what returns. By removing the dependence on paired training data while retaining
an interpretable lens model, this approach offers a route to source reconstruction for the lens
populations Euclid, LSST and Roman will deliver.

## 2 Data and methods

### 2.1 Dataset

We use simulated galaxy-galaxy strong lensing images generated with `lenstronomy` [10] to resemble
observations from the Roman Space Telescope. Images are 127 × 127 pixels at 0.10593 arcsec per pixel.
Each system consists of an elliptical-power-law deflector with external shear, which contributes no
light of its own, an elliptical Sérsic source [6], and a population of dark matter subhalos. Every
image stores the noiseless unlensed source alongside the observation, together with the generating
parameters, so both the source plane and the physical parameters can be scored against truth.

We verified the generator rather than assuming it: an analytic power-law-plus-shear convergence built
from the stored parameters matches the stored map at log-correlation 0.936, and an analytic
elliptical Sérsic matches the stored source at correlation 0.9989 once the instrument response is
included. Substructure is not a small perturbation here, so a smooth model cannot reach χ²/dof of 1
against these data by construction.

### 2.2 Forward model

The deflection is **β** = **θ** − **α**(**θ**) with six parameters: Einstein radius θ_E, radial slope
γ, two ellipticity components e₁ and e₂, and two external shear components. We require the deflection
to be differentiable and available in PyTorch, which rules out the Tessore and Metcalf [11] closed
form through `scipy.special.hyp2f1`, so we evaluate the underlying power series, which converges
geometrically for every physical axis ratio and agrees with `lenstronomy` to 2.7 × 10⁻¹⁵. Rays are
then shot, the sky is convolved with the instrument response, and the result is area-averaged onto
detector pixels, in that order, because the optics act before the detector integrates.

**Instrument response.** The response is measured rather than assumed, because the stored unlensed
array is the analytic source already convolved with the true kernel and therefore over-determines it.
A parametric sweep gives residual nmse 0.00707 at 0.10 arcsec, the value used by prior work on these
data, against a minimum of 0.00288 at 0.20 arcsec. A Wiener-regularized extraction stacked over 40
sources places the profile at 0.139 of its peak at 0.25 arcsec against 0.005 for the best-fit
Gaussian, a factor of 29. We use the extracted kernel throughout.

### 2.3 Objective and initialization

The objective is χ² = Σᵢ ((modelᵢ − dᵢ)/σ)² over a disk of radius 45 pixels, which contains 6,361
pixels, with σ estimated per image from a source-free annulus beyond a radius of 50 pixels. The
amortized runs replace σ with σ²_eff = σ²_bg + (0.02 · model)², an explicit multiplicative term that
admits the measured kernel-shape systematic. Without it, four images in every four hundred carry half
of the total batch gradient.

No labels, no high-resolution target and no simulation parameters enter the objective. Where a
starting value is needed we measure it from the image, fitting harmonics to the radius of peak
brightness along 72 azimuths; the m = 0 term estimates the Einstein radius to a median 0.48 pixels on
a ring of median radius 12.5 pixels.

**Why six lens parameters.** Holding the true source fixed and varying only the deflection model,
arc-weighted correlation with the truth runs 0.869 for a circular isothermal sphere, 0.973 for an
ellipsoid and 0.997 once external shear is added. The failure of the circular model is angular rather
than radial: it recovers the mean critical radius to 0.7 per cent while omitting a peak-to-peak
azimuthal swing of 5.4 pixels on a 12 pixel ring, against an arc width of about 2 pixels.

### 2.4 Source representations

The lens model is settled by the measurement above. The source is not, and we treat it as the open
design choice it is. Five representations are compared under the identical forward model, optimizer
and scorer, and are listed in Table 1. Two are free pixel grids, one at twice the detector sampling
with a total-variation prior and one produced by a fully convolutional decoder acting on the
observation back-projected into the source plane. Two are hybrids in which a fitted Sérsic carries
the wings and a network predicts only a bounded residual, differing in whether that residual is
decoded convolutionally or from a globally pooled vector. The fifth is the Sérsic alone, fit per
image by bounded nonlinear least squares.

The convolutional decoder follows an SRResNet and sub-pixel upsampling design [12, 13], with
GroupNorm in place of BatchNorm because arc brightness varies by a factor of 260 between images, and
softplus in place of ReLU so that a pixel starting negative can recover.

### 2.5 Evaluation

We score against source-plane truth, with the model source convolved by the instrument response
before comparison because the stored truth is post-convolution, and against the seven generating
physical parameters. Null baselines accompany every χ²/dof so that the absolute value has meaning: on
these data a constant image scores 37,511, a two-pixel blur of the observation 2,796 and a three-pixel
blur 4,091.

> **Figure 1.** Schematic of `SourceForm`. An observation enters, a lens and a source leave. Both
> inference paths, per-image bounded least squares and the amortized decoder, re-lens, convolve and
> bin, and close χ² against the same observed pixels. *[To draw; component panels exist in
> `figS5_sersic_stages.png`.]*

## 3 Results

### 3.1 Parameter and source recovery

On 800 validation images the seven physical parameters are recovered at Spearman +0.960 (θ_E), +0.954
(half-light radius), +0.945 (Sérsic index), +0.915 (source offset), +0.839 (lens ellipticity), +0.650
(external shear) and +0.538 (radial slope). None of these quantities was available to the fit. The
reconstructed source matches the stored truth at correlation 0.987 with a size ratio of 1.081, against
10.6 to 12.8 for the free-field baseline previously run on these data. Inference costs 12 ms for the
amortized decoder and 0.46 s for the refined per-image fit.

**Validation on quantities never fit.** The magnification field derived from the recovered lens
matches the truth at median log-correlation 0.929 over the arc annulus, total magnification is
recovered as 5.996 against a true 5.941, and the tangential critical curve is recovered to a radial
root mean square error of 0.63 pixels. Its azimuthal swing is 0.540 arcsec against a true 0.548
arcsec. A circular lens produces a swing of exactly zero, so recovering it indicates that the four
shape parameters were recovered jointly rather than in magnitude alone.

**Goodness of fit.** χ²/dof settles near 3,200 rather than 1. The model does not include substructure
while the data contain it, and the instrument response retains a shape error at the one to three per
cent level, which on arcs reaching peak-to-background ratios of 10³ to 10⁴ is a 50σ per-pixel
residual. What distinguishes these from an underestimated noise level is that the residual grows with
brightness: median χ²/dof runs 1,321, 2,706 and 7,140 across three signal-to-noise bins.

### 3.2 Which source representation the data support

| source representation | free source values | data per value | corr | size ratio | nmse |
|---|---|---|---|---|---|
| free field, 254², lens fixed and circular | 64,516 | 0.07 | – | 10.6 – 12.8 | – |
| free field, 62², convolutional decoder | 3,844 | 1.12 | 0.973 | 1.130 | 0.0527 |
| Sérsic + 48² residual, same decoder | 7 + 2,304 | 1.87 | 0.979 | 1.043 | **0.0408** |
| Sérsic + 32² correction from a 132-vector | 7 + 1,024 | 4.18 | 0.959 | 0.937 | 0.0799 |
| Sérsic only, 7 values | 7 | 616 | **0.987** | 1.081 | **0.0252** |

**Table 1.** Five source representations under one forward model, scored on the same 800 validation
images against the stored truth, convolution-matched. "Data per value" uses the 4,312 pixels above
3σ. Row 1 is the baseline previously run on these data; its size ratio is the range across a
hundredfold sweep of the regularization weight.

The last row sets how the rest should be read. The sources in this dataset are Sérsic profiles by
construction, so seven analytic parameters constitute the exactly correct functional form and 0.0252
is a ceiling rather than a competitor. The useful question is which free-form representation
approaches it, since that is the question that transfers to real galaxies where no analytic family
applies.

**Conditioning explains the failure.** The 254² field is underdetermined by a factor of fifteen and
inflates the reconstructed source tenfold to thirteenfold at every regularization strength across a
hundredfold sweep. This is not a tuning failure: the unconstrained minimizer of total variation is a
flat field, so increasing the weight moves toward the wash, and the most compact result in the sweep
was the least regularized one. When the forward model cannot place flux at the correct azimuth
whatever source is proposed, the best strategy under a pixel loss is to spread the source until the
lensed ring overlaps the true arc everywhere.

**Conditioning does not explain the remainder.** The two hybrid rows are both overdetermined, at 4.18
and 1.87 data per free value, yet they differ by a factor of two in source error, with the
better-conditioned one performing worse. What separates them is where spatial information is
permitted to go. The 32² correction is decoded from a global summary produced by average pooling, and
64 per cent of that model's 849,519 weights sit in the single dense layer that must regenerate a
spatial map from it; the correction it produces never exceeds 3 to 4 per cent of the Sérsic
amplitude. The 48² residual comes from a fully convolutional decoder acting on data already in the
source plane, so it begins in spatial register and needs only to deblur and sharpen. It carries fewer
weights, 0.63 M against 0.85 M, so capacity is not the explanation, although the two runs also differ
in box size and the attribution therefore rests on the mechanism as well as on the measured
saturation.

**A boxed free-form source trades coverage against resolution.** At the median, 28 per cent of the
true source flux falls outside a half-extent of 0.8 arcsec, and a box wide enough for the wings at
the same sampling would need roughly 9,600 unknowns against 4,300 informative pixels, returning to
the regime of row 1. Retaining the wings in seven parameters removes the trade.

**Better data fidelity is not better recovery.** The free-form run used more training images, more
epochs and a wider box, and reached the lower validation χ² of the two on the objective both were
trained on, 329 against 343. Its source is the worse of the two, at nmse 0.0527 against 0.0408 and
peak ratio 0.748 against 0.927. Every one of those advantages should have improved recovery as well
as fidelity, and none did.

### 3.3 Is the recovered sub-pixel detail real?

The physical warrant for reconstructing on a grid finer than the detector is the magnification field.
Median tangential stretch is 3.075 against a radial 0.994, so the data support a source grid roughly
three times finer than the detector, and both decoders request about 2.05 times, inside that ceiling.

Warrant is not evidence, so we test it directly. We add a Gaussian clump of full width at half
maximum 0.08 arcsec, which is 0.76 detector pixels, at 15 per cent of the source peak, re-render the
system through the lens, the instrument response and the detector, and add noise at the measured σ of
that image. The identical system without the clump is reconstructed under the same noise realization,
so the noise cancels in the difference, and we measure what survives in a small aperture at the clump
position. This is repeated for 2,400 injections across 60 images, with no retraining.

Between 17 and 22 per cent of the injected contrast is recovered, depending on local magnification.
Recovery improves where the lens magnifies more, at mean contrast 0.208 for μ ≥ 2 against 0.175 for
μ < 2, with a Mann-Whitney p-value of 5.3 × 10⁻⁵. Fractional reconstruction error over the whole
source falls from 0.39 in the μ = 1 to 2 bin to 0.038 in the μ = 16 to 32 bin, a power-law slope of
−0.78 across the five populated bins.

Magnification is nevertheless not the binding constraint. The rank correlation of recovered contrast
is +0.056 with local magnification and +0.152 with signal-to-noise, and stratified by the latter the
μ ≥ 2 advantage runs 1.25, then 1.22, then 1.08 as photons become plentiful.

> **Figure 2.** Recovered contrast against local magnification for 2,400 injections, with the
> signal-to-noise stratification showing the effect closing from 1.25 to 1.08.
> *[Source: `figMU1_resolution_vs_mu.png`, to regenerate at publication size.]*

### 3.4 Magnification-adaptive regularization

The measurement above also explains two attempts to build magnification into the prior, neither of
which paid. Weighting the correction penalty of the 32² decoder by local magnification gives source
nmse 0.0799 against a uniform control's 0.0789, and a structural gate on the 48² decoder gives 0.0414
against 0.0408. The gate is the more informative failure. It applies a sigmoid in log μ to the decoder
output before the forward model, removing sub-pixel detail wherever the lens delivered no resolution,
but because it precedes the likelihood and the network is trained through it the network learns its
inverse: the gated model emits 1.61 times the sub-pixel variance of the control before the gate, the
gate attenuates by 0.624, and the product is 1.005. A differentiable constraint placed before the
likelihood is not a constraint if the network is trained through it. The gate survives only where χ²
is indifferent to it, reducing speckle by 14 to 18 per cent in the faint outskirts while remaining
invisible to every flux-weighted metric.

### 3.5 Amortized inference as an initializer

A single forward pass of the amortized decoder costs 12 ms against 815 ms for the per-image fit, and
matches the fit on Einstein radius (+0.960) and half-light radius (+0.953). It fails on ellipticity,
at +0.269 against +0.762, and not through misorientation: the median phase error is 19 degrees while
the modulus is shrunk by a flat factor near 0.22 in every phase-error bin, including the bin in which
orientation is correct to within 5 degrees. That is the signature of a conditional-mean estimator
acting on a spin-2 quantity, and further training cannot remove it.

We therefore use the network to initialize the fit rather than to replace it. On 800 validation
images with identical bounds, tolerances and residual function, the cold fit requires 480 model
evaluations and 0.815 s across four staged parameter releases and reaches χ²/dof 3,302, while the
network-initialized fit requires 151 evaluations and 0.460 s in a single stage and reaches 3,228.
Recovery improves on six of seven parameters and the shrinkage is removed entirely, from +0.269
through +0.776 to +0.839.

## 4 Limitations

All results reported here come from a single dark matter class. The dataset loader sorted a
concatenated list of per-class paths before truncating, so every run drew from the axion class;
training and validation remain disjoint splits and this class carries the heaviest subhalo population
of the three, so the smooth-model results are conservative, but generalization across classes is
untested. We report point estimates and no posterior, which is the largest methodological gap and the
one our own results indicate most directly: the flat 0.22 shrinkage of Section 3.5 is a bias we can
measure and explain, and a density head over the fourteen parameters would report it as a width
rather than commit it as an error. Magnification is derived from the recovered lens rather than
measured independently. The last row of Table 1 is a ceiling specific to a dataset whose sources are
Sérsic profiles by construction, and the system has not been demonstrated on a galaxy that is not
one. All data are simulated, from one instrument, using one of two available bands, and the
approximately one per cent differences in Table 2 rest on a single seed per training run.

## 5 Conclusion and future work

We presented `SourceForm`, a label-free system that fits a lens and a source jointly to each strong
lensing observation using an analytic differentiable forward model as its only supervision. On
Roman-like simulations it recovers seven physical parameters from the pixels alone at Spearman +0.54
to +0.96, reduces the reconstructed-source size error to 1.08 times the truth against 10.6 for a
free-field baseline, and reproduces the magnification field, total magnification and tangential
critical curve without those quantities entering the fit. Comparing five source representations under
one forward model, we find that conditioning determines which representations fail and that decoder
architecture accounts for a further factor of two among those that do not, and we report a case in
which the representation achieving the lower validation χ² recovers the source substantially worse.
Injecting structure below the detector pixel scale, we measure that 17 to 22 per cent of it returns,
significantly more where the lens magnifies more, although photon count predicts recovery three times
more strongly.

Three directions follow. A density head or normalizing flow over the fourteen parameters would convert
the measured conditional-mean bias into a reported uncertainty. Rendering sources from a real galaxy
catalog rather than from analytic profiles would test the model-family ceiling that Table 1's last row
represents. Finally, both Roman bands are present in every image and the lens is achromatic, so a
two-band fit would constrain a single lens with twice the data while permitting the source to differ.

---

## References

[1] M. Bartelmann. Gravitational lensing. *Classical and Quantum Gravity*, 27(23):233001, 2010.

[2] S. Vegetti and L. V. E. Koopmans. Bayesian strong gravitational-lens modelling on adaptive grids. *Monthly Notices of the Royal Astronomical Society*, 392:945, 2009.

[3] F. M. Heinze, G. Despali and R. S. Klessen. Not all subhaloes are created equal. *Monthly Notices of the Royal Astronomical Society*, 527(4):11996, 2023.

[4] T. E. Collett. The population of galaxy-galaxy strong lenses in forthcoming optical imaging surveys. *The Astrophysical Journal*, 811:20, 2015.

[5] A. Shankar, M. W. Toomey and S. Gleyzer. Unsupervised physics-informed super-resolution of strong lensing images for sparse datasets. *Machine Learning and the Physical Sciences workshop, NeurIPS*, 2024.

[6] V. Cardone. The lensing properties of the Sérsic model. *Astronomy and Astrophysics*, 415:11, 2003.

[7] Y. D. Hezaveh, L. Perreault Levasseur and P. J. Marshall. Fast automated analysis of strong gravitational lenses with convolutional neural networks. *Nature*, 548:555, 2017.

[8] W. R. Morningstar et al. Data-driven reconstruction of gravitationally lensed galaxies using recurrent inference machines. *The Astrophysical Journal*, 883:14, 2019.

[9] A. Adam, L. Perreault Levasseur, Y. Hezaveh and M. Welling. Pixelated reconstruction of foreground density and background surface brightness in gravitational lensing systems using recurrent inference machines. *The Astrophysical Journal*, 925:124, 2022.

[10] S. Birrer and A. Amara. lenstronomy: multi-purpose gravitational lens modelling software package. *Physics of the Dark Universe*, 22:189, 2018.

[11] N. Tessore and R. B. Metcalf. The elliptical power law profile lens. *Astronomy and Astrophysics*, 580:A79, 2015.

[12] C. Ledig et al. Photo-realistic single image super-resolution using a generative adversarial network. *CVPR*, 2017.

[13] W. Shi et al. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. *CVPR*, 2016.

[14] S. H. Suyu, P. J. Marshall, M. P. Hobson and R. D. Blandford. A Bayesian analysis of regularized source inversions in gravitational lensing. *MNRAS*, 371:983, 2006.

[15] K. Karchev, A. Coogan and C. Weniger. Strong-lensing source reconstruction with variationally optimized Gaussian processes. *MNRAS*, 512:661, 2022.

*Add the Roman, Euclid and LSST mission references you prefer, and a citation for the simulation dataset.*

---

## Appendix

### A Full parameter recovery

Per-image bounded nonlinear least squares (trust-region reflective, not plain Levenberg-Marquardt,
since several parameters carry hard bounds), initialized from the amortized decoder, n = 800.

| parameter | fit median | true median | median abs. err | p90 abs. err | Spearman |
|---|---|---|---|---|---|
| θ_E (arcsec) | 1.370 | 1.320 | 0.047 | 0.179 | +0.960 |
| β (arcsec) | 0.292 | 0.305 | 0.033 | 0.091 | +0.915 |
| R_sersic (arcsec) | 0.505 | 0.484 | 0.041 | 0.159 | +0.954 |
| n_sersic | 1.140 | 1.005 | 0.138 | 0.786 | +0.945 |
| slope γ | 2.003 | 2.058 | 0.078 | 0.247 | +0.538 |
| lens abs(e) | 0.200 | 0.222 | 0.032 | 0.098 | +0.839 |
| shear abs(g) | 0.049 | 0.032 | 0.015 | 0.050 | +0.650 |

Source plane: correlation 0.9873, size ratio 1.0808, centroid error 0.506 pixels, peak ratio 0.834,
nmse 0.0252. χ²/dof median 3,227.9. The radial slope is the least constrained parameter and sits close
to isothermal; the lens-model ladder of Section 2.3 places the free slope at approximately +0.010 in
arc correlation and the external shear at approximately +0.014.

### B Instrument response calibration

Parametric Gaussian sweep, residual nmse against the stored convolved source, 25 compact sources:
0.01062, 0.00707, 0.00435, 0.00300, 0.00288, 0.00312, 0.00855 at full width at half maximum 0.00,
0.10, 0.14, 0.18, 0.20, 0.22 and 0.30 arcsec. The Wiener-regularized extraction stacked over 40
sources gives the wing measurement quoted in Section 2.2. Adding the multiplicative discrepancy term
of Section 2.3 flattens the per-image χ² across the signal-to-noise range from 1,411 to 8,455 down to
329 to 229, and reduces the p90 to p10 spread across images from 95-fold to 8.7-fold.

### C Resolution is not a rendering choice

The parametric source is continuous and can be sampled on any grid. Scored against the true source
rendered on the same grid, n = 2,000:

| factor | scale (arcsec/px) | corr | nmse | size ratio |
|---|---|---|---|---|
| 1× | 0.1059 | 0.9787 | 0.0420 | 0.979 |
| 2× | 0.0530 | 0.9773 | 0.0446 | 0.955 |
| 4× | 0.0265 | 0.9770 | 0.0452 | 0.942 |
| 8× | 0.0132 | 0.9769 | 0.0453 | 0.933 |

Correlation moves by 0.0017 across an eightfold range, indicating that the sampling does not limit
the reconstruction.

### D Validation

| test | asserts | measured |
|---|---|---|
| deflection | power law plus shear against `lenstronomy` 1.14.2 | ≤ 3 × 10⁻¹⁵ |
| source profile | elliptical Sérsic against `lenstronomy` | 0.00 |
| hypergeometric series | truncation against `scipy.special.hyp2f1` | ≤ 3 × 10⁻¹⁵ |
| magnification | autograd Jacobian against the analytic Hessian | ≤ 2 × 10⁻⁹ |
| supersampling | error against an S = 9 reference | 5.5 × 10⁻³ at S = 3 |
| backend parity | identical physics under NumPy and PyTorch | 10⁻⁶ |
| torch forward model | against the NumPy renderer | 5.5 × 10⁻¹⁶ |
| back-projection | round trip reproduces the source value, not only its shape | 3.3 per cent |

Two gates run before any training: one perturbs a single input pixel and requires the peak output
response to fall on the corresponding output pixel, and one requires no non-finite gradient on the
first step, which matters because the ellipticity parameterization is singular at e₁ = e₂ = 0 under
reverse-mode automatic differentiation.

### E Additional detail, outlined

- **E.1 Box size as a measured choice.** Median flux inside the box against half-extent: 0.715 at 0.8
  arcsec, 0.891 at 1.2 arcsec, 0.958 at 1.6 arcsec, with the corresponding data-per-unknown ratios.
- **E.2 Spin-2 shrinkage.** Modulus ratio by phase-error bin: 0.21, 0.23, 0.23 and 0.22 across 0 to 5,
  5 to 15, 15 to 30 and 30 to 90 degrees, with counts 120, 221, 200 and 259.
- **E.3 Injection detail.** Aperture definition, per-bin counts (253, 1,215, 572, 295, 65), the full
  signal-to-noise stratification, and the fixed-amplitude caveat: the clump is 15 per cent of the
  global source peak wherever it lands, giving Spearman +0.40 between radius and recovered contrast.
- **E.4 The magnification gate.** Pre-gate sub-block variance 0.0427 against the control's 0.0266,
  and the speckle statistics the gate does buy.
- **E.5 Reproduction.** Anonymized repository, exact commands, χ²/dof convention.

---

## Notes for the author (delete before submission)

**What changed from the previous version, and what did not.** Structure, headings and register only.
Every number, table, figure and claim is carried over unchanged. Section headings are now plain nouns
in the manner of both reference papers, the four contribution bullets became the last paragraph of
the introduction, the metaphor opening is gone, bold run-in labels carry the sub-points inside
Methods and Results, and there is a proper Conclusion and Future Work section, which the previous
draft lacked and both reference papers have.

**Still out of scope.** The lens-model ladder in Section 2.3 comes from the earlier diagnostics work
rather than from `superres/`. Re-run it with `superres/lens_ladder.py`, which now exists, or delete
the numbers and write that paragraph as motivation. The response sweep in Section 2.2 and Appendix B
just needs `superres/calibrate_psf.py` re-run.

**Three things to fix.** State the χ²/dof convention in a footnote once you have confirmed what
`eval_b4.py` uses. Section 3.4 reports two nulls, which is what the result files support: removing
Option 2 took the linear-inversion null with it, and the B4 uniform ablation quoted in
`B5_RESULTS.md` never existed, since those are B3's numbers and there is no `b4_uniform` metrics
file. And `SourceForm` is a placeholder name that must be decided or removed.

**Length. Read this before anything else.** Sections 1 to 5 run about **3,330 words** with two
figures and one table. Four NeurIPS pages in that configuration holds roughly **2,500**, so you are
about 800 words over and this will not fit as written.

I have already made the two cuts that do not lose a result: Figure 3 is gone (Section 3.5 states its
two numbers in text) and Table 2 is gone (Section 3.4 states its four numbers in one sentence).
Everything remaining costs you something scientific, so the choice is yours. In descending order of
what I would sacrifice first:

1. **Section 3.4 entirely, 175 words.** The two magnification nulls. Honest and interesting, and the
   least load-bearing thing in the paper. Replace with one sentence in Section 3.3 saying two
   attempts to make the prior magnification-aware changed nothing, and point at the appendix.
2. **The second paragraph of Section 2.1, 60 words.** Generator verification. Move to the appendix
   and leave one clause saying the generator was verified against analytic forms.
3. **Section 2.5, 40 words.** Fold the two sentences into the end of Section 2.3.
4. **The first paragraph of Section 3.3, 60 words.** The magnification ceiling. Appendix C carries
   the related flatness argument.
5. **The "boxed free-form" paragraph in Section 3.2, 60 words.** Down to its first sentence.
6. **The conclusion's middle sentences, 80 words.** Both reference papers keep their conclusions to
   two paragraphs; yours can lose the enumeration and just state the three findings.
7. **Section 2.4, 100 words.** Table 1 already names all five representations, so this section can
   become three sentences plus the architecture note.

That list comes to roughly 575. If you still overrun after all of it, drop Section 3.5 and keep the
initializer result as two sentences at the end of Section 3.1, which buys another 200.

**Do not cut** Table 1, Figure 2, the paragraph beginning "The last row sets how the rest should be
read", or the goodness-of-fit paragraph in Section 3.1.

**Anonymity.** Nothing above names a person, institution or program, reference [5] is written in the
third person, and Table 1's first row is described as "previously run on these data" without claiming
or disclaiming authorship. Keep it that way, use an anonymized repository link, and add no
Acknowledgements section until camera-ready.
