# Physics as the Only Label: Source Representations for Unsupervised Super-Resolution of Strong Lenses

> **Format note.** Drafted against the workshop template (`rps_2026_template.zip`, NeurIPS style) for *Representations for the Physical Sciences*, NeurIPS 2026. Four pages of main content; references and appendices excluded. Double-blind, non-archival. Sections 1 to 6 below are the four pages. Everything under "Appendix" is unlimited and is drafted, not just outlined, so material can be moved either way as the layout settles.
>
> **Alternative titles.** (a) *Conditioning is not enough: six source representations under one lensing forward model.* (b) *What should a lensed galaxy be made of, when the only supervision is physics?* (c) *Source representations for physics-supervised super-resolution of strong gravitational lenses.*

---

## Abstract

High-resolution imaging of strong gravitational lenses is scarce, and the surveys that will find the next hundred thousand lenses will not make it less scarce. We ask what representation a source galaxy should have when the only supervision available is physics. For each observation we fit a lens and a source jointly with no high-resolution target and no labels: an analytic, differentiable elliptical-power-law-plus-shear deflection field, a measured instrument response and detector binning close a χ² against the observed pixels. On Roman-like simulations with full generative ground truth we recover seven physical parameters from the pixels alone (Spearman +0.54 to +0.96) and cut the reconstructed-source size error from 10.6× to 1.08× against a free-field baseline on the same images. Comparing six source representations under this one forward model, we find that conditioning predicts which representations fail, and that decoder architecture accounts for a further factor of two among those that do not. Injecting structure smaller than a detector pixel, we measure that 17 to 22 per cent of it returns, significantly more where the lens magnifies more, though photon count predicts recovery three times more strongly.

---

## 1. Introduction

A strong gravitational lens is a telescope nobody built. The foreground mass spreads a patch of the background galaxy over many detector pixels, so the sky has already done the oversampling a super-resolution method would otherwise have to invent. Euclid, LSST and Roman will find lenses in numbers that put higher-resolution follow-up out of reach for all but a handful of systems, so the paired low- and high-resolution training sets that supervised super-resolution assumes will not exist at survey scale. Supervision has to come from somewhere else, and for lensing the natural candidate is the forward model: whatever source we propose must, after being lensed, blurred by the instrument and integrated by the detector, reproduce the pixels we observed.

That settles the loss and leaves open the question this paper is about. The lens is a small, identifiable, physically motivated family. The source is not. It can be a parametric profile with seven numbers, a free pixel grid with several thousand, or some hybrid, and the choice is usually made by convention rather than by measurement. We hold the data, the deflection model, the instrument model and the scoring function fixed, vary only the source representation, and report what changes. Our contributions:

1. **A per-image joint lens and source inversion trained without labels**, validated against full generative ground truth: seven physical parameters at Spearman +0.54 to +0.96 from the pixels alone, and a reconstructed-source size error of 1.08× where a free-field baseline on the same images sits at 10.6 to 12.8×.
2. **A controlled comparison of six source representations.** Conditioning predicts which representations fail. Among those that do not, it predicts almost nothing: two comfortably overdetermined representations differ by a factor of two in source error, and the difference is the decoder.
3. **A direct measurement of sub-detector-pixel recoverability.** Across 2,400 injections of a 0.76-pixel clump through the full instrument model at matched noise, 17 to 22 per cent of the injected contrast returns, significantly more where the lens magnifies more ($p = 5.3\times10^{-5}$), but photon count predicts it three times more strongly.
4. **Physics-only training makes a one-pass network an excellent initialiser and a poor point estimator**, with the mechanism identified and the consequence measured: the per-image fit drops from 480 model evaluations in four stages to 151 in one, at a slightly better χ².

We also report three independent null results on magnification-adaptive regularisation, one of which fails for a reason that is not specific to lensing.

---

## 2. One forward model, no labels

**Data.** Roman-like `lenstronomy` [2] simulations at 0.10593″ per pixel on a 127² grid: an elliptical-power-law-plus-shear deflector contributing no light, an elliptical Sérsic source, and a subhalo population. We checked the generator rather than assuming it (Appendix B). Substructure is not a perturbation here. Inside $r < 30$ px the r.m.s. difference between images with and without subhaloes is 4.33 against a background σ of 0.032, so a smooth model can never reach χ²/dof ≈ 1 against these data.

**Model.** $\boldsymbol\beta = \boldsymbol\theta - \boldsymbol\alpha(\boldsymbol\theta)$ with six lens parameters. We evaluate the Tessore and Metcalf [1] deflection as its underlying power series rather than through `scipy.special.hyp2f1`, which is neither differentiable nor available in torch; since $|w| = (1-q)/(1+q) < 1$ always, the series converges geometrically and the term count can be chosen adaptively. It agrees with `lenstronomy` to $2.7\times10^{-15}$. Ray-shooting is followed by convolution on the sky and then area-averaging onto detector pixels, in that order. The instrument response is measured rather than assumed, and two independent estimates put it at 0.18 to 0.20″ FWHM with wings 29 times higher than a Gaussian at $r = 0.25''$ (Appendix B). Every earlier pipeline on this dataset used 0.10″.

**Objective.** $\chi^2 = \sum_{i} \left((\text{model}_i - d_i)/\sigma\right)^2$ over a 45-pixel disc (6,361 pixels), with σ measured per image from a source-free annulus. The amortised runs use $\sigma_{\text{eff}}^2 = \sigma_{\text{bg}}^2 + (0.02\,\text{model})^2$, admitting the measured kernel-shape systematic; without it, four images in every 400 carry half the batch gradient.

**Starting the fit without labels.** The baseline pipeline we inherited on this dataset read $\theta_E$ from the simulation metadata, which guaranteed the correlation it then reported. We replace it with a measurement on the image: along 72 azimuths, record the radius of peak brightness and fit $r_0 + a_1\cos\varphi + b_1\sin\varphi + a_2\cos2\varphi + b_2\sin2\varphi$, weighted by ring brightness. The $m = 0$ term estimates $\theta_E$ to a median 0.48 px on a ring of median radius 12.5 px; the $m = 1$ term estimates the source offset at Spearman +0.875, which matters because the brightest annulus sits at $\theta_E + \beta$ and β has a median of 2.9 px.

**Why the lens is fitted and not fixed.** Holding the true source fixed and varying only the deflection, arc-weighted correlation with the truth runs 0.869 for a circular isothermal sphere, 0.973 for an ellipsoid and 0.997 with external shear. Refitting $\theta_E$ buys +0.02; adding ellipticity buys +0.10. The failure is angular, not radial: averaged over azimuth a circular lens gets the critical radius right to 0.7 per cent, while the peak-to-peak swing it omits is 5.4 px on a 12 px ring against a response-limited arc width near 2 px. It puts the arc in the wrong place by more than the arc's own width at most azimuths while getting the mean radius essentially exactly right, which no azimuthally averaged metric can see.

**Scoring.** We score against source-plane truth, convolution-matched, and against the seven physical parameters. Image-plane scores reward the wrong thing on noisy targets: a 2-pixel blur of the input scores 0.951 on the conventional skill metric while the exact physical model scores 0.630 (Appendix C).

> **Figure 1.** Observation → (a) per-image bounded least squares or (b) back-projection through the fitted lens into a fully convolutional decoder → source. Both branches re-lens, convolve and bin, and close χ² against the same observed pixels. No high-resolution target enters anywhere. *[Schematic to draw; panels available in `figS5_sersic_stages.png`.]*

---

## 3. Six source representations, one scoring function

Rows 2 to 6 of Table 1 see the same 800 validation images, the same deflection and instrument models and the same scorer. Where a lens is frozen it is the same fitted lens.

> **Table 1.** Source representations under one forward model. "Data per value" uses the 4,312 pixels above 3σ. Source metrics are medians against the stored truth, convolution-matched. Row 1 is the free-field baseline we inherited on this dataset, built on a fixed circular isothermal lens in the manner of [11], with the Einstein radius taken from simulation metadata rather than fitted; its size ratio is the range across a 100× sweep of the total-variation weight.
>
> | source representation | free source values | data per value | corr | size ratio | nmse |
> |---|---|---|---|---|---|
> | free field, 254², lens fixed and circular | 64,516 | 0.07 | – | 10.6 – 12.8 | – |
> | free field, 64², linear inversion on the fitted lens | 4,096 | 1.05 | 0.958 | 1.064 | 0.082 |
> | free field, 62², convolutional decoder on the back-projection | 3,844 | 1.12 | 0.973 | 1.130 | 0.0527 |
> | Sérsic + 48² residual, same decoder | 7 + 2,304 | 1.87 | 0.979 | 1.043 | **0.0408** |
> | Sérsic + 32² bounded correction from a 132-vector | 7 + 1,024 | 4.18 | 0.959 | 0.937 | 0.0799 |
> | Sérsic only, 7 values, per-image fit | 7 | 616 | **0.987** | 1.081 | **0.0252** |

**Conditioning explains the failures.** The 254² field is underdetermined 15 to 1 and inflates the source by 10.6 to 12.8× across a 100× sweep of the total-variation weight. That is not a tuning failure: the unconstrained minimiser of total variation is a flat field, so raising the weight moves toward the wash, and the most compact result in the sweep was the least regularised one. The mechanism is worth naming because it recurs. With a misspecified lens the model cannot put flux at the right azimuth whatever source it proposes, so under a pixel loss its best strategy is to spread the source until the lensed ring is fat enough to overlap the true arc everywhere. The 64² linear inversion [3] is nearly critically determined and produces speckle, at a median ray coverage of 1.44 per source pixel.

**Conditioning does not explain the rest.** The two hybrid rows are both overdetermined, at 4.18 and 1.87 data per value, and they differ by a factor of two in source error. The better-conditioned one is the worse. What separates them is where spatial information goes. The 32² correction is decoded from a 132-dimensional global summary produced by an average pool, and 64 per cent of that model's 849,519 weights sit in the one dense layer that has to regenerate a spatial map from it; measurably, its correction never exceeds 3 to 4 per cent of the Sérsic. The 48² residual comes from a fully convolutional decoder acting on the observation already back-projected into the source plane, so it begins in spatial register and only has to deblur and sharpen. It carries fewer weights, roughly 0.63 M against 0.85 M, so capacity is not the explanation, although it does take its lens from a separate per-image fit rather than predicting it.

**Free-form must trade coverage against resolution; a hybrid need not.** A boxed source cannot hold flux outside its box, and 28 per cent of the true source flux at the median lies outside ±0.8″. A box wide enough for the wings at ±2.5″ and the same 0.05″ per pixel needs about 9,600 unknowns against 4,300 informative data, which is the regime of Table 1's first row. The hybrid keeps the wings in seven numbers and spends its pixels only on the residual. Measured, not argued: the Sérsic-based run at half-extent 1.2″ shows no truncation artefact, while the free-form run at the larger 1.6″ still clips the lensed model to a rounded square.

**Fitting the data better is not recovering the source.** The free-form run had more images (8,000 against 6,000), more epochs (50 against 30) and a wider box, and it reached the lower validation χ² of the two (329 against 343, on the objective both were trained on). Its source is the worse one: nmse 0.0527 against 0.0408, peak ratio 0.748 against 0.927.

---

## 4. Does the physics predict where super-resolution works?

The warrant for claiming resolution finer than the detector is the magnification field: where $|\mu| \gg 1$ the lens has already spread a patch of source over many pixels. Across 2,000 fits the median tangential stretch is 3.075× and the radial stretch 0.994×, so the data support a source grid roughly 3× finer than the detector and no more. Both decoders ask for 2.05× and 2.07×. The fitted lens also reproduces magnification quantities it was never fitted to, including the tangential critical curve to 0.63 px r.m.s. and its azimuthal swing at 0.540″ against a true 0.548″, where a circular lens would give exactly zero (Appendix D). Since μ is a derived function of the six lens parameters, this is a consistency check and not an independent measurement of magnification.

**Injection.** To test recoverability directly we add a Gaussian clump of FWHM 0.08″, which is 0.76 detector pixels, at 15 per cent of the source peak; re-render the system through lens, instrument and detector; and add noise at that image's measured σ. The same system without the clump is reconstructed under the identical noise realisation, and we measure the difference in a small aperture at the clump position. 2,400 injections across 60 images, no retraining.

Sub-detector-pixel structure comes back, at 17 to 22 per cent of injected contrast depending on local magnification. Recovery is better where μ is larger: mean contrast 0.208 for $\mu \geq 2$ against 0.175 for $\mu < 2$, Mann-Whitney $p = 5.3\times10^{-5}$. Reconstruction error over the whole source falls from 0.39 in the $\mu = 1$ to 2 bin to 0.038 in the $\mu = 16$ to 32 bin, a power-law slope of $-0.78$ across the five populated bins. Magnification does predict where the reconstruction can be trusted.

It is not what limits it. Rank correlation of recovered contrast is +0.056 with local magnification and +0.152 with signal-to-noise. Stratified by signal-to-noise, the $\mu \geq 2$ advantage runs 1.25, then 1.22, then 1.08 as photons become plentiful. Magnification decides whether sub-pixel structure survives when photons are scarce, and these data are photon-rich nearly everywhere. That single measurement retro-explains three separate attempts to turn μ into a prior, none of which paid.

> **Table 2.** Three implementations of magnification-adaptive regularisation, three nulls. Source nmse, medians, same 800 validation images. Rows 2 and 3 are matched ablations against a control identical in every respect but the μ term.
>
> | mechanism | μ-adaptive | uniform |
> |---|---|---|
> | regularisation weight, 64² linear inversion | ~4% worse at matched data fidelity | – |
> | correction penalty, 32² decoder | 0.0799 | 0.0789 |
> | structural resolution gate, 48² residual decoder | 0.0414 | 0.0408 |

The last fails for a reason worth isolating. The gate is a sigmoid in $\log\mu$ applied to the decoder's output before the forward model, removing sub-pixel detail where the lens delivered no resolution. Because it sits before the likelihood and the network is trained through it, the network learns its inverse. Measured on 12 images, the gated model emits 1.61× the sub-pixel variance of the control before the gate, the gate attenuates by 0.624, and $1.61 \times 0.624 = 1.005$: the two end up with the same sub-pixel content to within 0.3 per cent. A differentiable constraint placed before the likelihood is not a constraint if the network is trained through it. The gate survives only where χ² is indifferent to it, buying 14 to 18 per cent less speckle below 5 per cent of peak surface brightness (Appendix G).

> **Figure 2.** Recovered contrast against local magnification, 2,400 injections, with the signal-to-noise stratification showing the effect closing from 1.25× to 1.08×. *[Source: `figMU1_resolution_vs_mu.png`.]*

---

## 5. Amortised inference: a good initialiser, a poor estimator

One forward pass of the amortised decoder costs 12 ms against 815 ms for the per-image fit, and it matches the fit on $\theta_E$ (Spearman +0.960) and half-light radius (+0.953). It fails on ellipticity, at +0.269 against +0.762. The failure is not orientation. Split into modulus and phase, the median phase error is 19°, and the modulus is shrunk by a flat factor near 0.22 in every phase-error bin, including the bin where orientation is right to within 5°. That is what a conditional-mean estimator does to a spin-2 quantity under orientation uncertainty: a large ellipticity pointed the wrong way costs more under a squared loss than none at all, so the expected-error-minimising answer shrinks toward zero. The per-image fit is a maximum-likelihood estimator on one image and has no such incentive. Both behave as their objectives specify, and more epochs will not change it.

So we stop asking the network to be the estimator and use it to start one. On 800 common validation images, with identical bounds, tolerances and residual function, the cold fit takes 480 model evaluations and 0.815 s across four staged parameter releases and reaches χ²/dof 3302; started from the network it takes 151 evaluations and 0.460 s in a single stage and reaches 3228. Recovery improves on six of seven parameters and the ellipticity shrinkage is undone entirely, from +0.269 through +0.776 to +0.839. The staged schedule exists only because a neutral start falls into poor minima. From the network's start it is unnecessary, and the warm fit lands in a better basin rather than reaching the same one faster, which is what made 6,000 training lenses affordable for the frozen-lens experiments of Section 3.

> **Figure 3.** Cost against accuracy for network, cold fit and warm fit, with the ellipticity scatter inset. *[Source: `figR1_cost.png`, `figR3_ellipticity.png`.]*

---

## 6. Limitations

Every number here comes from one dark-matter class: the dataset loader sorted a concatenated list of per-class paths before truncating, so all runs drew from the axion class. Training and validation remain disjoint splits, and this class carries the heaviest subhalo population of the three, so the smooth-model results are conservative. Cross-class generalisation is untested. The parametric row of Table 1 is super-resolution within a model family, and these sources genuinely are Sérsics, which is why it is the hardest available baseline and also why it cannot be the deployable answer for real galaxies. The magnification field is derived from the fitted lens rather than measured. χ²/dof settles near 3,200 rather than 1 because the instrument response shape is a systematic floor: the residual rises with signal-to-noise (Spearman +0.449), the signature of a systematic and not of noise, and part of what remains is the subhalo signal the smooth model does not try to fit. Everything is simulated, uses one of two available bands, and the roughly 1 per cent differences in Table 2 rest on a single seed.

---

## References

[1] N. Tessore and R. B. Metcalf. The elliptical power law profile lens. *Astronomy & Astrophysics*, 580:A79, 2015.

[2] S. Birrer and A. Amara. lenstronomy: Multi-purpose gravitational lens modelling software package. *Physics of the Dark Universe*, 22:189, 2018.

[3] S. Warren and S. Dye. Semilinear gravitational lens inversion. *The Astrophysical Journal*, 590:673, 2003.

[4] S. H. Suyu, P. J. Marshall, M. P. Hobson and R. D. Blandford. A Bayesian analysis of regularized source inversions in gravitational lensing. *Monthly Notices of the Royal Astronomical Society*, 371:983, 2006.

[5] S. Vegetti and L. V. E. Koopmans. Bayesian strong gravitational-lens modelling on adaptive grids. *Monthly Notices of the Royal Astronomical Society*, 392:945, 2009.

[6] Y. D. Hezaveh, L. Perreault Levasseur and P. J. Marshall. Fast automated analysis of strong gravitational lenses with convolutional neural networks. *Nature*, 548:555, 2017.

[7] L. Perreault Levasseur, Y. D. Hezaveh and R. H. Wechsler. Uncertainties in parameters estimated with neural networks: application to strong gravitational lensing. *The Astrophysical Journal Letters*, 850:L7, 2017.

[8] W. R. Morningstar et al. Data-driven reconstruction of gravitationally lensed galaxies using recurrent inference machines. *The Astrophysical Journal*, 883:14, 2019.

[9] A. Adam, L. Perreault Levasseur, Y. Hezaveh and M. Welling. Pixelated reconstruction of foreground density and background surface brightness in gravitational lensing systems using recurrent inference machines. *The Astrophysical Journal*, 925:124, 2022.

[10] K. Karchev, A. Coogan and C. Weniger. Strong-lensing source reconstruction with variationally optimized Gaussian processes. *Monthly Notices of the Royal Astronomical Society*, 512:661, 2022.

[11] A. Shankar, M. W. Toomey and S. Gleyzer. Unsupervised physics-informed super-resolution of strong lensing images for sparse datasets. *Machine Learning and the Physical Sciences workshop, NeurIPS*, 2024.

[12] P. Reddy et al. DiffLense: a conditional diffusion model for super-resolution of gravitational lensing data. *Machine Learning: Science and Technology*, 5:035061, 2024.

[13] C. Ledig et al. Photo-realistic single image super-resolution using a generative adversarial network. *CVPR*, 2017.

[14] W. Shi et al. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. *CVPR*, 2016.

[15] T. E. Collett. The population of galaxy-galaxy strong lenses in forthcoming optical imaging surveys. *The Astrophysical Journal*, 811:20, 2015.

*Still to add: the Roman, Euclid and LSST mission papers you prefer, and a citation for the simulation dataset itself.*

---

# Appendix

*Unlimited length. A to F below are drafted; G to K are outlined.*

## A. Full parameter recovery

Per-image bounded nonlinear least squares (trust-region reflective, not plain Levenberg-Marquardt, since several parameters carry hard physical bounds), warm-started from the amortised network, $n = 800$ validation images. Nothing in this table was available to the fit.

| parameter | fit median | true median | median &#124;err&#124; | p90 &#124;err&#124; | Spearman |
|---|---|---|---|---|---|
| $\theta_E$ (″) | 1.370 | 1.320 | 0.047 | 0.179 | +0.960 |
| β (″) | 0.292 | 0.305 | 0.033 | 0.091 | +0.915 |
| $R_{\text{sersic}}$ (″) | 0.505 | 0.484 | 0.041 | 0.159 | +0.954 |
| $n_{\text{sersic}}$ | 1.140 | 1.005 | 0.138 | 0.786 | +0.945 |
| slope γ | 2.003 | 2.058 | 0.078 | 0.247 | +0.538 |
| lens &#124;e&#124; | 0.200 | 0.222 | 0.032 | 0.098 | +0.839 |
| shear &#124;g&#124; | 0.049 | 0.032 | 0.015 | 0.050 | +0.650 |

Source plane, against the stored array, convolution-matched: correlation 0.9873, size ratio 1.0808, centroid error 0.506 px, peak ratio 0.834, nmse 0.0252. χ²/dof median 3227.9, against null baselines of 37,511 (constant image), 2,796 (2-pixel blur) and 4,091 (3-pixel blur). The cold fit at $n = 2{,}000$ gives the same picture with lens $|e|$ at +0.762 and χ²/dof 3085.

Three biases point the same way: $\theta_E$ high by 3.5 per cent, $R_{\text{sersic}}$ by 10 per cent, $n_{\text{sersic}}$ by 17 per cent, and peak ratio 0.82. All are consistent with an under-modelled response wing, which is a testable prediction rather than a shrug.

## B. Verifying the generator, and calibrating the instrument response

An analytic elliptical-power-law-plus-shear convergence built from the manifest matches the stored map at ratio 0.98 and log-correlation 0.936 over $r = 3$ to 40 px; an analytic `SERSIC_ELLIPSE` matches the stored source at correlation 0.967, rising to 0.9989 once the instrument response is included. The response itself is over-determined, since the stored unlensed array is that analytic source already convolved. A parametric sweep over Gaussian FWHM gives residual nmse 0.01062, 0.00707, 0.00435, 0.00300, 0.00288, 0.00312, 0.00855 at 0.00, 0.10, 0.14, 0.18, 0.20, 0.22 and 0.30″, a clean minimum at 0.18 to 0.20″ with 0.10″ a factor 2.4 worse. A Wiener-regularised Fourier extraction, stacked over 40 sources of differing size and assuming nothing about the shape, gives wings 29 times higher than a 0.18″ Gaussian at $r = 0.25''$. Arcs here reach peak-to-background ratios of $10^3$ to $10^4$, so a 1 per cent error in kernel shape is a 50σ per-pixel residual, which is the entire explanation for χ²/dof near 3,200 in a fit that recovers $\theta_E$ to 0.047″.

## C. Why we do not score in the image plane

The conventional skill score on this dataset, $1 - \text{MSE}/\text{mean}(L^2)$ against the min-max-normalised noisy observation:

| "model" | skill |
|---|---|
| flat constant image | 0.348 |
| 2-pixel Gaussian blur of the input | **0.951** |
| 3-pixel blur of the input | 0.931 |
| trained models | 0.85 to 0.91 |
| exact physical model (true source, true lens, true response, noiseless) | **0.630** |

Spearman(skill, signal-to-noise) $= -0.85$. A physical model cannot beat the truth, so everything above 0.63 is fitting the noise realisation, and the metric rewards smoothing hardest exactly where there is least signal. Two causes: the target is noisy, and the denominator is a mean square rather than a variance, so the min-max pedestal inflates it for free.

## D. The lens-model ladder and the critical curve

True source held fixed, deflection model varied, arc-weighted correlation with truth, $n = 60$:

| deflection model | corr | fraction below 0.9 |
|---|---|---|
| circular isothermal sphere, true $\theta_E$ | 0.869 | 55% |
| same, $\theta_E$ refitted per image | 0.888 | 55% |
| ellipsoid, true ellipticity | 0.973 | 20% |
| power law, true slope and ellipticity | 0.983 | 13% |
| ellipsoid + true external shear | **0.997** | 2% |

Split by lens ellipticity the circular model collapses where expected: 0.952, 0.921, 0.761, 0.658 across $|e|$ bins 0 to 0.1, 0.1 to 0.2, 0.2 to 0.3 and above 0.3.

Magnification recovery, $n = 2{,}000$: per-pixel correlation of $\log\mu$ over the arc annulus 0.929; total magnification 5.996 fitted against 5.941 true; critical curve radial r.m.s. 0.0668″ = 0.63 px; azimuthal swing 0.540″ against 0.548″; median tangential stretch 3.075×, radial 0.994×.

## E. Resolution is not a rendering choice

The parametric source is a continuous function of sky position, so it can be sampled on any grid. Rendered at 1×, 2×, 4× and 8× the detector sampling and scored against the true source rendered on the same grid, $n = 2{,}000$:

| factor | scale (″/px) | corr | nmse | size ratio |
|---|---|---|---|---|
| 1× | 0.1059 | 0.9787 | 0.0420 | 0.979 |
| 2× | 0.0530 | 0.9773 | 0.0446 | 0.955 |
| 4× | 0.0265 | 0.9770 | 0.0452 | 0.942 |
| 8× | 0.0132 | 0.9769 | 0.0453 | 0.933 |

Flatness across an 8× range is the argument: correlation moves by 0.0017. The sampling does not limit the reconstruction, the parameters do. Were correlation to fall materially with factor, the rendering grid would be misaligned.

## F. Validation ladder

| test | asserts | measured |
|---|---|---|
| deflection | elliptical power law + shear against `lenstronomy` 1.14.2 | ≤ 3e−15 |
| source profile | elliptical Sérsic against `lenstronomy` | 0.00e+00 |
| hypergeometric series | truncation against `scipy.special.hyp2f1` | ≤ 3e−15 |
| magnification | autograd Jacobian against the analytic Hessian | ≤ 2e−9 |
| supersampling | error against an $S = 9$ reference | 5.5e−3 at $S = 3$ |
| backend parity | identical physics under numpy and torch | 1e−6 |
| torch forward model | against the numpy renderer | 5.5e−16 |
| ring initialiser | against the manifest, 400 images | 0.48 px median |

The five gate checks that run before any training include one that pokes a single input pixel and requires the peak output response to land on the corresponding output pixel, and one that requires no non-finite gradient on step one. The second has repeatedly earned its keep: the ellipticity parameterisation is singular at $e_1 = e_2 = 0$ under reverse-mode autodiff, which is a legitimate place for the optimum to sit rather than merely a place to pass through.

## G to K. Outlined

- **G. The three magnification nulls in detail**, including the pre-compensation measurement (pre-gate sub-block variance 0.0427 against the control's 0.0266) and the speckle statistics the gate does buy (r.m.s. over peak 0.0213 → 0.0183 below 5 per cent of peak).
- **H. Spin-2 shrinkage.** Modulus ratio by phase-error bin: 0.21, 0.23, 0.23, 0.22 across 0 to 5°, 5 to 15°, 15 to 30° and 30 to 90°, with bin counts 120, 221, 200, 259.
- **I. Injection experiment details.** Aperture definition, per-bin counts (253, 1215, 572, 295, 65), the full signal-to-noise stratification, and the fixed-amplitude caveat: the clump is 15 per cent of the global source peak wherever it lands, so it is a larger local perturbation far from centre, and Spearman(distance, contrast) = +0.40.
- **J. Dataset note.** Two defects in the simulation products. The substructure-free arrays were rendered with a circular deflector, settled by a phase test rather than an amplitude test: the convergence quadrupole on the ring reads 0.172 at a phase error of 1.6° against the manifest for the full array, and 0.0000 at 44.3° for the substructure-free one, where random is 45°. Anyone isolating the macro lens by reaching for that array measures zero ellipticity and concludes their own model is broken.
- **K. Reproduction.** Anonymised repository, exact commands, and the χ²/dof convention used throughout.

---

## Notes for the author (delete before submission)

**Where the numbers came from.** Section 5 and Appendix A use the refined $n = 800$ evaluation (`refined_evaluation_results.txt`). The 1× to 8× flatness and all magnification-extraction numbers are the $n = 2{,}000$ run (`superres_metrics.json`, `magnification.json`). The injection statistics were recomputed directly from `results/mu_resolution.json`, all 2,400 rows, and **supersede the 480-injection tables still sitting in `MAGNIFICATION_SR.md`**. In particular the 3.1× low-signal-to-noise figure in that file appears nowhere above; the current data give 1.25×. Delete the stale tables before anyone reads the repository.

**Four things to settle before submission.**

1. **χ²/dof convention.** `evaluate.py` divides by $6{,}361 - 14$; `eval_pathb.py` divides by $6{,}361 - 14 - 1024$, a factor 1.192. Confirm what `eval_b4.py` uses before putting 3,492 and 3,228 in the same sentence, then state the convention once in a footnote. I kept the raw χ²/dof comparison out of the main text for this reason and used the floored validation χ² (329 against 343) instead, which both decoders were actually trained on.
2. **Three nulls, not four.** `B5_RESULTS.md` lists a fourth, "B4's `--reg-mode mu` vs `uniform`, nmse 0.0799 vs 0.0789", but those are B3's numbers and there is no `b4_uniform` metrics file in `results/`. Both B4 v2 runs used `--reg-mode mu` with no uniform control. Table 2 reports three. Run the control if there is time; a reviewer who counts result files will notice otherwise.
3. **The axion-only sentence.** It is in Section 6 and in the abstract's implied scope but not stated in the abstract itself. If the cross-class re-run lands before the deadline, fold it in. If not, consider adding five words to the abstract, since a reviewer who finds it only in Section 6 will read it as a concession rather than a disclosure.
4. **Anonymity.** Nothing above names a person, institution or programme, and [11] is written in the third person. Keep it that way and use an anonymised repository link. Hold acknowledgements for camera-ready.

**Length.** Sections 1 to 6 run about 2,620 words including the abstract, plus three figures and two tables. NeurIPS style at 10pt over a 5.5in measure holds roughly 800 words to a full text page, so four pages minus the title block, three figures at about 1.8in each and two tables leaves room for roughly 2,300. Budget on cutting 300 to 400 words once it is typeset and you can see the actual overflow. Cut in this order, and stop as soon as it fits:

1. The gate-mechanism paragraph at the end of Section 4. It is one of the better paragraphs in the paper, but Appendix G carries it in full and a single sentence pointing there costs almost nothing.
2. The "free-form must trade coverage against resolution" paragraph in Section 3, down to its first and last sentences.
3. The instrument-response sentences in Section 2, which Appendix B repeats verbatim.
4. Figure 3, replaced by the two numbers it carries. Section 5 already states them in the text.

**Do not cut Table 1 or Figure 2.** Table 1 is the paper's spine and Figure 2 is its most novel content.

**Figures.** Only Figure 1 needs drawing. Figures 2 and 3 exist in `results/` and need regenerating at publication size with larger fonts. If a fourth figure would help more than 150 words of text, the candidate is `figS5_sersic_stages.png` cropped to three panels, showing observation, back-projection and reconstructed source.
