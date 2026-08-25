# Unsupervised Super-Resolution and Analysis of Real Lensing Images
## Complete technical account: Option 1 / Path A, Path B (B3), and B4

**Prajwal Upadhyay — GSoC 2026, ML4SCI DeepLense**
Mentors: Michael Toomey (MIT), Pranath Reddy (NASA), Sergei Gleyzer (Alabama), Hamees Sayed (IIT Madras)
Document date: 21 August 2026 · Covers everything in `superres/` and the work it rests on

---

> **How to use this document.** §0 is the map — read it first and you can follow any part of the
> meeting. §1–§4 are the shared foundation (data, physics, forward model, unsupervised
> initialisation). §5 is Option 1 / Path A. §6 is Path B / B3. §7 is B4. §8–§13 are validation,
> reproduction, the honest ledger, and prepared answers to the questions I expect.
>
> Every number in this document is either (a) recomputed from files on disk during the writing of
> this document — those are marked **[verified]** — or (b) quoted from a results file or report,
> marked **[reported]** with its source. Nothing is estimated or remembered.

---

## 0. The map: five names, in one page

The project has accumulated five labels. They are not five projects; they are two axes crossed.

**Axis 1 — how the source is represented:**

| | representation | unknowns per image |
|---|---|---|
| **Option 1** | parametric: elliptical Sérsic, 7 numbers | 7 |
| Option 2 *(retired)* | free-form: 64×64 = 4,096 pixels, linear inversion on a frozen lens | 4,096 |

**Axis 2 — how the parameters are obtained:**

| | inference method | cost per image |
|---|---|---|
| **Path A** | per-image Levenberg–Marquardt fit | 877 ms **[verified]** |
| **Path B** | amortised: one CNN forward pass, weights shared across the dataset | 12 ms **[reported]** |

**The designs that exist:**

| name | = | status |
|---|---|---|
| **Option 1 = Path A** | Sérsic source + EPL+shear lens, fitted per image by LM | the validated result |
| Option 2 | free-form source on the Path A lens, semilinear inversion | retired to `Option2_failed/`, two negative results |
| **B3** | Path B with source = Sérsic(7) **+** bounded 32×32 correction map | trained, 50 epochs, evaluated |
| B2 | Path B with source = free-form 32×32 map only, no Sérsic | control run, mis-configured, not a fair test |
| **B4** | proposed: Anirudh's fully-convolutional SISR operating on the *back-projection* through **our** fitted lens | **designed, not built** |

**One sentence for each of the four things this document covers:**

- **Option 1 / Path A.** Fit 14 physical numbers (6 lens, 7 source, 1 sky) per image by
  Levenberg–Marquardt against the observed pixels, using an analytic EPL+external-shear
  ray-tracer, a measured PSF and a measured noise level. Nothing is read from the manifest. The
  fitted source is a *continuous function*, so it can be sampled on any grid — that is the
  super-resolution. Validated against ground truth: source size ratio **1.089** against the old
  pipeline's 10.6–12.8, source correlation **0.987**, seven parameters recovered with Spearman
  +0.57 to +0.96.
- **Path B / B3.** Replace the optimiser with a CNN that emits the same 14 numbers *plus* a
  32×32 source-plane correction map, trained with **no high-resolution target** — the only
  supervision is χ² of the re-lensed, re-blurred, re-binned prediction against the observation.
  The magnification field, computed from the same ray-shooting that renders the image, decides
  where the correction is allowed to add detail. 71× faster, matches Path A on θ_E, β and
  R_sersic, and **fails on ellipticity for a reason that is a property of the estimator, not of
  the training**.
- **B4.** The measured weakness of B3 is its decoder: it squeezes the image through a global
  average pool into a 128-vector and then hallucinates a spatial map from it. Anirudh Shankar's
  SISR architecture never does this — it back-projects into the source plane first and stays
  fully convolutional. B4 = his architecture on our fitted lens. One day of work; every piece
  already exists in the repo.
- **The result I would actually lead with.** The network is a bad *estimator* and an excellent
  *initialiser*. Warm-starting the LM fit from the network gives **identical χ², better
  parameters on five of seven, ~2× fewer model evaluations, and removes the staged optimisation
  schedule entirely**.

**What Option 2 was and why it is excluded** (asked for, so stated once and then dropped): with
the lens frozen from Path A the image is linear in the source pixels, so `s = (MᵀC⁻¹M + λH)⁻¹
MᵀC⁻¹d` has a closed form (Warren & Dye 2003 semilinear inversion). It ran, it was well posed on
paper (0.64 unknowns per datum), and it lost — nmse ≈ 0.08 against the parametric 0.024, because
Model_A's sources genuinely *are* Sérsics. It also produced the first of two independent negative
results on magnification-adaptive regularisation. It is not discussed further below except where
it is the evidence for a design decision in B3.

---

## 1. The data: Model_A

### 1.1 What it is

Roman Space Telescope-like simulations produced with `lenstronomy`, generator confirmed rather
than assumed to be **EPL + SHEAR + SERSIC_ELLIPSE**:

- an analytic EPL+shear convergence built from the manifest parameters matches the stored
  `kappa_nss` at ratio 0.98 and log-correlation 0.936 over r = 3–40 px **[reported:
  PROJECT_REPORT §5.1]**
- an analytic `SERSIC_ELLIPSE` matches the stored `unlensed` at correlation 0.967 **[reported:
  same]**, and at **0.9989** once the PSF is included **[reported: superresolve.py header]**

| property | value |
|---|---|
| grid | 127 × 127 px |
| pixel scale | 0.10593 ″/px |
| bands | 2 (F062, F087); we use band 0 = F062 |
| classes | axion / cdm / wdm |
| per class | 10,000 train, 2,000 val, 2,000 test **[verified]** |
| lens galaxy light | **none** — fitting a deflector Sérsic returns amplitude ≈ 0 |

Arrays inside each `.npz`: `image`, `image_nss`, `unlensed`, `kappa`, `kappa_nss`, `kappa_sub`,
`psi`, `psi_nss`, `psi_sub`. Scalars: `theta_E`, `host_e1/e2`, `gamma1/2_ext`, `host_slope`,
`source_x/y`, `source_R_sersic`, `source_n_sersic`, `source_e1/e2`, `snr_max`, `z_lens`,
`z_source`, `num_subhalos`, `log10_m_axion`, `host_mass_log10`, `lens_mass_log10`,
`deflector_*`, `log_mhigh`, `uid`, `instrument`, `light_profile`, `bands_csv`, `dm_type`.

`unlensed` is the true source **already convolved with the instrument PSF**, on the image-plane
angular grid. This detail caused a real bug in the previous evaluation code (§5.6).

### 1.2 The population, measured on the 2,000 val images we actually use **[verified]**

```
theta_E          median 1.320"   ( = 12.46 px )
lens |e|         median 0.214    -> q median ~0.647
|gamma_ext|      median 0.033
host_slope       median 2.053    (range 1.90 - 2.20)
beta = |source offset|   median 0.311"  ( = 2.94 px )
source R_sersic  median 0.488"
source n_sersic  median 0.999
snr_max          median 10.1     p10 4.4    p90 23.3
num_subhalos     median 6093     range 25 - 39835
pixels above 3 sigma  ~4312 of 16129        [reported: PROJECT_REPORT §5.1]
fitted disc r <= 45 px                 = 6361 pixels  [verified]
```

Substructure is **not** a small perturbation on the axion class: rms(`image` − `image_nss`)
inside r < 30 px is 4.33 against a background σ of 0.032 **[reported: fit_per_image.py header]** —
a factor of 135. A smooth model can therefore *never* reach χ²/dof ≈ 1 against `image`, and that
is by design: the substructure is what is left in the residual.

### 1.3 Two defects in the dataset that I found and reported

**(a) `image_nss` and `kappa_nss` were rendered with a circular lens.** This is in
`sie_pipeline/MENTOR_BUGREPORT.md`, addressed to the four of you, dated 18 August.

The decisive test is the **phase** of the m = 2 term of the convergence measured on the ring
r = θ_E, compared against the position angle implied by `host_e1`/`host_e2`. Amplitude alone is
ambiguous because substructure adds azimuthal power at all harmonics; orientation is not.

| array | m₂ amplitude | Spearman vs manifest \|e\| | **phase error** |
|---|---|---|---|
| `kappa` (with substructure) | 0.172 | +0.938 | **1.6°** |
| `kappa_nss` (no substructure) | 0.0000 | +0.155 | 44.3° (random = 45°) |

The ruler was calibrated first on analytic EPL convergence maps, so m₂ reads directly as an |e|
(0.02→0.0200, 0.05→0.0499, 0.10→0.0999, 0.22→0.2209, 0.35→0.3549). `kappa_nss` reads 0.0000 on
every split and every DM class, 150 images checked.

Confirmed independently from the images, same code and settings, only the fitted array differing:

| target | fitted \|e\| median | manifest \|e\| median | Spearman |
|---|---|---|---|
| **`image`** | **0.196** | 0.221 | **+0.782** |
| `image_nss` | 0.032 | 0.221 | −0.120 |

**Why this matters for us:** anyone testing a smooth lens model reaches for `image_nss`, because
it isolates the macro lens from substructure. If they do, they measure zero ellipticity and
conclude their own model is broken. That is exactly what happened to me — I retracted a correct
result on 17 August and had to un-retract it on 18 August. **We fit `image`.**

**(b) `psi` is not usable as stored.** The scale factor needed to satisfy ∇²ψ = 2κ varies from
about −25 to −45 across images, so ψ cannot be ray-traced without pinning its normalisation
first. `kappa` is fine — it matches θ_E/(2r) at r = θ_E to better than 1%.

Also noted: `snr_max` is not peak/σ_bg. Measured peak/σ_bg spans 100–18,000 while `snr_max` spans
2.5–21, with a non-constant ratio (41–1018). We use it only as a stratification variable.

### 1.4 A defect in *my* evaluation that I found while preparing this document — please read

**Every result quoted as "all three DM classes" is in fact axion-only.** **[verified]**

`data_a.ModelADataset` builds its file list as

```python
paths = []
for c in classes:                       # ["axion", "cdm", "wdm"]
    paths += glob.glob(root/model/split/c/"*.npz")
paths = sorted(paths)                   # <-- sorts the CONCATENATED list
...
    if limit and len(keep) >= limit: break
```

`sorted()` orders by full path string, so every `.../axion/...` sorts before every `.../cdm/...`.
There are 10,000 axion train files and 2,000 axion val files, so:

| run | requested | actually loaded |
|---|---|---|
| Path A `--n 2000` (val) | 2000 across 3 classes | **2000 axion** |
| B3 training `--n-train 6000` | 6000 across 3 classes | **6000 axion (train split)** |
| B3 validation `--n-val 800` | 800 across 3 classes | **800 axion (val split)** |
| B2 evaluation, n = 800 | 800 across 3 classes | **800 axion** |

Confirmed by reading the `path` field of every row in `fits_img.json` (2000/2000 axion),
`fits_pb3_mu.json` (800/800 axion) and `fits_pb2.json` (800/800 axion).

**What this does and does not invalidate.**

- It does **not** invalidate any number. Train and val are different *splits*, so there is no
  train/test leakage; the fits and the recovery statistics are correct for what they measured.
- It does **not** weaken the physics: axion is the *hardest* class (median 6,093 subhalos,
  κ_sub reaching ±0.66), so the smooth-model results are, if anything, conservative.
- It **does** invalidate the sentence "all three DM classes" wherever it appears — in
  `README.md`, `sie_pipeline/README.md`, `RESULTS_V3.md`, `MENTOR_BUGREPORT.md` §3 and the
  `--classes axion cdm wdm` command lines. Those must be corrected before anything is submitted.
- It **does** mean **cross-class generalisation is untested**. That is a one-line fix
  (interleave the class lists, or take `limit` per class) and a ~30-minute re-run of
  `fit_per_image.py` at n = 600 (200 per class) to check that θ_E/|e| recovery holds on cdm and
  wdm. I would do that before the paper, not after.

This is the single most important correction in this document.

---

## 2. The physics, from first principles

### 2.1 The lens equation

Mass between us and a distant galaxy bends light. In the thin-lens approximation all the
deflection happens in one plane, and the mapping from where light *came from* (source-plane
angular position **β**) to where we *see* it (image-plane position **θ**) is

$$\boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}(\boldsymbol{\theta})$$

This one equation is the entire forward model. Everything else is a choice of what **α** is.

Two consequences used throughout:

- **The map is many-to-one.** Several θ map to the same β — that is what makes arcs, rings and
  multiple images. Forward (source → image) is single-valued and cheap. Backward (image →
  source) is the inverse problem.
- **Surface brightness is conserved.** I_image(θ) = I_source(β(θ)). You *never* multiply an
  intensity by the magnification. μ is a change of solid angle; a lensed arc has more total flux
  because it subtends more sky, not because any patch got hotter. This is one of the most common
  errors in lensing ML code. In this pipeline μ appears in exactly two places — as a diagnostic,
  and as a *prior weight* telling the source-plane regulariser where the data are informative —
  and never as a multiplicative factor on a pixel value.

### 2.2 Potential, convergence, shear

**α** is the gradient of a 2-D potential ψ, and the potential's Laplacian is the projected mass:

$$\boldsymbol{\alpha} = \nabla\psi, \qquad \kappa = \tfrac{1}{2}\nabla^{2}\psi$$

κ (*convergence*) is the projected surface mass density in units of the critical density Σ_cr;
κ = 1 is the strong-lensing threshold. The shear is the traceless part of the same tensor:

$$\gamma_1 = \tfrac{1}{2}(\psi_{,11} - \psi_{,22}), \qquad \gamma_2 = \psi_{,12}$$

κ focuses isotropically (a circle stays a circle and grows); γ stretches anisotropically (a
circle becomes an ellipse).

### 2.3 The lens models, in increasing generality

**SIS — singular isothermal sphere.** Flat rotation curve, one parameter:

$$\kappa(\theta) = \frac{\theta_E}{2\theta}, \qquad
\boldsymbol{\alpha}(\boldsymbol{\theta}) = \theta_E\,\hat{\boldsymbol{\theta}}$$

The deflection has **constant magnitude θ_E everywhere**, pointing radially inward. β = 0 gives a
perfect Einstein ring; β ≠ 0 gives two images at θ_E + β and −(θ_E − β).

> **The single most useful fact in the whole project:** a circular lens produces **at most two**
> images of a point source, and they are diametrically opposite. `count_arcs` finds **≥ 3
> distinct arcs in 98% of Model_A images**. The images themselves refute SIS, with no reference
> to the manifest. This is what killed the previous pipeline's lens model.

**SIE — singular isothermal ellipsoid.** Same radial profile, elliptical isodensity contours;
axis ratio q and a position angle, carried as a spin-2 pair (e₁, e₂) so the parameterisation is
continuous. Real ellipticals have q ≈ 0.6–0.8; Model_A's median q is ≈ 0.65.

**EPL — elliptical power law.** Adds a free radial slope, κ ∝ r^(1−γ); γ = 2 recovers SIE.

**External shear.** Mass outside the frame — a neighbouring group, large-scale structure — adds a
pure quadrupole with no convergence:

$$\boldsymbol{\alpha}_{\rm ext} = \begin{pmatrix}\gamma_1 & \gamma_2\\ \gamma_2 & -\gamma_1\end{pmatrix}\boldsymbol{\theta}$$

Nearly every real lens needs it. In our code the shear acts about the *grid origin*, not the lens
centre, matching `lenstronomy`'s `SHEAR` with `ra_0 = dec_0 = 0`, which is how Model_A was
generated. A shear about a different point differs by a constant deflection, which is exactly
degenerate with the source position — so the choice costs no generality.

**How much each term is worth, measured.** Take the *true* source (analytic Sérsic, manifest
parameters), ray-trace it through a candidate lens at 3× supersampling, convolve, bin, no noise;
reference is the same thing through the true EPL+shear lens; correlate, arc-weighted, n = 60
**[reported: PROJECT_REPORT §7.1]**:

| deflection model | corr with truth | frac < 0.9 |
|---|---|---|
| SIS, true θ_E | 0.869 | 55% |
| SIS, θ_E refit per image | 0.888 | 55% |
| SIE (true ellipticity, no shear) | 0.973 | 20% |
| EPL (true slope + ellipticity, no shear) | 0.983 | 13% |
| **SIE + true external shear** | **0.997** | 2% |

Split by ellipticity, SIS collapses exactly where you would expect:

| \|e\| | n | SIS | SIE | SIE+shear |
|---|---|---|---|---|
| 0.0–0.1 | 9 | 0.952 | 0.975 | 0.995 |
| 0.1–0.2 | 21 | 0.921 | 0.969 | 0.999 |
| 0.2–0.3 | 15 | **0.761** | 0.988 | 0.997 |
| 0.3+ | 15 | **0.658** | 0.972 | 0.995 |

**Refitting θ_E buys +0.02. Adding ellipticity buys +0.10.** That single comparison is why the
whole architecture was replaced: the previous pipeline spent its effort on an operator bank
indexed by θ_E, i.e. refining the axis that was already correct to 2%.

### 2.4 What SIS actually gets wrong: angular, not radial

The Einstein ring is the tangential critical curve, 1 − κ − |γ| = 0. Its radius as a function of
azimuth is

- **SIS:** r_crit(φ) = θ_E — a perfect circle.
- **SIE/EPL + shear:** r_crit(φ) ≈ θ_E(1 + e·cos 2(φ − φ₀)) — period 180°, because ellipticity is
  spin-2.

Measured on 60 truth lenses **[reported: PROJECT_REPORT §7.2]**:

| quantity | value |
|---|---|
| mean over φ of r_crit/θ_E | **0.993** — SIS is right to 0.7% |
| peak-to-peak swing of r_crit over φ | **5.4 px** (median) |
| same, \|e\| > 0.3 | **9.9 px** |
| mean ring radius | 12.2 px |
| PSF-limited arc width | ≈ 2 px |

So a circular lens puts the arc in the wrong place by **more than the arc's own width** at most
azimuths, while getting the mean radius essentially exactly right. Any metric that averages over
azimuth will not see this.

### 2.5 EPL deflection: the closed form, and why it is a power series in our code

Tessore & Metcalf (2015), eqs. 22–23, in the major-axis frame:

$$\alpha = \frac{2}{1+q}\left(\frac{b}{R}\right)^{t} Z \;{}_2F_1\!\left(1, \tfrac{t}{2};\, 2-\tfrac{t}{2};\, w\right)$$

with

$$Z = q x' + i y', \quad R = |Z|, \quad w = -\frac{1-q}{1+q}\cdot\frac{Z}{\bar Z},
\quad b = \theta_E\sqrt{q}, \quad t = \gamma - 1$$

and the convergence in the same frame

$$\kappa = \frac{2-t}{2}\left(\frac{b}{\sqrt{q^2x'^2+y'^2}}\right)^{t}$$

`lenstronomy` evaluates ₂F₁ with `scipy.special.hyp2f1`, which is neither differentiable nor
available in torch. Because (1)ₙ/n! = 1 the series collapses to a plain geometric-like sum

$$_2F_1 = \sum_n c_n w^n, \qquad c_0 = 1, \qquad \frac{c_{n+1}}{c_n} = \frac{t/2 + n}{2 - t/2 + n}$$

and |w| = (1−q)/(1+q) < 1 always, so it converges geometrically. Model_A's most elliptical lens
has |e| = 0.393 → q = 0.436 → |w| = 0.393, and 0.393⁴⁰ ≈ 1e−17. `_n_terms_for` chooses the term
count adaptively as n = log(tol)/log|w| — 30 terms for the worst lens, 8 for a near-circular one,
which is worth a factor 2–5 inside an optimiser loop. `tests/test_lens_models.py` **measures**
the truncation error against `scipy.special.hyp2f1` rather than assuming it.

Written with explicit real and imaginary parts, not complex tensors, so the autograd path has no
Wirtinger-derivative subtleties.

**Agreement with lenstronomy 1.14.2: 2.7 × 10⁻¹⁵** on the deflection field **[reported:
README.md]**. The Sérsic matches to **0.00e+00** — exactly, because we deliberately reproduce
lenstronomy's own b(n) = 1.9992n − 0.3271 rather than the more accurate Ciotti & Bertin expansion.
The goal is to reproduce the simulator, not to be independently more correct; a different b(n)
would show up as a systematic bias in recovered R_sersic.

### 2.6 Two singularities that cost me days, and how they are fixed

Both live in `ellipticity_to_phi_q(e1, e2)`, and both are **invisible in the forward pass**.

At e₁ = e₂ = 0 the value is perfectly well defined (q = 1, circular), but two operations are 0/0
under reverse-mode autodiff:

$$\frac{\partial}{\partial e_1}\sqrt{e_1^2+e_2^2} = \frac{e_1}{\sqrt{e_1^2+e_2^2}},
\qquad
\frac{\partial}{\partial e_2}\operatorname{atan2}(e_2,e_1) = \frac{e_1}{e_1^2+e_2^2}$$

This is not hypothetical. The earlier `train_amortised.Encoder` zero-initialised its final Linear
weight *and* bias, so e₁ = 0.6·tanh(0) = 0 and e₂ = 0 **exactly** on step one. The first backward
pass produced NaN gradients on the ellipticity head; `clip_grad_norm_` does not filter NaN, so it
scaled every parameter by a NaN coefficient; Adam wrote NaN into the whole head; on step two q was
NaN and `_n_terms_for` died inside `math.ceil(nan)` with *"cannot convert float NaN to integer"* —
an error message pointing at the hypergeometric term count, six frames from the actual cause.

The fixes, all in `lens_models.py`:

1. `EPS_E2 = 1e-24` added inside the sqrt, so the derivative at the origin is 0 (finite and
   correct — |e| genuinely has a minimum there) instead of inf. Shifts |e| by ≤ 1e−12.
2. `E_FLOOR = 1e-10`: a degenerate (e₁, e₂) is pushed onto the +e₁ axis with `where` **before**
   either singular operation, so the degenerate branch contributes exactly zero gradient. φ is
   arbitrary there because q = 1 makes the frame rotation a no-op. Perturbs the deflection by
   ~3e−10, two orders below the 1e−8 agreement the tests require.
3. `E_MAX = 0.9` cap (q = 0.053), far outside Model_A's range but far enough from q = 0 that
   (b/R)^t and the 1/(1+q) prefactor stay finite if a free parameter transiently wanders.
4. `_n_terms_for` raises an explicit `FloatingPointError` naming the real cause if it is handed a
   non-finite |w|, because Python's `min`/`max` propagate NaN silently.

**And it is not merely an initialisation accident.** e = 0 is a legitimate place for the optimum
to sit, so the fit is expected to *sit on* the singular point, not merely pass through it.

### 2.7 Magnification

Differentiate the lens equation:

$$A = \frac{\partial\boldsymbol{\beta}}{\partial\boldsymbol{\theta}} = I - \frac{\partial\boldsymbol{\alpha}}{\partial\boldsymbol{\theta}}
= \begin{pmatrix}1-\kappa-\gamma_1 & -\gamma_2\\ -\gamma_2 & 1-\kappa+\gamma_1\end{pmatrix},
\qquad \mu = \frac{1}{\det A} = \frac{1}{(1-\kappa)^2 - |\gamma|^2}$$

det A = 0 is the **critical curve** = the Einstein ring, where μ diverges. The eigenvalues are

$$\lambda_t = 1 - \kappa - |\gamma| \quad\text{(tangential)}, \qquad
\lambda_r = 1 - \kappa + |\gamma| \quad\text{(radial)}$$

and a source patch is stretched by 1/|λ| along each eigendirection.

**For a circular SIS**, κ = |γ| = θ_E/2r, so λ_r = 1 exactly, λ_t = 1 − θ_E/r, and all the
magnification is tangential. That is the formula hard-coded in
`physics_losses.fixed_sis_magnification` in the old pipeline — correct for a circular SIS and for
nothing else. **For SIE or SIE+shear both statements fail**: λ_r ≠ 1, and, more subtly, the
*eigenvectors* of A are no longer radial and tangential — external shear picks its own axis,
independent of where the lens centre is. Any code that decomposes μ along radial/tangential
directions is projecting onto the wrong basis.

Two implementations, and the relationship between them is the point:

- `hessian_analytic()` — closed form (Tessore & Metcalf eq. 17 in the corrigendum form used by
  `lenstronomy`'s `EPLMajorAxis.hessian`), then rotated out of the major-axis frame. γ is spin-2,
  so the frame rotation is γ₁′ = γ₁cos2φ − γ₂sin2φ, γ₂′ = γ₁sin2φ + γ₂cos2φ; κ is a scalar and is
  unchanged. External shear adds directly to (γ₁, γ₂) and contributes nothing to κ. The **sign**
  of that rotation is fixed by `tests/test_lens_models.py` against lenstronomy's own hessian, not
  by argument.
- `magnification_autograd()` — two vector-Jacobian products through the deflection field that is
  *actually applied*. This works because the map is pointwise (α at one pixel does not depend on
  θ at another), so `grad(sum(α_x))` w.r.t. x is dα_x/dx everywhere with no cross-pixel
  contamination. That assumption is checked against `hessian_analytic` in
  `tests/test_jacobian.py`.

**The structural point:** because the production path differentiates the deflection actually used
by the ray-tracer, μ and the ray-tracer are *incapable* of describing different lenses. That was
precisely the defect in the old pipeline, where a hard-coded circular formula had no link to the
sparse operators in use.

`magnification()` clips |μ| at 50 by default and returns |μ| unless `signed=True`.

### 2.8 Why magnification is the scientific premise of the project

Where |μ| ≫ 1 the lens has spread a small patch of source over many detector pixels — **the sky
has already done the oversampling for us**. So the source can legitimately be reconstructed
finer than the detector *in those regions specifically*, and a super-resolution prior should know
that the information content is not uniform across the source plane.

This is what converts "we rendered the source on a 4× finer grid" (a numerical statement) into
"the lens delivered this resolution" (a physical one). It is also why the magnification work and
the super-resolution work belong in the same paper rather than in two.

Measured on 2,000 val images from the fitted lens **[verified from `magnification.json`]**:
median tangential stretch **3.075×**, median radial stretch **0.994×** (i.e. essentially none —
consistent with a nearly isothermal profile). So the data support a source grid up to ~3× finer
than the detector, and no more. B3 asks for 2.05×, which sits comfortably inside that, and
`train_pathb.py` prints the comparison at startup and **warns if you ask for more than the lens
can deliver**.

### 2.9 The degeneracies — and why they are the whole problem

Given one observed image, the pair (lens, source) is not unique:

- **Mass-sheet degeneracy.** κ → λκ + (1−λ) with the source rescaled leaves every image position
  and flux ratio unchanged. Only time delays or an absolute source size break it.
- **Source-position transformation.** A broader family with the same effect.
- **Lens/source trade-off.** A bigger, smoother source through a stronger lens imitates a compact
  source through a weaker one.

The previous architecture's response was to **fix the lens** and solve only for the source. That
removes the degeneracy but replaces it with a worse risk: **if the fixed lens is wrong, the
source absorbs the error.** That is exactly what happened — see §5.1.

Our response instead: keep the lens family small enough to be identifiable (6 parameters), keep
the source family small enough to be identifiable (7 parameters, or 7 + a *bounded* correction),
and fit both jointly per image. 14 unknowns against 6,361 fitted pixels is overdetermined ~450:1,
so the degeneracies that matter in the pixellated case never get room to act.
---

## 3. The forward model — the observation operator

```
source (sky) --lens--> lensed sky --PSF--> blurred sky --pixel binning--> data
```

The order is physical and not negotiable: lensing remaps the sky, the PSF is an optical
convolution that happens *on the sky*, and pixelation is an integration performed *by the
detector*. Blurring after binning would be a different — and wrong — instrument.

`raytrace.render(source, lens_params, n_pix, pixel_scale, psf, supersample, grid, background)`:

1. build the image-plane grid of (sub)pixel centres,
2. `bx, by = ray_shoot(X, Y, lens_params)` → β = θ − α,
3. `sky = source.at(bx, by)` — for a Sérsic this is a **closed form evaluated directly at the
   ray-shot coordinates**, so there is no source grid and hence *no interpolation error*,
4. convolve with the PSF (`replicate` padding, not zeros — a zero pad darkens the border and
   would bias the background level that the χ² partly measures),
5. area-average S×S blocks — the detector's pixel integration,
6. add a free additive `background`.

### 3.1 Coordinate conventions (verified against the truth arrays, not assumed)

- x → array axis 1, y → array axis 0; arcsec; origin at the grid centre with the **(N−1)/2**
  convention — the same one used by `data_manifest.ManifestDataset._crop`, `evaluate_sis.py` and
  `build_sis_mappings.py`, so indices remain comparable with every earlier evaluation.
- ellipticity: φ = arctan2(e₂, e₁)/2, c = hypot(e₁, e₂), q = (1−c)/(1+c) (minor/major).
- frame rotation matches `lenstronomy Util.util.rotate`: x′ = x cos φ + y sin φ,
  y′ = −x sin φ + y cos φ.
- lens equation β = θ − α.
- Sérsic convention matched to `SERSIC_ELLIPSE` with its default `sersic_major_axis=False`,
  because that is what generated Model_A:
  norm = √|1 − e₁² − e₂²|, x′ = ((1−e₁)dx − e₂dy)/norm, y′ = (−e₂dx + (1+e₁)dy)/norm,
  R = hypot(x′, y′), I = amp·exp(−b(n)[(R/R_s)^{1/n} − 1}).

### 3.2 Supersampling and flux conservation

The old sparse operators conserved flux exactly because they accumulated area fractions. Neither
bilinear sampling nor point-sampling a continuous profile does. The mitigation is supersampling:
evaluate the image plane at S×S subpixels, then area-average. The residual error falls as O(1/S²)
and is **measured** in `tests/test_raytrace.py` against an S = 9 reference:
**S = 3 gives 5.5e−3** **[reported: README.md]**, well below Model_A photon noise.

Practical note: the headline `fits_img.json` run uses **S = 1**, because `--psf-mode empirical`
requires it — the stored empirical kernel is defined on the detector grid, not a supersampled
one. That is a deliberate trade: an exactly measured kernel at S = 1 beats an assumed Gaussian at
S = 2. Both are available; `fit_one`'s default is S = 2 with a Gaussian.

### 3.3 The PSF, measured rather than assumed

Every previous run used `--psf-fwhm-arcsec 0.10`. That number was never measured. It can be,
because `unlensed` is the source **already convolved** with the instrument PSF and the manifest
gives the analytic Sérsic that generated it. So the PSF is the ratio of the two, and it is
over-determined. Two independent estimates, in `calibrate_psf.py`:

**A. Parametric sweep.** Render the analytic Sérsic, convolve with a trial Gaussian, minimise the
residual against `unlensed`, n = 25 compact sources (R_sersic < 0.8″) **[reported: PROJECT_REPORT
§7.4]**:

```
FWHM(")   residual nmse
 0.00      0.01062
 0.10      0.00707    <- what every SIS run used
 0.14      0.00435
 0.18      0.00300
 0.20      0.00288    <- minimum
 0.22      0.00312
 0.30      0.00855
```

A clean single minimum at 0.18–0.20″; 0.10″ is 2.4× worse. Roman F062 at 0.106″/px is
undersampled, so the effective PSF (optics ⊕ pixel response) being broader than the diffraction
limit is physically sensible.

**B. Empirical Fourier extraction, assuming nothing about the shape.** Since
`unlensed = sersic ⊛ psf`,

$$\widehat{\rm PSF} = \frac{\hat U\,\overline{\hat S}}{|\hat S|^2 + \epsilon\,\max|\hat S|^2}$$

a Wiener-regularised division, stacked over 40 images with different source sizes so that
different parts of the frequency plane get filled in. Result: the wings are **far** broader than
a Gaussian — at r = 0.25″ the empirical profile is 0.139 of its peak against 0.005 for a 0.18″
Gaussian, **a factor of 29**, rising further outward **[reported: raytrace.py / evaluate.py]**.
That is the signature of a Moffat.

**Why this matters more than it looks.** The residual floor of estimate A is nmse 0.003 rather
than ~0, which already says the kernel is not Gaussian. Model_A arcs reach peak/σ_bg of 10³–10⁴,
so a **1% error in the PSF shape is a ~50σ per-pixel residual**. This is the entire explanation
for χ²/dof ≈ 3000 in a fit that recovers θ_E to 0.046″ — the parameters are right and the kernel
shape is the systematic floor. `load_psf` centre-crops the stored 127² kernel to 25² (the profile
is below 1e−3 of peak beyond r ≈ 0.5″ = 5 px), which is ~25× faster per model evaluation.

Three implementations are available: `gaussian_psf`, `moffat_psf` (α = fwhm/(2√(2^{1/β}−1)),
β = 2.5 default, β→∞ recovers a Gaussian), and `load_psf` for the empirical kernel. The headline
results use **empirical**.

### 3.4 The noise model

Model_A ships no noise map. σ is measured per image as the standard deviation of the background
annulus **r > 50 px** (`data_a.ModelADataset.sigma`). The arcs sit at r < 30 px on a 127 px grid,
so that annulus is source-free; measured this way σ_bg ≈ 0.032 in native units, and `image` vs
`image_nss` differ there by ≈ √2 × that value — confirming the two arrays carry *independent*
noise realisations and that the estimate is picking up real detector noise rather than residual
signal.

**Images are used in NATIVE units, not min–max normalised.** This matters and is not cosmetic.
`data_manifest.ManifestDataset.__getitem__` returns (a − min)/(max − min). That is a per-image
affine map whose scale is set by the single brightest and single faintest pixel — both
noise-dominated at snr_max ≈ 10 — so the noise level after normalisation is itself a random
variable, and a χ² built on it is meaningless. The measured damage: because `mean(obs²)` includes
the induced pedestal, the old `skill` metric ended up anti-correlated with SNR at **Spearman
−0.85** (§5.7). Here the affine freedom lives in the *model*, as a free `amp` and a free
`background`, where it can be fitted, rather than in the data where it corrupts the noise scale.

**The σ-floor (Path B only, §6.7).** σ_bg alone claims the only uncertainty is background noise.
It is not: the PSF wings are measured wrong at the 1–3% level, and that error scales with model
flux. So Path B uses

$$\sigma_{\rm eff}^2 = \sigma_{\rm bg}^2 + (f\cdot\text{model})^2, \qquad f = 0.02$$

This is the standard way to admit a multiplicative systematic. On the per-image fits it flattens
χ² across SNR from 1411 → 8455 down to **329 → 229**, and cuts the p90/p10 spread from 95× to
8.7× **[reported: train_pathb.py header]**.

---

## 4. Closing the unsupervised leak: initialisation from the ring geometry

### 4.1 What the leak was

`train_sis_bank.py` line 344:

```python
t_px = ds.theta_E / args.resolution     # <- the MANIFEST column
bins = assign_bins(t_px, alphas)
```

and that integer then (a) selected the backward operator that produced the network's input,
(b) was broadcast into the network as a third input channel, and (c) selected the forward
operator that graded the network's output. `evaluate_sis.py` lines 586–588 repeated it.

**Consequence:** Spearman(predicted ring radius, true θ_E) = +0.91/+0.92/+0.95 across three runs
— *guaranteed by construction*, because the ring radius of the prediction is set by which forward
operator was applied, and that operator was chosen using the answer. Contrast the quantity that
was **not** handed over, the source position: Spearman(predicted source centroid, true β) =
−0.130 / +0.032 / −0.081; the predicted centroid sat at ~1 px on every image against a true
median β of 2.90 px. **The network placed the source at the centre of every image.**
**[reported: PROJECT_REPORT §6]**

### 4.2 The replacement: `theta_e_init.py`

θ_E is now a **free parameter of the forward model**, fitted against the image. This module only
supplies a starting point, and it does so from the pixels alone.

For each of 72 azimuths, walk outward and record the radius of peak brightness, r(φ) — bilinear
interpolation along each ray, `dr = 0.5` px, r from 1.5 px to (N/2 − 1.5) px. Then fit, weighted
by the ring brightness (dark azimuths carry a meaningless radius and must not get equal say):

$$r(\varphi) = r_0 + a_1\cos\varphi + b_1\sin\varphi + a_2\cos 2\varphi + b_2\sin 2\varphi$$

The three harmonics separate three physically distinct things:

| harmonic | quantity | what it is |
|---|---|---|
| m = 0 | r₀ → **θ_E** | a **lens** property — what the model needs |
| m = 1 | (a₁, b₁) → **β**, as a *vector* | a **source** property. For a circular lens the brighter image lies on the same side as the source, so r(φ) peaks toward it |
| m = 2 | hypot(a₂, b₂) → quadrupole | zero for a circular lens at *any* source position, so it is a model-free ellipticity/substructure indicator |

**Why not just take the brightest annulus.** It sits at r ≈ θ_E + β, and β has median 2.90 px
against a 1 px operator bin — so a typical image would land three bins away. Separating m = 0
from m = 1 is exactly the correction that makes the estimator a *lens* measurement.

**Accuracy, measured against the manifest (evaluation only, never used in the fit)**, 400 val
images **[reported: theta_e_init.py header / PROJECT_REPORT §6]**:

```
median |theta_E_est - theta_E_true| = 0.48 px      p90 = 1.49 px
   on a ring of median radius 12.5 px
ring completeness: median 0.90 on Model_A   (vs 0.25 on Model_4)
```

That 0.90 completeness is why the estimator is far more reliable here than on the 12.5%-of-images
subset it could be checked on before.

**What is deliberately NOT done:** m₂ is reported but **not** converted into an e₁/e₂ guess. The
ring quadrupole responds to substructure as well as to macro ellipticity (axion κ_sub reaches
±0.66 against a macro κ of ~0.6), so turning it into a lens ellipticity would import an
interpretation the fit is supposed to make for itself. Everything not measurable from the ring
starts at a **population-neutral** constant, explicitly not drawn from the manifest:
`{amp: 1.0, R_sersic: 0.45, n_sersic: 1.0, se1: 0, se2: 0, sx: 0, sy: 0}`, with amp overridden by
the 99.5th percentile of the image and background by its median.

This module is numpy on purpose: it produces an initialisation, so it sits outside the gradient
path entirely. In Path B the same four numbers become network *inputs* (§6.3) — which is feature
engineering, not leakage, because every one of them is measured from the pixels.

---

## 5. OPTION 1 = PATH A — the per-image physical fit

`superres/fit_per_image.py`, `evaluate.py`, `superresolve.py`, `magnification_extract.py`.

### 5.1 What problem this solves

The previous pipeline gave a network **254² = 64,516 free source pixels** against ~4,300
informative data pixels — 15:1 underdetermined — held down only by a hand-tuned total-variation
prior. Measured outcome: `src_truth_size_ratio` = **10.6–12.8 across a 100× sweep of the TV
weight** (0, 0.03, 0.1, 0.3, 3.0), with **zero** recovered information about the source position
**[reported: PROJECT_REPORT §5.2]**.

That is not a regularisation-strength failure. TV's unconstrained minimiser is a *flat field*, so
raising the weight moves toward the wash, not away from it — the most compact result in the whole
sweep was at tv = 0.03 and it was still 10.6× too large. The real mechanism: **with a
misspecified lens the network cannot put flux at the right azimuth no matter what source it
proposes, so its best strategy under a pixel-MSE objective is to hedge — spread the source until
the forward-lensed ring is fat enough to overlap the true arc at every azimuth.** The 11×
inflation is the predictable *optimal* response to an unfittable forward model.

Option 1's answer: make the problem overdetermined and the lens correct.

### 5.2 The parameter vector

Fourteen numbers per image, in a fixed order (the optimiser works on a flat vector):

| # | name | meaning | bounds |
|---|---|---|---|
| 1 | `theta_E` | Einstein radius (″) | 0.20 – 4.00 |
| 2 | `gamma` | EPL radial slope; 2 = isothermal | 1.40 – 2.60 |
| 3 | `e1` | lens ellipticity, spin-2 component 1 | −0.60 – 0.60 |
| 4 | `e2` | lens ellipticity, spin-2 component 2 | −0.60 – 0.60 |
| 5 | `g1` | external shear component 1 | −0.30 – 0.30 |
| 6 | `g2` | external shear component 2 | −0.30 – 0.30 |
| 7 | `amp` | Sérsic amplitude | 1e−6 – 1e6 |
| 8 | `R_sersic` | half-light radius (″) | 0.03 – 3.00 |
| 9 | `n_sersic` | Sérsic index | 0.30 – 6.00 |
| 10 | `se1` | source ellipticity 1 | −0.60 – 0.60 |
| 11 | `se2` | source ellipticity 2 | −0.60 – 0.60 |
| 12 | `sx` | source offset x (″) | −2.00 – 2.00 |
| 13 | `sy` | source offset y (″) | −2.00 – 2.00 |
| 14 | `background` | additive sky | free |

Lens centre is pinned at cx = cy = 0. Derived reported quantities: |e| = hypot(e₁,e₂),
|g| = hypot(g₁,g₂), β = hypot(sx,sy).

Bounds are wide enough not to bind on Model_A (max |e| in the dataset is 0.393, slope range
1.90–2.20) and tight enough to keep the profile exponent and the ellipse well conditioned.

### 5.3 The objective and the optimiser

$$\chi^2(\mathbf{v}) = \sum_{i \in \text{disc}} \left(\frac{\text{model}_i(\mathbf{v}) - d_i}{\sigma}\right)^2,
\qquad \chi^2/\text{dof} = \frac{\chi^2}{N_{\rm pix} - 14}$$

- **Fitted region:** a disc of radius 45 px about the centre → **6,361 pixels** **[verified]**.
  The arcs live at r < 30 px; including the far corners adds thousands of pure-noise pixels that
  dilute χ² without constraining anything. The `background` parameter is still well constrained
  because the disc contains plenty of blank sky.
- **Optimiser:** `scipy.optimize.least_squares`, `method="trf"` (trust-region reflective — the
  bounded Levenberg–Marquardt variant), xtol = ftol = gtol = 1e−8, `max_nfev = 60 × n_free`.
- **Why not SGD.** This is a small, smooth, overdetermined least-squares problem with an analytic
  model. LM converges in tens of iterations; gradient descent on the same objective would need
  thousands and would still need a schedule. Autograd earns its keep in Path B, where the unknown
  is a network's weights rather than fourteen numbers.

### 5.4 Staged parameter release

Releasing all fourteen at once from a neutral start lands in local minima: with the source shape
free, the fit can trade source ellipticity against lens shear before the ring radius is even
right. The schedule fixes the geometry first:

| stage | free parameters |
|---|---|
| 1. `geometry` | theta_E, amp, sx, sy, R_sersic, background |
| 2. `+shape` | + n_sersic, se1, se2 |
| 3. `+shear` | + g1, g2 |
| 4. `all` | + gamma, e1, e2 — **everything free** |

Each stage warm-starts from the previous one. **The staging is a convergence device only** — the
final stage has every parameter free, so nothing is pinned at a value the data did not choose.
(§8 shows the schedule becomes unnecessary once the network provides the start.)

Note the circular fast path in `deflection_epl`: for q = 1 the hypergeometric argument is
identically zero and ₂F₁ = 1 exactly, so the series is pure overhead — which matters because the
first three stages hold e₁ = e₂ = 0.

### 5.5 The result — parameter recovery

**Recomputed from `superres/results/fits_img.json` + the truth scalars during the writing of this
document. n = 2000 val images (axion — see §1.4). Nothing below was available to the fit.**
**[verified]**

| parameter | fit median | true median | median \|err\| | p90 \|err\| | **Spearman** |
|---|---|---|---|---|---|
| θ_E (″) | 1.3658 | 1.3200 | 0.0464 | 0.1859 | **+0.956** |
| β (″) | 0.3016 | 0.3110 | 0.0346 | 0.1010 | **+0.914** |
| R_sersic (″) | 0.5369 | 0.4879 | 0.0461 | 0.2211 | **+0.924** |
| n_sersic | 1.1744 | 0.9988 | 0.1666 | 0.9073 | **+0.944** |
| slope γ | 1.9996 | 2.0530 | 0.0789 | 0.2465 | +0.571 |
| lens \|e\| | 0.1921 | 0.2142 | 0.0348 | 0.1141 | **+0.762** |
| shear \|γ_ext\| | 0.0510 | 0.0334 | 0.0160 | 0.0549 | +0.611 |

```
chi2/dof     median 3085.2   p10 247.7   p90 18160.9
cost         median 0.877 s/image, 468 model evaluations
fitted pixels 6361 (disc r <= 45 px), 14 free parameters
```

These match the numbers quoted in `PATHB_AND_MAGNIFICATION.md` §6 exactly.

**The three biases, all pointing the same way.** θ_E +3.5%, R_sersic +10%, n_sersic +17%,
peak_ratio 0.82 — all consistent with **under-modelled PSF wings**: the fit compensates for a
kernel that is too narrow in the wings by making the configuration slightly larger and brighter.
This is a testable prediction, and the cleanest single improvement available: fit the Moffat β,
or fit a two-component kernel.

### 5.6 The result — source plane

The fitted Sérsic, **convolved with the PSF**, against the stored `unlensed` array
**[verified from `superres_metrics.json`, n = 2000]**:

```
correlation   0.9865        size_ratio   1.0887
```

versus **10.65** for the old SIS pipeline. Additional source metrics, same n = 2000 run
**[reported: PATHB_AND_MAGNIFICATION.md §6]**: centroid error 0.523 px, peak_ratio 0.8208,
nmse 0.0271.

**The PSF-matching is not a detail.** `evaluate_sis.source_truth_metrics` compared the network's
*intrinsic* (pre-PSF) source against `unlensed`, which is *post*-PSF — an apples-to-oranges
difference of one PSF width (~10% for a 4.7 px source and a 1.7 px kernel) before any model error
is counted. Small next to a factor of 11, but it has to go before `size_ratio` can serve as the
calibration criterion its own docstring proposes. `metrics.source_truth()` convolves the model
source first.

Also in `metrics.centroid_size`: the second moment is taken about the object's **own** centroid,
not the grid centre. A correct reconstruction of a source at offset β sits off-centre *by
construction*, and β is a recovered physical quantity, not an error. Measuring about the grid
centre penalises the right answer.

### 5.7 The metric that had to be replaced

`evaluate_sis.py` reported `skill = 1 − MSE(pred, L)/mean(L²)` against the min–max-normalised
**noisy** observation. Measured on Model_A **[reported: PROJECT_REPORT §7.3]**:

| "model" | skill |
|---|---|
| flat constant image | 0.348 |
| **2-px Gaussian blur of the input itself** | **0.951** |
| 3-px blur of the input | 0.931 |
| the trained models | 0.85 – 0.91 |
| **the exact physical model** (true source, true lens, true PSF, noiseless) | **0.630** |

Spearman(skill, snr_max) = **−0.85**.

A physical model cannot beat the truth. Everything above 0.63 is fitting the noise realisation,
and the metric rewards smoothing most exactly where there is least signal. Two causes: the target
is noisy, and the denominator is mean(obs²) rather than a variance, so the min–max pedestal
inflates it for free.

Replaced by: **χ²/dof** (absolute meaning, real noise estimate), **source_truth** (PSF-matched,
against the stored truth), **parameter_recovery** (the metric Model_4 could not produce at all),
and **stratification by snr_max** (a single median hid ρ = −0.85).

**Null baselines are reported alongside every χ², so a reviewer can see what "1.0" is worth on
this data** **[reported: evaluate.py output]**: constant 37,511 · 2-px blur 2,796 · 3-px blur
4,091. At 3,085 the physical fit is at the level of a good matched blur — and unlike the blur, it
recovers seven physical parameters.

**Why χ²/dof does not reach 1, stated as a measurement rather than an excuse.** The PSF *shape*
is the systematic floor (§3.3): a 1% kernel error on an arc with peak/σ of 10³–10⁴ is a ~50σ
per-pixel residual. The signature of a systematic rather than of noise is that **χ² rises with
SNR** — measured Spearman(χ²/dof, snr_max) = **+0.449** **[verified]**, and stratified:
1,951 (snr < 6) → 7,642 (6–15) → 75,864 (> 15) **[reported]**. Report the recovery table and the
residual figure; do not chase χ² = 1.

### 5.8 The super-resolution step itself

This is the part that is easy to state and easy to get wrong.

`fit_per_image.py` returns seven numbers describing the source. Those seven numbers are a
**continuous function of sky position**, not a pixel grid. So once fitted, the source can be
evaluated on any grid — 2×, 4×, 8× finer than the detector that observed it. **We never learn an
image prior; we invert a calibrated physical forward model and then sample its solution as finely
as we like.**

Rendering finely is trivial. The question is whether the fine detail is *correct*. Model_A stores
`unlensed` only on the 127 px grid, which validates 1× only. To validate at 4× we render the
**true** source — from the manifest Sérsic parameters — on the same fine grid and compare. That
is legitimate because (a) the manifest parameters are used *only* here, for scoring, never in the
fit, and (b) those parameters reproduce the stored `unlensed` at correlation 0.9989, so the
analytic truth and the stored truth are the same object.

**Measured, n = 2000** **[verified from `superres_metrics.json`]**:

| factor | grid | scale (″/px) | corr | nmse | size_ratio |
|---|---|---|---|---|---|
| 1× | 127² | 0.1059 | 0.9787 | 0.0420 | 0.9786 |
| 2× | 254² | 0.0530 | 0.9773 | 0.0446 | 0.9550 |
| 4× | 508² | 0.0265 | 0.9770 | 0.0452 | 0.9416 |
| 8× | 1016² | 0.0132 | 0.9769 | 0.0453 | 0.9334 |

**Flatness across factors is the entire argument.** The fitted source is an analytic function, so
sampling it more finely adds no error of its own; any degradation would mean the *parameters*,
not the sampling, are the limit. corr moves by 0.0017 and nmse by 0.003 over an 8× range. The
gentle size_ratio drift 0.979 → 0.933 is the finite truth grid, not the model.

**Red flag to watch for:** if corr *drops* materially with factor (0.98 → 0.90), the rendering
grid is misaligned.

**The honest limitation, which must be in the paper.** This is super-resolution **within the
model family**. A Sérsic can only produce Sérsic-shaped things. Model_A's sources genuinely *are*
Sérsics, so here the recovered fine detail is real and the comparison is exact. On a galaxy with
spiral arms or a companion, a parametric reconstruction would smooth them away and this
measurement would not detect it. Testing that needs a free-form source — which is what Option 2
and B2/B3's correction map are for.

### 5.9 Magnification extraction — the physical warrant for the resolution claim

`magnification_extract.py`, n = 2000 **[verified from `magnification.json`]**. Four quantities:

**1. Magnification map.** Per-pixel corr(log μ_fit, log μ_true) over the arc annulus
(0.45 θ_E < r < 1.9 θ_E): **median 0.9293**. This tests the whole lens, not just θ_E, because μ
depends on κ *and* γ.

**2. Total magnification** μ_tot = (lensed flux)/(unlensed flux) — the number observers actually
use, since it converts an observed brightness into an intrinsic luminosity. Computed by
ray-shooting the source through the lens at 3× supersampling and comparing integrals on the same
grid, so pixel areas cancel and the ratio is exactly the definition. **fitted 5.996 vs true
5.941** (median), i.e. ~1% median bias.

**3. Critical curve.** Solve 1 − κ − |γ| = 0 along 72 rays by sign change plus linear
interpolation:

```
rms radial error of the recovered critical curve  = 0.0668"  = 0.63 px
azimuthal swing (peak-to-peak):  fitted 0.5397"   true 0.5482"
```

**The swing is the discriminating number.** A circular lens has swing exactly zero. Recovering
0.540″ against a true 0.548″ — a 1.5% error — means e₁, e₂, g₁, g₂ were recovered *jointly*, not
just in magnitude.

**4. Resolution gain.** Median tangential stretch **3.075×** (true 3.07×), radial **0.994×**.
This is the number that licenses the super-resolution: the lens has already spread each source
patch over ~3 detector pixels, so a source grid up to ~3× finer than the detector is supported by
the data rather than by interpolation.

**Two different "μ on the ring" numbers appear in the repo and they are not in conflict.**
`superresolve.py` reports median |μ| = **6.77** in a ±3 px band centred on θ_E — i.e. essentially
*on* the critical curve, where μ diverges. `magnification_extract.py` reports **3.06** over the
much wider annulus 0.45–1.9 θ_E, which includes low-μ regions. Different windows, both correct;
the paper should quote one and define it.

**The honest caveat, stated first in the code and repeated here.** μ is a *derived* quantity of
the lens: given (θ_E, γ, e₁, e₂, g₁, g₂) it is fully determined. Since the fit already recovers
those well, μ from the fitted lens is close to the true μ more or less automatically. **This is
not an independent measurement of magnification.** The correct claim is: *"we compute the
magnification field from the fitted lens and verify that the recovered lens reproduces the true
magnification to ~1% in total magnification and 0.63 px on the critical curve"*. What makes it
worth reporting anyway is item 4 — it is what converts a numerical statement about grids into a
physical statement about what the data support.

**One visible bias:** in `figM3` the μ_tot cloud sits above the 1:1 line at high μ. Same cause as
θ_E (+3.5%) and R_sersic (+10%) — the under-modelled PSF wings.

### 5.10 Figures produced

`make_figures.py` recomputes nothing — it reads files already on disk, so the figures cannot
disagree with the numbers.

| figure | content | what "right" looks like |
|---|---|---|
| `fig1_superres.png` | observation → reconstructed source at 8× → true source at 8× → stored `unlensed` | columns 2 and 3 look like the same object; column 1 is visibly coarse |
| `fig2_recovery.png` | fitted vs true, 7 panels, 1:1 line | tight diagonals for θ_E, β, R_sersic; looser for \|e\|; a cloud for slope |
| `fig3_sizeratio.png` | source size ratio, old SIS vs this work | two separated humps, old near 11, new near 1 |
| `fig4_residuals.png` | data / model / residual, 4 examples | residual shows **arc-shaped structure**, not a smooth halo — that structure is the substructure |
| `fig5_magnification.png` | observation next to log₁₀\|μ\| | bright ring in μ coinciding with the arc |
| `figM1`–`figM4` | μ maps, critical curves, μ scatter, resolution gain | |
---

## 6. PATH B, design B3 — amortised physics-constrained super-resolution

`superres/train_pathb.py` (the model + training loop), `eval_pathb.py` (scoring),
`make_figures_pathb.py`, `tests/test_pathb.py`.

### 6.1 The model in one picture

```
image ──CNN──┬──▶ 14 lens+source parameters ──▶ Sérsic(β)          ┐
             │                                                      ├──▶ source(β)
             └──▶ 32×32 correction map C ──────▶ C(β)              ┘
                                                        │
                          β = θ − α(θ ; predicted lens) │
                                                        ▼
                            PSF convolve → pixel bin → χ² vs the observation
```

**There is no high-resolution target anywhere.** The only supervision is the physics: whatever
the network emits must, after being lensed, blurred and binned, reproduce the observed image.
That is what makes this unsupervised super-resolution rather than a supervised upscaler.

### 6.2 Why a Sérsic **plus** a correction, and not free pixels

Two earlier attempts bracket this design and both failed for the *same* reason in different
clothes:

| attempt | unknowns per image | data pixels | outcome |
|---|---|---|---|
| original SIS pipeline | 254² = 64,516 free source pixels | ~4,300 | source **10.6–12.8× too large** across a 100× TV sweep |
| Option 2 (linear inversion, correct lens) | 64² = 4,096 free pixels | ~4,300 | median ray coverage 1.44/pixel; CG failed on ~30%; reconstructions were **speckle** |
| **B3** | **14 + 32² = 1,038**, and the 1,024 are *bounded and penalised* | ~4,300 | **4.14 : 1 overdetermined** **[verified]** |

The lesson from both is that the source plane does not contain enough independent information to
support thousands of *free* parameters per image. B3 splits the source into the part the physics
can predict and the part it cannot:

$$\text{source} = \underbrace{\text{Sérsic}(7)}_{\text{predictable}} \;+\;
\underbrace{\text{correction}(32^2,\ \text{bounded},\ \mu\text{-penalised})}_{\text{residual}}$$

and it **degrades gracefully**: if the correction learns nothing it collapses to the parametric
solution, which already works. The network cannot do *worse* than Path A on the source; it can
only add what the Sérsic missed.

**Why a network at all, when the linear solve failed?** A per-image solve has only that image's
~4,300 pixels to constrain its unknowns. A network shares one set of weights across the whole
training set, so it carries a **learned prior** over what galaxies look like. That is a genuinely
different resource, and it is exactly what the linear inversion lacked. The two failures have
different causes, so Option 2's failure is not evidence against B3.

### 6.3 The input

`build_input(X, S, R)` → a (B, 5, 127, 127) tensor.

**Channel 0: `arcsinh(image / σ_bg)`, not `image / σ_bg`.** The raw signal-to-noise peak varies by
**260×** between images (p10 251, max 65,087). A convolutional filter bank is *shared* across all
of them, so the same physical feature — an arc edge — arrives at wildly different activation
scales and one filter cannot serve both. arcsinh is the standard astronomical stretch: linear
near zero, so the noise scale is preserved and a 1σ fluctuation still reads as 1, and logarithmic
in the wings. **It compresses the across-image range from 260× to 1.9×** **[reported:
train_pathb.py]**.

**Channels 1–4: the four ring statistics, broadcast as constant planes.**
`ring_summary()` = `theta_e_init.initial_guess()` returns
`[θ_E, |m₁|, |m₂|/θ_E, ring_completeness]`. All four are measured from the pixels — nothing
touches the manifest, so feeding them is feature engineering, not leakage. Only
**rotation-invariant** quantities are used (magnitudes, not vectors), so they survive the
dihedral augmentation unchanged and need no transformation alongside the image.

**Why bother, when a CNN "should" learn these itself?** Measured on 400 val images, the ring
dipole |m₁| predicts the true source offset at **Spearman +0.875** and the quadrupole |m₂|
predicts the true |e| at **+0.512**, while the v1 network — handed only θ_E — returned **+0.228**
and **+0.313**. It reproduced almost exactly the one number it was given and learned little else
from the pixels. `eval_pathb.py` §0b now prints this **ring-only baseline** next to the network,
so *"did the network add anything?"* is answerable rather than assumed.

**The bypass.** The ring scalars are *also* concatenated straight onto the pooled feature vector,
skipping the trunk. Without the bypass they have to survive four strided convolutions and a
global average pool, competing with ~16,000 image pixels for room in a 128-dimensional
bottleneck — and measurably they do not: a v2 network handed |m₁| at ρ = +0.832 emitted a source
offset at ρ = +0.049. **It destroyed a signal it had been given.**

### 6.4 The architecture, exactly

```
trunk   Conv(5→32, k5, s2, p2)   GroupNorm(8,32)   SiLU      127 → 64
        Conv(32→64, k3, s2, p1)  GroupNorm(8,64)   SiLU       64 → 32
        Conv(64→128, k3, s2, p1) GroupNorm(8,128)  SiLU       32 → 16
        Conv(128→128,k3, s2, p1) GroupNorm(8,128)  SiLU       16 → 8
pool    AdaptiveAvgPool2d(1) → Flatten            → 128
feat    concat(pool, ring[4])                     → 132

head 1  Linear(132→128)  SiLU  Linear(128→14)               the physical parameters
head 2  Linear(132→64·8·8) → view(64,8,8)
        ConvT(64→32, k4, s2, p1) GroupNorm(8,32)  SiLU        8 → 16
        ConvT(32→16, k4, s2, p1) GroupNorm(4,16)  SiLU       16 → 32
        Conv(16→1, k3, p1)                                    the 32×32 correction map
```

**Parameter count, computed exactly [verified]: 849,519** (trunk 244,672 · parameter head 18,830
· decoder FC 544,768 · decoder convs 41,249). At fp32 that is 3.398 MB, matching the 3.41 MB
checkpoints on disk.

Three things worth saying about this table in the meeting:

1. **The trunk is deliberately small.** The task is not visual recognition; it is reading a
   handful of geometric quantities off an arc. A larger network mostly buys overfitting on 6,000
   images.
2. **GroupNorm, not BatchNorm.** Batch statistics would couple images together, and the previous
   pipeline had a documented problem with BatchNorm interacting with a bin-homogeneous sampler.
3. **`dec_fc` alone is 64.1% of all the weights [verified].** A single dense 132 → 4096 layer.
   That is the architectural weak point, and it is exactly what B4 removes (§7). The transposed
   convolutions after it impose locality, which is a far better prior than an unstructured dense
   map — but they are decoding from a 132-dimensional *global summary* of the image, which has
   already thrown away where everything was.

`base = n_c // 4` so the two stride-2 transposed convolutions land on exactly `n_c` and nothing
is bilinearly resized; a resize would make the effective number of degrees of freedom smaller
than n_c², quietly contradicting the conditioning arithmetic printed at startup.

**Initialisation: small random (std = 1e−3), NOT zeros.** Zero-init is what let the previous
amortised run sit at its neutral output for every epoch (§2.6).

### 6.5 Constrained output parameterisation — `to_params`

Every physical quantity is mapped through tanh/sigmoid/softplus into its physical range, so the
network **cannot** emit a negative Einstein radius or n_sersic = 0 — either of which produces NaN
inside (b/R)^t or R^{1/n} *before any gradient exists to correct it*.

| output | mapping | range |
|---|---|---|
| θ_E | `ring_θ_E · exp(0.3 tanh z₀)` | ×[0.74, 1.35] about the ring estimate |
| γ | `2.0 + 0.5 tanh z₁` | 1.5 – 2.5 |
| e₁, e₂ | `0.6 tanh z₂,₃` | ±0.6 |
| g₁, g₂ | `0.3 tanh z₄,₅` | ±0.3 |
| amp | `softplus(z₆ + 1)` | > 0 |
| R_sersic | `0.05 + 2.0 σ(z₇ − 1)` | 0.05 – 2.05 |
| n_sersic | `0.3 + 5.0 σ(z₈ − 1)` | 0.3 – 5.3 |
| se₁, se₂ | `0.6 tanh z₉,₁₀` | ±0.6 |
| sx, sy | `2|m₁| · tanh z₁₁,₁₂` | ±2·(ring dipole) |
| background | `tanh z₁₃` | ±1 |

**Two quantities are anchored on the ring geometry rather than predicted from scratch**, because
both estimators are good and the *residual* is far better conditioned than the absolute value:

- θ_E anchored on the ring m = 0 term (ring-only ρ = +0.963)
- |β| **scaled** by the ring dipole |m₁| (ring-only ρ = +0.832)

> *Consistency note:* the code quotes the |m₁| ring baseline as **+0.832** in `to_params` and
> **+0.875** in `ring_summary`. Both are ring-only baselines on 400 val images and both appear
> in docstrings; the number to quote anywhere else is the one `eval_pathb.py` §0b actually
> prints at evaluation time, since that one is recomputed on the images being scored.

The anchoring is why θ_E works and why, in v2, β did not: θ_E was anchored and came out at
+0.955; β was free and came out at **+0.049** despite |m₁| being available as an input. The
*angle* of β is still learned from the pixels — only the magnitude is anchored — so the network
still has real work to do, and §0b prints the ring-only control.

**Why `sx = 2|m₁| tanh(z)` and not `r·cos φ, r·sin φ`.** Forcing the magnitude with a learned
angle φ = atan2(z, z′) reintroduces exactly the EPS_E2 disease: atan2 is singular at the origin
and a small-initialised head lands there. `tests/test_pathb.py` check 5 went straight back to NaN
when that was tried. Scaling has no singularity, still gives the network a dimensionless residual
to learn, and still lets it move β by up to a factor ~2.8 either way.

**`background` is bounded by tanh.** v2 left it as a free `0.1·z` and it drifted to a median of
**−0.98** (Path A gives +0.018) — the network dug a negative sky pedestal so it could paint an
over-bright, over-large source on top of it.

### 6.6 The forward model in torch, and the magnification machinery

`render_batch()` is the same physics as `raytrace.render`, batched:

```python
ax, ay = deflection(X[None], Y[None], lens)     # analytic EPL + shear, IN the graph
bx, by = X[None] - ax, Y[None] - ay             # ray shooting
sky    = SersicSource(src).at(bx, by)           # b3
C_eff  = corr_scale * amp * tanh(C)             # bounded, scales with source brightness
sky    = sky + sample_correction(C_eff, bx, by, half_extent)
sky    = conv2d(pad(sky, 'replicate'), psf)
pred   = area_downsample(sky, S) + background
```

`tests/test_pathb.py` check 1 proves this torch forward model equals the numpy
`raytrace.render()` to **5.5e−16** **[reported]**, so the network is optimising the same physics
the paper describes.

**`sample_correction`.** Bilinear `grid_sample` at the ray-shot positions. `align_corners=False`
treats ±1 as the *outer edges* of the border pixels while our map spans [−half, +half] to pixel
*centres*, so coordinates are divided by `half·(1 + 1/(n−1))`. Getting this wrong shifts the
correction by half a pixel relative to the Sérsic. `padding_mode='zeros'`: outside the map the
correction is zero, i.e. the source reverts to the pure Sérsic — the intended behaviour.

**`ray_coverage` — where the magnification comes from.** A nearest-pixel histogram of the ray
landing points, per image:

$$\text{coverage}_j = \#\{\text{image sub-pixels that ray-trace into source pixel } j\}$$

**That count IS the magnification of that source pixel**, up to the constant sub-pixel area — and
it comes from the *same* ray-shooting that renders the image, so μ and the ray-tracer are
structurally incapable of describing different lenses. That was precisely the defect in
`physics_losses.fixed_sis_magnification`, which hard-coded det A = 1 − θ_E/r with no link to the
sparse operator actually in use.

Three implementation decisions in that function, each of which was forced by a measurement:

1. **Detached on purpose.** This is a specification of the prior ("where is the source well
   sampled?"), not a quantity we want gradients through. Letting it be differentiable would let
   the network lower its own penalty by *moving the lens*, which is not a physical incentive.
2. **Rays that land outside the map are discarded, not clamped.** Only ~34% of image-plane rays
   land inside a 0.8″ half-width source map. Clamping the rest onto the border made **the edge of
   the map the highest-coverage region**, inverting the whole point of the weighting — the
   network would have been least penalised exactly where the data say nothing.
   `tests/test_pathb.py` check 3 caught this.
3. **A 3×3 box smooth.** The histogram is a Monte-Carlo estimate of μ from a finite number of
   rays. At supersample = 1 there are ~5.4 rays per source pixel **[verified: 0.34·127²/1024]**,
   so the raw count carries substantial shot noise. A 3×3 average keeps the large-scale structure
   — which is what μ actually is — and removes the aliasing. It creates no information; the real
   fix for genuinely low coverage is `--supersample 2`, and the startup banner reports whether
   you need it.

**The penalty.**

$$R = \sum_j w_j\,\left(\frac{C_j}{\text{amp}}\right)^2, \qquad
w_j = \left(\frac{\text{median coverage}}{\text{coverage}_j}\right)^{0.5},\ \text{clipped to }[1/5,\,5]$$

Physically: where |μ| is large the data genuinely constrain fine structure, so penalise the
correction *less*; where |μ| ≈ 1 they do not, and anything painted in there is invention.

**Power 0.5 rather than 1.0, for a measured reason.** Coverage spans ~1000× across the source
plane. With raw 1/μ (power 1) the thin high-μ strip beside the caustic ends up effectively
unregularised — in the Option 2 experiment that dropped source correlation from **0.96 to 0.74**.

`--reg-mode uniform` runs the identical model with w ≡ 1, so the contribution of the magnification
term is *measured* rather than asserted. That is the ablation.

**`curvature(C)`** — discrete Laplacian energy, the standard source-plane regulariser (Suyu et al.
2006). Optional in b3, where the Sérsic already supplies smoothness; **essential in b2**, where
without it a free source grid reproduces the speckle of Option 2 exactly.

### 6.7 The loss

$$\mathcal{L} = \chi^2 + \chi^2_{\rm detached}\left(\lambda_{\rm corr} R + \lambda_{\rm curv} \text{Curv}\right),
\qquad \chi^2 = \operatorname{mean}_{i \in \text{disc}}\left(\frac{\text{pred}_i - d_i}{\sigma_{{\rm eff},i}}\right)^2$$

with σ_eff² = σ_bg² + (0.02·pred)² (§3.4).

**λ is expressed as a FRACTION of χ² and multiplied by a detached χ², so it is scale-free.** In v1
λ = 1.0 against a χ² of ~10⁵ meant the penalty was six orders of magnitude too small to do
anything — which is why the μ and uniform runs came out identical. At full saturation the b3
penalty is λ_corr · corr_scale² = 3 × 0.09 = 27% of χ².

### 6.8 Training configuration (the 50-epoch v3 runs)

| flag | value | note |
|---|---|---|
| `--n-train` | 6000 | train split — **axion only**, see §1.4 |
| `--n-val` | 800 | val split — axion only |
| `--epochs` | 50 | |
| `--warmup-epochs` | 5 | correction switched **off**; the parametric part converges first |
| `--batch-size` | 16 | |
| `--lr` | 3e−4, Adam, cosine annealing to 0 over T_max = epochs | |
| `--weight-decay` | 1e−4 | |
| `--n-c` | 32 | correction map side |
| `--half-extent` | 0.8″ | → 0.05161 ″/px → **2.052× finer than the detector** [verified] |
| `--max-gain` | 3.1 | measured median tangential stretch; the code **warns** if you exceed it |
| `--corr-scale` | 0.30 | a *safety bound*, not an operating point |
| `--lambda-corr` | 3.0 (× χ²) | |
| `--lambda-curv` | 0.0 (b3) / 3.0 (b2) | |
| `--sigma-floor` | 0.02 | **the single most important flag in the file** |
| `--reg-mode` | `mu` / `uniform` | the ablation |
| `--reg-power` | 0.5 | |
| `--augment` | 1 | random 90° rotations + flips |
| `--supersample` | 1 | |
| `--fit-radius-px` | 45 | 6,361 pixels |
| `--stretch` | 1 | arcsinh |
| `--seed` | 0 | |

**Curriculum.** `--warmup-epochs 5` trains the parametric part alone, so the correction can only
add what the *converged* Sérsic missed rather than racing it to explain the same flux.

**Augmentation is free and exactly valid here** because the objective is self-supervised: a
rotated image is a legitimate member of the data distribution and needs **no label
transformation** — the network predicts the lens of whatever image it is shown, and χ² is taken
against that same image. `n_pix` is odd, so the grid centre is a pixel centre and `rot90` is
exact. The ring features are rotation-invariant magnitudes, so they need no transformation
either. The first run overfitted 2.9×; with augmentation the train/val gap is 0.97×.

**Guards.** Non-finite loss or gradient → the batch is skipped *and reported* (not silently
dropped); `clip_grad_norm_(5.0)`; best-val checkpoint saved separately; tanh saturation printed
**every epoch**.

### 6.9 What went wrong in v1, and why the diagnosis matters more than the fix

The first full 40-epoch run (both reg modes) produced a round blob: |e| = 0.048 against a true
0.216, n_sersic Spearman −0.014, χ²/dof 12,998 against the per-image fit's 3,085. One cause
explains all of it, and it is **not** the architecture and **not** the number of epochs.

> **The χ² was never normalised.** σ was the background std alone, so a bright arc produced
> residuals of 10³–10⁴σ and a faint one ~10.

Measured on the 400 validation images **[reported: PATHB_POSTMORTEM.md]**:

```
per-image chi2:  p10 656   median 12,998   p90 206,917   max 21,953,076
  top  1% of images (4)   carry 50.8% of the total chi2
  top 10% of images (40)  carry 87.4%
expected share of the batch gradient taken by the single largest member
of a random batch of 16:   54.1%      (uniform would be 6.2%)
```

**The effective batch size was ~2.** Three consequences, all confirmed:

**(a) The magnification term never acted.** λ_corr = 1.0 against χ² ~10⁵ is six orders of
magnitude too small. Proof: the μ and uniform runs are the *same network* —
Pearson(μ, uniform) = +0.9992 on θ_E, +0.9937 on R_sersic, +0.9973 on n_sersic — and
corr(|C|, log μ) was −0.034 in the μ run and −0.038 in the uniform run. **The entire μ-vs-uniform
ablation was void. Magnification is not what broke this — it never got a chance to do anything,
good or bad.**

**(b) The tanh saturated.** With no effective penalty, **75%** of every correction map sat within
1% of the bound, where the gradient is ~0. The correction stopped learning *and* starved the
parametric head at the same time. You can see the square edge of the saturated map in figB1.

**(c) It overfitted** 2.9× train/val, invisible until the run was over.

**The v2/v3 fixes, each addressing one measured cause:**

1. `--sigma-floor 0.02` — the correct noise model, not a fudge (§3.4).
2. λ as a fraction of χ², multiplied by `chi2.detach()` — scale-free, cannot be silently six
   orders out again.
3. Saturation measured and **printed every epoch**, warned about at the end.
4. `--augment` — free, and exactly valid for a self-supervised objective.
5. `--source-mode b2` in the same file, so B2-vs-B3 is a controlled experiment rather than an
   argument.
6. Ring dipole/quadrupole added as inputs *and* bypassed to the head; β anchored on |m₁|;
   background bounded.

**This is the part of the project I would most want a reviewer to see**, because the failure was
diagnosed by measurement — gradient share, χ² percentiles, saturation fraction,
Pearson(μ-run, uniform-run) — rather than by trying more epochs.

### 6.10 The v3 result

50 epochs, 5 warm-up, 6000 train / 800 val. Same 800 val images throughout.
**Path A column = per-image LM, n = 2000; B columns n = 800.**

| | Path A | B3 v1 (broken) | **B3 μ** | **B3 uniform** | B2 |
|---|---|---|---|---|---|
| θ_E ρ | +0.956 | +0.956 | **+0.960** | +0.960 | +0.949 |
| β ρ | +0.914 | +0.228 | **+0.826** | +0.835 | +0.641¹ |
| R_sersic ρ | +0.924 | +0.147 | **+0.953** | +0.956 | n/a¹ |
| n_sersic ρ | +0.944 | −0.014 | +0.475 | +0.476 | n/a¹ |
| slope γ ρ | +0.571 | +0.193 | +0.132 | +0.083 | +0.205 |
| **lens \|e\| ρ** | **+0.762** | +0.313 | **+0.269** | +0.256 | +0.130 |
| \|g\| ρ | +0.611 | +0.142 | +0.142 | +0.143 | +0.083 |
| source corr | 0.9865 | 0.9132 | 0.9566 | 0.9569 | 0.860² |
| source size_ratio | 1.0887 | 1.3421 | **0.9493** | 0.9699 | 0.682² |
| source peak_ratio | 0.8208 | 0.5175 | **0.9905** | 0.9526 | 1.814² |
| source nmse (Sérsic only) | **0.0271** | 0.1677 | 0.0845 | 0.0832 | 0.261² |
| source nmse (**+ correction**) | — | — | **0.0799** | 0.0789 | — |
| χ²/dof, plain σ | 3085 | 12,998 | 9430 | 9732 | 12,921 |
| χ²/dof, 2% floor | **210** | — | **337–346** | 337–346 | 402–417 |
| ms / image | 877 | 12 | **12** | 12 | 12 |

¹ B2 has no Sérsic, so its β/R_sersic/n_sersic entries are *unused network outputs*, not
predictions. β correlating at +0.641 is an artefact of the |m₁| anchoring: sx = 2|m₁|tanh(z), so
even an untrained z inherits the ring's correlation with the true offset.
² `evaluate.py` **cannot** score B2 — it builds a Sérsic from those same unused parameters. The
real B2 source numbers are `source_truth_corrected` in `fits_pb2.json`.

The B3-μ, B3-uniform and B2 columns were re-read from the recorded `evaluate.py` output during
the writing of this document, and the source rows were additionally recomputed from the
`source_truth_base` / `source_truth_corrected` blocks of `fits_pb3_mu.json`,
`fits_pb3_uniform.json` and `fits_pb2.json`. All match **[verified]**. The "2% floor" χ² row is
**[reported: RESULTS_V3.md]** and was not independently recomputed — it is the one row in this
table I cannot vouch for personally.

**Three pieces of fine print that must not go into a paper unflagged:**

1. **The two χ² columns are not on the same denominator.** `eval_pathb.py` divides by
   (6361 − 14 − 1024) = 5,323 while Path A divides by (6361 − 14) = 6,347 — a factor **1.192**
   **[verified]**. So B3's plain-σ 9,430 is inflated ~19% relative to Path A's 3,085 by
   bookkeeping alone. Whether counting 1,024 bounded, penalised correction pixels as full degrees
   of freedom is right is arguable; either way, say which convention you used.
2. **"source nmse" appears twice and means two different things.** `evaluate.py` §2 scores the
   **Sérsic only** (0.0845), because it rebuilds the source from the parameters. `eval_pathb.py`
   §A scores **Sérsic + correction** (0.0799). Quoting one where the other belongs would misstate
   the entire super-resolution claim.
3. **All of it is axion-only** (§1.4).

### 6.11 What improved, and by how much

The v2/v3 loss fixes worked. B3 went from a round blob to a genuine reconstruction:

- **β +0.228 → +0.826**; **R_sersic +0.147 → +0.953** — *better than Path A's +0.924*;
  n_sersic −0.014 → +0.475
- source nmse **0.168 → 0.085** (2× better); corr 0.913 → 0.957
- **size_ratio 1.342 → 0.949 and peak_ratio 0.518 → 0.991** — both now *closer to 1 than Path
  A's* (1.089 and 0.821). The network's source is less biased in size and peak than the
  optimiser's, which is a real and slightly surprising result.
- tanh saturation **75% → 0.0%**; train/val gap **2.9× → 0.97×**; val χ² flat from epoch ~42
- on the metric it actually optimised (2% floor), B3 is **1.6×** Path A (346 vs 210), not the
  3.1× the plain-σ column suggests — that column is dominated by a handful of bright arcs.

**And the correction now helps**, which is the super-resolution claim in one line
**[verified from `fits_pb3_*.json`, medians over 800 images]**:

| | corr | size_ratio | peak_ratio | **nmse** |
|---|---|---|---|---|
| B3 μ, Sérsic only | 0.9566 | 0.9493 | 0.9905 | 0.0845 |
| B3 μ, **Sérsic + correction** | 0.9591 | 0.9373 | 1.0068 | **0.0799** (−5.4%) |
| B3 uniform, Sérsic only | 0.9569 | 0.9699 | 0.9526 | 0.0832 |
| B3 uniform, **Sérsic + correction** | 0.9593 | 0.9566 | 0.9595 | **0.0789** (−5.2%) |

at a correction amplitude of **3.81% / 3.17% of the Sérsic** and **0.00% saturation**. That is
precisely the regime the design was aiming for: small, unsaturated, and helping. Note the
correction also moves `peak_ratio` from 0.9905 to 1.0068 — i.e. it sharpens the core, which is
what a super-resolving correction should do.

### 6.12 What is still wrong: both ellipticities are shrunk ~4×

```
lens   |e|    network 0.049    Path A 0.192    truth 0.222     rho +0.269 vs +0.762
source |se|   network 0.065    Path A 0.245    truth 0.217     rho +0.326 vs +0.874
```

**It is not an orientation failure.** Decompose into modulus and phase:

| | modulus ρ | median phase error (random = 45°) |
|---|---|---|
| B3 lens \|e\| | +0.269 | 19.0° |
| B3 shear \|g\| | +0.142 | 24.9° |
| Path A lens \|e\| | +0.776 | 3.9° |
| B3 β (a vector) | +0.826 | 9.3° direction error (random = 90°) |

The network gets the *direction* roughly right and the *magnitude* badly wrong. And the shrinkage
is a flat factor of ~0.22 in **every** phase-error bin:

```
   phase-err bin    n   median |e| pred   median |e| true   ratio
         0-5 deg  120           0.0576            0.2683    0.21
        5-15 deg  221           0.0536            0.2380    0.23
       15-30 deg  200           0.0503            0.2184    0.23
       30-90 deg  259           0.0393            0.1775    0.22
```

**This is what a squared loss is supposed to do.** A network trained on χ² is a
**conditional-mean estimator**: E[e | image]. For a spin-2 quantity under orientation
uncertainty, a large ellipticity pointed the wrong way costs *more* than no ellipticity at all,
so the expected-error-minimising answer shrinks the modulus toward zero. The LM fit is a
**maximum-likelihood** estimator on one image and has no such incentive. **Both are behaving
exactly as their objectives specify.** More epochs will not fix this; it is a property of the
estimator, not of the optimisation.

Three principled routes out, in increasing ambition: (i) use the network as an initialiser and
let LM finish — **already demonstrated, §8**; (ii) train a *likelihood* rather than a point
estimate (a density head, or a normalising flow over the 14 parameters) so the shrinkage becomes
a reported width rather than a bias; (iii) unroll a few optimisation steps inside the network
(a Recurrent Inference Machine, Adam et al. 2022), which makes the network an *optimiser* rather
than a regressor.

### 6.13 The B2 control, and why it is not a fair test

B2 = `--source-mode b2`: `source = amp · softplus(C)`, no Sérsic. Same physics, same optimiser,
same data, same code path — so B2-vs-B3 is a controlled experiment.

It came out badly (source corr 0.86, size_ratio 0.68, peak_ratio 1.81), and **the run is
mis-configured, not the design refuted**. Three reasons, in order of importance:

**(a) The "Sérsic only" panel is zero by construction.** In b2 `eval_pathb.source_maps()` returns
a zero map for `base`. That panel is *supposed* to be blank, and it is why `evaluate.py` reports
nonsense (source corr 0.7793) — it renders a Sérsic from parameters the model never used.

**(b) The map is 17×17 px inside a 127×127 frame.** ±0.8″ = ±7.55 px covers **1.8%** of the
panel, so the figure's percentile stretch computes p1 = p99.5 = 0 over a 98%-zero image and the
panel collapses to near-black with a tiny saturated square.

**(c) It is genuinely truncated.** Measured on 200 images, the fraction of true source flux
inside a ±0.8″ box: **p10 0.426, median 0.722, p90 0.927**. **28% of the source flux, at the
median, cannot be represented at all.** In B3 that costs nothing because the Sérsic carries the
wings and the map only adds a core correction. **In B2 the map *is* the source.**

**And here is the trap.** A fair B2 needs ~±2.5″. Keeping the same 0.0516 ″/px resolution over
±2.5″ needs n_c ≈ 98 → **9,604 unknowns against ~4,300 data pixels = 2.2:1 underdetermined** —
straight back into the regime that produced size_ratio 10.6–12.8 in the original pipeline. B3 at
±0.8″/32 is 4.1:1 *over*determined.

> **That trade-off is the actual argument for the parametric+correction split: free-form must
> choose between coverage and resolution; the hybrid does not have to.**

This is a *result*, not an excuse — and it is one of the more quotable sentences in the project.

### 6.14 The magnification ablation: a negative result, twice

Now that λ is scale-free, the ablation is finally valid. **All four columns below were
recomputed from `fits_pb3_mu.json` and `fits_pb3_uniform.json` for this document [verified]:**

| | correction rms | saturation | corr(\|C\|, log μ) | source nmse |
|---|---|---|---|---|
| B3 μ | 3.81% | 0.0% | **+0.575** | 0.0799 |
| B3 uniform | 3.17% | 0.0% | +0.459 | 0.0789 |

The μ weighting **does work mechanically** — it concentrates the correction more strongly in
high-magnification regions (+0.575 vs +0.459) — and it produces **no measurable benefit**
(0.0799 vs 0.0789, i.e. 1.3% *worse*). The two networks remain nearly identical
(Pearson +0.9999 on θ_E, +0.976 on e₁).

**This is the second independent negative result on the same idea.** Option 2's linear inversion
found magnification-adaptive regularisation ~4% worse than uniform at matched data fidelity,
across every fidelity level.

Both should be reported. A well-motivated idea that measurably does not pay, tested twice with
two completely different estimators, is a legitimate contribution — and it is *falsifiable*,
which is more than most positive results in this space.

**Why it might not pay here, worth one sentence in the paper:** median ray coverage is low, so
most of the source plane sits in the regime where the weighting does nothing useful; and the
high-μ strip beside the caustic is thin, so relaxing smoothing there mostly admits noise. On a
dataset with a genuinely clumpy source and higher magnification it could still win — that is a
testable prediction, not a defence.

### 6.15 The evaluation protocol, and why it is designed this way

`eval_pathb.py` writes its results in **exactly the schema of `fits_img.json`**, the file produced
by the per-image LM fit. So the network is scored by the *same code* that scored the optimiser:

```bash
python superres/eval_pathb.py --ckpt superres/results/pb3_mu_best.pt --root . --out .../fits_pb3_mu.json
python superres/evaluate.py   --fits superres/results/fits_pb3_mu.json --root .
```

That is deliberate. If the network were scored by a bespoke script, any difference from the
per-image fit could be a difference *in the scoring* rather than in the model, and there would be
no way to tell which.

On top of that, `eval_pathb.py` reports four things `evaluate.py` cannot know about:

- **§0 Collapse check.** Across-image standard deviation of every predicted parameter against the
  truth's. The previous amortised network emitted the *same numbers for every image* and its
  medians still looked reasonable. A ratio < 0.05 is flagged as COLLAPSED and the recovery table
  is declared unreportable.
- **§0b Ring-only control.** Rank correlation of the *input ring statistic alone* with the truth,
  next to the network's own prediction, with an explicit verdict of `network ADDS` /
  `no gain (echoes input)` / `network LOSES info`. This is the control that matters most, because
  four measured ring statistics are fed in as inputs.
- **§A The correction's contribution.** The source is scored twice, Sérsic-only and
  Sérsic+correction, on the same images with the same PSF. **The difference is the entire
  super-resolution claim, isolated.** If it is zero the network has learned nothing beyond the
  parametric model, and the honest thing is to say so.
- **§C Where the correction went.** corr(|correction|, log ray coverage). Positive means the
  network added detail where the lens actually delivered resolution; near zero means it spread
  detail uniformly, i.e. it is inventing texture.
---

## 7. B4 — the proposed hybrid (designed, not built)

### 7.1 Where the idea comes from

The repository this project started in is Anirudh Shankar's grid-based unsupervised
super-resolution work (`sisr.py`, `train.py` at the repo root; ML4PS @ NeurIPS 2024 with Toomey
and Gleyzer). His loop:

```
lr_image ──backward operator──▶ reconstructed_source (crude, IN THE SOURCE PLANE)
                                        │
        cat([reconstructed_source, lr_image]) ──▶ SISR (fully conv + PixelShuffle)
                                        │
                                   fine_source (HR)
                                        │
        forward operator ─▶ PSF ─▶ downsample ─▶ weighted MSE vs lr_image
```

**The self-supervision idea is the same as ours** — no HR target, the loss is the re-degraded
prediction against the observed LR image. Two things differ, one in his favour and one in ours:

| | Anirudh | ours (B2/B3) |
|---|---|---|
| network input | the image **already back-projected into the source plane** | the raw image |
| architecture | fully convolutional, residual blocks + PixelShuffle, **spatial correspondence preserved end to end** | conv trunk → **global average pool → 128-vector** → decode a 32×32 map |
| lens | fixed SIS, precomputed sparse operator bank, θ_E from the manifest | EPL + shear, **fitted per image, in-graph, analytic** |
| source-plane validation | not available (Model_IV has no truth) | size_ratio 0.95, corr 0.96 against stored truth |

**His architecture is the better super-resolver; ours is the better lens model.** His network
never has to learn the geometry — it is applied analytically before and after, so the SISR only
has to deblur and upsample a *spatially registered* image, which is exactly what SRResNet-style
networks are good at. Ours squeezes everything through a global pool and then hallucinates a
spatial map from a 132-dimensional summary. Recall that **64% of B3's weights are in that one
dense layer** (§6.4). That is almost certainly why the correction map only reaches 3–4% and why
n_sersic and |e| stay weak.

### 7.2 The design

```
1. predict the lens (14 params) with the current trunk   -- OR take it from Path A
2. back-project the observed image through THAT lens into the source plane
     (this is exactly the L^T operator already written in Option2_failed/pixel_source.py)
3. feed cat([back_projection, image]) into sisr.SISR
     -- fully convolutional, PixelShuffle x2, NO global pool anywhere
4. forward-lens with our analytic differentiable EPL+shear -> PSF -> bin -> chi^2
     with the 2% sigma floor
```

This keeps his spatial architecture and our correct, per-image, differentiable lens. **Every
piece already exists in the repo**: the adjoint operator Lᵀ (validated to 7.8e−15 in
`test_pixel_source.py`), the analytic forward model, the σ-floor loss, the shared scorer.

### 7.3 Why it should work, and what would falsify it

**The hypothesis:** B3's correction is small because the decoder cannot place structure, not
because the data cannot support it. **The test:** B4 should raise the correction amplitude and
the Δnmse without raising the tanh saturation, and should improve n_sersic and the source
peak/size ratios. **The falsifier:** if B4's correction is also ~3–4% and Δnmse is also ~−5%, then
the data genuinely do not contain more, and B3's small correction was the right answer all along
— which would itself be a clean, publishable statement about the information content of the
source plane at Roman resolution.

**Cost:** roughly one day. **Priority:** *not* the highest-value use of the remaining time before
the deadline — see the strategy document. It is the most interesting remaining idea and the most
likely to improve the source map, but it is a new experiment, and a new experiment eight days
before a deadline is how papers get withdrawn.

---

## 8. The result I would lead with: amortised initialisation + per-image refinement

`superres/refine_pathb.py`. **This is the piece that turns Path B from "the network is worse" into
a contribution.**

Stop asking the network to *be* the estimator (§6.12 says it structurally cannot be). Use it for
what it is good at — producing, in 12 ms, a starting point far better than any hand-written one —
and let LM do the last mile.

**Measured, n = 25 [reported: RESULTS_V3 §3; `_refine_test.json` on disk confirms n = 25]:**

```
                       method  model evals  LM seconds  total ms/img   chi2/dof
                network alone            -           -          12.0       6817
           cold LM (4 stages)          362       0.842         842.5       2995
            network + warm LM          136       0.432         444.2       2995

   speedup 1.95x wall clock, 2.66x model evaluations,  chi2 ratio 1.000
```

**Identical χ² — and better parameters than the cold fit on five of seven:**

```
   parameter      network   cold LM   warm LM     truth  |   net rho  cold rho  warm rho
   theta_E         1.4798    1.4867    1.4867    1.4100  |   +0.964    +0.953    +0.943
   beta            0.2398    0.3223    0.3064    0.3327  |   +0.814    +0.937    +0.944
   e               0.0442    0.1860    0.1860    0.2162  |   +0.075    +0.848    +0.849
   g               0.0367    0.0715    0.0542    0.0366  |   -0.049    +0.477    +0.668
   R_sersic        0.3843    0.4792    0.4480    0.4432  |   +0.922    +0.848    +0.958
   n_sersic        1.1231    1.0803    1.0765    0.9908  |   +0.310    +0.842    +0.988
   gamma           1.8150    2.0126    2.0126    2.0504  |   +0.255    +0.370    +0.378

   source:            corr  size_ratio    nmse   peak_ratio
   network alone    0.9464      0.9580  0.1033      1.0517
   cold LM          0.9810      1.1214  0.0373      0.7560
   network + warm   0.9831      1.0939  0.0332      0.7556
```

Three readings:

1. **The ellipticity shrinkage is completely undone**: +0.075 → +0.849. The conditional-mean bias
   lives in the network's *output*, not in its usefulness as a starting point.
2. **The warm start needs ONE stage instead of four.** The staged schedule in `fit_per_image.py`
   exists only because a neutral start falls into local minima. From the network's start it is
   unnecessary — which is why warm *beats* cold on g, n_sersic and R_sersic: **a better basin,
   not just a faster route to the same one.**
3. χ² ratio 1.000 — the refinement is not cheaper by being worse.

**The claim to put in the paper:**

> Amortised inference supplies an initialisation good enough to eliminate the staged-optimisation
> schedule and halve the cost of the per-image fit at identical final accuracy, while improving
> parameter recovery on five of seven parameters.

**Not** "the network replaces the fit." It does not, and the ellipticity numbers say so plainly.

**Status: n = 25 only.** `refine_pathb.py --n 800` is ~15 minutes and is the single highest-value
command still unrun. (An older n = 14 run is quoted in the file's docstring with slightly
different numbers; use the n = 25 table above, or better, re-run at 800 and use that.)

---

## 9. The validation ladder — every test and what it asserts

`superres/tests/run_all.py` runs five checks that need no torch (~3 min). All must PASS; if any
fails, everything downstream is meaningless.

| test | asserts | measured |
|---|---|---|
| `test_lens_models.py` | EPL+shear deflection vs `lenstronomy` 1.14.2; hypergeometric truncation vs `scipy.special.hyp2f1`; hessian sign convention | **≤ 3e−15** |
| `test_sources.py` | elliptical Sérsic vs `lenstronomy SERSIC_ELLIPSE` | **0.00e+00** (exact) |
| `test_jacobian.py` | `magnification_autograd` vs `hessian_analytic` | **≤ 2e−9** |
| `test_raytrace.py` | supersampling error vs an S = 9 reference; flux conservation of `area_downsample` | **S=3 → 5.5e−3** |
| `test_theta_e_init.py` | ring estimator against the manifest | **~0.45 px median** |
| `test_backend_parity.py` (torch) | identical physics functions under numpy and torch | **1e−6** |
| `test_pathb.py` (torch, 5 checks) | (1) torch forward model = numpy `render()` → **5.5e−16**; (2) shapes/ranges of `to_params`; (3) `ray_coverage` does not put maximum coverage on the map border; (4) `sample_correction` alignment; (5) **no NaN gradients on step one** | all PASS |

Check 5 of `test_pathb.py` is the one that has repeatedly earned its keep: it caught the atan2
re-introduction of the ellipticity singularity (§6.5) immediately.

`backend.py` exists for the same reason: the physics is written **once** against eleven
elementwise operations, and `test_backend_parity.py` runs the identical function under numpy and
torch requiring 1e−6 agreement. Writing the lens twice guarantees the two copies drift — and a
silent drift in a deflection field is precisely the class of bug the operator sign audit already
caught once (three of five mapping directories had forward and backward operators with the *same*
sign, geometrically impossible, and **invisible in the loss**).

---

## 10. Reproduction — exact commands, in order

All from the **`Grid_Based_Experiment` root**, not from inside `superres/`.

```bash
pip install numpy scipy lenstronomy matplotlib     # torch only for Path B

# 0. sanity-check the physics  (~3 min)   -- if anything FAILS, stop
python superres/tests/run_all.py

# 1. measure the PSF           (~3 min)   -- output already in results/
python superres/calibrate_psf.py --root . --n 40 \
       --save superres/results/psf_empirical.npy

# 2. OPTION 1 / PATH A: the per-image fit  (~30 min for n=2000 at 0.88 s/image)
python superres/fit_per_image.py --root . --n 2000 --classes axion cdm wdm \
       --target image --psf-mode empirical --supersample 1 \
       --out superres/results/fits_img.json

# 3. score it against truth    (~1 min)
python superres/evaluate.py --fits superres/results/fits_img.json --root .

# 4. super-resolve and validate at 1x/2x/4x/8x  (~2 min)
python superres/superresolve.py --fits superres/results/fits_img.json --root . \
       --factors 1 2 4 8 --n-save 6

# 5. magnification extraction  (~10 min at n=2000)
python superres/magnification_extract.py --fits superres/results/fits_img.json --root .

# 6. figures
python superres/make_figures.py --root .
python superres/make_figures_magnification.py --root .

# ---- PATH B ----
python superres/tests/test_pathb.py          # five gate checks, run BEFORE training

python superres/train_pathb.py --root . --classes axion cdm wdm \
    --n-train 6000 --n-val 800 --epochs 50 --warmup-epochs 5 \
    --sigma-floor 0.02 --lambda-corr 3.0 --augment 1 \
    --reg-mode mu --out superres/results/pb3_mu.pt

python superres/train_pathb.py ... --reg-mode uniform --out superres/results/pb3_uniform.pt
python superres/train_pathb.py ... --source-mode b2 --lambda-curv 3.0 --warmup-epochs 0 \
    --out superres/results/pb2.pt

python superres/eval_pathb.py --ckpt superres/results/pb3_mu_best.pt --root . \
    --out superres/results/fits_pb3_mu.json --out-npz superres/results/pb3_mu.npz
python superres/evaluate.py --fits superres/results/fits_pb3_mu.json --root .
python superres/make_figures_pathb.py --pathb superres/results/fits_pb3_mu.json \
    --npz superres/results/pb3_mu.npz --ckpt superres/results/pb3_mu.pt --root .

# ---- the amortised-initialisation result ----
python superres/refine_pathb.py --pathb superres/results/fits_pb3_mu.json --root . --n 800
```

**Two housekeeping items before anything is submitted:**
`figB1_mu_reconstruction.png` and `figB1_uniform_reconstruction.png` (timestamps 13:07, 13:28) are
from the **failed v1 run**; the v3 ones are `*_mu3_*` and `*_uniform3_*`. Delete the stale pair so
they cannot end up in the paper. And add a line to `eval_pathb.py`'s b2 branch noting that
`evaluate.py` §2 is invalid for `--source-mode b2`.

---

## 11. Honest ledger

### Validated by measurement

| claim | evidence |
|---|---|
| EPL+shear deflection matches lenstronomy | 2.7e−15, `test_lens_models.py` |
| Sérsic matches lenstronomy exactly | 0.00e+00, `test_sources.py` |
| torch forward model = numpy forward model | 5.5e−16, `test_pathb.py` check 1 |
| autograd magnification = analytic Hessian | ≤ 2e−9, `test_jacobian.py` |
| Model_A generator = lenstronomy EPL+SHEAR+SERSIC_ELLIPSE | κ ratio 0.98 / log-corr 0.936; source corr 0.967, 0.9989 with PSF |
| SIS explains Model_A arcs at 0.87; SIE+shear at 0.997 | analytic ladder, n = 60 |
| Model_A PSF is 0.18–0.20″, not 0.10″, with Moffat wings | two independent estimates |
| The ring θ_E estimator is accurate to ~0.48 px median | 400 images vs manifest |
| θ_E is **fitted**, not read — the unsupervised leak is closed | `fit_per_image.py` never opens a truth array |
| Source size_ratio 10.65 → **1.089**, corr → **0.987** | n = 2000, `superres_metrics.json` |
| Seven parameters recovered, ρ = +0.57 to +0.96 | n = 2000, recomputed for this document |
| Super-resolution fidelity is **flat** across 1×–8× | n = 2000 |
| Critical curve recovered to 0.63 px; swing 0.540 vs 0.548″ | n = 2000 |
| μ_tot recovered to ~1% median | n = 2000 |
| `image_nss`/`kappa_nss` were rendered with a circular lens | phase test, 1.6° vs 44.3° |
| `skill` is anti-correlated with SNR at ρ = −0.85 | null-baseline study |
| v1 Path B failed because χ² was unnormalised | gradient-share and χ²-percentile measurements |
| B3 does not collapse; every parameter varies | `eval_pathb.py` §0 |
| B3 correction helps (−5.4% nmse) at 3–4% amplitude, 0% saturation | `eval_pathb.py` §A |
| μ-adaptive regularisation gives no benefit — **twice** | Option 2 (−4%) and B3 (−1.3%) |
| Network warm-start halves LM cost at identical χ² | n = 25 |

### Not validated / open

| item | status |
|---|---|
| **Cross-class generalisation (cdm, wdm)** | **untested — every run is axion-only (§1.4)** |
| `refine_pathb` at n = 800 | not run; n = 25 only |
| Whether B4 helps | designed, not built |
| Substructure recovery from the residual | not attempted; the residual figure is qualitative |
| Real-survey performance (HSC/HST/DES) | not attempted |
| Whether the correction can recover **injected** sub-Sérsic structure | not attempted — this is the single most convincing missing experiment |
| A source-plane uncertainty / posterior | not attempted |
| Model_3 (HST-like) or Model_4 re-run with the new pipeline | not attempted |

---

## 12. The limits to state before a reviewer finds them

1. **Super-resolution within the model family.** A Sérsic can only make Sérsic-shaped things.
   Model_A's sources genuinely are Sérsics, so the comparison here is exact — and that is
   *also* the honest ceiling: on this dataset the correction map cannot find real sub-structure
   that the data do not contain. The claim is "the method learns a source representation finer
   than the detector, constrained only by the lensing physics, and the magnification field
   determines where that extra resolution is admissible; on Model_A the correction stays small,
   as it should when the true source is parametric — which is itself evidence the regularisation
   is calibrated rather than free to hallucinate."
2. **μ is derived from the fitted lens**, not measured independently (§5.9).
3. **χ²/dof ≈ 3000 is a PSF-shape systematic**, not a parameter failure — and the fact that it
   *rises* with SNR (ρ = +0.449) is the signature that proves it.
4. **θ_E (+3.5%), R_sersic (+10%), n_sersic (+17%), peak_ratio 0.82** are all biased the same way,
   all consistent with under-modelled PSF wings.
5. **B3's ellipticity shrinkage is a property of the estimator**, not of the training.
6. **The B2 run is mis-configured** and should be presented as such, not as a refutation of
   free-form sources.
7. **Everything is simulation.** No real data has been touched. The domain-gap experiment
   (degrade Model_A to HSC sampling and seeing, show recovery survives, *then* real cutouts) is
   designed but not run.
8. **Axion-only** (§1.4).

**The one experiment that would most improve the paper:** inject sub-Sérsic structure — a clump,
a spiral arm — into a rendered source, run the whole pipeline, and show the correction map
recovers it while the pure Sérsic cannot. It is an afternoon of work with `raytrace.render()`,
and it converts "the correction stays small, as it should" from a defence into a demonstration.

---

## 13. Questions I expect, with answers

**"Is this actually unsupervised?"**
Yes, and the leak that made the previous version not-unsupervised is documented and closed. The
loss never sees a high-resolution target, a source truth or a manifest value. θ_E and β are
initialised from the ring geometry *of the image itself* (`polar_ridge` + `fourier_rphi`);
everything else starts at a population-neutral constant. Truth is loaded only by `evaluate.py`,
after the fit is written to disk. For Path B the four ring statistics are network *inputs*, all
measured from pixels — and `eval_pathb.py` §0b prints the ring-only baseline so the network has
to *beat its own inputs* to have added anything.

**"Where does the extra resolution come from? Isn't it just interpolation?"**
Three separate answers. (i) The fitted source is a continuous analytic function; sampling it more
finely adds no error of its own, which is *measured* — corr is flat to 0.0017 across 1×→8×.
(ii) The lens itself is the super-resolving element: the measured median tangential stretch is
3.07×, so the sky has already spread each source patch over ~3 detector pixels. (iii) B3's source
grid is 2.05× finer than the detector, and the code **warns** if you ask for more than the
measured stretch supports. Asking for a finer grid than the lens can deliver is interpolation
dressed up as physics, and the startup banner says so.

**"χ²/dof is 3000. Doesn't that mean the model is wrong?"**
The model is wrong in a known, measured way: the PSF *shape*. Arcs reach peak/σ_bg of 10³–10⁴, so
a 1% kernel error is ~50σ per pixel. The signature that distinguishes a systematic from noise is
that χ² *rises* with SNR — measured ρ = +0.449 — and it does. The null baselines (constant 37,511;
2-px blur 2,796; 3-px blur 4,091) put 3,085 at the level of a good matched blur, while also
recovering seven physical parameters. Also, some of the residual is *supposed* to be there: it is
the substructure, at rms 4.33 against σ_bg 0.032.

**"Why is the network worse than the optimiser? Train it longer."**
More epochs will not fix it, and I can show why. The shrinkage factor is a flat ~0.22 in every
phase-error bin, which is the signature of a conditional-mean estimator on a spin-2 quantity, not
of under-training. The fix is to change what is being estimated (a likelihood, or an unrolled
optimiser), or to use the network as an initialiser — which is measured to give identical χ² at
half the cost and better parameters on 5/7.

**"What did the magnification weighting buy?"**
Mechanically it works — it concentrates the correction where μ is high (+0.575 vs +0.459). On
this data it buys nothing measurable (1.3% worse), and an independent linear-inversion experiment
found ~4% worse. I would report both as a negative result with a stated mechanism, not bury them.

**"Why should I believe the lens is right and not just fitting noise?"**
Because it reproduces things it was never fitted to: the critical curve's *azimuthal swing*
(0.540″ recovered vs 0.548″ true — a circular lens has swing exactly zero), the total
magnification to ~1%, and the per-pixel log-μ map at correlation 0.93. And because the source it
implies matches the stored `unlensed` array at correlation 0.987 with size_ratio 1.089.

**"How does this compare with the previous DeepLense super-resolution work?"**
Same self-supervised premise (no HR target), three differences: the lens is fitted per image as
EPL+shear rather than fixed circular SIS with θ_E from the manifest; the evaluation is against
*source-plane ground truth and physical parameters* rather than image-plane MSE/SSIM/PSNR (and
§5.7 shows why image-plane metrics on noisy targets are dangerous — a 3-px blur of the input beats
the exact physical model); and the resolution claim is tied to a *measured* magnification field
rather than to a chosen upscaling factor.

**"What would you do with another month?"**
In order: (1) fix the class sampling and re-run to cover cdm/wdm; (2) the injected-clump
experiment; (3) B4; (4) the HSC domain-gap experiment on degraded Model_A, which is fully
controlled because there is truth on both sides; (5) a likelihood head so the shrinkage becomes a
reported uncertainty rather than a bias.

---

## Appendix A. Symbol and parameter glossary

| symbol | meaning |
|---|---|
| θ | image-plane angular position (what we observe) |
| β | source-plane angular position; also used for \|source offset\| = hypot(sx, sy) |
| α | scaled deflection angle; β = θ − α(θ) |
| ψ | lensing potential; α = ∇ψ |
| κ | convergence = ½∇²ψ; projected mass in units of Σ_cr |
| γ₁, γ₂ | shear (spin-2); here `g1, g2` denote *external* shear |
| θ_E | Einstein radius (″) |
| γ (`gamma`) | EPL radial slope; κ ∝ r^(1−γ); γ = 2 is isothermal |
| e₁, e₂ | lens ellipticity, spin-2; \|e\| = hypot; q = (1−\|e\|)/(1+\|e\|) |
| se₁, se₂ | **source** ellipticity |
| q | axis ratio, minor/major |
| b | EPL normalisation = θ_E√q |
| t | EPL exponent = γ − 1 |
| μ | magnification = 1/det A |
| λ_t, λ_r | tangential/radial eigenvalues of A: 1 − κ ∓ \|γ\| |
| A | lens Jacobian ∂β/∂θ |
| R_sersic | Sérsic half-light radius (″) |
| n_sersic | Sérsic index |
| b(n) | Sérsic normalisation, 1.9992n − 0.3271 (lenstronomy's form, deliberately) |
| σ_bg | background rms from the r > 50 px annulus |
| σ_eff | √(σ_bg² + (0.02·model)²) — Path B only |
| C | the 32×32 source-plane correction map (raw network output) |
| C_eff | corr_scale · amp · tanh(C) — the correction actually applied |
| n_c | correction map side (32) |
| half_extent | source-plane half-width (0.8″) |
| coverage_j | rays landing in source pixel j = the magnification of that pixel |
| ρ | Spearman rank correlation |
| size_ratio | recovered source rms size / true rms size; target 1 |
| nmse | amplitude-matched normalised mean square error |

## Appendix B. File map

**`superres/` — the deliverable, self-contained**

| file | role |
|---|---|
| `backend.py` | numpy/torch shim so the physics is one source file |
| `lens_models.py` | EPL/SIE/SIS + shear deflection, Hessian, magnification (both routes) |
| `sources.py` | elliptical Sérsic; `PixelSource` stub |
| `raytrace.py` | source → lens → PSF → detector; Gaussian/Moffat/empirical kernels |
| `data_a.py` | Model_A reader; raw units; per-image σ; **truth is evaluation-only** |
| `theta_e_init.py` | ring geometry → θ_E and β; closes the unsupervised leak |
| `metrics.py` | χ²/dof, PSF-matched source truth, parameter recovery, SNR stratification, null baselines |
| `calibrate_psf.py` | measures the PSF two ways |
| **`fit_per_image.py`** | **Option 1 / Path A — the 14-parameter joint fit** |
| `evaluate.py` | the shared scorer; **truth enters only here** |
| **`superresolve.py`** | **render the fitted source at 1×/2×/4×/8× and validate each** |
| **`magnification_extract.py`** | μ map, μ_tot, critical curve, resolution gain |
| `make_figures.py`, `make_figures_magnification.py` | figures, from files on disk |
| **`train_pathb.py`** | **Path B: model, forward model, μ-weighted loss, training loop** |
| **`eval_pathb.py`** | collapse check, ring-only control, correction contribution, speed |
| **`refine_pathb.py`** | **network → warm-start LM: the amortisation result** |
| `make_figures_pathb.py` | figB1–figB4 |
| `tests/` | `run_all.py` + seven test files |
| `Option2_failed/` | the retired free-form linear inversion, with its negative results |

**Documentation on disk**

| file | content |
|---|---|
| `superres/README.md` | Option 1 / Path A: result, files, run order, expected numbers |
| `superres/PATHB_AND_MAGNIFICATION.md` | magnification extraction + the B3 design and thresholds |
| `superres/PATHB_POSTMORTEM.md` | the v1 failure, measured |
| `superres/RESULTS_V3.md` | the v3 scoreboard, the hybrid, B4, recommendations |
| `superres/Option2_failed/OPTION2.md` | Option 2 and its two negative results |
| `sie_pipeline/MENTOR_BUGREPORT.md` | the `_nss` circular-lens dataset defect |
| `sie_pipeline/IMPLEMENTATION.md` | the full account of the rewrite |
| `diagnostics_2026_08_12/PROJECT_REPORT.md` | the 60-page diagnosis of the previous pipeline |
| `diagnostics_2026_08_12/EXPLAINER.md`, `DIAGNOSIS.md`, `BANK_EXPLAINER.md` | supporting material |
| `STAGE_D_E_FIXED_SIS.md` | the earlier fixed-SIS magnification work |
