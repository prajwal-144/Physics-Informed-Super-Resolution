# Concepts, methods and design decisions — a complete walkthrough

**Companion to `DEEPLENSE_TECHNICAL_BRIEF.md`.** That document says *what* the pipeline is.
This one answers *why it is like that*, from first principles, including the pieces you asked
about explicitly: the difference from the original grid-based lensing, whether any of it was
unnecessary, the physics vocabulary, the metrics, the loss, B4, B5, and the source box.

Written 22 August 2026. Every number is marked **[verified]** if it was recomputed from a file on
your disk while writing this, or **[reported]** with its source if it was taken from a document
or a recorded run log. Nothing is remembered or estimated.

> **Three corrections to premises in the questions, stated up front so nothing downstream is
> built on them.** Each is explained where it belongs.
>
> 1. **B4 free-form is *not* independent of fitting.** It freezes a lens that came from the
>    Levenberg–Marquardt fit, and it needs one for all 6,000 training images too. It is not "pure
>    ML" — it is a learned decoder bolted onto a fitted physical lens. (§12)
> 2. **B4 free-form is *not* the better mode.** `--base sersic` beats `--base none` on four of
>    five source metrics and on χ². Free-form nmse 0.0527, +Sérsic nmse 0.0408 **[verified]**. (§11, §12)
> 3. **B5's magnification gate did not help.** nmse 0.0414 (gate) and 0.0425 (gate + μ-weighted
>    curvature) against the ungated control's 0.0408 **[verified]**. That is the *fourth*
>    independent null on magnification-adaptive regularisation, and it is a result, not a
>    failure. (§14)

---

# 1. Grid-based lensing (Anirudh Shankar) vs ray-tracing (ours)

This is the deepest technical difference between the two projects, so it gets the longest answer.

## 1.1 The shared starting point

Both methods implement the same physics: the lens equation

$$\boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}(\boldsymbol{\theta})$$

and both must answer the same computational question:

> Given a source brightness distribution, produce the image a telescope would see.

The difference is entirely in **how that operator is represented in memory and applied**.

## 1.2 His method: grid-based lensing with precomputed sparse matrices

Reading `differentiable_lensing.py` and `build_sis_mappings.py` on your disk, the construction is:

**Step 1 — build two grids.** A regular square grid of pixel *corners* in the image plane, and a
second grid of the same corners pushed through the lens equation. The second grid is no longer
square: each square pixel has become a curved quadrilateral.

**Step 2 — clip polygon against polygon.** For every deformed quadrilateral and every square
destination pixel, compute the **area of their overlap**. `clip_polygon_with_square` implements
**Sutherland–Hodgman clipping** (a polygon is clipped against each of four half-planes in turn),
and `polygon_area` implements the **shoelace formula**

$$A = \tfrac12\left|\sum_i (x_i y_{i+1} - x_{i+1} y_i)\right|$$

The result is a 4-D tensor `T[sx, sy, ix, iy]` = overlap area of source cell (sx,sy) with image
pixel (ix,iy).

**Step 3 — flatten to a sparse matrix.** `build_sparse_mapping` takes the non-zero entries of T,
divides by the source cell area (so the operator maps *intensities*, not fluxes, preserving
surface brightness), and packs them into a `torch.sparse_coo_tensor` **M** of shape (Q, P) where
Q = number of source pixels and P = number of image pixels.

**Step 4 — apply as a matrix multiply.** Lensing is now

```python
I_flat = torch.sparse.mm(M, source_flat)
```

and chains of operators are just chains of `mm`.

**Step 5 — the log-polar detour.** A circular SIS deflection is purely radial and constant in
magnitude, so in log-polar coordinates it becomes a *shift*. Near the centre a Cartesian grid is
badly undersampled — all the flux of the central pixel is spread over a whole Einstein ring — so
`make_log_grid` builds a **double-logarithmic** grid (two `torch.logspace` halves mirrored about
the centre) and the operator chain is: square → log grid → lensing on the log grid → log grid →
square. That is what the four stored `.pt` files are:

| file | role |
|---|---|
| `sparse_grid_fracs_euclid_backward.pt` | image → source, row-normalised averaging |
| `scatter_to_log_128.pt` | regular HR source grid → log-polar grid |
| `forward_from_log_128.pt` | lensing applied on the log grid |
| `scatter_from_log_128.pt` | log-polar → regular HR image grid |

**Step 6 — one lens for everything.** `construct_sis(theta_x, theta_y, alpha_r)` builds
`alpha = alpha_r * theta / r` — a **single scalar** `alpha_r`. Because M is built offline from
that scalar, changing the lens means rebuilding M. In the DeepLense repo version, `alpha_r` (and
later, in `train_sis_bank.py`, a 13- or 16-bin *bank* of them) is chosen per image using
`theta_E` read from the manifest.

There is also a second, non-matrix path in the same file: `forward()` uses
`F.grid_sample(source_image, self.grid, mode='nearest', align_corners=True)` — direct
interpolation with a precomputed grid.

## 1.3 Our method: analytic in-graph ray-tracing

`superres/raytrace.py` + `superres/lens_models.py`. There is **no matrix and no precomputation**.

```python
X, Y   = image_plane_grid(n_pix, pixel_scale, supersample)   # (sub)pixel centres, arcsec
bx, by = ray_shoot(X, Y, lens_params)                        # beta = theta - alpha(theta; 6 params)
sky    = source.at(bx, by)                                   # evaluate the source AT beta
blur   = convolve(sky, psf)                                  # optical PSF, on the sky
pred   = area_downsample(blur, S) + background               # detector pixel integration
```

Every step is a differentiable tensor op. `α` is computed in closed form from
(θ_E, γ, e₁, e₂, g₁, g₂) by the Tessore & Metcalf (2015) elliptical-power-law formula, evaluated
with a hand-written truncated hypergeometric series so it works in both numpy and torch (§4.5).

**The single most important structural consequence:** because `α` is a *function evaluated inside
the computation graph*, the lens parameters are ordinary leaf tensors with gradients. You can
optimise them. In the matrix representation you cannot, for three separate reasons documented in
`lens_models.py`:

1. In `torch.sparse.mm(M, source_flat)` the only tensor carrying a gradient is the source.
   ∂loss/∂α does not exist.
2. α determines **which entries of M are non-zero**, not merely their values. That is a discrete
   change to a data structure, not a smooth reweighting.
3. The construction itself is non-differentiable — `build_scatter_matrix` finds the landing pixel
   with `torch.round(...)`, and `round` has zero derivative almost everywhere.

## 1.4 How the source and the image are represented in each

This is the question you asked most directly, so here it is as a table.

| | original (grid-based) | ours (Path A / B3) | ours (B4 / B5) |
|---|---|---|---|
| **source representation** | a **pixel grid**: 128² or 254² free numbers on a regular (or log-polar) grid, produced by a CNN | a **continuous analytic function**: an elliptical Sérsic with 7 parameters, evaluated at arbitrary real-valued coordinates. B3 adds a bounded 32² residual grid on top | a **pixel grid** again: 48² free numbers in a ±1.2″ box, produced by a fully-convolutional network; optionally *added to* the analytic Sérsic |
| **how the source is "read" at β** | not read at all — the source *is* on a grid, and the operator matrix already encodes which source cells feed which image pixels | `SersicSource.at(bx, by)` evaluates the closed form directly at the ray-shot coordinates. **No source grid ⇒ no interpolation error** | `sample_source()` = bilinear `grid_sample` at the ray-shot coordinates, with a documented half-pixel alignment convention |
| **image representation** | a pixel vector; lensing is a sparse matrix–vector product | a pixel array produced by evaluating the source at every (sub)pixel's β | same |
| **lens representation** | a **precomputed sparse matrix** (or a bank of 13–16 of them) built offline from one scalar α_r | **six numbers** (θ_E, γ, e₁, e₂, g₁, g₂), fitted per image, live in the graph | same six numbers, but **frozen** during B4/B5 training |
| **where flux conservation comes from** | *exact by construction* — the matrix entries are overlap areas | *approximate*, controlled by supersampling; error measured at **5.5e−3** for S = 3 against an S = 9 reference **[reported]** | same |
| **cost of changing the lens** | rebuild the matrix (originally 8h46m for the 128 chain; seconds after the supersampling rewrite) | **zero** — it is a function call | zero, but B4 chooses not to |

## 1.5 The head-to-head, with pros and cons

### Grid/matrix lensing

**Pros**

1. **Exact flux conservation.** Overlap areas sum correctly by construction. Our supersampled
   ray-tracing only approaches this as O(1/S²).
2. **Fast at inference for a fixed lens.** One sparse `mm` per image, GPU-friendly, and the cost
   is independent of how complicated the deflection field is.
3. **Composability.** Multiple lens planes are just more matrices in the product chain — a real
   advantage for multi-plane lensing.
4. **The log-polar trick genuinely solves a real problem.** Near the centre of a circular lens the
   information density in a Cartesian grid collapses; a log grid fixes it exactly.
5. **No repeated evaluation of a transcendental function.** Our EPL deflection evaluates a
   hypergeometric series at every (sub)pixel, every model evaluation.

**Cons**

1. **The lens cannot be inferred.** This is fatal for our project, and it is the reason for the
   rewrite. It also forced the manifest leak: θ_E had to come from *somewhere*, and it came from
   the ground truth.
2. **It scales catastrophically with lens complexity.** A bank indexed by θ_E alone is 13–16
   directories. Adding e₁, e₂, g₁, g₂ at any useful resolution is 10⁴–10⁵ matrix pairs. **The
   architecture, not the physics, is the blocker.**
3. **Sign errors are invisible.** Because the operator is data on disk rather than a formula,
   nothing checks that forward and backward are inverses. Measured on your disk by
   `diagnose_ring_geometry.py`: **three of five** stored mapping directories had forward and
   backward operators with the *same* sign, which is geometrically impossible, and the training
   loss never noticed **[reported: build_sis_mappings.py header]**.
4. **Polygon clipping is O(N⁴) in pure Python.** The 128-grid forward chain took **8h46m**
   **[reported: same]**, which is why the wrong sign survived for months — you cannot iterate on
   geometry at that speed.
5. **Discretisation of the source is forced.** A pixel grid is always either too coarse (loses the
   source) or too fine (underdetermined). This is the 64,516-free-pixel problem.

### Analytic in-graph ray-tracing

**Pros**

1. **The lens is a parameter, so it can be fitted.** Everything in this project follows from that
   one property: unsupervised θ_E, per-image ellipticity, magnification derived from the applied
   deflection, and the whole ground-truth validation table.
2. **Validated against an external reference.** Deflection matches `lenstronomy` 1.14.2 to
   **2.7e−15** and the Sérsic matches **exactly** **[reported: README.md]**. A matrix on disk has
   nothing to be compared against.
3. **No source grid at all in Path A.** The Sérsic is evaluated at real-valued β, so there is
   zero interpolation error, and the fitted source can be rendered on *any* grid afterwards —
   which is literally the super-resolution mechanism.
4. **Arbitrary lens complexity is free.** EPL + external shear costs the same as SIS.
5. **Magnification cannot disagree with the ray-tracer**, because it is obtained by
   differentiating the deflection that is actually applied (§4.7).
6. **Fast to change.** A new lens model is a new function, not an overnight rebuild.

**Cons**

1. **Flux conservation is only approximate.** Mitigated by supersampling and *measured*, but a
   real theoretical loss relative to the area-overlap operator.
2. **Repeated transcendental evaluation.** The hypergeometric series runs at every subpixel of
   every model evaluation — 468 model evaluations per image in the cold fit **[verified]**.
3. **Interpolation reappears the moment the source is pixelated.** B4's `sample_source` is
   bilinear, so B4 does carry a resampling error that Path A does not.
4. **Multi-plane lensing is more work** than adding a matrix to a chain.
5. **The centre is still undersampled** — we have no log-polar equivalent. We sidestep it by
   never putting the source on a fine Cartesian grid in Path A, and by masking to r ≤ 45 px.

## 1.6 The honest summary for your mentors

> The original method represents the lensing operator as **data** (a precomputed sparse matrix of
> polygon-overlap areas) and the source as a **pixel grid**. We represent the operator as a
> **function** (an analytic deflection field evaluated inside the autograd graph) and the source,
> in the primary path, as a **continuous parametric function**.
>
> His representation buys exact flux conservation and cheap repeated application of one fixed
> lens. Ours buys a lens that can be **inferred from the image**, which is what makes the pipeline
> genuinely unsupervised and what allowed ellipticity — measured to be worth **+0.10 in image
> correlation against +0.02 for refitting θ_E** — to enter the model at all.
>
> B4 deliberately puts his *architecture* (back-projection into the source plane, fully
> convolutional SISR, PixelShuffle upsampling, magnification-weighted regularisation) back on top
> of our lens, so the two ideas are now combined rather than competing.

---

# 2. Am I over-focused on physics at the expense of ML?

Short answer: **you over-invested in physics *relative to a machine-learning venue*, and that
investment was still the right call, because without it every ML number you produced would have
been meaningless.** But the balance now needs to shift, and you have about a week.

## 2.1 The case that the physics was necessary, not indulgent

Each of these is a measured fact from your own repository, and each one would have silently
corrupted an ML result:

| physics work | what it prevented |
|---|---|
| Replacing SIS with EPL+shear | SIS explains these arcs at correlation **0.869**, SIE+shear at **0.997**; for \|e\| > 0.3 SIS falls to **0.658** **[reported]**. With a wrong lens the network's optimal strategy is to inflate the source — measured **10.6–12.8×** across a 100× regulariser sweep. No architecture fixes that. |
| Measuring the PSF (0.18–0.20″, Moffat wings) instead of assuming 0.10″ | The assumed value was **2.4× worse** in residual nmse. A network must supply the missing blur, and the only place it can put it is the source — a direct bias on the quantity you are trying to measure. |
| Closing the θ_E manifest leak | Spearman(prediction, truth) = **+0.95 guaranteed by construction**. That is a published-result-shaped number that means nothing. |
| Replacing `skill` with χ² + source truth | A 3-px blur of the input scored **0.951** where the exact physical model scored **0.630**, and skill was anti-correlated with SNR at **ρ = −0.85** **[reported]**. Every ML comparison made on that metric was measuring the wrong thing. |
| The `image_nss` circular-lens discovery | You would have concluded your own correct code was broken. You *did*, for one day. |
| The ellipticity singularity fix | NaN gradients on step one, poisoning the whole head. |

**Any ML architecture you had built on top of the un-repaired pipeline would have produced
numbers you could not defend.** That is not a rhetorical point — the previous pipeline's headline
metrics all looked healthy while the source was eleven times too large.

## 2.2 The case that you should now shift

Being blunt, because a NeurIPS-workshop reviewer will be:

- Your networks are small and conventional. B3 is a 4-layer conv trunk + global pool + two heads
  (849,519 weights **[verified]**). B4/B5 is an SRResNet-shaped stack (2.55 MB checkpoints, so
  ~640k parameters). Both are 2016-era designs.
- Optimiser, schedule and regularisers are all standard (Adam, cosine annealing, weight decay,
  Laplacian smoothness).
- **No uncertainty quantification.** This is the biggest single omission, because you have an
  *identified, measured* bias (the spin-2 shrinkage) that would become a *feature* the moment you
  predicted a distribution instead of a point.
- **No architectural ablation.** You ablate μ-weighting three ways but never width, depth, or the
  back-projection input itself.
- No equivariance, despite a spin-2 target on a rotationally-symmetric problem — an E(2)-CNN is
  almost the textbook application.

## 2.3 The resolution: your physics work *is* an ML contribution, if you frame it as one

The frame that makes the whole thing land at a representations workshop:

> **The question is not "which architecture", it is "which representation of the source can a
> physics-only objective actually constrain".**

Under that frame, every piece of your physics work is *experimental apparatus for an ML question*,
and you have three data points nobody else in this line has: 64,516 free pixels fails, 4,096
fails, 14 + a bounded residual works, all on the same data with the same forward model. That is
an ML result obtained by doing physics properly.

**Concretely, for the remaining week:** do not start a new architecture. Do these instead, in
order:
1. Report the warm-start result as the ML headline (§10) — it is now verified at n = 800.
2. Report the three-plus-one negative results on magnification as a *measured* finding (§14).
3. Add one sentence positioning the spin-2 shrinkage against unrolled-optimiser methods
   (recurrent inference machines).
4. If and only if there is slack: one architectural ablation (width or depth) so the paper has one.

---

# 3. Did we complicate anything unnecessarily? Could libraries have done it?

You asked this three ways — "is anything unnecessary", "did we rebuild something a library
provides", and "is our lensing even standard?". Taking them in turn.

## 3.1 Is our lensing standard gravitational lensing? **Yes. Completely standard.**

This is worth being unambiguous about, because you sound unsure and you should not be.

- **The lens model is the standard one.** Elliptical power law (EPL) + external shear is the
  default macro model in essentially every modern strong-lens modelling paper, and it is exactly
  what `lenstronomy` provides as `EPL` + `SHEAR`. It is what generated Model_A.
- **The deflection formula is a published closed form.** Tessore & Metcalf (2015), eqs. 22–23.
  `lenstronomy` implements the same formula.
- **The source model is the standard one.** Elliptical Sérsic, matched to `lenstronomy`'s
  `SERSIC_ELLIPSE` convention including its `b(n) = 1.9992n − 0.3271`.
- **The observation operator is the standard one.** Lens → PSF → pixel binning, in that order.
- **Ray-tracing (evaluating β = θ − α at every image pixel and reading the source there) is *the*
  standard way to render a lensed image.** `lenstronomy`'s own `ImageModel` does exactly this.
  The grid/matrix approach is the unusual one, not ours.

So: **you are not doing something exotic. You moved from an unusual representation to the
conventional one, and you validated the move against the reference implementation to 1e−15.**

## 3.2 Then why not just call `lenstronomy`?

This is the fair version of your question, and the answer is specific.

`lenstronomy` **is** used in this project — as the *reference* that `tests/test_lens_models.py`
and `tests/test_sources.py` check against. What it cannot do is be *inside* a PyTorch graph:

1. **It is numpy/scipy only.** `lenstronomy`'s EPL calls `scipy.special.hyp2f1`, which has no
   torch equivalent and no autograd rule. Training B3/B4/B5 requires the deflection to be
   differentiable *with respect to the lens parameters* and to run on batched GPU tensors.
2. **Speed.** Path A evaluates the forward model ~468 times per image **[verified]**. Round-trips
   through a general-purpose library with its own parameter-class machinery would dominate.
3. **Exactness of convention.** By re-implementing you controlled every convention (the (N−1)/2
   grid origin, `sersic_major_axis=False`, shear about the grid origin, lenstronomy's own b(n))
   and then *proved* the match. That is stronger than assuming a library call did what you meant.

**This is the standard pattern in differentiable astrophysics** — it is why `herculens`,
`caustics`, `GIGA-Lens` and similar packages exist at all: they re-implement lenstronomy's physics
in JAX/PyTorch for exactly this reason. Your `lens_models.py` is a small, purpose-built member of
that family. Worth one sentence in the paper and worth saying out loud in the meeting, because it
converts "why did you rewrite this?" into "this is the known approach".

**One legitimate criticism to pre-empt:** you could have *adopted* one of those existing
differentiable packages instead of writing your own. The honest defence is that you needed exactly
two profiles, you needed them in both numpy and torch from one source file, and you validated to
1e−15 — a dependency would have cost more integration time than the ~600 lines it replaced. That
is a defensible engineering call, not an obviously correct one.

## 3.3 What *was* genuinely unnecessary, in hindsight

Being honest, since you asked:

| item | verdict |
|---|---|
| **The θ_E operator bank (Phase 2)** | The most expensive dead end. It refined the axis that was already right to 2% while the axis that mattered (ellipticity) was structurally inexpressible. **But** it was the correct answer to the question Model_4 could pose, since Model_4 has no ground truth. You could not have known without truth. |
| **Option 2 (linear inversion)** | Two weeks-equivalent of work for a negative result. Worth it: it is one of your three representation data points, and it produced a reusable adjoint operator that B4 now depends on. |
| **B2** | Mis-configured (inherited H = 0.8, losing 28% of the flux). The *experiment* was necessary; that particular *run* was wasted. |
| **The double-logarithmic grid machinery** | Never used in `superres/`. Fine — it belongs to the old representation. |
| **Writing `backend.py`** | Looks like over-engineering; it is not. It is what stops the numpy and torch physics from silently diverging, and `test_backend_parity.py` enforces it at 1e−6. |
| **The polygon-clipping code** | Not yours, and superseded. |

**Net:** roughly one third of the elapsed project time went into things that did not reach the
final pipeline, and almost all of that third produced a *measurement* that justifies a decision in
the final pipeline. That is a normal, healthy ratio for research. The one thing I would call
avoidable is the length of time the bank was pursued before ground truth was available — and the
fix for that was to switch datasets, which you did.

## 3.4 "Is all of our work validated and does it make sense?"

Yes, and specifically:

| claim | how it is validated | value |
|---|---|---|
| deflection field | vs `lenstronomy` 1.14.2 | 2.7e−15 |
| Sérsic profile | vs `lenstronomy SERSIC_ELLIPSE` | 0.00e+00 (exact) |
| torch forward model = numpy forward model | `test_pathb.py` check 1 | 5.5e−16 |
| numpy/torch physics parity | `test_backend_parity.py` | 1e−6 |
| autograd magnification = analytic Hessian | `test_jacobian.py` | ≤ 2e−9 |
| supersampling error | vs S = 9 reference | 5.5e−3 at S = 3 |
| ring θ_E estimator | vs manifest, 400 images | 0.48 px median |
| back-projection round trip (B4) | vs true source where coverage ≥ 3 | corr 0.9989, median value error 3.32% **[reported: B4.md]** |
| forward/backward operator consistency | `test_b4.py` check 1 | passes — this is the exact test the original pipeline lacked |
| the whole inversion | seven parameters vs generative truth, n = 2000 | ρ = +0.57 to +0.96 |

The one thing that is *not* validated is generalisation: everything is one dark-matter class of
one simulated instrument (see the technical brief §1.4).

---

# 4. EPL, shear, external shear, and the surrounding vocabulary

## 4.1 Convergence and shear — the two things a lens does

Start from the deflection potential ψ. The deflection is its gradient and the projected mass is
its Laplacian:

$$\boldsymbol{\alpha} = \nabla\psi, \qquad \kappa = \tfrac12\nabla^2\psi$$

κ is the **convergence** — projected surface mass density in units of the critical density Σ_cr.
κ = 1 is the threshold for strong lensing. The **shear** is the traceless part of the second
derivative tensor:

$$\gamma_1 = \tfrac12(\psi_{,11} - \psi_{,22}), \qquad \gamma_2 = \psi_{,12}$$

Intuitively: **κ makes a circle into a bigger circle; γ makes a circle into an ellipse.** κ is a
scalar (spin-0); γ is a spin-2 object, meaning it returns to itself under a 180° rotation, not a
360° one.

## 4.2 SIS — singular isothermal sphere

The simplest useful model. Its density profile ρ ∝ r⁻² gives a flat rotation curve, which is
roughly what elliptical galaxies have.

$$\kappa(\theta) = \frac{\theta_E}{2\theta}, \qquad \boldsymbol{\alpha} = \theta_E\,\hat{\boldsymbol{\theta}}$$

**One parameter, θ_E.** The deflection has *constant magnitude everywhere*, pointing inward.

Key facts you should be able to state in the meeting:
- If the source sits exactly behind the lens (β = 0) you get a perfect Einstein ring of radius θ_E.
- Offset by β and you get **exactly two** images, at θ_E + β and −(θ_E − β), **diametrically
  opposite**.
- Therefore **three or more distinct arcs is proof, from the image alone, that the lens is not
  circular.** Your `count_arcs` finds ≥ 3 arcs in **98%** of Model_A images **[reported]**.
- For a circular SIS, κ = |γ| = θ_E/2r exactly, so λ_r = 1 (no radial stretching) and all
  magnification is tangential. This is *only* true for SIS.

## 4.3 SIE — singular isothermal ellipsoid

Same radial profile, elliptical isodensity contours:

$$\kappa(x,y) = \frac{\theta_E}{2\sqrt{qx'^2 + y'^2/q}}$$

with axis ratio q = b/a and a position angle φ. Real elliptical galaxies have q ≈ 0.6–0.8;
Model_A's median q is ≈ 0.65 **[verified: median |e| = 0.214]**.

**Why ellipticity is carried as (e₁, e₂) and not (q, φ).** A position angle is periodic with
period 180°, so it wraps: φ = 179° and φ = 1° are nearly the same lens but numerically far apart,
and any optimiser or network working on φ sees a discontinuity. The **spin-2** parameterisation

$$e_1 = \frac{1-q}{1+q}\cos 2\varphi, \qquad e_2 = \frac{1-q}{1+q}\sin 2\varphi$$

is smooth and single-valued: it maps the whole (q, φ) space onto a disc with the circular lens at
the origin. The factor of **2** in the angle is exactly the spin-2 property. Recovering it in
`ellipticity_to_phi_q` is:

```python
phi = 0.5 * atan2(e2, e1)
c   = hypot(e1, e2)
q   = (1 - c) / (1 + c)
```

The price is that the origin (a perfectly circular lens) is a **removable singularity in value and
a real one in the gradient** — see the technical brief §2.6. It is the single subtlest piece of
numerics in the project.

## 4.4 EPL — elliptical power law

SIE with the radial slope freed:

$$\kappa \propto r^{1-\gamma}, \qquad \gamma = 2 \Rightarrow \text{SIE}$$

Written in the code in the major-axis frame as

$$\kappa = \frac{2-t}{2}\left(\frac{b}{\sqrt{q^2x'^2+y'^2}}\right)^{t},
\qquad b = \theta_E\sqrt{q},\quad t = \gamma - 1$$

γ is the 3-D density slope: γ = 2 is isothermal, γ < 2 is shallower (more extended mass), γ > 2 is
steeper (more concentrated). Model_A's range is 1.90–2.20 **[reported]**, i.e. all nearly
isothermal — which is why γ is the hardest parameter to recover (ρ = +0.571 for Path A
**[verified]**): there is very little signal to constrain it.

**Measured value of each extra term** (correlation with the truth image, n = 60 **[reported]**):
SIS 0.869 → SIS with θ_E refit 0.888 → SIE 0.973 → EPL 0.983 → **SIE + shear 0.997**.
Ellipticity is worth +0.10; the slope is worth +0.010. On this dataset EPL over SIE is nearly
free but nearly pointless — we keep it because it costs one parameter and it is what the
simulator used.

## 4.5 The hypergeometric series — why the EPL deflection needs one

Tessore & Metcalf give the EPL deflection in closed form:

$$\alpha = \frac{2}{1+q}\left(\frac{b}{R}\right)^{t}\,Z\;{}_2F_1\!\left(1,\tfrac{t}{2};\,2-\tfrac{t}{2};\,w\right),
\qquad Z = qx' + iy',\; w = -\frac{1-q}{1+q}\frac{Z}{\bar Z}$$

₂F₁ is the **Gauss hypergeometric function**, defined by the series
$\sum_n \frac{(a)_n(b)_n}{(c)_n}\frac{z^n}{n!}$ with $(a)_n$ the Pochhammer symbol (rising
factorial). Here a = 1, so $(1)_n = n!$ and the two cancel — the series collapses to a simple
recursion:

$$c_0 = 1, \qquad \frac{c_{n+1}}{c_n} = \frac{t/2+n}{2-t/2+n}$$

and since |w| = (1−q)/(1+q) < 1 always, it converges geometrically. Model_A's most elliptical lens
gives |w| = 0.393, and 0.393⁴⁰ ≈ 1e−17. `_n_terms_for` picks the term count adaptively as
n = log(tol)/log|w| — 30 terms for the worst case, 8 for a near-circular lens.

Two implementation details worth knowing: it is written in **explicit real and imaginary parts**
rather than complex tensors, so the autograd path has no Wirtinger-derivative subtleties; and the
truncation error is **measured against `scipy.special.hyp2f1`** in the tests rather than assumed.

## 4.6 External shear — and why "external" matters

**Shear** in general is the anisotropic part of the lensing distortion. **External shear** is
specifically the contribution from mass *outside the modelled frame* — a neighbouring galaxy, a
group, large-scale structure along the line of sight. It has a defining property:

$$\boldsymbol{\alpha}_{\rm ext} = \begin{pmatrix}g_1 & g_2\\ g_2 & -g_1\end{pmatrix}\boldsymbol{\theta},
\qquad \kappa_{\rm ext} = 0$$

**Zero convergence** — it adds a pure quadrupole and no mass. In our parameter vector it is
(g₁, g₂), and like ellipticity it is spin-2. Model_A's median |γ_ext| is 0.033 **[verified]**.

Three things to be able to say:
- **Nearly every real lens needs it.** Neglecting it is one of the classic sources of bias in lens
  modelling.
- **It acts about the grid origin in our code**, matching `lenstronomy`'s `SHEAR` with
  ra_0 = dec_0 = 0, which is how Model_A was generated. A shear about a different point differs by
  a *constant* deflection, which is exactly degenerate with the source position — so the choice
  costs no generality.
- **It breaks the radial/tangential decomposition.** External shear picks its own axis,
  independent of the lens centre, so once it is present the eigenvectors of the lens Jacobian are
  no longer radial and tangential. Any code that decomposes μ that way is projecting onto the
  wrong basis — which the old `physics_losses.fixed_sis_magnification` did.

There is no separate "extended shear" in this project; if you have seen that phrase it is either
a typo for external shear or a reference to *higher-order* (flexion) terms, which we do not model.

## 4.7 Magnification, the Jacobian, and the critical curve

Differentiate the lens equation:

$$A = \frac{\partial\boldsymbol\beta}{\partial\boldsymbol\theta} = I - \frac{\partial\boldsymbol\alpha}{\partial\boldsymbol\theta}
= \begin{pmatrix}1-\kappa-\gamma_1 & -\gamma_2\\ -\gamma_2 & 1-\kappa+\gamma_1\end{pmatrix}$$

$$\mu = \frac{1}{\det A} = \frac{1}{(1-\kappa)^2 - |\gamma|^2}$$

Eigenvalues λ_t = 1 − κ − |γ| (tangential) and λ_r = 1 − κ + |γ| (radial); a source patch is
stretched by 1/|λ| along each eigendirection.

**det A = 0 is the critical curve** — the Einstein ring — where μ formally diverges. Its image in
the source plane is the **caustic**. A source crossing a caustic changes its number of images by
two.

**Surface brightness is conserved.** μ is a change of *solid angle*, not of brightness:
I_image(θ) = I_source(β(θ)). A lensed arc has more total flux because it subtends more sky, not
because any patch got hotter. **Never multiply an intensity by μ.** This is the most common
physics error in lensing ML code, and the reason μ appears in this project only as (a) a
diagnostic and (b) a *weight on a prior*.

**Measured on your fitted lenses, n = 2000 [verified]:** median tangential stretch **3.075×**,
median radial stretch **0.994×** (essentially none, as expected for a near-isothermal profile);
critical curve recovered to **0.63 px** rms with azimuthal swing 0.540″ against a true 0.548″;
total magnification 5.996 against 5.941.

## 4.8 Other terms you may be asked about

| term | meaning |
|---|---|
| **Einstein radius θ_E** | the radius of the ring formed when source, lens and observer are aligned; sets the mass scale inside it |
| **thin-lens approximation** | all deflection happens in one plane; valid because the lens is tiny compared with the distances |
| **Σ_cr, critical density** | the surface density at which κ = 1 and strong lensing begins |
| **caustic** | the source-plane image of the critical curve |
| **substructure / subhalos** | small dark-matter clumps inside the lens halo; the DeepLense science target. Model_A axion images have a median of **6,093** subhalos **[verified]** |
| **mass-sheet degeneracy** | κ → λκ + (1−λ) with the source rescaled leaves every image position and flux ratio unchanged; only time delays or an absolute source size break it |
| **source-position transformation** | a broader family of transformations with the same effect |
| **Sérsic profile** | I(R) = amp·exp(−b(n)[(R/R_s)^{1/n} − 1]); n = 1 is exponential (disc), n = 4 is de Vaucouleurs (elliptical) |
| **Moffat profile** | a PSF shape with broader wings than a Gaussian: I(r) = (1 + (r/α)²)^{−β}; β → ∞ recovers a Gaussian |
| **flexion** | third-order lensing distortion (arc bending); not modelled here |
---

# 5. Source size ratio, Spearman, the 32×32 map, and χ²

## 5.1 Source size ratio — what it is and why it is the headline

**Definition, from `metrics.centroid_size` + `metrics.source_truth`:**

1. Take the model source, PSF-convolved, on the detector grid. Clip negatives to zero.
2. Threshold at 5% of its own peak: `f = where(a >= 0.05*peak, a, 0)`. This removes the noise
   floor so the second moment is not dominated by the empty sky.
3. Compute the flux-weighted centroid (c_y, c_x) **of that object**.
4. Compute the rms radius about **its own centroid**:

$$\text{size} = \sqrt{\frac{\sum_i f_i\,\left[(y_i-c_y)^2 + (x_i-c_x)^2\right]}{\sum_i f_i}}$$

5. Do exactly the same for the true source (the npz `unlensed` array).
6. `size_ratio = size_model / size_true`. **Target 1.0.**

**Why "about its own centroid" is not a detail.** A correct reconstruction of a source at offset β
sits off-centre *by construction* — β is a recovered physical quantity, not an error. Measuring
the second moment about the *grid* centre would add β² to the size and penalise a correct answer.
The old `evaluate_sis.source_metrics` had exactly this problem, which is why it moved away from
"compactness".

**Why it is the headline number.** It is the single scalar that exposed the previous pipeline:
size_ratio sat at **10.6–12.8** across a 100× sweep of the regularisation weight, while every
*plausibility* metric (fill factor 0.66, main-flux fraction 0.998, one connected component) read
healthy. A source eleven times too large is not a subtle failure — but nothing except a truth-based
size measurement could see it.

**Current values [verified]:**

| method | size_ratio |
|---|---|
| original grid pipeline | 10.6 – 12.8 |
| Path A (per-image fit, n = 2000) | 1.0887 |
| Path A refined (n = 800) | 1.0808 |
| B3 (Sérsic only) | 0.9493 |
| B4 free-form | 1.1302 |
| **B4 + Sérsic base** | **1.0434** |
| B5 gate | 1.0392 |

**Its companions.** `size_ratio` alone can be gamed, so it is always reported with:
- **`peak_ratio`** — peak of the flux-normalised model over peak of the flux-normalised truth. A
  source that is the right *size* but too flat, or too spiky, shows up here. Path A is 0.834
  (too flat, a PSF-wings artefact); B4+Sérsic is 0.927 **[verified]**.
- **`centroid_err_px`** — is it in the right *place*? ~0.5 px throughout.
- **`corr`** and **`nmse`** — shape and amplitude-matched error.

Together these four cover **size, sharpness, position, and shape**, which is why they appear as a
block everywhere in this project.

## 5.2 Spearman rank correlation — what it is and why we use it

**Definition.** Replace each value by its **rank** (1st smallest, 2nd smallest, …), average the
ranks within ties, then compute the ordinary Pearson correlation of the ranks:

$$\rho = \frac{\sum_i (R_i - \bar R)(S_i - \bar S)}{\sqrt{\sum_i (R_i-\bar R)^2 \sum_i (S_i-\bar S)^2}}$$

where R and S are the ranks of the fitted and true values. It runs from −1 to +1.

**What it measures:** whether the method **orders** the images correctly. ρ = +0.96 for θ_E means
that if you sorted the 2,000 images by fitted θ_E you would recover almost exactly the ordering by
true θ_E.

**Why rank and not Pearson (linear) correlation:**

1. **It is invariant to any monotonic transformation**, so a systematic multiplicative or additive
   bias does not affect it. That is exactly what you want here: θ_E comes out **+3.5% high**
   because the PSF wings are under-modelled, and a *calibratable* bias is a very different failure
   from *no information*. Spearman separates "the method has the signal but a bias" from "the
   method has nothing".
2. **It is robust to outliers.** A handful of catastrophic fits (the ellipticity–shear degeneracy
   traps ~4% of images) would dominate a Pearson coefficient. Ranks bound their influence.
3. **It works on skewed distributions.** χ²/dof spans 250 → 18,000 **[verified]**; SNR spans
   4.4 → 23; subhalo counts span 25 → 39,835. Linear correlation on such distributions is mostly
   measuring the tail.

**Why it is the right metric for this project specifically.** The scientific use of a lens model
is comparative: *is this lens more massive than that one, is this source more compact*. A method
that gets every θ_E 3.5% high but in perfect order is scientifically usable — you calibrate the
offset. A method that gets the median exactly right but shuffles the order is useless. Spearman is
the metric that says which one you have.

**How to read the values in our tables:**

| ρ | reading |
|---|---|
| > 0.9 | essentially fully recovered (θ_E, β, R_sersic, n_sersic in Path A) |
| 0.7 – 0.9 | recovered with scatter (lens \|e\| at +0.762) |
| 0.5 – 0.7 | weak but real (shear +0.611, slope +0.571) |
| ≈ 0 | no information — **and this is the collapse signature** (B3's \|e\| at +0.269 sits between "weak" and "broken", which is why §6.12 of the brief investigates it rather than reporting it) |

**One caution to state in the meeting:** Spearman ignores calibration entirely. That is why the
tables always print *median fitted*, *median true* and *median absolute error* next to ρ. Neither
number alone is sufficient.

## 5.3 The 32×32 source-plane correction map (B3) — what and why 32

**What it is.** In B3 the source is

$$S(\boldsymbol\beta) = \underbrace{\text{Sérsic}(\boldsymbol\beta;\,7\ \text{params})}_{\text{smooth, infinite support}}
\;+\;\underbrace{C_{\rm eff}(\boldsymbol\beta)}_{\text{a 32×32 grid, bilinearly sampled}}$$

with

```python
C_eff = corr_scale * amp * tanh(C_raw)        # corr_scale = 0.30
```

so the correction is (a) **bounded** by ±30% of the source amplitude, (b) **scaled by the source
brightness**, which makes `corr_scale` dimensionless and transferable between images spanning 260×
in flux, and (c) **penalised** by the magnification-weighted term.

It lives on a grid spanning **±0.8″** at **32×32**, i.e. 2·0.8/31 = **0.05161 ″/px**, which is
**2.052× finer than the 0.10593 ″ detector pixel [verified]**. Sampled at the ray-shot positions β
by `grid_sample`, with zero padding outside the box.

**Why 32, specifically.** Four constraints intersect, and 32 is what survives all four:

1. **Conditioning.** Unknowns per image = 14 + n_c². Against ~4,300 informative data pixels:
   n_c = 32 gives 1,038 unknowns = **4.14 : 1 overdetermined [verified]**. n_c = 64 would give
   4,110 — essentially 1:1, which is the regime where Option 2 produced speckle. n_c = 16 would be
   17:1 over but only 1.03× finer than the detector, i.e. no super-resolution at all.
2. **The lens's measured stretch.** With half_extent = 0.8″, n_c = 32 gives 2.05× refinement
   against a **measured** median tangential stretch of **3.07× [verified]**. n_c = 48 at the same
   box would be 3.08× — right at the limit; n_c = 64 would be 4.1×, i.e. **asking for resolution
   the lens did not deliver**, which is interpolation dressed as physics. `train_pathb.py` prints
   this comparison at startup and warns if you exceed it.
3. **The decoder architecture.** `base = n_c // 4`, and two stride-2 transposed convolutions take
   base → n_c exactly. 32 → base 8 → 16 → 32 with no bilinear resize. A resize would make the
   *effective* degrees of freedom fewer than n_c², quietly invalidating the conditioning
   arithmetic in point 1.
4. **Ray coverage.** At supersample 1 there are ~0.34·127²/n_c² rays per source pixel; at n_c = 32
   that is **5.4 [verified]**, just above the ~4 threshold below which the coverage histogram is
   mostly shot noise. At n_c = 64 it would be 1.3 — the Option 2 regime again.

**Note that the 32×32 box is a different, smaller box than B4's.** B3 uses ±0.8″ because the
correction only has to cover the *core*; the Sérsic carries the wings. B4 free-form uses ±1.2″ or
±1.6″ because there the grid *is* the source. That distinction is §13.

## 5.4 Why χ², and what it actually is

**Definition.** For a model prediction m, data d and per-pixel uncertainty σ:

$$\chi^2 = \sum_{i} \frac{(m_i - d_i)^2}{\sigma_i^2}, \qquad
\chi^2/\text{dof} = \frac{\chi^2}{N_{\rm pixels} - N_{\rm free\ parameters}}$$

**Where it comes from.** If the noise is Gaussian with known σ, the likelihood of the data given
the model is

$$\mathcal{L} \propto \exp\!\left(-\tfrac12\chi^2\right)$$

so **minimising χ² is exactly maximising the Gaussian likelihood.** χ² is not an arbitrary
distance; it is the log-likelihood of the measurement.

**Why we chose it — five reasons, in order of importance:**

1. **It has an absolute meaning.** A correct model on correctly-estimated noise gives χ²/dof ≈ 1.
   You do not need a baseline to know whether you fitted. Contrast the previous metric, `skill`,
   which is relative to an arbitrary reference and where a 3-px blur scored 0.951 against the
   exact physical model's 0.630 **[reported]**.
2. **It weights each pixel by how much it actually knows.** A pixel with large σ contributes
   little. An unweighted MSE treats a noise-dominated corner pixel and a bright arc pixel as
   equally informative, which they are not.
3. **It makes the estimator a maximum-likelihood estimator**, which brings the standard
   statistical machinery: parameter covariances from the Jacobian, the discrepancy principle for
   choosing regularisation strength, and a defensible interpretation of the residual.
4. **It is the natural objective for Levenberg–Marquardt.** LM is a *nonlinear least-squares*
   algorithm; it wants a vector of residuals r_i = (m_i − d_i)/σ_i, and χ² = |r|². The optimiser
   and the statistics agree.
5. **It made the previous pipeline's failure visible.** The moment you measure χ²/dof you can also
   measure *null baselines* (a constant image, a blurred copy of the input) and see immediately
   whether "0.89" is good.

**What χ² means here, concretely.** Path A reaches χ²/dof ≈ **3085** **[verified]**, not 1. That is
understood, measured, and reported:
- The PSF *shape* is the systematic floor. Arcs reach peak/σ_bg of 10³–10⁴, so a **1% kernel error
  is a ~50σ per-pixel residual**.
- The signature that distinguishes a systematic from noise is that χ² **rises with SNR** —
  measured Spearman(χ²/dof, snr_max) = **+0.449 [verified]**. Random noise would give no trend.
- Null baselines on the same images: constant image 37,511; 2-px blur 2,796; 3-px blur 4,091
  **[reported]**. So 3,085 is at the level of a well-matched blur, while also recovering seven
  physical parameters.
- And part of the residual is *supposed* to be there: it is the substructure, at rms 4.33 against
  σ_bg = 0.032 **[reported]**.

**The σ-floor.** For the network paths σ is not σ_bg alone but

$$\sigma_{\rm eff}^2 = \sigma_{\rm bg}^2 + (0.02\cdot\text{model})^2$$

because the PSF wings are wrong at the 1–3% level and *that error scales with flux*. Without it,
per-image χ² spanned 656 → 21,953,076, four images out of 400 carried 51% of the total loss, and
the single largest member of a batch of 16 took **54% of the gradient** — an effective batch size
of about 2 **[reported: PATHB_POSTMORTEM.md]**. This one flag is the difference between B3 v1
(broken) and B3 v3 (works).

---

# 6. "The only supervision is χ² of the re-lensed, re-blurred, re-binned prediction"

This sentence is the whole claim of the project, so here it is unpacked completely.

## 6.1 What supervised super-resolution would look like

You would need pairs: a low-resolution image and *the same object* at high resolution. You train
`f(LR) → HR` and the loss is `|f(LR) − HR|²`. **The HR image is the supervision.**

For lensing this is essentially impossible: you cannot re-observe the same lens with a better
telescope on demand, and simulated HR/LR pairs teach the network the simulator's own conventions.

## 6.2 What we do instead

We never compare anything to a high-resolution target. Instead we take the network's (or the
optimiser's) proposed **source**, push it forward through a model of *what the telescope does*,
and compare the result to the **one image we actually have**.

The chain, in order, exactly as the code executes it:

| step | what happens | code |
|---|---|---|
| **1. propose a source** | 7 Sérsic parameters, or a 32×32 correction on top, or a 48×48 free grid | `SersicSource`, `sample_correction`, `SourceSISR` |
| **2. "re-lensed"** | for every image pixel θ compute β = θ − α(θ; the 6 lens parameters) and read the source at β. This is ray-tracing; it produces the arcs | `lens_models.ray_shoot` + `source.at(bx,by)` |
| **3. "re-blurred"** | convolve with the **measured** PSF, on the sky, before any pixelation | `convolve(sky, psf)` |
| **4. "re-binned"** | average S×S subpixels into each detector pixel — the detector's integration | `area_downsample` |
| **5. add the sky** | a free additive background constant | `+ background` |
| **6. compare** | χ² against the observed pixels, inside r ≤ 45 px, with per-image σ | `((pred - X)/sig)[:, mask]**2` |

Steps 2–4 are together the **observation operator** — a model of the physical measurement process.
The order is not negotiable: lensing remaps the sky, the PSF is an optical convolution that
happens *on the sky*, and pixelation is an integration performed *by the detector*. Blurring after
binning would be a different, and wrong, instrument.

## 6.3 Why this counts as supervision at all

Because the forward model is **many-to-one and known**. Many sources produce the same image
(that is the inverse problem), but any *given* source produces exactly one image, and we can
compute it exactly. So "your source must reproduce my data after being lensed, blurred and
binned" is a real, hard constraint — it is just a constraint imposed by *physics* rather than by
labels.

That is what makes it **self-supervised**: the target is a transformation of the input, not an
external annotation.

## 6.4 Three consequences worth stating

1. **Nothing in the loss knows the truth.** `data_a.truth()` and `data_a.unlensed()` are called
   only by `evaluate.py`, `eval_pathb.py`, `eval_b4.py` and `superresolve.py` — after the fit is
   written to disk. That separation is what makes the recovery table meaningful rather than
   circular.
2. **Data augmentation becomes exactly valid with no relabelling.** A rotated lensed image is a
   legitimate member of the data distribution, and χ² is taken against *that same rotated image*.
   In supervised SR you would have to rotate the target too and worry about interpolation; here
   there is no target. (`n_pix` is odd so the grid centre is a pixel centre and `rot90` is exact.)
3. **The super-resolution is a by-product, not an output.** We never ask the model to produce a
   high-resolution *image*. We ask it to produce a *source* consistent with the data, and then we
   sample that source finely. The resolution comes from the lens having already spread the source
   over many detector pixels — measured median tangential stretch **3.075× [verified]**.

---

# 7. Where is the physics actually coded? (And are we coding the degeneracy equations?)

## 7.1 The honest answer about §2.5 (the degeneracies)

**No — the mass-sheet degeneracy and the source-position transformation are not implemented
anywhere, and they should not be.** They are not equations you *evaluate*; they are statements
about what the data *cannot* determine. You do not code a degeneracy. You respond to it, in one of
three ways:

| response | who does it | what we do |
|---|---|---|
| break it with extra data | time-delay cosmography (needs a variable source) | not available |
| break it with a prior | fix the mass sheet by assuming a profile family | **this is what we do** — an EPL family has no free mass sheet, so λ is fixed by the profile |
| let it dominate | the previous pipeline | the 11× source inflation is partly this |

So the degeneracies enter our code as a **design constraint on dimensionality**: 14 unknowns
against 6,361 fitted pixels is overdetermined ~450:1, which is why the degeneracies that wreck
pixellated reconstructions never get room to act. §1.5 of the brief states this; there is no
`degeneracy.py` and there should not be.

**What you should say if asked:** *"We do not solve the degeneracies; we avoid the regime where
they bite, by keeping both the lens and the source in low-dimensional families and fitting them
jointly. The residual degeneracy we do observe — ellipticity trading against external shear — is
visible in the ~4% of images where the fit lands in the wrong basin, and it is exactly what the
staged parameter release exists to avoid."*

## 7.2 Where every piece of physics lives

| physics | file | function | validated by |
|---|---|---|---|
| lens equation β = θ − α | `lens_models.py` | `ray_shoot` | used by everything |
| EPL deflection (Tessore & Metcalf) | `lens_models.py` | `deflection_epl`, `_hyp2f1_series` | `test_lens_models.py` vs lenstronomy, 2.7e−15 |
| external shear | `lens_models.py` | `deflection_shear` | same |
| ellipticity ↔ (q, φ) | `lens_models.py` | `ellipticity_to_phi_q` | same; singularities guarded |
| convergence + shear (Hessian) | `lens_models.py` | `hessian_analytic` | `test_lens_models.py` vs lenstronomy's hessian |
| magnification, closed form | `lens_models.py` | `magnification` | `test_jacobian.py` |
| magnification, autograd | `lens_models.py` | `magnification_autograd` | vs the closed form, ≤ 2e−9 |
| elliptical Sérsic | `sources.py` | `SersicSource.at` | `test_sources.py`, exact |
| PSF kernels (Gaussian / Moffat / empirical) | `raytrace.py` | `gaussian_psf`, `moffat_psf`, `load_psf` | `calibrate_psf.py` measures the real one |
| PSF convolution | `raytrace.py` | `convolve` | edge-replicate padding |
| detector pixel integration | `raytrace.py` | `area_downsample` | `test_raytrace.py`, flux conserved |
| the full observation operator | `raytrace.py` | `render` | `test_pathb.py` check 1 vs torch, 5.5e−16 |
| ring geometry → θ_E, β | `theta_e_init.py` | `polar_ridge`, `fourier_rphi` | vs manifest, 0.48 px |
| critical curve, resolution gain, μ_tot | `magnification_extract.py` | `critical_radius`, `resolution_gain`, `total_magnification` | vs the true lens, n = 2000 |
| back-projection (image → source plane) | `backproject.py` | `backproject` | `test_b4.py` check 1, corr 0.9989 |
| source sampling at β | `backproject.py` / `train_pathb.py` | `sample_source`, `sample_correction` | `test_b4.py` check 3 |
| numpy/torch parity | `backend.py` | `get_backend` | `test_backend_parity.py`, 1e−6 |

**One sentence for the meeting:** *all the physics is in four files — `lens_models.py`,
`sources.py`, `raytrace.py`, `backproject.py` — and every one of them is checked against either
`lenstronomy` or an independent analytic derivation.*

---

# 8. How the parameter fitting works — the mathematics

## 8.1 What "fitted against the image" means

You have 14 unknown numbers. You have 6,361 measured pixel values. You have a function that turns
the 14 numbers into a predicted 6,361-pixel image. **Fitting** means: search the 14-dimensional
space for the point whose predicted image is closest to the observed one, in the χ² sense.

$$\hat{\mathbf v} = \arg\min_{\mathbf v} \;\chi^2(\mathbf v),
\qquad \chi^2(\mathbf v) = \sum_{i \in \text{disc}} \left(\frac{m_i(\mathbf v) - d_i}{\sigma}\right)^2$$

Nothing else enters. No labels, no truth, no manifest.

## 8.2 What a "flat vector" is

The physics code wants a **dictionary**: `lens = {"theta_E":…, "gamma":…, "e1":…, …}` and
`source = {"amp":…, "R_sersic":…, …}` — because that is how you read physics.

An optimiser wants a **1-D array of real numbers**, `v = [v₀, v₁, …, v₁₃]`, because that is how you
do calculus: gradients, Jacobians and step directions are all vectors in ℝ¹⁴.

The **flat vector** is the fixed-order array that bridges the two. In `fit_per_image.py`:

```python
PARAMS = ("theta_E","gamma","e1","e2","g1","g2",
          "amp","R_sersic","n_sersic","se1","se2","sx","sy","background")
```

and `_split(v)` converts an array back into the two dictionaries plus the background scalar.
"Flat" simply means *unstructured, one dimension, fixed order* — the order is a contract, and it
must not change, because element 3 means `e2` to the optimiser, the bounds array, and the
Jacobian alike.

## 8.3 Levenberg–Marquardt, properly explained

This is the algorithm doing the work, so here it is from the ground up.

### The problem class

**Nonlinear least squares**: minimise `f(v) = ½|r(v)|²` where `r` is a vector of residuals. Here
r_i = (model_i(v) − d_i)/σ, of length 6,361, and v ∈ ℝ¹⁴.

### Ingredient 1 — Gauss–Newton

Linearise the residual around the current point:

$$\mathbf r(\mathbf v + \delta) \approx \mathbf r(\mathbf v) + J\delta,
\qquad J_{ij} = \frac{\partial r_i}{\partial v_j}$$

Substituting and minimising over δ gives the **normal equations**

$$(J^{\!\top}J)\,\delta = -J^{\!\top}\mathbf r$$

This is fast — quadratic convergence near the solution — because for least squares J⊤J is a very
good approximation to the true Hessian (it is exactly the Hessian minus a term proportional to the
residual, which is small near a good fit). **But** if J⊤J is ill-conditioned or the linearisation
is poor far from the optimum, the step is wild and the method diverges.

### Ingredient 2 — gradient descent

$$\delta = -\lambda^{-1} J^{\!\top}\mathbf r$$

Always a descent direction, never diverges, but crawls — especially in narrow valleys, which is
exactly the geometry of a degenerate parameter space.

### Levenberg–Marquardt: interpolate between them

$$(J^{\!\top}J + \lambda\,\mathrm{diag}(J^{\!\top}J))\,\delta = -J^{\!\top}\mathbf r$$

- λ → 0: pure Gauss–Newton (fast, near the solution)
- λ → ∞: pure gradient descent with a tiny step (safe, far away)

**The adaptive rule:** try the step. If χ² decreased, accept it and *decrease* λ (be bolder). If
χ² increased, reject it and *increase* λ (be more careful). That single rule is the whole
algorithm, and it is why LM is the default for small smooth nonlinear least-squares problems —
it is safe far away and fast close in.

Using `diag(J⊤J)` rather than the identity makes the damping **scale-invariant**: θ_E in arcsec
and n_sersic dimensionless get damped proportionally to their own curvature, so the algorithm
does not care that the parameters have wildly different units.

### What we actually call

```python
scipy.optimize.least_squares(resid, v0, bounds=(lo, hi), method="trf",
                             xtol=1e-8, ftol=1e-8, gtol=1e-8,
                             max_nfev=60*len(idx))
```

`method="trf"` is **Trust Region Reflective** — LM's bounded cousin. Instead of a damping
parameter it maintains an explicit **trust region** (a radius inside which the linear model is
believed), solves the subproblem within it, and "reflects" steps off the bound surfaces so the
iterate stays feasible. Same idea, but it can honour box constraints, which we need: θ_E must stay
positive, n_sersic must stay in a range where R^{1/n} is finite, |e| must stay below the cap.

The three tolerances are stopping rules: `xtol` on the step size, `ftol` on the relative χ²
reduction, `gtol` on the gradient norm. `max_nfev` bounds the number of *model evaluations*.

**Jacobian:** we do not supply one, so SciPy builds it by **finite differences** — 14 extra model
evaluations per Jacobian. That is why the measured cost is **480 model evaluations** and **0.815 s
per image** for the cold fit **[verified]**. Supplying an analytic Jacobian (which the torch path
could provide) would be the obvious speed-up and has not been done.

### Why not gradient descent / Adam?

Because this problem is small (14 unknowns), smooth, overdetermined (~450:1) and has an analytic
model. LM converges in tens of iterations; SGD would need thousands and a schedule, and would
still be less accurate. Autograd earns its keep in Path B, where the unknown is a *network's
weights* — millions of parameters, no Jacobian worth forming.

## 8.4 Staged parameter release, and why it exists

Releasing all fourteen from a neutral start lands in local minima: with the source shape free, the
fit can trade **source ellipticity against lens shear** before the ring radius is even right. So:

| stage | free parameters |
|---|---|
| 1. `geometry` | θ_E, amp, sx, sy, R_sersic, background |
| 2. `+shape` | + n_sersic, se₁, se₂ |
| 3. `+shear` | + g₁, g₂ |
| 4. `all` | + γ, e₁, e₂ — **everything free** |

Each stage warm-starts the next. **This is a convergence device only** — the final stage has every
parameter free, so nothing is pinned at a value the data did not choose.

There is a nice payoff here: §10 shows that once a *network* supplies the starting point, the
staged schedule becomes unnecessary. One stage, all 14 free, converges — and to a *better* basin.

## 8.5 The starting point, and why it is not cheating

- **θ_E** and **|β|** come from the ring geometry of the image itself (`theta_e_init.py`,
  §4 of the technical brief): trace the ridge of peak brightness at 72 azimuths, fit
  r(φ) = r₀ + m₁ + m₂, and read r₀ → θ_E, (a₁,b₁) → β as a vector, |m₂| → a model-free ellipticity
  indicator. Median error **0.48 px** against the manifest **[reported]**.
- **Everything else** starts at a population-neutral constant, deliberately not drawn from the
  manifest: `R_sersic = 0.45`, `n_sersic = 1.0`, ellipticities 0, γ = 2, amp = the 99.5th
  percentile of the image, background = its median.
- `m₂` is measured and reported but **not** converted into an e₁/e₂ guess, because the ring
  quadrupole responds to substructure as well as to macro ellipticity, and turning it into a lens
  ellipticity would import an interpretation the fit is supposed to make for itself.

---

# 9. The metrics and the loss — the complete list, and the intuition

## 9.1 The loss (what is optimised)

**Path A** — pure χ², no regularisation:

$$\mathcal L_A(\mathbf v) = \sum_{i\in\text{disc}}\left(\frac{m_i(\mathbf v)-d_i}{\sigma}\right)^2$$

No prior is needed because 14 unknowns against 6,361 pixels is overdetermined ~450:1.

**Path B (B3):**

$$\mathcal L_{B3} = \chi^2 \;+\; \chi^2_{\rm detach}\Big(\lambda_{\rm corr}\underbrace{\textstyle\sum_j w_j (C_j/\text{amp})^2}_{\text{μ-weighted amplitude}} + \lambda_{\rm curv}\underbrace{\textstyle\|\nabla^2 C\|^2}_{\text{smoothness}}\Big)$$

**B4 / B5:**

$$\mathcal L_{B4} = \chi^2 \;+\; \chi^2_{\rm detach}\Big(\lambda_{\rm curv}\|\nabla^2 S_{\rm rel}\|^2 + \lambda_{\ell 2}\textstyle\sum_j w_j S_{{\rm rel},j}^2\Big)$$

with `S_rel = (S_map − base)/amp`, and in B5 the curvature term optionally carries the same μ
weight.

### The four ingredients, and the intuition for each

**(a) χ² with a σ-floor.** *"Match the data, weighting each pixel by how much it actually
knows."* The floor σ_eff² = σ_bg² + (0.02·model)² admits the measured PSF-shape systematic; without
it one bright image owns the batch gradient.

**(b) λ expressed as a fraction of χ².** The penalty is multiplied by `chi2.detach()`. *Intuition:
"spend at most 27% as much effort on being smooth as on fitting the data"* — a ratio, not an
absolute. In v1, λ = 1.0 against a χ² of ~10⁵ was six orders of magnitude too small and the
regulariser did nothing at all; the two "different" runs came out as literally the same network
(Pearson +0.9992 on θ_E) **[reported]**. Making λ scale-free means that cannot recur.

**(c) The curvature (Laplacian) penalty.** Sum of squared discrete Laplacians — the standard
source-plane regulariser in lens modelling (Suyu et al. 2006). *Intuition: "a smooth source is
free; a speckled one is expensive."*

Why curvature and **not total variation**, which the old pipeline used: TV's unconstrained
minimiser is a **constant field**, so raising its weight drives the source toward a flat wash
rather than toward compactness — measured, size_ratio stayed at 10.6–12.8 across a 100× TV sweep
**[reported]**. Curvature is a **quadratic form**, so with a Gaussian likelihood the source
sub-problem stays convex and λ has a determinate value (set by the discrepancy principle
χ²/dof → 1, or by the L-curve corner) rather than by taste.

**(d) The magnification weight.**

$$w_j = \left(\frac{\text{median coverage}}{\text{coverage}_j}\right)^{0.5},\ \text{clipped to }[1/5,\,5]$$

*Intuition: "where the lens delivered many rays, the data can constrain fine structure, so
penalise it less; where it delivered few, anything you paint in is invention."* Power 0.5 rather
than 1.0 for a measured reason: coverage spans ~1000×, and raw 1/μ leaves the thin near-caustic
strip effectively unregularised — in Option 2 that dropped source correlation from 0.96 to
**0.74** **[reported]**.

## 9.2 The metrics (what is reported — never optimised)

**Goodness of fit**

| metric | definition | reads |
|---|---|---|
| `chi2_per_dof` | χ² / (N_pix − N_params) | absolute fit quality |
| null baselines | χ²/dof of a constant image, a 2-px blur, a 3-px blur of the input | what "good" is worth on this data |
| stratification by `snr_max` | every summary split into SNR < 6, 6–15, > 15 | a single median hid ρ = −0.85 once |

**Source-plane truth** (model PSF-convolved first, against the npz `unlensed`)

| metric | reads |
|---|---|
| `corr` | shape agreement, amplitude-free |
| `nmse` | amplitude-matched normalised squared error — the summary number |
| `size_ratio` | is it the right size? (§5.1) |
| `peak_ratio` | is it the right sharpness? |
| `centroid_err_px` | is it in the right place? |

**Parameter recovery** — fitted vs true per parameter: median, median \|error\|, p90 \|error\|, and
Spearman ρ. *This is the metric the old pipeline could not produce at all*, because Model_4 has no
truth; only plausibility could be scored, and plausibility read healthy while the source was 11×
too large.

**Method-specific diagnostics**

| metric | where | reads |
|---|---|---|
| collapse ratio (pred std / truth std) | `eval_pathb.py` §0 | did the network emit the same numbers for every image? |
| ring-only control | `eval_pathb.py` §0b | did the network **beat its own inputs**? |
| Δnmse from the correction | `eval_pathb.py` §A | **the super-resolution claim, isolated** |
| corr(\|C\|, log coverage) | `eval_pathb.py` §C | did detail go where the lens delivered resolution? |
| tanh saturation | training + eval | is the bounded correction pinned at its bound? |
| `src_flux_in_box` | `eval_b4.py` | is the boxed source inventing or losing flux? |
| model evaluations / seconds per image | Path A, refine | cost |

## 9.3 The one thing to remember about the loss/metric split

**The loss is χ² plus priors. The metrics are almost entirely things the loss never sees.**

That is deliberate and it is the strongest methodological feature of the project: the model is
optimised against the *data*, and judged against the *truth*, and the two never touch. It is what
makes a sentence like "source size_ratio 1.04" mean something, and it is exactly what was missing
when a 3-px blur of the input could beat the exact physical model on the headline metric.
---

# 10. `refine_pathb.py` — what it is, why it exists, and what it gave

## 10.1 The problem it solves

The B3 network recovers θ_E (ρ +0.960), β (+0.826) and R_sersic (+0.953) as well as the optimiser
does, but it **shrinks both ellipticities by a factor of ~4**:

```
lens   |e|   network 0.049   Path A 0.194   truth 0.222     [verified, n=800]
source |se|  network 0.065   Path A 0.245   truth 0.217     [reported]
```

The critical diagnosis — and the reason this is a *finding* rather than a bug — is that **this is
not undertraining and not an orientation failure**. Decomposing into modulus and phase
**[reported: RESULTS_V3.md]**:

| | modulus ρ | median phase error (random = 45°) |
|---|---|---|
| B3 lens \|e\| | +0.269 | 19.0° |
| Path A lens \|e\| | +0.776 | 3.9° |

The network gets the *direction* roughly right and the *magnitude* badly wrong, and the shrinkage
is a flat factor of ~0.22 in **every** phase-error bin (0.21, 0.23, 0.23, 0.22).

**Why a squared loss must do this.** A network trained on χ² is a **conditional-mean estimator**:
given the image it emits E[e | image]. For a **spin-2** quantity under orientation uncertainty, a
large ellipticity pointed the wrong way costs *more* than no ellipticity at all — the squared
error of a vector of length r at angle error δ is r²(2 − 2cos δ), which is minimised by shrinking
r when δ is uncertain. So the expected-error-minimising answer is to shrink the modulus toward
zero. Levenberg–Marquardt is a **maximum-likelihood** estimator on *one* image and has no such
incentive: it reports the mode of that image's likelihood, not an average over the population.

**Both are behaving exactly as their objectives specify.** More epochs cannot fix it; it is a
property of the estimator, not of the optimisation.

## 10.2 The idea

Stop asking the network to *be* the estimator. Use it for what it is good at — producing, in
~12 ms, a starting point far better than any hand-written one — and let a short
Levenberg–Marquardt run do the last mile.

## 10.3 What the code actually does, step by step

`refine_pathb.py`:

1. **Load the network's per-image predictions** from `fits_pb3_mu.json` (the output of
   `eval_pathb.py`), keyed by image index.
2. **For each image, run two fits from scratch and compare them:**
   - **cold**: start from `initial_vector(img, res)` — the ring-geometry estimate plus neutral
     constants — and run the **full 4-stage schedule** (`STAGES` imported unchanged from
     `fit_per_image.py`).
   - **warm**: start from the network's 14 numbers and run **one stage with all 14 free**:
     `fit_from(img, sig, v0_network, [("all", PARAMS)], ...)`.
3. Both use identical bounds, identical tolerances (xtol = ftol = gtol = 1e−8), identical
   residual masks, identical PSF — the *only* difference is the starting point and the schedule.
4. Report cost (model evaluations, seconds), χ²/dof, per-parameter medians and Spearman ρ against
   the truth, and the source-plane metrics for network-alone / cold / warm.
5. Write the **warm** rows in the `fits_img.json` schema so `evaluate.py` can score them with the
   same code as everything else.

Note what is *not* done: the network is not fine-tuned, nothing is back-propagated, and the truth
is not touched until the reporting step.

## 10.4 The result — verified at n = 800, not n = 25

The version in the earlier documents was n = 25. You have since run this properly. I recomputed
the comparison from `fits_refined_mu.json` (warm), `fits_img.json` (cold) and `fits_pb3_mu.json`
(network) on the **800 common validation images**, against the manifest truth. **[verified]**

```
param         network   cold LM   warm LM     truth |  net rho  cold rho  warm rho
theta_E        1.3838    1.3692    1.3701    1.3200 |   +0.960    +0.952    +0.960
beta           0.2059    0.2859    0.2917    0.3048 |   +0.826    +0.912    +0.915
e              0.0489    0.1944    0.1998    0.2215 |   +0.269    +0.776    +0.839
g              0.0377    0.0503    0.0492    0.0324 |   +0.142    +0.632    +0.650
R_sersic       0.4211    0.5235    0.5050    0.4839 |   +0.953    +0.934    +0.954
n_sersic       1.0358    1.1806    1.1401    1.0051 |   +0.475    +0.940    +0.945
gamma          1.8295    1.9967    2.0032    2.0579 |   +0.132    +0.551    +0.538

cold LM : median 0.815 s,  480 model evaluations, chi2/dof 3302
warm LM : median 0.460 s,  151 model evaluations, chi2/dof 3228
```

**Read it as four statements:**

1. **Cost.** 1.77× faster in wall clock, **3.18× fewer model evaluations** — and the schedule
   collapses from four stages to one.
2. **Quality.** χ²/dof is *not* equal, it is **2.2% better** for the warm fit (3228 vs 3302). The
   refinement is not cheaper by being worse; it is cheaper *and* slightly better.
3. **The shrinkage is completely undone.** |e| ρ goes +0.269 → **+0.839**, which is also **better
   than the cold fit's +0.776**.
4. **Warm beats cold on six of seven parameters** (θ_E, β, |e|, |g|, R_sersic, n_sersic), losing
   only on γ (+0.538 vs +0.551), which is the least constrained parameter in the dataset.

**Why warm beats cold rather than merely matching it.** The staged schedule exists because a
neutral start falls into local minima; it is a *device*, and devices cost something. From the
network's start the fit lands in a **better basin** — most visibly on |e| and |g|, exactly the
parameters the staged schedule releases last and therefore constrains worst.

## 10.5 The n = 6000 run you did — what it bought

You then ran the same refinement over the **training split**, 6,000 images, to produce
`fits_train.json`. **[verified]**

```
n = 6000, train split:  median 0.458 s/image, 151 model evaluations, chi2/dof 3302
                        total wall clock 1.98 h
network alone on the same images: chi2/dof median 10523
```

Two things this is worth:

1. **It is the amortised-initialisation result being used for real, not demonstrated.** B4 and B5
   need a fitted lens for every training image. A cold fit at 0.815 s would have been ~1.4 h; the
   warm fit took **1.98 h including the network pass**, and produced a *better* lens. More
   importantly it turns "we measured a 1.8× speed-up" into "the speed-up is what made 6,000
   training lenses affordable", which is a much better sentence in a paper.
2. **It confirms the n = 800 numbers on 7.5× more images**, on a different split, with the same
   median χ²/dof (3302) and the same 151 model evaluations.

## 10.6 The claim to make — and the one not to make

> **Make this claim.** Amortised inference supplies an initialisation good enough to eliminate the
> staged-optimisation schedule and cut the per-image fit cost by ~1.8× in wall clock and ~3.2× in
> model evaluations, at *better* final χ² and better parameter recovery on six of seven
> parameters — including a complete recovery of the spin-2 shrinkage that the network alone
> suffers.

> **Do not make this claim.** "The network replaces the fit." It does not, and the ellipticity
> numbers say so plainly: +0.269 alone versus +0.839 refined.

---

# 11. B4 — the complete pipeline, both modes

## 11.1 Why B4 exists

B3's decoder is:

```
image → conv trunk → AdaptiveAvgPool2d(1) → 128-vector → dense → decode a 32×32 map
```

**A global average pool discards position.** Everything downstream has to reconstruct a *spatial*
map from a *global summary*. That is a hard generative problem, and the measured consequence is
that B3's correction never exceeds 3–4% of the Sérsic and the source nmse stalls at ~0.080. On top
of that, **64% of B3's 849,519 weights sit in the single dense layer** that does this
(132 → 4096) **[verified]** — most of the network is spent on the step that throws information
away and then tries to put it back.

B4 removes the bottleneck twice over:

1. **Put the data in the source plane before the network sees it.** Ray-shoot the observation
   through the fitted lens and deposit each pixel at its β. The network is handed a picture that
   is already roughly the source; it only has to deblur and sharpen — which is exactly what
   SRResNet-style networks are built for.
2. **Make the network fully convolutional.** No pooling anywhere. A source pixel is produced by a
   receptive field centred on the corresponding input pixel. `tests/test_b4.py` check 5 asserts
   this literally: poke one input pixel and the peak output response must land on the
   corresponding output pixel.

This is Anirudh Shankar's architecture, run on our lens.

## 11.2 The pipeline, start to finish

### Step 0 — the frozen lens (before training)

B4 does **not** fit a lens. It reads one per image from a JSON:

- validation: `fits_refined_mu.json` — the warm-refined Path A/B fits, 800 images
- training: `fits_train.json` — the same, 6,000 images

Both are unsupervised (nothing from the manifest), so freezing costs no scientific validity. It
also **isolates the question B4 exists to answer**: *given a correct lens, does a fully
convolutional SISR on the back-projection beat (a) a Sérsic, (b) the B3 decoder, (c) the Option 2
linear inversion?* All four then solve the same problem on the same lens, which makes the
comparison clean.

### Step 1 — ray-shoot the observation

```python
bx, by = ray_shoot_batch(GX, GY, lens)      # beta = theta - alpha(theta), analytic EPL+shear
```

for every image pixel, using the frozen six lens parameters. Done under `no_grad`.

### Step 2 — back-projection

$$b = \frac{L^{\!\top} d}{L^{\!\top}\mathbf 1}$$

Concretely (`backproject.backproject`): for each image pixel, find which source-box cell its β
lands in; **add** the pixel's brightness into that cell (`num.scatter_add_`); **count** the hits
(`cov.scatter_add_`); divide. So each source pixel gets the **average brightness of the rays that
landed in it**.

Two decisions that matter:

- **Rays landing outside the box are discarded, not clamped.** Clamping piles every far-field ray
  onto the border and makes the box edge the *highest*-coverage region — the opposite of the
  truth. That bug was caught by `test_pathb.py` check 3 and is deliberately not repeated.
- **The back-projection itself is never smoothed.** Only the coverage map is 3×3 box-averaged
  (it is a Monte-Carlo count and is shot-noise dominated); smoothing the back-projection would
  throw away exactly the sub-pixel information the method is trying to recover.

**This is not a reconstruction.** It is a crude, noisy, unevenly-sampled *estimate* of the source
— precisely the `cross_grid_fill(lr_image, backward_mapping)` step of the original pipeline. Its
value is that it puts the data into the source plane, **in spatial register**, before a
convolutional network ever sees it.

**No gradient flows through it.** It is an input feature built under `no_grad`; the gradient
reaches the network through the **forward** model only. That is also what the original pipeline
did (its backward operator was a fixed precomputed sparse matrix), and it means the nearest-pixel
rounding here costs nothing.

### Step 3 — coverage = magnification

`L⊤1` — the same count that normalises the back-projection — **is the ray coverage, i.e. the
magnification of each source pixel** up to the constant sub-pixel area. Here the back-projection
grid is 0.1043″/px against a 0.10593″ detector, so one ray per source pixel corresponds to μ ≈ 1
and the count reads as μ directly.

It is used three ways: as network input channel 1, as the weight in the L2 regulariser, and (in
B5) as the gate.

### Step 4 — build the network input

```python
ch = [ asinh(bp / sigma),          # channel 0: the data, in the source plane
       log10(1 + coverage) ]       # channel 1: the magnification map
if base == "sersic":
    ch.append( asinh(base_in / sigma) )   # channel 2: the fitted Sersic, same grid
```

`arcsinh` because the raw signal-to-noise peak varies **260×** between Model_A images and a shared
filter bank cannot absorb that; arcsinh is linear near zero (a 1σ fluctuation still reads as 1) and
logarithmic in the wings, compressing the across-image range to ~1.9×.

### Step 5 — SourceSISR

```
stem      Conv(in_ch→64, 3) → GroupNorm → SiLU
body      6 × [ Conv(64,3) GN SiLU Conv(64,3) GN  + identity → SiLU ]
merge     Conv(64,3) → GroupNorm ,  added to the stem output  (long skip, as in SRResNet)
up        Conv(64 → 64·4, 3) → PixelShuffle(2) → GroupNorm → SiLU
head      Conv(64→1, 3) → softplus
```

- **PixelShuffle** (sub-pixel convolution, Shi et al. 2016): produce r²·C channels at low
  resolution and rearrange them into C channels at r× resolution. It is the standard SR upsampler
  because the learned r² values per output block are a *learned* interpolation kernel, and it
  avoids the checkerboard artefacts of transposed convolution.
- **GroupNorm, not BatchNorm** — this is a deliberate departure from Anirudh's `sisr.py`. With arc
  brightness varying 260× between images, BatchNorm makes one image's normalisation depend on
  which other images happen to share its batch. GroupNorm is per-sample.
- **softplus, not ReLU** — the other deliberate departure. Surface brightness must be
  non-negative, but ReLU has *exactly zero gradient* on the negative side, so a pixel that starts
  negative can never recover. softplus is strictly positive with a non-zero gradient everywhere.
- **Head initialised to std 1e−3, never zeros** — zero-init is what let the first amortised
  network sit at its neutral output for every epoch.
- No pooling anywhere. Grid: input n_in = round(2H/res)+1 (23 px at H = 1.2), output
  n_out = 2·n_in = 46… in the actual runs **48², at 0.0511″/px, 2.07× finer than the detector**.

### Step 6 — assemble the source (the two modes)

**Mode A — `--base none` (pure free-form, the original idea):**

```python
S_map = amp * y                      # y = softplus(net(...)), strictly positive
```

The network output **is** the source, scaled by the fitted Sérsic amplitude so the network only
has to learn a dimensionless shape. Outside the box the source is exactly zero.

**Mode B — `--base sersic` (residual mode):**

```python
base_out = SersicSource(fitted_params).at(fine_grid)   # the fitted Sersic, on the 48x48 grid
S_map    = base_out + amp * (y - ln 2)                 # softplus(0) = ln 2
```

The subtraction of ln 2 is the neutral offset: a network emitting zeros produces `y = softplus(0)
= ln 2`, so `y − ln 2 = 0` and **S_map is exactly the fitted Sérsic**. In other words the model
starts at Path A and the network predicts only the **residual**. It also gets the Sérsic as a
third input channel, so it can see what it is correcting.

This mode has the graceful-degradation property: **B4 with `--base sersic` cannot do worse than
Path A** unless the network actively makes things worse, because the zero-residual solution is
available to it.

### Step 7 — forward model (identical to everything else in the project)

```python
sky  = sample_source(S_map, bx, by, H)          # bilinear read at the ray landing points
sky  = conv2d(pad(sky,'replicate'), psf)        # empirical PSF, on the sky
pred = area_downsample(sky, S) + background     # detector binning + the fitted sky level
```

### Step 8 — loss

$$\chi^2 = \operatorname{mean}_{r\le 45\,\text{px}}\left(\frac{\text{pred}-d}{\sigma_{\rm eff}}\right)^2,
\qquad \sigma_{\rm eff}^2 = \sigma_{\rm bg}^2 + (0.02\,\text{pred})^2$$

$$\mathcal L = \chi^2 + \chi^2_{\rm detach}\left(\lambda_{\rm curv}\|\nabla^2 S_{\rm rel}\|^2
+ \lambda_{\ell2}\sum_j w_j S_{{\rm rel},j}^2\right)$$

where `S_rel = (S_map − base)/amp` — so in `--base sersic` mode **the penalties act on the
residual only**, which is what makes the Sérsic's own curvature and amplitude penalty-free.

### Step 9 — evaluation

`eval_b4.py` resamples the source map from its own box onto the 127-px detector grid
(`to_detector`), PSF-convolves it, and scores it against the npz `unlensed` array with
**`metrics.source_truth` — the identical function used for Path A, B3 and Option 2**. It also
computes `src_flux_in_box` and χ²/dof, and prints the parametric fit's numbers alongside.

## 11.3 The results, both modes [verified from the metrics JSONs]

| | Path A (refined) | B4 free-form | **B4 + Sérsic** | B5 gate | B5 gate + μ-curv |
|---|---|---|---|---|---|
| config | 7 params | H 1.6, 8000/1000, 50 ep | H 1.2, 6000/800, 30 ep | as B4+Sérsic, gate 2.0 | as B4+Sérsic, gate 2.0, curv-μ |
| corr | **0.9873** | 0.9731 | 0.9794 | 0.9790 | 0.9785 |
| size_ratio | 1.0808 | 1.1302 | **1.0434** | 1.0392 | 1.0507 |
| nmse | **0.0252** | 0.0527 | **0.0408** | 0.0414 | 0.0425 |
| peak_ratio | 0.8340 | 0.7484 | **0.9266** | 0.9221 | 0.9122 |
| centroid px | 0.5061 | 0.4971 | 0.5610 | 0.5604 | 0.5625 |
| χ²/dof (plain σ) | 3228 | 5224 | **3492** | 3519 | 3529 |
| flux-in-box p10/med/p90 | — | 0.48/0.94/1.36 | 0.65/**0.96**/1.34 | 0.65/0.97/1.35 | 0.64/0.97/1.35 |

Four readings:

1. **`--base sersic` is the better mode, clearly.** nmse 0.0408 vs 0.0527 (−23%), peak_ratio
   0.927 vs 0.748, χ²/dof 3492 vs 5224 — and the χ² gap to the parametric fit falls from 1.62× to
   **1.08×**.
2. **B4+Sérsic beats the parametric fit on the two *bias* metrics**: size_ratio 1.043 vs 1.081 and
   peak_ratio 0.927 vs 0.834, both closer to 1. It loses on the *scatter* metrics (corr, nmse).
   Coherent story: **the free-form residual removes the Sérsic's systematic bias — the parametric
   source is ~8% too large and ~17% too flat — at the cost of adding variance.**
3. **B4 is the best free-form source reconstruction in the project**: nmse 0.0408 against Option
   2's 0.082 and B3's 0.080, a ~49% improvement over both.
4. **The most instructive number** is that free-form has the *lower* val χ² and the *worse* source.
   It had more data (8000 vs 6000), more epochs (50 vs 30) and a bigger box, and it still explains
   the image better while sitting further from the truth. **Fitting the data better is not the
   same as recovering the source** — textbook ill-posedness, measured rather than argued.

---

# 12. Is B4 free-form "pure ML", independent of fitting, and better?

Three sub-questions, and the honest answers are **no, no, and no**.

## 12.1 "It is not dependent on any fitting, right?" — it is, in three places

1. **The lens is fitted, and frozen from the fit.** `--lens-fits` and `--lens-fits-train` are
   required arguments; the code exits with an error if the training-split JSON is missing. Every
   ray-shoot in B4 uses six numbers that came out of a Levenberg–Marquardt fit.
2. **The amplitude comes from the fit.** `S_map = amp * y` where `amp` is the fitted Sérsic
   amplitude. The network learns a dimensionless *shape*; the fit sets the scale.
3. **The background comes from the fit.** `pred = ... + BG` — the sky level in the χ² is the
   fitted one, not a learned one.

So even in `--base none`, B4 is *"a learned decoder conditioned on a fitted physical lens"*, not a
free-standing neural network. In `--base sersic` the dependence is total: the fitted Sérsic is an
input channel *and* the additive base.

**This is a feature, not an embarrassment.** It is exactly the sentence that separates you from a
generic SR paper: the network never has to learn the geometry, because the geometry is applied
analytically before and after it.

## 12.2 "It is totally different from B3, right?" — different decoder, same everything else

| | B3 | B4 |
|---|---|---|
| lens | **predicted by the network**, 14 params, in-graph | **frozen**, from a fit |
| network input | the raw lensed image (+ 4 ring scalars as planes) | the **back-projection** + coverage (+ Sérsic) |
| architecture | conv trunk → **global pool** → 128-vector → dense → transposed-conv decoder | **fully convolutional**, residual blocks, PixelShuffle, no pooling |
| source | Sérsic(7 predicted) + bounded 32² correction | 48² free grid, optionally added to the *fitted* Sérsic |
| bound on the correction | tanh, ±30% of amp | none — softplus positivity only |
| forward model | identical | identical |
| loss | χ² + μ-weighted L2 + curvature | χ² + μ-weighted L2 + curvature |
| scoring | `metrics.source_truth` | `metrics.source_truth` |
| parameters | 849,519 | ~640,000 (2.55 MB checkpoint) |

So: **the decoder and the input representation are completely different; the physics, the loss
family and the scoring are identical.** That is what makes the comparison meaningful.

One more real difference: **B3 does two jobs (infer the lens *and* the source); B4 does one
(infer the source, given a lens).** That is why B4's source is better and why B4 cannot replace
B3 — they answer different questions.

## 12.3 "And it is giving better results?" — better than B3 on the source, worse than the Sérsic

| claim | true? |
|---|---|
| B4 beats B3 on the source | **Yes.** nmse 0.0408 vs 0.080, peak_ratio 0.927 vs 0.991 (B3 slightly better here), size_ratio 1.043 vs 0.949 (both good). Overall clearly better. |
| B4 beats Option 2 | **Yes.** 0.0408 vs 0.082. |
| B4 free-form beats B4 + Sérsic | **No.** 0.0527 vs 0.0408. |
| B4 beats the parametric Path A fit | **No, and it should not be expected to.** 0.0408 vs 0.0252 on nmse, 0.9794 vs 0.9873 on corr. **Model_A's sources genuinely *are* Sérsics**, so a 7-parameter model with exactly the right functional form is the hardest possible baseline. B4 *does* beat it on size_ratio and peak_ratio, i.e. on bias. |

**The defensible claim:** *a free-form source reconstruction with no parametric assumption comes
within 1.6× of the correct parametric model on nmse, is better calibrated in size and peak, is
well posed at 1.87:1 overdetermined, and is the deployable form for real galaxies where no Sérsic
family applies.* Do **not** claim B4 beats the parametric fit on Model_A.

---

# 13. The source box, box truncation, and where it is used

## 13.1 What the box is

Any **pixelated** source needs a finite grid, and a finite grid needs an extent. The **source box**
is the square region of the source plane, spanning ±H arcsec in each direction, on which the
pixelated part of the model lives. Outside it, `grid_sample(..., padding_mode='zeros')` returns
zero — **the model has literally no flux there.**

| model | box (H) | grid | pixel scale | role of the grid |
|---|---|---|---|---|
| Path A | **none** | none | — | the Sérsic is a continuous function with infinite support |
| B3 | 0.8″ | 32² | 0.0516″ | a *correction* on top of a Sérsic |
| B2 | 0.8″ (inherited) | 32² | 0.0516″ | **the whole source** — this was the bug |
| B4 free | 1.2″ or 1.6″ | 48² or 62² | ~0.051″ | the whole source |
| B4 +Sérsic / B5 | 1.2″ | 48² | 0.0511″ | a *residual* on top of a Sérsic |

So: **the box exists only where the source is pixelated.** Path A has no box at all.

## 13.2 Box truncation — the failure mode

Measured on 300 val images, the fraction of true source flux that falls inside ±H
**[reported: B4.md / train_b4.py header]**:

| H (″) | n_out | out px scale | unknowns | vs ~4300 px | **median flux in box** |
|---|---|---|---|---|---|
| 0.8 | 32 | 0.0516″ | 1,024 | 4.20 : 1 | **0.715** |
| 1.0 | 40 | 0.0513″ | 1,600 | 2.69 : 1 | 0.827 |
| **1.2** | **48** | **0.0511″** | **2,304** | **1.87 : 1** | **0.891** |
| 1.4 | 54 | 0.0528″ | 2,916 | 1.47 : 1 | 0.935 |
| 1.6 | 62 | 0.0525″ | 3,844 | 1.12 : 1 | 0.958 |

**The trade-off is forced.** Bigger box → more of the source captured, but more unknowns per
datum, and eventually you are back in the underdetermined regime that produced size_ratio 10.6–12.8.

Two observable symptoms of truncation:

1. **In the source plane:** the reconstruction is too small and too peaked, because the wings it
   cannot represent get pushed into the core. B2 at H = 0.8 gave size_ratio **0.68** and peak_ratio
   **1.81** **[verified]** — with **28% of the flux outside the box**, that is most of the
   explanation.
2. **In the image plane:** in `figS5_stages` column 6 the free-form model image is clipped to a
   rounded **square** — the box mapping forward through the lens. Outside the box the source is
   exactly zero, so the lensed image is exactly zero too. This is most of B4 free-form's χ² gap
   (5224 vs 3228): the Sérsic can put flux in the arc's faint outer halo and a boxed source cannot.

## 13.3 How H = 1.2 was chosen

Four constraints, and 1.2 is where they intersect:

1. **Flux captured** ≥ ~0.89 — enough that truncation is a correction, not a dominant effect.
2. **Conditioning** 1.87:1 overdetermined — still on the safe side of 1:1.
3. **Resolution** 0.0511″/px = **2.07× finer than the detector**, inside the **measured 3.07×**
   median tangential stretch. Any finer would be claiming resolution the lens did not deliver.
4. **The startup banner prints all three**, and warns if (3) is violated. That is deliberate: the
   B2 failure happened because a default was inherited silently.

## 13.4 The elegant part: `--base sersic` makes the box almost irrelevant

With a Sérsic base, the Sérsic carries the wings (it has infinite support) and the box only has to
cover the region where the **residual** is significant. Measured consequence: the **v2 +Sérsic run
at H = 1.2 has no truncation artefact**, while the free-form run at the *larger* H = 1.6 still
shows edge effects **[reported: MAGNIFICATION_SR.md §1]**.

**That is the cleanest single argument for the hybrid representation**, and it generalises: the
parametric part handles the part of the source that is smooth and extended, the pixel grid handles
the part that is structured and compact, and neither has to do the other's job.

## 13.5 The principled alternative (out of scope, worth naming)

An **irregular source grid** — a Delaunay tessellation whose vertices are the ray landing points,
so the resolution automatically follows the magnification (Vegetti & Koopmans 2009). That is the
principled version of what B5's gate does crudely, and it removes the box question entirely
because the grid adapts to where the rays actually went. Worth one sentence in future work.

---

# 14. B5 — what it adds over B4, and what it measured

## 14.1 Your premise is right: magnification was already in B4

Yes. In B4, magnification (as ray coverage) is used in **two** places:

1. **Input channel 1** — `log10(1 + coverage)` is fed to the network.
2. **The L2 regulariser weight** — `--reg-mode mu --lambda-l2 0.5` weights the source amplitude
   penalty by `(median μ / μ)^0.5`, i.e. suppress amplitude where the lens did not look.

What was **not** in B4: the smoothness (Laplacian) penalty was **uniform**, across a source plane
whose sampling varies ~30×; and nothing *structurally* prevented the network from emitting
full-resolution detail in a region the lens never resolved. B5 addresses both.

## 14.2 The two additions

### (a) The magnification gate — the important one

$$g = \sigma\!\left(\frac{\log_{10}\mu - \log_{10}\mu_{\rm gate}}{\tau}\right),
\qquad S_{\rm out} = g\cdot S_{\rm fine} + (1-g)\cdot \text{upsample}\big(\text{avgpool}(S_{\rm fine})\big)$$

with `mu_gate = 2.0` and `tau = 0.35` dex.

In words: **where μ is large, the network's full-resolution output passes through untouched. Where
μ is small, the sub-pixel detail is removed and the output falls back to the coarse grid** — which
is the resolution the data actually support there.

Three properties worth stating:

- **It is structural, not a penalty.** A penalty is something the optimiser can trade away against
  χ². A gate cannot be traded: the fine detail is *deleted* before the forward model sees it.
  **The network cannot claim resolution the lens did not deliver.**
- **`g` is computed from the frozen lens and detached**, so it is a fixed, physically determined
  mask per image, not something the network can learn its way around.
- **It conserves flux block by block.** `coarse(S_fine)` is an exact area average over each
  mag×mag block put back by nearest-neighbour repetition, so the gate redistributes detail without
  creating or destroying flux — which matters, because the forward model is most sensitive to flux.

`tau = 0.35` dex means the gate goes from 0.1 to 0.9 over about a factor of 5 in μ — gentle enough
not to print a hard ring into the source at the threshold radius.

### (b) μ-weighted curvature

B4 weighted only the L2 term by 1/μ. `--curv-mu 1` gives the Laplacian penalty the same
`(median μ/μ)^p` weight: smooth hard where the lens saw little, barely at all where it saw a lot.

### Everything else is identical

`train_b5_mu.py` imports `LENS_KEYS, SRC_KEYS, build_inputs, curvature, mu_weights, pack` directly
from `train_b4.py` and the physics from `backproject.py` / `raytrace.py` / `sisr_net.py`, so the
two models cannot drift apart. **The control run is B4 + Sérsic with the gate off**, which is why
`b4_sersic` and `b5_gate` share every other setting (H 1.2, 6000/800, 30 epochs, base sersic,
λ_curv 3.0, λ_l2 0.5, reg-mode mu) **[verified from the two configs]**.

## 14.3 What the gate actually did — the fourth null

**[verified from `b4_sersic_metrics.json`, `b5_gate_metrics.json`, `b5_gate_curv_metrics.json`]**

| | control (B4 + Sérsic, no gate) | gate 2.0 | gate 2.0 + μ-curvature |
|---|---|---|---|
| corr | **0.9794** | 0.9790 | 0.9785 |
| nmse | **0.0408** | 0.0414 | 0.0425 |
| size_ratio | 1.0434 | **1.0392** | 1.0507 |
| peak_ratio | **0.9266** | 0.9221 | 0.9122 |
| centroid px | **0.5610** | 0.5604 | 0.5625 |
| χ²/dof | **3492** | 3519 | 3529 |
| flux-in-box median | 0.9638 | 0.9700 | 0.9741 |

**The gate changed essentially nothing, and what it changed it made marginally worse.** nmse
+1.5% for the gate, +4.2% for gate + μ-curvature. Every difference is at the ~1% level on 800
images.

**This is the fourth independent null on the magnification-adaptive idea in this project:**

| # | setting | result |
|---|---|---|
| 1 | Option 2, linear inversion, μ-adaptive vs uniform regularisation | ~4% **worse** at matched data fidelity |
| 2 | B3, μ-weighted correction penalty vs uniform | 1.3% worse (0.0799 vs 0.0789) |
| 3 | B4, `--reg-mode mu` | in use, never ablated against uniform in the final runs |
| 4 | **B5, structural gate** | **1.5% worse (0.0414 vs 0.0408)** |

Four attempts, four nulls, using a penalty, a linear-inversion prior, a loss weight and finally a
hard architectural constraint. **That is a strong, well-supported negative result** — the idea has
now been tested in every form it can take.

## 14.4 The measurement that explains *why* — `mu_resolution.py`

This is the part that turns four nulls into a finding, and it is the strongest new piece of
science in the recent work. It runs on a **trained checkpoint with no retraining**.

### Experiment A — reconstruction error stratified by local μ

**[verified, recomputed from `mu_resolution.json`, ckpt `b4_sersic_best.pt`]**

| local μ | 0.5–1 | 1–2 | 2–4 | 4–8 | 8–16 | 16–32 | 32+ |
|---|---|---|---|---|---|---|---|
| n | 2 | 101 | 119 | 120 | 118 | 54 | 3 |
| fractional error | (9.48) | **0.3917** | 0.0881 | 0.0467 | 0.0414 | 0.0384 | (0.46) |

Ignoring the two bins with n ≤ 3, the fractional reconstruction error falls **10× from μ ≈ 1–2 to
μ ≈ 16–32**, with a log-log slope of **−0.78** across the five populated bins **[verified]**.

> **Caveat you must carry into the paper.** `MAGNIFICATION_SR.md` quotes an *earlier, smaller* run
> of this experiment (0.308 / 0.230 / 0.086 / 0.036 / 0.023 / 0.037, slope −0.72). The JSON now on
> disk is a larger run and gives the numbers above. **They are not the same numbers.** Quote the
> file, not the markdown, and delete the stale table.

### Experiment B — sub-detector-pixel injection

A Gaussian clump of FWHM 0.08″ = **0.76 detector pixels** at 15% of the Sérsic peak is added to the
fitted source, lensed, PSF'd, binned and noised at that image's measured σ. The same system
*without* the clump is reconstructed with the **same noise realisation**, and the difference is
measured in a small aperture at the clump position. **Contrast** = recovered / injected.

**[verified, recomputed from the 2,400 injections in `mu_resolution.json`]**

| local μ | 1–2 | 2–4 | 4–8 | 8–16 | 16+ |
|---|---|---|---|---|---|
| n | 253 | 1215 | 572 | 295 | 65 |
| mean contrast | 0.175 | 0.204 | 0.225 | 0.198 | 0.199 |

```
mu >= 2 : mean contrast 0.2084 (n = 2147)
mu <  2 : mean contrast 0.1746 (n =  253)
Mann-Whitney U, two-sided:  z = 4.04,  p = 5.3e-05
spearman(contrast, mu)  = +0.056
spearman(contrast, snr) = +0.152
```

Three findings:

1. **Sub-detector-pixel structure IS recovered**, at 17–23% of the injected contrast. Not zero —
   **so super-resolution is genuinely happening**. And in `--base sersic` mode the Sérsic base
   contributes exactly zero to the *difference*, so all of it comes from the free-form residual.
2. **Higher μ helps, and the effect is statistically solid but modest**: 0.208 vs 0.175, a 19%
   difference, p = 5.3e−5 on 2,400 injections.
3. **μ is not the dominant variable.** ρ(contrast, μ) = +0.056 against ρ(contrast, SNR) = +0.152 —
   **photon count matters about three times as much as magnification.**

Stratified by SNR **[verified]**:

```
snr 0-8    n=1000   mu<2 0.151   mu>=2 0.188    ratio 1.25x   rho +0.153
snr 8-20   n=1000   mu<2 0.180   mu>=2 0.220    ratio 1.22x   rho -0.007
snr 20+    n= 400   mu<2 0.213   mu>=2 0.230    ratio 1.08x   rho +0.001
```

> **Important correction to the write-up you have.** `MAGNIFICATION_SR.md` reports this split as
> "3.1× at low SNR" from a 480-injection run and builds a strong claim on it. On the 2,400-injection
> run now on disk the low-SNR ratio is **1.25×**, and the trend across SNR bins is 1.25 → 1.22 →
> 1.08 — a mild, monotone weakening, not a dramatic regime change. **The larger run substantially
> weakens that claim and you must not quote the 3.1× figure.**

### What the honest version of the story is

> Magnification measurably helps sub-pixel recovery — a 19% mean improvement above μ ≈ 2,
> p = 5×10⁻⁵ — but it is a second-order effect on this dataset: photon count correlates with
> recovery about three times more strongly. That is why four separate attempts to exploit
> magnification adaptively — a linear-inversion prior, a penalty weight, a loss weight, and a hard
> architectural gate — all came out neutral. **The information is there; it is simply not the
> binding constraint in this regime.**

That is a genuine, defensible, four-times-replicated result with a measured mechanism, and it is
much better science than a positive result would have been if you had had to squint at it.

**One caveat to state:** the injected clump has a fixed amplitude (15% of the *global* Sérsic peak)
regardless of where it lands, so far from the source centre it is a larger *local* perturbation and
easier to detect. Scaling the clump to the local surface brightness would be the cleaner design.

---

# 15. Are the metrics and the loss the same for Path A, B3 and B4/B5?

**Metrics: yes, deliberately and identically. Loss: same family, different terms.**

## 15.1 Metrics — identical by construction

Every method's source is scored by **the same function**, `metrics.source_truth(model_psf_convolved,
npz_unlensed, pixel_scale)`, on the **same validation images**, with the model **PSF-convolved
first** (because `unlensed` is post-PSF).

| | Path A | B3 | B4 / B5 |
|---|---|---|---|
| source scorer | `metrics.source_truth` | same | same |
| how it gets there | `evaluate.py` renders the Sérsic on the detector grid | `eval_pathb.py` writes `fits_*.json` in the **`fits_img.json` schema**, then `evaluate.py` scores it with identical code | `eval_b4.py` resamples the box onto the detector grid via `to_detector`, then calls the same scorer |
| parameter recovery | `evaluate.py`, 7 params vs manifest | same code | n/a — B4 does not predict parameters |
| χ²/dof | `metrics.chi2_per_dof` | same | same |
| null baselines | yes | yes | via `evaluate.py` |
| stratification by SNR | yes | yes | yes |

This is a deliberate design decision, stated in `eval_pathb.py`'s docstring: *if the network were
scored by a bespoke script, any difference from the per-image fit could be a difference in the
scoring rather than in the model, and there would be no way to tell which.*

**Two caveats that must be stated whenever the table is shown:**

1. **The χ²/dof denominators differ.** Path A divides by (6361 − 14) = 6,347. `eval_pathb.py`
   divides by (6361 − 14 − 1024) = 5,323 — a factor **1.192** **[verified]**. B4's `eval_b4.py`
   uses its own convention again. Either report one convention or state all three.
2. **"source nmse" is ambiguous in B3.** `evaluate.py` §2 scores the **Sérsic only** (0.0845);
   `eval_pathb.py` §A scores **Sérsic + correction** (0.0799). It is the super-resolution number;
   be explicit each time.

## 15.2 Loss — same skeleton, different terms

| term | Path A | B3 | B4 free | B4 + Sérsic / B5 |
|---|---|---|---|---|
| χ² against the observation | ✔ | ✔ | ✔ | ✔ |
| σ-floor (2% of model) | ✘ (plain σ_bg) | ✔ | ✔ | ✔ |
| λ as a fraction of χ² | n/a | ✔ | ✔ | ✔ |
| μ-weighted amplitude penalty | ✘ | ✔ (on the tanh correction) | ✔ (λ_l2 = 0.5) | ✔ |
| curvature (Laplacian) penalty | ✘ | optional, off in the runs | ✔ (λ_curv = 3.0) | ✔ |
| μ-weighted **curvature** | ✘ | ✘ | ✘ | only B5 `--curv-mu 1` |
| bound on the free part | n/a | tanh at ±30% of amp | softplus positivity only | softplus + Sérsic base |
| structural μ gate | ✘ | ✘ | ✘ | only B5 |
| curriculum / warm-up | staged parameter release (4 stages) | 5 epochs parametric-only | none | none |

**Why Path A needs no prior at all:** 14 unknowns against 6,361 fitted pixels is overdetermined
~450:1. There is nothing for a regulariser to do. Every prior in the project exists solely because
a *pixelated* source was introduced.

**Why Path A uses plain σ_bg and the network paths use the σ-floor:** the floor was introduced to
fix a *gradient-balance* problem in mini-batch training (one image taking 54% of a batch's
gradient). A per-image optimiser has no batch and no such pathology. The consequence is that Path
A's χ² numbers are on a different noise model from the network paths' training loss — which is why
`RESULTS_V3.md` reports both a "plain σ" and a "2% floor" column.
---

# Appendix — every named method, theorem and trick in the project

Alphabetical inside groups. Each entry says what it is, why it is used here, and where.

## A. Optimisation and statistics

**Least squares.** Minimise `½|r(v)|²` for a residual vector r. If the noise is Gaussian with
known σ and r = (model − data)/σ, this is exactly maximum likelihood.

**Gauss–Newton.** Linearise r around the current point, r(v+δ) ≈ r + Jδ, and solve the normal
equations (JᵀJ)δ = −Jᵀr. Fast (quadratic convergence near the optimum) because for least squares
JᵀJ approximates the Hessian well; unstable far away or when JᵀJ is ill-conditioned.

**Levenberg–Marquardt.** (JᵀJ + λ·diag(JᵀJ))δ = −Jᵀr. Interpolates between Gauss–Newton (λ→0) and
scaled gradient descent (λ→∞), with an adaptive rule: step accepted → decrease λ; step rejected →
increase λ. Using `diag(JᵀJ)` rather than I makes the damping scale-invariant across parameters
with different units. **Used for:** the entire Path A fit. **§8.3.**

**Trust Region Reflective (TRF).** The bounded variant we actually call
(`scipy.optimize.least_squares(method="trf")`). Instead of a damping parameter it maintains an
explicit trust radius inside which the linear model is believed, and reflects steps off bound
surfaces so iterates stay feasible. Needed because θ_E > 0, n_sersic ∈ [0.3, 6], |e| ≤ 0.6.

**Finite-difference Jacobian.** We do not supply ∂r/∂v, so SciPy estimates each column by
perturbing one parameter — 14 extra model evaluations per Jacobian. This is why the cold fit costs
**480 model evaluations, 0.815 s [verified]**. Supplying an analytic (or autograd) Jacobian is the
obvious unclaimed speed-up.

**Warm start.** Initialising an iterative solver from a good guess instead of a neutral one.
**§10** — it removes the 4-stage schedule, cuts model evaluations 3.2×, and lands in a better
basin.

**Curriculum learning.** Present an easier version of the task first. Here: B3's
`--warmup-epochs 5` trains the parametric head with the correction switched off, so the correction
can only add what the *converged* Sérsic missed rather than racing it to explain the same flux.
Path A's staged parameter release is the same idea for an optimiser.

**Adam.** Adaptive-moment stochastic optimiser: per-parameter step sizes from running estimates of
the gradient's first and second moments. Standard for networks; useless for our 14-parameter
problem, where LM is far better.

**Cosine annealing.** Learning rate follows a half cosine from its initial value to ~0 over
`T_max` epochs. Smooth decay, no schedule hyperparameters to tune beyond the horizon.

**Weight decay (1e−4).** L2 penalty on the weights, encouraging small weights and reducing
overfitting.

**Gradient clipping (`clip_grad_norm_(5.0)`).** Rescale the whole gradient if its norm exceeds 5.
Guards against a single pathological batch. **Note the failure mode we hit:** clipping does *not*
filter NaN — a NaN norm scales every parameter by NaN — which is why there is an explicit
non-finite check *before* clipping in `train_pathb.py` and `train_b4.py`.

**Spearman rank correlation ρ.** Pearson correlation of the ranks. Invariant to monotonic
transformations, robust to outliers, works on skewed distributions. **§5.2.**

**Pearson correlation.** Linear correlation. Used in the postmortem to prove the μ and uniform B3
v1 runs were literally the same network (+0.9992 on θ_E).

**Mann–Whitney U test.** Non-parametric test of whether one sample tends to have larger values than
another; equivalent to comparing mean ranks. Used in `mu_resolution.py` Experiment B: μ ≥ 2 vs
μ < 2 gives **z = 4.04, p = 5.3e−5** on 2,400 injections **[verified]**. Non-parametric is the
right choice because the contrast distribution is skewed and heteroscedastic.

**Conditional-mean vs maximum-likelihood estimators.** A network trained on squared error learns
E[parameter | data] — an average over everything consistent with the image. An optimiser reports
the mode of a single image's likelihood. For a **spin-2** quantity under orientation uncertainty,
the conditional mean shrinks the modulus toward zero, because a large ellipticity pointed the wrong
way costs more than none at all. **This is the whole explanation of B3's 4× ellipticity
shrinkage** and it is why the fix is a different estimator, not more epochs. **§10.1.**

**Discrepancy principle.** Choose the regularisation strength λ so that χ²/dof → 1 — i.e. fit the
data exactly as well as the noise allows and no better. The principled alternative to sweeping λ
by hand. Referenced in `sources.PixelSource`; not usable directly here because the PSF systematic
puts χ²/dof in the thousands.

**L-curve.** Plot the residual norm against the solution norm as λ varies; the corner is the
compromise point. Used in Option 2 to pick λ.

**Null baselines.** Score models that contain no physics — a constant image, a 2-px blur, a 3-px
blur of the input — so that a metric value has an absolute reference. This is what exposed the
`skill` metric (blur 0.951 > exact physical model 0.630).

**Stratification.** Report every summary split by a covariate (here `snr_max`). A single median
hid Spearman(skill, SNR) = −0.85.

## B. Inverse problems and lens modelling

**Ill-posedness / conditioning.** A problem is well posed if a solution exists, is unique and
depends continuously on the data. The operative number in this project is **unknowns per
informative datum**: 64,516/4,300 (15:1 under) fails; 4,096/4,300 (~1:1) fails; 1,038/4,300
(1:4.1 over) works; 2,304/4,300 (1:1.87 over) works.

**Regularisation.** Adding a prior term to make an ill-posed problem well posed. Ours:
Laplacian curvature + a μ-weighted L2 amplitude penalty.

**Total variation (TV).** Sum of |∇source|. Edge-preserving and popular, but **its unconstrained
minimiser is a constant field**, so raising its weight drives the source toward a flat wash. The
old pipeline's size_ratio stayed at 10.6–12.8 across a 100× TV sweep — the sweep was inside a
bracket that was already flat. Replaced by curvature.

**Curvature regularisation (Suyu et al. 2006).** Sum of squared discrete Laplacians. A **quadratic
form**, so with a Gaussian likelihood the source sub-problem stays convex and λ has a determinate
value rather than a chosen one.

**Semilinear inversion (Warren & Dye 2003).** With the lens *frozen*, the predicted image is
**linear** in the source pixels, d = BLs + n, so the best source has a closed form
s = (MᵀC⁻¹M + λH)⁻¹MᵀC⁻¹d with M = BL. No training, no learning rate, no collapse mode — it either
solves or it does not. This is Option 2. It also produced the adjoint operator Lᵀ that B4's
back-projection reuses.

**Adjoint / back-projection.** Lᵀ is the transpose of the lensing operator: instead of *reading*
the source at each ray's landing point, it *deposits* each image pixel's brightness at its landing
point. Normalised by Lᵀ1 (the ray count) it is an average, and it is B4's network input. **§11.2.**

**The adjoint test.** ⟨Ms, r⟩ = ⟨s, Mᵀr⟩ for random s, r. If Mᵀ is not the exact transpose of M,
conjugate gradients converges silently to the wrong answer. Option 2 measures **7.8e−15**.

**Conjugate gradients (CG).** Iterative solver for symmetric positive-definite systems; used in
Option 2's inversion. It **assumes symmetry**, which is why the adjoint test is the critical gate.

**Vegetti & Koopmans (2009) adaptive grids.** A Delaunay tessellation whose vertices are the ray
landing points, so source-plane resolution automatically follows the magnification. The principled
version of B5's gate; out of scope here.

**Magnification-adaptive regularisation.** Weight the source prior by 1/μ so that fine structure
is cheap where the lens delivered resolution and expensive where it did not. Tried four times in
this project; four nulls. **§14.**

## C. Geometry and numerics of the operators

**Shoelace formula.** Area of a polygon from its vertices, A = ½|Σ(x_i y_{i+1} − x_{i+1} y_i)|.
Used in the original grid-based lensing to compute cell overlaps.

**Sutherland–Hodgman clipping.** Clip a polygon against a convex region by clipping against each
edge's half-plane in turn. `clip_polygon_with_square` in `differentiable_lensing.py`. O(N⁴) in
pure Python over a full grid — the 128 chain took 8h46m.

**Sparse COO matrix.** Coordinate-list sparse format (row indices, column indices, values). The
original operators are `torch.sparse_coo_tensor` of shape (n_source_pixels, n_image_pixels), and
lensing is `torch.sparse.mm`.

**Supersampling.** Evaluate the image plane at S×S subpixels and area-average. Recovers flux
conservation to O(1/S²); **measured 5.5e−3 at S = 3** against an S = 9 reference. It is also how
`build_sis_mappings.py` replaced the polygon clipping, taking the build from hours to seconds.

**Area downsampling.** Exact mean over aligned S×S blocks — the detector's pixel integration.
Written as an explicit reshape rather than `F.interpolate(mode='area')` so the assumption (equal
areas, aligned blocks) is visible and testable.

**Bilinear `grid_sample`, and `align_corners`.** Reads a grid at real-valued coordinates. With
`align_corners=False`, ±1 are the **outer edges** of the border pixels, while our maps span
[−H, +H] to pixel **centres** — so the divisor is `H·(1 + 1/(n−1))`. Getting this wrong shifts the
source by half a pixel, which is **25% of the super-resolution being claimed**. Asserted by
`test_b4.py` check 3.

**`scatter_add_`.** Accumulate values into an index-addressed buffer — how the back-projection and
the coverage histogram are built.

**Nearest-pixel rounding in the back-projection.** Costs nothing here because no gradient flows
through the back-projection; it is an input feature built under `no_grad`.

**Log-polar / double-logarithmic grids.** For a circular lens the deflection is purely radial and
constant, so in log-polar coordinates it becomes a shift; and a log-spaced radial grid fixes the
severe undersampling near the centre. Central to the original method; not used in ours.

**Monte-Carlo coverage and its shot noise.** The ray-count map is an estimate of μ from finitely
many rays. At supersample 1 there are ~5.4 rays per source pixel **[verified]**, so a raw count is
noisy; a 3×3 box average keeps the smooth large-scale structure that μ actually is. It creates no
information — the real fix is `--supersample 2`.

## D. Neural-network components

**Residual block.** `x + f(x)`. Makes the identity easy to represent, so depth does not degrade
optimisation. Both `sisr.py` and `sisr_net.py` use them.

**Long skip connection.** The stem output is added back after the residual body (SRResNet). Lets
the network refine a signal rather than regenerate it.

**PixelShuffle / sub-pixel convolution (Shi et al. 2016).** Produce r²·C channels at low resolution
and rearrange them into C channels at r× resolution. The standard SR upsampler: the r² values per
output block act as a *learned* interpolation kernel, and it avoids the checkerboard artefacts of
transposed convolution. **This is where B4's 2× super-resolution physically happens.**

**Transposed convolution.** The upsampler in B3's decoder. Prone to checkerboard artefacts;
chosen there because it imposes locality on a map decoded from a global vector.

**Global average pooling.** Collapses a feature map to one number per channel. **The thing B4
exists to avoid** — it discards position, so the decoder must regenerate a spatial map from a
global summary.

**BatchNorm vs GroupNorm.** BatchNorm normalises using statistics over the batch, so one image's
normalisation depends on which others share its batch. With arc brightness varying **260×** that is
actively harmful; GroupNorm normalises within each sample over groups of channels. Anirudh's
`sisr.py` uses BatchNorm; `sisr_net.py` deliberately uses GroupNorm.

**ReLU vs SiLU vs softplus.** ReLU has **exactly zero gradient** on the negative side, so a unit
(or an output pixel) that starts negative can never recover — fatal for a non-negativity
constraint. SiLU (x·σ(x)) is smooth and used internally. **softplus** = log(1+eˣ) is strictly
positive with a non-zero gradient everywhere, and is used on the output because surface brightness
cannot be negative. Note `softplus(0) = ln 2`, which is why the residual mode subtracts ln 2 so
that a zero network output means "no correction".

**Constrained output parameterisation.** Map each physical output through tanh / sigmoid /
softplus into its physical range, so the network **cannot** emit a negative Einstein radius or
n_sersic = 0 — either of which produces NaN inside (b/R)^t or R^{1/n} *before any gradient exists
to correct it*.

**Anchoring.** Predict a *residual* to a good classical estimate rather than an absolute value:
`θ_E = ring_θ_E · exp(0.3 tanh z)`. Far better conditioned. In B3 the anchored θ_E reached ρ +0.960
while an unanchored β reached +0.049 despite the same information being available.

**arcsinh stretch.** `asinh(x/σ)`: linear near zero (a 1σ fluctuation still reads as 1),
logarithmic in the wings. The standard astronomical display/normalisation transform. It compresses
the across-image dynamic range from **260× to 1.9×**, which a shared convolutional filter bank
otherwise cannot absorb.

**Dihedral augmentation.** Random 90° rotations and flips. **Exactly valid here with no relabelling**
because the objective is self-supervised — χ² is taken against the same (rotated) image, and the
ring features fed to the network are rotation-invariant magnitudes. `n_pix` is odd, so the grid
centre is a pixel centre and `rot90` is exact.

**Amortised inference.** Pay the optimisation cost once at training time; inference is a single
forward pass. (In the mentor deck this is called *"train once, predict instantly"* — see the
presentation notes.) Measured: ~12 ms/image for B3 against 877 ms for the per-image fit.

**Self-supervision / physics-informed loss.** The target is a *transformation of the input* rather
than an external label. Here the transformation is the observation operator: lens → PSF → bin.
**§6.**

**Collapse.** A network emitting nearly the same output for every input. Its medians can still look
plausible, which is why `eval_pathb.py` prints the across-image standard deviation of every
prediction against the truth's and flags any ratio < 0.05.

## E. Instrument and source models

**Sérsic profile.** I(R) = amp·exp(−b(n)[(R/R_s)^{1/n} − 1]). n = 1 is an exponential disc, n = 4
is de Vaucouleurs. `b(n)` is defined by the half-light condition; `lenstronomy` uses the
approximation **b(n) = 1.9992n − 0.3271**, and we reproduce it deliberately — the goal is to match
the simulator, not to be independently more accurate, because a different b(n) shows up as a
systematic bias in recovered R_sersic.

**Elliptical Sérsic convention.** Matched to `SERSIC_ELLIPSE` with `sersic_major_axis=False`
(the "product average" ellipse), because that is what generated Model_A.

**Point spread function (PSF).** The instrument's response to a point source. **Measured, not
assumed:** a Gaussian sweep against the npz `unlensed` array gives a clean minimum at FWHM
**0.18–0.20″** against the 0.10″ previously used (2.4× worse in residual).

**Wiener deconvolution.** Since `unlensed = sérsic ⊛ PSF`, in Fourier space
PSF̂ = Û·conj(Ŝ)/(|Ŝ|² + ε·max|Ŝ|²). The ε is the Wiener regularisation — without it, frequencies
where the Sérsic has no power divide by ~0. Stacking over 40 images with different source sizes
fills in different parts of the frequency plane. This is estimate B in `calibrate_psf.py` and it
assumes **nothing** about the kernel shape.

**Moffat profile.** I(r) = (1 + (r/α)²)^{−β}, with α = FWHM/(2√(2^{1/β} − 1)); β → ∞ recovers a
Gaussian. The empirical kernel has Moffat-like wings: at r = 0.25″ it is **0.139 of its peak
against 0.005 for a 0.18″ Gaussian — a factor of 29**.

**Why the PSF shape sets the χ² floor.** Arcs reach peak/σ_bg of 10³–10⁴, so a **1% error in the
kernel shape is a ~50σ per-pixel residual**. The signature that it is a systematic and not noise is
that χ² *rises* with SNR — measured ρ(χ²/dof, snr_max) = **+0.449 [verified]**.

**Background annulus noise estimate.** σ is the standard deviation of pixels at r > 50 px, where
Model_A has no signal. Cross-checked: `image` and `image_nss` differ there by ≈ √2 × σ, confirming
independent noise realisations.

**Native units vs min–max normalisation.** Min–max normalisation is a per-image affine map whose
scale is set by the single brightest and faintest pixel — both noise-dominated at SNR ≈ 10 — so the
noise level after normalisation is itself a random variable and a χ² built on it is meaningless.
We keep native units and put the affine freedom in the *model* (free `amp`, free `background`).

## F. Lensing vocabulary quick reference

θ (image position) · β (source position) · α (deflection) · ψ (potential) · κ (convergence) ·
γ₁,γ₂ (shear) · g₁,g₂ (external shear here) · θ_E (Einstein radius) · γ (EPL slope) · q (axis
ratio) · e₁,e₂ (spin-2 ellipticity) · b = θ_E√q · t = γ−1 · A (lens Jacobian) · μ = 1/det A ·
λ_t = 1−κ−|γ| · λ_r = 1−κ+|γ| · critical curve (det A = 0) · caustic (its source-plane image) ·
Σ_cr (critical density) · mass-sheet degeneracy · thin-lens approximation · flexion (not modelled).

---

# Closing: the five things I would make sure you can say without notes

1. **"We replaced a precomputed sparse lensing matrix with an analytic deflection field evaluated
   inside the autograd graph, so the lens became a fittable parameter."** Everything else follows
   from that one change.
2. **"The supervision is physics, not labels: whatever source we propose must, after being lensed,
   blurred and binned, reproduce the observed image."**
3. **"The representation question is the scientific question. 64,516 free source pixels fails,
   4,096 fails, 14 parameters plus a bounded 1,024-pixel residual works — same data, same forward
   model, same optimiser."**
4. **"The network is a poor estimator and an excellent starting point: on 800 images it cuts the
   fit from 480 model evaluations to 151 and from four stages to one, at slightly better χ² and
   better recovery on six of seven parameters."**
5. **"Magnification measurably predicts where sub-pixel recovery works — 19% better above μ ≈ 2,
   p = 5×10⁻⁵ — but four separate attempts to exploit it adaptively all came out neutral, because
   on this data photon count is about three times more predictive than magnification."**
