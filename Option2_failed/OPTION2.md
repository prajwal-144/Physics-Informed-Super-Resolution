# Option 2 — free-form source reconstruction on the frozen lens

Three new files. Nothing existing was modified.

| file | what it does |
|---|---|
| `pixel_source.py` | the machinery: builds the sparse lensing operator L from a frozen lens, applies M = B·L and its exact adjoint matrix-free, solves by conjugate gradients, chooses λ by the L-curve corner, and implements magnification-adaptive regularisation |
| `fit_pixel_source.py` | the driver: reads `results/fits_img.json`, solves per image in both regularisation modes, scores against `unlensed`, writes JSON + example arrays |
| `tests/test_pixel_source.py` | four checks: adjoint, H symmetry, round-trip on a known source, and recovery of sub-pixel structure |

---

## What it does, physically

`fit_per_image.py` gives a 7-parameter Sérsic. A Sérsic can only make
Sérsic-shaped things. Here the source becomes a free 64×64 grid — 4,096 unknowns
— and we solve for all of them.

**The trick that makes this tractable:** with the lens **frozen** from Path A, the
predicted image is *linear* in the source pixels:

```
d = B L s + n
```

so the best source has a closed form,

```
s = (Mᵀ C⁻¹ M + λ H)⁻¹ Mᵀ C⁻¹ d ,     M = B L
```

This is the semilinear inversion of Warren & Dye (2003). **No training, no
learning rate, no collapse mode.** It either solves or it doesn't.

**Why it's well posed now and wasn't before.** The old pipeline had 254² = 64,516
free source pixels against ~4,300 informative data pixels — 15:1 underdetermined
— with a *wrong* lens and θ_E leaked from the manifest. Here it's 4,096 against
6,361 (**0.64 unknowns per datum**) with a lens fitted per image that recovers
ellipticity at ρ = 0.76.

**Magnification, used rather than reported.** Where |μ| is large the sky has
already oversampled the source, so fine source pixels are well constrained there
and should be smoothed less. We get |μ| free and self-consistently: the **column
sums of L** count how many image subpixels read from each source pixel, and that
count *is* the magnification. No separate μ computation, so no possibility of the
μ map and the ray-tracer describing different lenses — exactly the defect in
`physics_losses.fixed_sis_magnification`.

---

## What to run, in order

From the **`Grid_Based_Experiment` root**. Requires `results/fits_img.json` to
exist (Path A).

### Step 1 — validate the machinery (~1 min)

```bash
python superres/tests/test_pixel_source.py
```

**Expect, and these are the numbers I measured:**

```
adjoint test 0:  <Ms,r> = +1.41762545e+00   <s,M^T r> = +1.41762545e+00   rel 7.8e-15
H symmetry:      rel 4.2e-16
source grid  64^2 at 0.0381 arcsec/px
detector     127^2 at 0.1059 arcsec/px   -> 2.78x finer
ray coverage per source pixel: median 1.4   max 24.1   (this IS the magnification)

smooth Sersic          :  corr 0.9998   nrmse 0.0160
Sersic + sub-pixel blob:  corr 0.9999   nrmse 0.0157
PASS
```

**The adjoint test is the one that matters.** CG assumes the operator is
symmetric; if `MT()` isn't the exact transpose of `M()`, CG converges silently to
the wrong answer. 1e-14 agreement means it's right.

**Honest caveat on test 3.** The sub-pixel blob is a small perturbation on a much
brighter Sérsic, so a correlation of 0.9999 is dominated by the Sérsic and does
*not* isolate the blob's recovery. The test shows the machinery runs on such a
source; it does not prove the blob was recovered. Don't quote it as a
super-resolution result.

### Step 2 — a small run first (~2 min for n = 8)

```bash
python superres/fit_pixel_source.py --fits superres/results/fits_img.json \
    --root . --n 8 --n-lambda 7 --n-save 4 \
    --out superres/results/pixel_source_small.json \
    --out-npz superres/results/pixel_examples_small.npz
```

~20 s/image (two regularisation modes × a λ sweep each). **Check the header
reads `0.64 unknowns per datum`** — if it's above ~1.5 the problem is
underdetermined and you should reduce `--n-src`.

### Step 3 — the real run (~35 min for n = 100)

```bash
python superres/fit_pixel_source.py --fits superres/results/fits_img.json \
    --root . --n 100 --n-lambda 9 --n-save 6
```

---

## What to expect — including the result you may not want

Measured on n = 4 (indicative; the n = 100 run is what to quote):

```
      reg mode      corr   size_ratio   centroid px      nmse
            mu    0.7395       1.0664        1.1447    0.4463
       uniform    0.9578       1.0638        0.7023    0.0818

MAGNIFICATION-ADAPTIVE vs UNIFORM, matched on data fidelity
      chi2 / chi2_min    nmse mu   nmse unif    change
                 1.02     0.5145      0.4930      +4.4%
                 1.05     0.5082      0.4883      +4.1%
                 1.10     0.4722      0.4525      +4.4%
                 1.25     0.3911      0.3759      +4.0%
```

**Two findings, both negative, both worth reporting honestly.**

### (a) Magnification-adaptive regularisation does not help here

At matched data fidelity it is consistently **~4% worse** than uniform
smoothing, across every fidelity level. That comparison is the fair one:
comparing each mode at its own L-curve λ isn't valid because the two have
different H, so the same λ means different amounts of smoothing. Targets are set
relative to each image's own best achievable χ², because absolute targets like
χ²/dof = 1 are unreachable (the PSF systematic alone puts it in the thousands).

The idea is physically sound and it is what the original pipeline's
`magnification_adaptive_source_grid` was reaching for. On this data it does not
pay. Possible reasons worth one sentence in the paper: median ray coverage is
only 1.4 per source pixel, so most of the source plane is in the low-μ regime
where the weighting does nothing useful; and the high-μ strip next to the caustic
is thin, so relaxing smoothing there mostly admits noise. **A measured negative
result on a well-motivated idea is a legitimate contribution — do not bury it.**

### (b) The free-form source is worse than the parametric one — and that is expected

nmse ≈ 0.08 (uniform) against **0.024** for the parametric Sérsic. The
parametric model wins because **Model_A's sources genuinely are Sérsics**, so it
has exactly the right prior and only 7 unknowns. A 4,096-pixel free-form solve
cannot beat that on data drawn from the parametric family.

**So do not frame Option 2 as beating Option 1 on Model_A. It can't, and
claiming otherwise would be wrong.** Frame it as:

> The free-form reconstruction agrees with the parametric one on the same frozen
> lens, which confirms the lens rather than the source prior is doing the work.
> Free-form is the deployable form for real galaxies, where no parametric family
> applies — and we show it is well posed once the lens is correct, which it was
> not in the previous pipeline.

That is true, useful, and defensible.

---

## Tuning knobs, if you want to explore

```
--n-src 48         fewer unknowns -> better conditioned, coarser source
--half-extent 1.2  source-plane half-width in arcsec
--reg-power 0.5    0 = uniform, 1 = full 1/mu. See pixel_source._reg_weights
--reg-clip 5.0     bounds the weight range
--reg-modes mu uniform
```

`--reg-power` was set to 0.5 rather than 1.0 for a measured reason: with the raw
1/μ weighting (power = 1) the source correlation dropped to 0.74 against 0.96 for
uniform, because the coverage spans ~1000× and the high-μ strip ended up
effectively unregularised. A sweep over `--reg-power 0 0.25 0.5 1.0` would be a
clean one-figure ablation if you have time.
