# `superres/` — the super-resolution deliverable

Self-contained. `sie_pipeline/` is untouched and can be deleted or kept; this
folder carries its own copies of everything it needs, so neither can break the
other.

**What this folder does, in one line:** fit a physical lens + source model to a
single low-resolution image with no labels, then sample the fitted source on a
grid finer than the detector — that is the super-resolution.

---

## The result

`results/fits_img.json`, n = 200 val images, all three DM classes, target
`image` (the real observation, substructure included). Nothing below was
available to the fit.

| parameter | recovered median | true median | median error | Spearman |
|---|---|---|---|---|
| θ_E | 1.381″ | 1.340″ | 0.048″ | **+0.938** |
| β (source offset) | 0.277″ | 0.307″ | 0.031″ | **+0.900** |
| R_sersic | 0.495″ | 0.442″ | 0.045″ | **+0.917** |
| n_sersic | 1.126 | 0.989 | 0.160 | **+0.895** |
| lens \|e\| | 0.196 | 0.221 | 0.035 | **+0.782** |
| external shear \|γ\| | 0.058 | 0.039 | 0.016 | +0.629 |
| slope γ | 1.998 | 2.046 | 0.084 | +0.451 |

Source plane vs the stored `unlensed` truth: correlation **0.987**,
size ratio **1.079** (the old SIS pipeline sat at 10.65), centroid error 0.53 px.

Super-resolution fidelity, fitted source vs the analytic truth rendered on the
same grid:

| factor | grid | scale | corr | nmse | size ratio |
|---|---|---|---|---|---|
| 1× | 127² | 0.1059″ | 0.9779 | 0.045 | 0.969 |
| 2× | 254² | 0.0530″ | 0.9764 | 0.046 | 0.939 |
| 4× | 508² | 0.0265″ | 0.9762 | 0.047 | 0.925 |
| 8× | 1016² | 0.0132″ | 0.9760 | 0.047 | 0.914 |

Flat across factors, which is the point: the fitted source is a continuous
function, so sampling it more finely adds no error of its own. The parameters,
not the sampling, are the limit.

Median |μ| on the ring, from the fitted lens: **6.6**. The lens is the
super-resolving element — it spreads a small patch of source over many detector
pixels. Surface brightness is conserved; μ is never multiplied into an intensity.

---

## Files

### The physics (copied from `sie_pipeline/`, all validated)

| file | what it does |
|---|---|
| `backend.py` | tiny numpy/torch shim so the physics is one source file, not two that drift |
| `lens_models.py` | deflection for SIS / SIE / EPL + external shear; convergence, shear, magnification. **Matches lenstronomy to 2.7e-15** |
| `sources.py` | elliptical Sérsic profile. **Matches lenstronomy exactly (0.00e+00)**. Also `PixelSource`, a free-form source stub that is *not used* |
| `raytrace.py` | the observation operator: source → lens → PSF → detector pixels. Gaussian, Moffat and empirical PSF kernels |
| `data_a.py` | Model_A reader. Raw units, per-image noise from the background annulus. Truth is evaluation-only |
| `theta_e_init.py` | measures θ_E and the source offset from the ring geometry of the image itself (`polar_ridge` + `fourier_rphi`). **0.45 px median error.** This is what makes the pipeline unsupervised |
| `metrics.py` | source-plane scoring against `unlensed`, PSF-matched |
| `calibrate_psf.py` | measures the PSF instead of assuming it. Finds 0.20″ (not the 0.10″ previously used) and Moffat-like wings |

### The pipeline

| file | what it does |
|---|---|
| `fit_per_image.py` | **the core.** Fits 14 numbers per image — 6 lens, 7 source, 1 sky — by Levenberg–Marquardt on the noise-weighted residual. ~1.1 s/image. Nothing read from the manifest |
| `evaluate.py` | scores a fit against ground truth. **Truth enters only here**, after the fit is written to disk |
| `superresolve.py` | **the new part.** Takes the fitted source and renders it at 1×/2×/4×/8×, scores each against the analytic truth on the same grid, computes the magnification map, saves example arrays |
| `make_figures.py` | all five paper figures, from files already on disk. Recomputes nothing, so the figures cannot disagree with the numbers |

### Tests

`tests/run_all.py` runs five checks that need no torch (~3 min). All should PASS.
`tests/test_backend_parity.py` needs torch and is only relevant if you go back to
the amortised network.

---

## What to run, in order

Everything is run from the **`Grid_Based_Experiment` root**, not from inside
`superres/`.

```bash
pip install numpy scipy lenstronomy matplotlib
```

### Step 1 — sanity check the physics (~3 min)

```bash
python superres/tests/run_all.py
```

**Expect:** five blocks, all ending `PASS`. Key numbers: deflection vs
lenstronomy ≤ 3e-15, Sérsic 0.00e+00, Jacobian ≤ 2e-9, supersampling S=3 error
5.5e-3, θ_E estimator ~0.45 px.

**If anything FAILS, stop.** Everything downstream is meaningless.

### Step 2 — measure the PSF (~3 min)

```bash
python superres/calibrate_psf.py --root . --n 40 --save superres/results/psf_empirical.npy
```

**Expect:** a clean single minimum at FWHM **0.18–0.20″**, residual nmse ≈ 0.0029
(against 0.0071 at 0.10″). Then a radial profile table where the empirical/Gaussian
ratio climbs to ~29 at r = 0.25″ — the kernel has Moffat-like wings.

*A copy of the output kernel is already in `results/`, so this step is optional if
you just want to reproduce the figures.*

### Step 3 — fit (~4 min for 200 images)

```bash
python superres/fit_per_image.py --root . --n 200 --classes axion cdm wdm \
    --target image --psf-mode empirical --supersample 1 \
    --out superres/results/fits_img.json
```

**Expect:** progress every 5 images at ~1.1 s/image, χ²/dof median settling
around 3000–5000.

**Use `--target image`, not `image_nss`.** The `_nss` arrays were rendered with a
circular lens and are inconsistent with `image` — see
`sie_pipeline/MENTOR_BUGREPORT.md`. Fitting `image_nss` returns |e| ≈ 0.03 with
Spearman ≈ 0; fitting `image` returns |e| ≈ 0.20 with Spearman +0.78.

*A copy of the output is already in `results/`.*

### Step 4 — score against truth (~1 min)

```bash
python superres/evaluate.py --fits superres/results/fits_img.json --root .
```

**Expect** (n = 200, all three classes):

| quantity | expect | red flag |
|---|---|---|
| θ_E Spearman | 0.92 – 0.96 | < 0.85 |
| θ_E median error | 0.04 – 0.07″ | > 0.12″ |
| β Spearman | 0.85 – 0.93 | < 0.7 |
| R_sersic Spearman | 0.88 – 0.94 | < 0.8 |
| n_sersic Spearman | 0.85 – 0.93 | < 0.75 |
| **lens \|e\| Spearman** | **0.72 – 0.84** | < 0.5 |
| shear Spearman | 0.55 – 0.70 | < 0.3 |
| slope Spearman | 0.35 – 0.55 | — expected to be weak |
| **source size_ratio** | **0.95 – 1.20** | **> 2** |
| source correlation | 0.97 – 0.995 | < 0.95 |
| χ²/dof | 3000 – 6000, **rising** with SNR | falling with SNR |

χ²/dof does not reach 1 and that is understood: the PSF *shape* is the systematic
floor (Model_A arcs reach peak/σ of 10³–10⁴, so a 1 % kernel error is a ~50σ
per-pixel residual). It rises with SNR, which is the signature of a systematic
rather than noise. Report the recovery table and the residual figure; do not
chase χ² = 1.

### Step 5 — super-resolve (~2 min)

```bash
python superres/superresolve.py --fits superres/results/fits_img.json --root . \
    --factors 1 2 4 8 --n-save 6
```

**Expect:** the fidelity table above — corr ≈ 0.976–0.978 and nmse ≈ 0.045–0.047
**flat across all four factors**, size ratio drifting gently 0.97 → 0.91. Median
|μ| on the ring ≈ 6–7. Then a 1× cross-check against the stored `unlensed`:
corr ≈ 0.987, size_ratio ≈ 1.08.

**Red flag:** if corr *drops* materially with factor (say 0.98 → 0.90), the
rendering grid is misaligned. It should be flat.

Writes `results/superres_metrics.json` and `results/superres_examples.npz`.

### Step 6 — figures (~1 min)

```bash
python superres/make_figures.py --root .
```

**Expect** five PNGs in `results/`:

| figure | what it shows | what "right" looks like |
|---|---|---|
| `fig1_superres.png` | observation → reconstructed source at 8× → true source at 8× → stored `unlensed` | columns 2 and 3 should look like the same object; column 1 is visibly coarse and noisy |
| `fig2_recovery.png` | fitted vs true, 7 panels, 1:1 line | tight diagonals for θ_E, β, R_sersic; visible but looser for \|e\|; a cloud for slope |
| `fig3_sizeratio.png` | source size ratio, old SIS vs this work | two well-separated humps, old near 11, new near 1 |
| `fig4_residuals.png` | data / model / residual, 4 examples | residual should show arc-shaped structure, not a smooth halo — that structure is the substructure |
| `fig5_magnification.png` | observation next to log₁₀\|μ\| | bright ring in μ coinciding with the arc |

---

## The honest limitation, for the paper

This is super-resolution **within the model family**. A Sérsic can only produce
Sérsic-shaped things. Model_A's sources genuinely are Sérsics, so here the
recovered fine detail is real and the comparison in step 5 is exact. On a galaxy
with spiral arms or a companion, a parametric reconstruction would smooth them
away and this measurement would not detect it. Testing that needs a free-form
(pixellated) source — `sources.PixelSource` is a stub for it and has never been
run.

Say this explicitly rather than letting a reviewer find it.
