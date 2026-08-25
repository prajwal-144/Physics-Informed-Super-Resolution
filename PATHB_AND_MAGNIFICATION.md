# Magnification extraction + Path B (design B3)

Everything added in this round, what each file does, what to run, and what
counts as a good result.

---

## 1. Files: created vs modified

### Created — magnification

| file | purpose |
|---|---|
| `superres/magnification_extract.py` | computes four magnification quantities per image from the fitted lens and scores them against the true lens |
| `superres/make_figures_magnification.py` | figures `figM1`–`figM4` from the JSON that script wrote. Recomputes no physics |

### Created — Path B

| file | purpose |
|---|---|
| `superres/train_pathb.py` | the network: CNN → 14 physical parameters + a 32×32 source correction map, trained by χ² against the observation with a magnification-weighted penalty on the correction |
| `superres/eval_pathb.py` | scores a checkpoint; writes results in the **exact schema of `fits_img.json`** so `evaluate.py` can score the network with the same code that scored the optimiser |
| `superres/make_figures_pathb.py` | figures `figB1`–`figB4` |
| `superres/tests/test_pathb.py` | five gate checks — run this before training |

### Modified — two files, both small, both for a stated reason

**`superres/lens_models.py`** — added one constant and used it in one line:

```python
EPS_E2 = 1e-24                                   # new
...
c = xp.clip(xp.sqrt(e1 * e1 + e2 * e2 + EPS_E2), None, max_c)   # was: sqrt(e1*e1 + e2*e2)
```

At exactly `e1 = e2 = 0` the derivative `d/de1 sqrt(e1²+e2²) = e1/sqrt(e1²+e2²)`
is `0/0 = inf`. The forward pass is fine (q = 1, the circular fast path fires),
so nothing looks wrong until `.backward()` is called. `train_amortised.Encoder`
zero-initialised its final `Linear`, so the network started at *exactly* e = 0
and the first backward pass wrote NaN into the whole ellipticity head. That is
the mechanical cause of the earlier amortised collapse.

`1e-24` and not `1e-12`: I tried `1e-12` first and it broke
`tests/test_lens_models.py` (errors 2e-6 against a 1e-8 threshold). With `1e-24`
the test passes — **3.55e-15** for the elliptical cases, **1.03e-10** for the
exactly-circular one.

**`superres/train_pathb.py`** — `ray_coverage()` was changed twice during
development, both times because `tests/test_pathb.py` failed:

1. Out-of-map rays were being **clamped** onto the border pixels. Only ~34% of
   image-plane rays land inside a 0.8″ source map, so the other 66% piled up on
   the edge and made the map **border** the highest-coverage region — inverting
   the entire weighting. They are now discarded. (Edge coverage went from 34.6
   to 3.0 against an interior of 18.9.)
2. A 3×3 box average was added. The histogram is a Monte-Carlo estimate of μ
   from a finite ray count; at low supersampling it is a checkerboard of 0s and
   1s that has nothing to do with the smooth underlying field. The smoothing
   removes the aliasing without inventing information — the real fix for
   genuinely low coverage is `--supersample 2`, and the startup banner now
   reports rays-per-source-pixel and warns below 4.

### Moved

`superres/Option2_failed/` now contains a **copy** of every Option 2 file:
`pixel_source.py`, `fit_pixel_source.py`, `eval_pixel_source.py`,
`make_figures_option2.py`, `OPTION2.md`, `tests/test_pixel_source.py`, and the
nine `results/` files.

⚠️ **The sandbox could not delete the originals** (`Operation not permitted`).
Delete these by hand so there is only one copy:

```
superres/OPTION2.md
superres/pixel_source.py
superres/fit_pixel_source.py
superres/eval_pixel_source.py
superres/make_figures_option2.py
superres/tests/test_pixel_source.py
superres/results/pixel_source.json
superres/results/pixel_source_small.json
superres/results/pixel_examples.npz
superres/results/pixel_examples_small.npz
superres/results/fig6_pixel_source.png
superres/results/fig7_lcurve.png
superres/results/fig8_magnification.png
superres/results/fig9_mu_vs_uniform.png
superres/results/_ps_smoke.json
superres/results/_ps_smoke.npz
superres/results/_ps_test.log
```

Also safe to delete — my smoke-test leftovers:
`superres/results/_pathb_smoke.{pt,json,npz}`, `superres/results/_pathb_dev.{pt,json,npz}`.

---

## 2. `magnification_extract.py`

### The physics

Differentiate the lens equation β = θ − α(θ). The Jacobian

```
A = ∂β/∂θ = I − ∂α/∂θ = [[1−κ−γ₁,  −γ₂   ],
                          [ −γ₂  , 1−κ+γ₁]]
```

maps a small source patch to a small image patch, so the area ratio is

```
μ = 1 / det A = 1 / [(1−κ)² − |γ|²]
```

`det A = 0` is the **critical curve** — the Einstein ring — where μ diverges.

**Surface brightness is conserved.** μ is a change of solid angle, not of
brightness. It is never multiplied into an intensity anywhere in this pipeline.
A lensed arc is brighter in *total flux* because it subtends more sky, not
because any patch of it got hotter. (The old `physics_losses.py` multiplied a
hard-coded μ into a loss; that is the error being avoided.)

### The four quantities, and why each one

| # | quantity | what it tests | why an observer cares |
|---|---|---|---|
| 1 | **μ map**, per-pixel `corr(log μ_fit, log μ_true)` over the arc | the whole lens at once — μ depends on κ **and** γ | it is the resolution/flux map of the system |
| 2 | **total magnification** μ_tot = (lensed flux)/(unlensed flux) | flux conservation of the whole forward model | converts an observed brightness into an **intrinsic luminosity** — the number lensing papers actually quote |
| 3 | **critical curve** radius per azimuth | e₁, e₂, g₁, g₂ **jointly** | its azimuthal **swing** is exactly zero for a circular lens, so a non-zero swing recovered correctly proves the lens *shape* was recovered |
| 4 | **resolution gain** 1/\|1−κ−\|γ\|\| | how much the lens stretches the source | **this is the licence to super-resolve** — the lens has already dithered the source across many detector pixels |

### The code

- μ_tot is computed by *actually ray-tracing* the source through the lens on a
  3× supersampled grid and dividing the integrals — the definition, no shortcut.
  No PSF, because the PSF conserves flux and would cancel anyway.
- The critical curve is found by walking 72 rays outward, evaluating
  λ_t = 1−κ−|γ| at 400 radii, and locating the sign change with linear
  interpolation. No root-finder, no failure mode beyond "no crossing in range",
  which returns NaN and is excluded.
- κ, γ come from `hessian_analytic()` — the closed-form EPL+shear Hessian,
  already validated against lenstronomy in `tests/test_lens_models.py`.

### Measured result (n = 900 val images, already run)

```
                                 fitted    true   med |rel err|   Spearman
μ on the arc annulus              3.073    2.692      16.7%        +0.470
TOTAL magnification μ_tot         6.122    5.981      13.0%        +0.827
critical curve, mean radius       1.347    1.291       3.4%        +0.955   arcsec
critical curve, azimuthal swing   0.548    0.557      11.9%        +0.873   arcsec
tangential stretch 1/|λ_t|        3.070    2.814      10.3%        +0.414
radial stretch    1/|λ_r|         0.998    0.955       6.2%        +0.550

per-pixel corr(log μ_fit, log μ_true):  median 0.9301   p10 0.6374
rms radial error of the critical curve: median 0.0663 arcsec = 0.63 px
```

**The honest framing.** μ is a *derived* quantity of the lens: given
(θ_E, γ, e₁, e₂, g₁, g₂) it is fully determined. Since `fit_per_image.py`
already recovers those (θ_E ρ = 0.956, |e| ρ = 0.762 at n = 2000), μ agreeing is
partly automatic. The correct claim is:

> *"We compute the magnification field from the fitted lens and verify that the
> recovered lens reproduces the true total magnification to 13% and the critical
> curve to 0.63 pixels."*

Not *"we measured the magnification independently."* Two of the six Spearmans
(+0.470 and +0.414) are weak, and the reason is mechanical rather than physical:
both are medians over a **fixed annulus**, so they barely vary between images and
the rank correlation is dominated by noise. The relative errors (16.7%, 10.3%)
are the meaningful numbers for those two rows.

**A χ² quality cut helps the tail, not the typical image.** Dropping the worst
10% by χ²/dof raises the swing Spearman 0.873 → 0.903 and μ_tot 0.827 → 0.845.
That is the ellipticity–shear degeneracy: a handful of images land in a wrong
minimum, and χ² flags them **without truth**, which is what makes the cut usable
on real data. The median barely moves, because the χ² floor is dominated by the
PSF-shape systematic that every image shares.

**One visible bias.** In `figM3` the μ_tot cloud sits above the 1:1 line at high
μ. This is the same bias as θ_E (+3.5%) and R_sersic (+10%) in `evaluate.py`, and
the same cause: the empirical PSF's wings are under-modelled, so the fit
compensates with a slightly larger, slightly brighter configuration.

---

## 3. Path B, design B3

### What it is, in one picture

```
image ──CNN──┬──▶ 14 lens+source parameters ──▶ Sérsic(β)   ┐
             │                                              ├──▶ source(β)
             └──▶ 32×32 correction map C ──────▶ C(β)       ┘
                                                     │
                       β = θ − α(θ; predicted lens)  │
                                                     ▼
                        PSF convolve → pixel bin → χ² vs the observation
```

**There is no high-resolution target anywhere.** The only supervision is the
physics: whatever the network emits must, after being lensed, blurred and binned,
reproduce the observed image. That is what makes this unsupervised
super-resolution rather than a supervised upscaler.

### Why Sérsic **plus** a correction, and not free pixels

Two earlier attempts bracket this design, and both failed for the *same* reason
in different clothes:

| attempt | unknowns per image | data pixels | outcome |
|---|---|---|---|
| original SIS pipeline | 254² = 64,516 free source pixels | ~4,300 | source came out **10.6–12.8× too large** across a 100× TV sweep |
| Option 2 (linear inversion) | 64² = 4,096 free pixels | ~4,300 | median ray coverage 1.44/pixel; CG failed on ~30%; reconstructions were **speckle** |
| **B3** | **14 + 32² = 1,038**, and the 1,024 are *bounded and penalised* | ~4,300 | **4.1 : 1 overdetermined** |

The source plane does not contain enough independent information to support
thousands of *free* parameters per image. B3 splits the source into the part the
physics can predict and the part it cannot:

```
source = Sérsic(7 parameters)  +  correction(32², bounded, μ-penalised)
```

and it **degrades gracefully**: if the correction learns nothing it collapses to
the parametric solution, which already works (source corr 0.9865, size_ratio
1.0887 at n = 2000). The network cannot do *worse* than Path A; it can only add
what the Sérsic missed.

### Why a network at all, when the linear solve failed

A per-image solve has only that image's ~4,300 pixels to constrain its unknowns.
A network shares one set of weights across the whole training set, so it carries
a **learned prior** over what galaxies look like. That is a genuinely different
resource, and it is exactly what the linear inversion lacked. Same reason the
failures are not evidence against this attempt.

### The ML concepts, named

- **Amortised inference.** The expensive optimisation is paid once at training
  time; inference is one forward pass. Measured: **12.4 ms/image** (batched, CPU)
  against **877 ms/image** for Levenberg–Marquardt = **71×** on CPU, more on GPU.
  This is what makes the method usable on a survey rather than a hand-picked
  sample.
- **Physics-informed / differentiable forward model.** The loss is not a
  regression onto labels; it is χ² through an analytic, differentiable EPL+shear
  ray-tracer. `tests/test_pathb.py` check 1 proves the torch forward model equals
  the numpy `raytrace.render()` to **5.5e-16**, so the network is optimising the
  same physics the paper describes.
- **Hybrid parametric + non-parametric decoder.** The transposed-conv decoder
  imposes locality on the correction (a far better prior than a dense 128→1024
  map), and the tanh bound plus the penalty keep it subordinate to the Sérsic.
- **Constrained output parameterisation.** Every physical quantity is mapped
  through tanh/sigmoid/softplus into its range, so the network *cannot* emit a
  negative Einstein radius or n_sersic = 0 — either of which produces NaN inside
  `(b/R)^t` before any gradient exists to correct it. θ_E is a *multiplicative
  correction* to the ring-geometry estimate (already good to 0.44 px), so the
  network learns a small residual instead of an absolute value.
- **Curriculum / staged release.** `--warmup-epochs 5` trains the parametric
  part alone first, so the correction can only add what the converged Sérsic
  missed rather than racing it to explain the same flux.

### What, specifically, is the super-resolution

The source is represented on a **32×32 grid spanning ±0.8″ = 0.0516″/pixel**,
against a detector pixel of 0.10593″ — **2.05× finer**.

That number is not a free choice. `magnification_extract.py` measures a median
tangential stretch of **3.07×**, so the lens has already spread each source patch
over ~3 detector pixels and the data support a source grid up to ~3× finer.
2.05× sits comfortably inside that, and `train_pathb.py` prints the comparison at
startup and **warns if you ask for more than the lens can deliver**. This is the
link that turns "we rendered on a finer grid" (a numerical statement) into "the
lens delivered this resolution" (a physical one) — and it is the reason the
magnification work and the super-resolution work belong in the same paper.

### Magnification in the loss

```
R = Σⱼ wⱼ Cⱼ² ,    wⱼ = (median coverage / coverageⱼ)^0.5,  clipped to [1/5, 5]
```

where `coverageⱼ` counts how many image sub-pixels ray-trace into source pixel j.
**That count *is* the magnification of that pixel**, up to the constant sub-pixel
area — and it comes from the *same* ray-shooting that renders the image, so μ and
the ray-tracer cannot describe different lenses. That was precisely the defect in
`physics_losses.fixed_sis_magnification`, which hard-coded `det A = 1 − θ_E/r`
with no link to the operator actually in use.

Physically: where |μ| is large the data genuinely constrain fine structure, so
penalise the correction less; where |μ| ≈ 1 they do not, and anything painted in
there is invention.

Power 0.5 rather than 1.0 for a measured reason: coverage spans ~1000×, and with
raw 1/μ the thin high-μ strip beside the caustic ends up effectively
unregularised — in the Option 2 experiment that dropped source correlation from
0.96 to **0.74**.

`--reg-mode uniform` runs the identical model without the weighting, so the
contribution is **measured, not asserted**. Run both.

---

## 4. Evaluation scripts

| script | what it answers |
|---|---|
| `tests/test_pathb.py` | is the machinery correct at all? (5 checks) |
| `eval_pathb.py` | did the network collapse? did the correction help? where did it put the detail? how much faster? |
| `evaluate.py` (existing, unmodified) | the **shared** scorer — parameter recovery, source truth, χ², SNR strata |
| `magnification_extract.py` | is the recovered magnification right? (it evaluates as it computes) |
| `make_figures_pathb.py`, `make_figures_magnification.py` | figures, from the JSONs only |

`eval_pathb.py` writing the `fits_img.json` schema is deliberate: if the network
were scored by a bespoke script, any difference from the per-image fit could be a
difference in the *scoring* rather than in the model, with no way to tell which.

**The collapse check runs first and prints before anything else.** The previous
amortised network emitted identical numbers for every image and its medians still
looked plausible. `eval_pathb.py` prints the across-image standard deviation of
every predicted parameter next to the truth's, and flags any ratio below 0.05.

---

## 5. What to run, in order

All commands from the **`Grid_Based_Experiment` root**.

### Step 0 — the gates (~1 min)

```bash
python superres/tests/run_all.py
python superres/tests/test_backend_parity.py     # needs torch; run_all skips it
python superres/tests/test_pathb.py
```

Expect `PASS` from all three. `test_backend_parity.py` should print (measured):

```
deflection  tE=1.34 e=(0.0,0.0)      max |np - torch| = 2.22e-16
deflection  tE=1.34 e=(0.22,-0.11)   max |np - torch| = 6.66e-16
sersic                               max |np - torch| = 1.33e-15
magnification autograd vs analytic   max |diff|       = 8.75e-08
```

The `e=(0.0, 0.0)` line is the one that matters here — it exercises the exact
point the `EPS_E2` fix guards, under torch, and agrees with numpy to 2e-16.

`test_pathb.py` should print:

```
[PASS] render_batch == raytrace.render with zero correction   max rel diff 5.47e-16
[PASS] a spike reads back at its own pixel centre             value 1.000000
[PASS] only in-range rays are counted, and each exactly once
[PASS] coverage is higher near the caustic than at the map edge  inner 18.9 vs edge 3.0
[PASS] weight DEcreases with coverage                         corr = -0.986
[PASS] all gradients finite
```

If check 1 fails, **stop** — the network would be fitting different physics from
the paper. If check 5 fails, the `EPS_E2` fix has been reverted.

### Step 1 — magnification (~5 min for all 2000)

```bash
python superres/magnification_extract.py --fits superres/results/fits_img.json --root . --n 2000
python superres/make_figures_magnification.py
```

Already run at n = 900; `results/magnification.json` and `figM1`–`figM4` exist.
Re-run at n = 2000 for the paper.

### Step 2 — train Path B

Small first, to confirm it moves (~2 min CPU):

```bash
python superres/train_pathb.py --root . --classes axion cdm wdm \
    --n-train 300 --n-val 60 --epochs 8 --warmup-epochs 3 --lr 1e-3 \
    --out superres/results/_pathb_dev.pt
```

Then the real run:

```bash
python superres/train_pathb.py --root . --classes axion cdm wdm \
    --n-train 6000 --n-val 400 --epochs 40 --warmup-epochs 5 \
    --reg-mode mu --out superres/results/pathb_mu.pt

python superres/train_pathb.py --root . --classes axion cdm wdm \
    --n-train 6000 --n-val 400 --epochs 40 --warmup-epochs 5 \
    --reg-mode uniform --out superres/results/pathb_uniform.pt
```

**Cost.** Measured 13 s/epoch for 300 images on CPU → ~4.3 min/epoch at 6000, so
~3 h per run on CPU. On a GPU it is minutes. If you only have CPU time, use
`--n-train 2000 --epochs 25` (~1 h) — the training set is images, and 2000
distinct lenses is already a lot of prior.

The checkpoint is written **every epoch**, so you can stop early and still
evaluate.

### Step 3 — evaluate

```bash
python superres/eval_pathb.py --ckpt superres/results/pathb_mu.pt --root . \
    --out superres/results/fits_pathb_mu.json --out-npz superres/results/pathb_mu.npz

python superres/evaluate.py --fits superres/results/fits_pathb_mu.json --root .

python superres/make_figures_pathb.py --pathb superres/results/fits_pathb_mu.json \
    --npz superres/results/pathb_mu.npz --ckpt superres/results/pathb_mu.pt --root .
```

Repeat for `pathb_uniform.pt`. The μ-vs-uniform comparison is the ablation.

---

## 6. What counts as a "good result"

### Path A baseline to beat or match (n = 2000, measured, already on disk)

```
theta_E   ρ +0.956      source corr        0.9865
beta      ρ +0.914      source size_ratio  1.0887
R_sersic  ρ +0.924      source nmse        0.0271
n_sersic  ρ +0.944      source peak_ratio  0.8208
|e|       ρ +0.762      centroid err       0.523 px
|g|       ρ +0.611      chi2/dof median    3085
```

### Path B thresholds

| check | good | acceptable | bad — do not report |
|---|---|---|---|
| collapse ratio (pred std / truth std) | 0.7 – 1.3 on all 7 | > 0.3 | any parameter < 0.05 |
| θ_E Spearman | ≥ 0.90 | ≥ 0.80 | < 0.6 |
| \|e\| Spearman | ≥ 0.60 | ≥ 0.40 | ≈ 0 (that is the collapse signature) |
| source size_ratio | 0.95 – 1.15 | 0.9 – 1.3 | > 2 |
| **correction Δnmse** | **≤ −10%, helping > 70% of images** | ≤ −3% | ≥ 0 |
| corr(\|correction\|, log μ) | ≥ +0.3 | ≥ +0.1 | ≤ 0 (it is painting texture) |
| correction rms as % of Sérsic | 3 – 15% | up to 25% | ≈ 0%, or pinned at the 30% cap |
| speedup vs per-image fit | ≥ 50× | ≥ 10× | — |
| χ²/dof median | within ~2× of 3085 | within 5× | ≫ 10× the per-image fit |

**On χ²/dof: it will not reach 1, and that is measured, not excused.**
`calibrate_psf.py` extracts the PSF empirically and finds Moffat-like wings — at
r = 0.25″ the true kernel is 0.139 of its peak against 0.005 for a 0.18″
Gaussian, a factor of 29. Model_A arcs reach peak/σ_bg of 10³–10⁴, so a 1%
kernel-shape error is a ~50σ per-pixel residual. The null baselines are
`constant 37511`, `blur2 2796`, `blur3 4091`, so 3085 is already at the level of
a good matched blur. **The PSF shape is the floor, not the parameters.**

### What I actually measured on a deliberately undertrained run

300 training images, 8 epochs, CPU, ~100 s total — nowhere near converged:

```
0. COLLAPSE CHECK      No collapse: every parameter varies across images.
A. correction Δnmse    -8.9%   (helps on 87% of images)
   correction rms      19.2% of the Sérsic  (cap 30%)
C. corr(|C|, log μ)    +0.316
B. speed               12.4 ms/image vs 877 ms  =  71x
```

So the mechanism works: the correction improves the source on 87% of images and
concentrates where the magnification is high. **But read the caveat.** At 19.2%
the correction is close to its cap, which at this stage means it is compensating
for a *badly fitted Sérsic* rather than adding real sub-structure — the
parametric part had only 8 epochs. **In a converged run the correction should
shrink** (toward 3–15%) while Δnmse stays negative. If it stays pinned near 30%
after 40 epochs, the parametric head has not converged and the "super-resolution"
is really error compensation. That is the single most important thing to check
before claiming anything.

### On Model_A specifically — do not overclaim

Model_A's sources genuinely **are** Sérsics. A correction map cannot find real
sub-structure that the data do not contain, so on this dataset the honest ceiling
is *"the correction is small and does not hurt"*. The claim to make is:

> The method learns a source representation finer than the detector, constrained
> only by the lensing physics, and the magnification field determines where that
> extra resolution is admissible. On Model_A the correction stays small, as it
> should when the true source is parametric — which is itself evidence the
> regularisation is calibrated rather than free to hallucinate.

If you want a stronger claim, the right experiment is to **inject** sub-Sérsic
structure (a clump, a spiral arm) into a rendered source, run the whole pipeline,
and show the correction recovers it while the pure Sérsic cannot. That is one
afternoon of work with `raytrace.render()` and it would be the single most
convincing figure in the paper.

---

## 7. Known limits, stated up front

1. **μ is derived from the fitted lens**, not measured independently (§2).
2. **χ²/dof ≈ 3000 is a PSF-shape systematic**, not a parameter failure.
3. θ_E (+3.5%), R_sersic (+10%), n_sersic (+17%) are biased high and peak_ratio
   is 0.82 — all consistent with the same under-modelled PSF wings.
4. **`train_pathb.py` and `eval_pathb.py` have been run only as smoke tests**
   (24–300 images, 3–8 epochs) on CPU with torch 1.13. They run end-to-end, the
   loss decreases monotonically on train and val, no batches are skipped, and no
   NaNs appear; `test_backend_parity.py` and `test_pathb.py` both pass under
   torch. But **no converged run exists yet** — treat every Path B number in §6
   as a threshold to check, not a result to quote. Also verify on your own torch
   version: mine was 1.13, and `eval_pathb.py`/`make_figures_pathb.py` fall back
   gracefully if `torch.load(..., weights_only=)` is unavailable.
5. `results/figB1`–`figB4` currently show the 8-epoch dev run. **Regenerate them
   after the real training run** or they will misrepresent the method.
6. Option 2 (free-form) is retired to `Option2_failed/` with its negative results
   documented in `Option2_failed/OPTION2.md`. The magnification-adaptive
   regularisation was ~4% *worse* than uniform there at matched data fidelity —
   report that honestly alongside whatever Path B shows.
