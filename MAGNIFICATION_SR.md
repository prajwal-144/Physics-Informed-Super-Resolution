# B4 v2 results, and magnification as a measured quantity

---

## 1. The two v2 runs

| | Sérsic (Path A) | B4 v1 free | **B4 v2 free** | **B4 v2 +sérsic** |
|---|---|---|---|---|
| config | — | H 1.2, l2 0 | H 1.6, l2 0.5, curv 3, 50 ep, 8000 | H 1.2, **base sérsic**, l2 0.5, curv 3, 30 ep, 6000 |
| corr | **0.9873** | 0.9698 | 0.9731 | 0.9794 |
| size_ratio | 1.0808 | 1.1037 | 1.1302 | **1.0434** |
| nmse | **0.0252** | 0.0589 | 0.0527 | 0.0408 |
| peak_ratio | 0.8340 | 0.7848 | 0.7484 | **0.9266** |
| centroid px | 0.5061 | 0.5433 | **0.4971** | 0.5610 |
| χ²/dof (plain σ) | 3228 | 5431 | 5224 | **3492** |
| val χ² (floored) | ~210 | 344 | **329** | 343 |
| flux in box p10/med/p90 | — | 0.63/1.03/1.59 | 0.48/0.94/1.36 | **0.65/0.96/1.34** |

### What the numbers say

**`--base sersic` is the winner, and it is not close.** nmse 0.0408 against free-form's
0.0527 (−23%), peak_ratio 0.927 against 0.748, and χ²/dof **3492 against the
parametric fit's 3228 — a 1.08× gap, down from 1.68× in v1.** The box-truncation
artefact is gone too: in `figS5_sersic_stages` column 6 the lensed model is a
full round arc, where the free-form version was clipped to a rounded square.
That is exactly what the Sérsic base was added to fix, since it has infinite
support and the residual only has to live inside the box.

**B4 v2 +sérsic beats the parametric fit on the two bias metrics**: size_ratio
1.043 vs 1.081 and peak_ratio 0.927 vs 0.834, both closer to 1. It loses on the
scatter metrics (corr, nmse). Coherent story: the free-form residual removes the
Sérsic's systematic bias — the parametric source is 8% too large and 17% too
flat — at the cost of adding some variance.

**The most instructive number is that free-form has the LOWER val χ² (329) and
the WORSE source (nmse 0.0527 vs 0.0408).** It had more data (8000 vs 6000), more
epochs (50 vs 30) and a bigger box (1.6 vs 1.2), and it still explains the image
better while sitting further from the truth. That is textbook ill-posedness,
measured rather than argued, and it is worth a sentence in the paper: **fitting
the data better is not the same as recovering the source.**

Both runs converged (best val at epoch 22/30 and 47/50, gaps 1.09× and 1.03×).

---

## 2. Why the back-projection background looks noisy

Three reasons, all expected:

1. **The back-projection is data, the true source is a model.** `unlensed` in the
   npz is rendered analytically with no noise — measured on a blank corner, its
   std is 5.9e-5 against the observation's 3.3e-2, a factor of ~560. The
   back-projection is built from the *observed* pixels, which carry background
   noise everywhere, so every source pixel it fills receives noise.

2. **The noise is loudest exactly where the coverage is lowest.** The
   back-projection is `(Lᵀd)/(Lᵀ1)` — an average of the rays landing in each
   source pixel. Averaging N samples of noise gives σ/√N, and N is the ray
   coverage. In the outer box N is 1–2, so those pixels are essentially raw
   single noise samples; near the caustic N is 10–30 and the background there is
   ~4× quieter. That is the mottling you see, and its texture is literally the
   inverse-magnification map.

3. **It is the network's job to remove it.** The back-projection is an *input*,
   not an output. The speckle still visible in the reconstructed source
   (`figS4`, `figS6`) is the network failing to fully denoise it — which is what
   the μ-gate in §5 is designed to stop structurally.

So: not a bug, and not something to "fix" in the back-projection. The right
comparison is back-projection → reconstruction, not back-projection → truth.

---

## 3. Colour vs greyscale

**We are not training on colour.** Every array fed to the network is
single-channel intensity. The colour in the figures is a matplotlib *colormap*
(`magma`) applied at display time, plus a `sqrt` percentile stretch. Change it
and nothing about the model changes.

For display, my recommendation: keep a perceptually uniform map (`magma`,
`viridis`, `inferno`) for diagnostic figures — the eye resolves far more levels
in them than in grey, which matters when you are hunting speckle and edge
artefacts. For the paper's main figure either is defensible; a lot of lensing
papers use greyscale (`gray_r`) for the observation panels and a colour map for
residual/magnification panels. The **stretch matters far more than the
colormap**: everything here uses a sqrt stretch between the 1st and 99.5th
percentile, and that choice changes the visual impression much more than the
palette does.

### But there IS a colour issue, and it is a real one

```
npz 'image' shape: (2, 127, 127)     bands_csv: F062,F087     instrument: Roman_VIS
band 0: max 13.63   bg std 0.0164   peak/bg 833
band 1: max 23.20   bg std 0.0281   peak/bg 826
correlation between bands: 0.752
```

**Every Model_A image has two Roman bands, and this entire project uses band 0
only** (`data_a._band` takes `a[self.band]`, default 0). The bands correlate at
0.75, so band 1 carries roughly 25% independent information, and it is the
*brighter* of the two.

This matters more than it might sound, because **the lens is achromatic**: the
deflection field is identical in both bands. So a two-band fit constrains one
lens with twice the data, while allowing the source to differ between bands.
That is a clean, physically motivated upgrade — and it is a bigger lever than
anything else left on the table. It is not a small change though (dataset, the
forward model, and the source representation all need a band axis), so it is a
judgement call whether to attempt it before the deadline.

---

## 4. Is the source box necessary?

For a **free-form** source, yes: a pixelated source needs a finite grid, and the
grid needs an extent. The trade-off is forced and was measured in `B4.md` —
bigger box means more flux captured but fewer data per unknown.

For **`--base sersic`, effectively no.** The Sérsic has infinite support and
carries the wings; the box only has to cover the region where the *residual* is
significant. That is why the v2 +sérsic run has no truncation artefact at
H = 1.2 while the free-form run still shows edge effects at H = 1.6. If you keep
`--base sersic`, stop worrying about the box.

The other standard answer is an **irregular source grid** — a Delaunay
tessellation whose vertices are the ray landing points, so the resolution
automatically follows the magnification (Vegetti & Koopmans 2009). That is the
principled version of what the μ-gate below does crudely, and it is out of scope
for this deadline.

---

## 5. Magnification: what we already use, and what is new

### Already in use (both v2 runs)

1. `log10(1 + coverage)` is **input channel 1** to the network.
2. `--reg-mode mu --lambda-l2 0.5` weights the source L2 penalty by
   `(median μ / μ)^0.5` — suppress amplitude where the lens did not look.

Coverage *is* the magnification here: the back-projection grid is
0.1043″/px against a 0.10593″ detector, so one ray per source pixel corresponds
to μ ≈ 1 and the count reads as μ directly.

### New: `mu_resolution.py` — does μ actually predict where SR works?

Everything so far *asserted* the link. This measures it, on a trained
checkpoint, with no retraining.

**Experiment A — reconstruction error stratified by local μ** (30 images):

```
   local mu      0.5-1    1-2     2-4     4-8    8-16   16-32
   frac. error   0.308   0.230   0.086   0.036   0.023   0.037
   power-law slope d(log error)/d(log mu) = -0.72
```

**A 10× reduction in fractional error from μ ≈ 1 to μ ≈ 8–16.** Note this is
normalised *inside each bin* — the first version normalised by the whole-map
mean and reported a slope of **+1.05**, which was measuring brightness, not
resolution. Worth knowing if you ever re-derive this.

**Experiment B — sub-detector-pixel injection** (480 injections, 30 images).
A Gaussian clump of FWHM 0.08″ = **0.76 detector pixels** is added to the fitted
Sérsic, lensed, PSF'd, binned, and noised at that image's measured σ; the same
system without the clump is reconstructed with the **same noise realisation**;
the difference is measured in a small aperture at the clump position.

```
   local mu      1-2     2-4     4-8    8-16   16+
   contrast     0.145   0.216   0.261   0.173  0.172

   mu >= 2 vs mu < 2:  0.220 vs 0.145,  Mann-Whitney p = 2.7e-4
```

Three findings, and the third is the interesting one:

- **Sub-detector-pixel structure IS recovered**, at 15–26% of injected contrast.
  Not zero — so super-resolution is genuinely happening. And the Sérsic base
  contributes exactly zero to the difference, so all of it comes from the
  free-form residual.
- **Recovery requires μ ≳ 2** and peaks at μ ≈ 4–8. It falls again at μ > 8,
  which is physically sensible: the highest magnifications occur in a thin strip
  at the caustic where the patch is smeared along one direction, so an isotropic
  aperture loses it.
- **Stratified by SNR the effect splits in two:**

```
   snr 0-8    mu<2: 0.046   mu>=2: 0.141    3.1x  (rho +0.194)
   snr 8-20   mu<2: 0.260   mu>=2: 0.231    none  (rho -0.032)
   snr 20+    all points have mu>=2          none  (rho -0.047)

   spearman(contrast, mu)  = +0.134
   spearman(contrast, snr) = +0.429
```

> **When photons are scarce, magnification decides whether sub-pixel structure is
> recoverable — a factor of three. When photons are plentiful it does not: SNR is
> the binding constraint, not the lensing.**

That single sentence retro-explains **all three** previous null results on
magnification-adaptive regularisation (Option 2's linear inversion, B3's
correction penalty, B4's `--reg-mode mu` vs `uniform`). Most of Model_A sits in
the photon-rich regime where μ is simply not what is limiting you. Three
negatives plus one measured explanation is a much better story than three
negatives.

**Caveat to state:** the injected clump has a fixed amplitude (15% of the global
Sérsic peak) regardless of where it lands, so far from the source centre it is a
larger *local* perturbation and easier to detect — which shows up as
`spearman(distance, contrast) = +0.322`. Scaling the clump to the local surface
brightness would be the cleaner design if there is time.

### New: `train_b5_mu.py` / `eval_b5_mu.py` — μ in the architecture

Two additions, both switchable so each can be ablated:

1. **The magnification gate.**
   ```
   g     = sigmoid((log mu - log mu_gate) / tau)          on the coarse grid
   S_out = g * S_fine + (1 - g) * upsample(avgpool(S_fine))
   ```
   Where μ is large the full-resolution output passes through. Where μ is small
   the sub-pixel detail is **removed** and the source falls back to the coarse
   grid. This is not a penalty the optimiser can trade away — it is structural.
   The network cannot claim resolution the lens did not deliver. `g` is computed
   from the frozen lens and detached. Default `--mu-gate 2.0`, which is where
   Experiment B measures recovery to collapse.

2. **μ-weighted curvature.** B4 weighted only the L2 term by 1/μ and left the
   Laplacian uniform across a source plane whose sampling varies ~30×.
   `--curv-mu 1` gives the smoothness prior the same weight.

Smoke-tested end to end; mean gate value came out 0.62, i.e. the gate is active
on ~38% of the box — a sensible operating point rather than a no-op.

---

## 6. What to run, in order

Nothing below modifies an existing file.

### Step 1 — the measurement (no training, ~4 min)

```bash
python superres/mu_resolution.py --ckpt superres/results/b4_sersic_best.pt \
    --root . --n-images 60 --n-pos 20
```

Produces `results/mu_resolution.json` and `results/figMU1_resolution_vs_mu.png`.
**Run this first** — it is the paper's magnification result and it needs no
further training. Expect Experiment A's slope negative (≈ −0.7) and Experiment B
to show μ ≥ 2 beating μ < 2 with p < 0.01.

### Step 2 — the gated model, and its ablation

```bash
# control: reproduces B4 v2 +sersic exactly
python superres/train_b5_mu.py --root . --classes axion cdm wdm \
    --lens-fits superres/results/fits_refined_mu.json \
    --lens-fits-train superres/results/fits_train.json \
    --n-train 6000 --n-val 800 --epochs 30 --base sersic \
    --mu-gate 0 --curv-mu 0 --out superres/results/b5_control.pt

# gate only
python superres/train_b5_mu.py ... --mu-gate 2.0 --curv-mu 0 \
    --out superres/results/b5_gate.pt

# gate + mu-weighted curvature
python superres/train_b5_mu.py ... --mu-gate 2.0 --curv-mu 1 \
    --out superres/results/b5_gate_curv.pt
```

Each ~1 h on your GPU. Then:

```bash
python superres/eval_b5_mu.py --ckpt superres/results/b5_gate_best.pt --root . \
    --out superres/results/b5_gate_metrics.json \
    --out-npz superres/results/b5_gate_examples.npz
python superres/make_figures_b4.py --metrics superres/results/b5_gate_metrics.json \
    --npz superres/results/b5_gate_examples.npz
```

### What counts as success for B5

| check | good | means |
|---|---|---|
| nmse | ≤ 0.038 | beats B4 v2 +sérsic (0.0408) |
| peak_ratio | 0.90–1.05 | keeps B4's gain (0.927) |
| flux in box p10–p90 | inside 0.75–1.15 | tighter than B4's 0.65–1.34 |
| χ²/dof | ≤ 3600 | no worse than B4's 3492 |
| mean gate value | 0.4–0.8 | the gate is doing something but not everything |
| speckle in figS4 | visibly reduced | the point of the gate |

**If the gate changes nothing**, that is the fourth null on magnification-adaptive
regularisation — and after Experiment B you can now say *why*: Model_A is mostly
photon-rich, and in that regime μ is not the binding constraint. Report it.

---

## 7. Priority, with the deadline in view

1. **`mu_resolution.py` at n = 60.** Zero training, and it turns the
   magnification half of the paper from an assertion into a measurement with a
   p-value. Do this today.
2. **Freeze B4 v2 +sérsic as the super-resolution result.** It is good: 1.08×
   the parametric fit's χ², better on two of five source metrics, and it beats
   every previous free-form attempt by ≥ 23% on nmse.
3. **B5 ablation** if there is time. Two runs (control + gate) is enough.
4. **Two-band** (§3) is the biggest remaining lever and the least likely to fit
   in the time. Note it as future work unless the writing is already done.
