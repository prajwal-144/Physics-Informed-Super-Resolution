# B4 results — 50 epochs, 6000/800

## 1. Verdict: this worked

Same 800 validation images, same scoring, same frozen lens.

| method | corr | size_ratio | nmse | peak_ratio | centroid px |
|---|---|---|---|---|---|
| original grid pipeline | — | **10.6–12.8** | — | — | — |
| Option 2, 64² linear inversion | 0.958 | 1.064 | 0.082 | — | — |
| B3 decoder (Sérsic + 32² correction) | 0.959 | 0.949 | 0.080 | 0.991 | — |
| **B4 (back-projection + SISR)** | **0.9698** | **1.1037** | **0.0589** | 0.7848 | 0.5433 |
| Sérsic, 7 params (refined Path A) | 0.9873 | 1.0808 | 0.0252 | 0.8340 | 0.5061 |

**B4 is the best free-form source reconstruction in this project.** Its nmse
beats the linear inversion (0.082) and the B3 decoder (0.080) by ~27%, and it
sits 2.3× above the parametric fit — which has exactly the right functional form
for Model_A and is therefore the hardest possible baseline.

Against the thresholds set in `B4.md` before the run:

| check | result | verdict |
|---|---|---|
| size_ratio | 1.104 | **good** (0.9–1.15) |
| source corr | 0.9698 | **good** (≥ 0.96) |
| source nmse | 0.0589 | **good** (≤ 0.06) |
| centroid error | 0.543 px | **good** (≤ 0.8) |
| val/train gap | 1.06× | **good** (< 1.3) |
| peak_ratio | 0.785 | acceptable (target 0.85–1.15) |
| flux in box | median 1.030 | acceptable, but see §3 |
| χ²/dof, plain σ | 5431 vs 3228 | 1.68×, borderline good/acceptable |

Five "good", three "acceptable", none failed. And on the loss it actually
optimised, B4's val χ² is **344**, against B3's **337** — a dead heat on data
fidelity while producing a substantially better source. That is precisely the
architectural claim: the bottleneck was never the fit, it was the decoder.

Inference is **2.1 ms/image on GPU**.

---

## 2. More epochs will not help

```
   ep      train        val    gap       pen
    1      665.2      502.6   0.76   0.00558
    3      470.5      440.6   0.94   0.00742
   25      362.9      371.3   1.02   0.00869
   50      334.4      355.9   1.06   0.00929
   best val 344.1 @ epoch 24    last 355.9
```

Validation bottomed out at **epoch 24 of 50** and drifted slightly *up* over the
remaining 26. Train/val gap is 1.06 — no overfitting, no underfitting, simply
converged. Extra epochs buy nothing.

---

## 3. The two real defects, with evidence

### (a) The source box truncates the arc — visible in `figS5_stages`, column 6

Column 6 (lensed, no PSF, supersampled) shows the model image clipped to a
rounded **square**. That is the ±1.2″ source box mapping into the image plane:
outside the box the source is exactly zero, so the lensed image is exactly zero
too. The Sérsic has infinite support and B4 does not.

This is most of the χ² gap (5431 vs 3228): the parametric model can put flux in
the arc's faint outer halo and B4 cannot.

### (b) Speckle where the data say nothing — visible in `figS4`, `figS6`

The reconstructed sources carry low-amplitude mottled texture in their
surroundings, while the true sources are smooth. The network is painting
structure in the low-coverage part of the box.

Measured consequences — the flux budget is not tight:

```
recovered flux / true flux:  p10 0.625   median 1.030   p90 1.591   p99 3.085
  31.2% of images above 1.2      13.4% above 1.5      15.6% below 0.7
stratified:  SNR 0-6  flux 0.814   SNR 6-15  1.051   SNR >15  1.105
size_ratio:  SNR 0-6  0.951        SNR 6-15  1.096    SNR >15  1.245
```

The ceiling is ~0.89 (that fraction of the true flux is inside the box), so a
median of 1.03 means ~16% too much flux, and the **spread** is the real problem:
at low SNR the source comes out too faint and small, at high SNR too bright and
too big. One global regularisation strength cannot serve images spanning 260× in
brightness.

**The knob designed for exactly this was switched off.** The run used
`lambda_l2: 0.0` — only the curvature penalty was active. `--lambda-l2` with
`--reg-mode mu` penalises source amplitude weighted by 1/coverage, i.e. it
suppresses flux precisely where the lens never looked. That is the
magnification-as-information-map idea finally having a specific pathology to
fix, rather than being a general-purpose smoother (where it twice came out
neutral-to-negative).

---

## 4. What to try, ranked by expected value

| # | change | why | cost |
|---|---|---|---|
| 1 | `--base sersic` | the fitted Sérsic carries the wings (infinite support, fixes 3a) and the amplitude (fixes 3b); the network predicts only the residual, so B4 **cannot do worse than Path A** | same |
| 2 | `--lambda-l2 0.5 --reg-mode mu` | directly targets the speckle and the flux scatter | same |
| 3 | `--lambda-curv 3` | current 1.0 is clearly too weak given the mottling | same |
| 4 | `--half-extent 1.6` | 95.8% of the flux in box instead of 89%, at 1.12:1 — only if **not** using `--base sersic` | same |
| 5 | `--supersample 2` | 4× more rays: coverage goes from ~5.4 to ~21 per source pixel, so the back-projection is far less aliased | 4× compute (fine on your GPU) |
| — | more epochs | **no** — converged at epoch 24 | — |

Suggested next run, one command:

```bash
python superres/train_b4.py --root . --classes axion cdm wdm \
    --lens-fits superres/results/fits_refined_mu.json \
    --lens-fits-train superres/results/fits_train.json \
    --n-train 6000 --n-val 800 --epochs 30 \
    --base sersic --lambda-l2 0.5 --lambda-curv 3.0 --reg-mode mu \
    --out superres/results/b4_sersic.pt
```

Then the ablation that makes it publishable — the same run with
`--reg-mode uniform`. If the μ weighting helps *here* it is a positive result on
an idea that was neutral in two previous settings, and that is worth a paragraph.

**Target for the improved run:** peak_ratio ≥ 0.85, flux-in-box p10–p90 inside
0.75–1.15, nmse ≤ 0.045, χ²/dof within 1.3× of 3228. If `--base sersic` does not
at least match the Sérsic on every metric, something is wrong with the residual
path — that mode is constructed so it cannot lose.

---

## 5. Why the "true source" looks pixelated — it is not a bug

Model_A stores `unlensed` on the **same 127×127 detector grid as the image**, at
0.10593″/pixel, already PSF-convolved. Cropped to the ±1.2″ source box that is
**23×23 pixels**. B4's output is **48×48** over the same box.

So the reference is genuinely coarser than the reconstruction, and **there is no
high-resolution ground truth anywhere in the dataset**. That is the entire point
of unsupervised super-resolution: if a high-resolution truth existed, we would
train on it directly and none of this would be necessary.

`figS5_stages` column 4 now re-renders the true Sérsic analytically from the
manifest parameters on the 48×48 grid, so the eye can compare like with like.
It is **display only** — every number in `eval_b4.py` is scored against the npz
array (column 3), because that is the honest comparison.

---

## 6. New figures

```bash
python superres/make_figures_b4_stages.py --ckpt superres/results/b4_best.pt \
    --root . --indices 0 2 5
```

- **`figS5_stages.png`** — eight columns: LR observation | back-projection
  (network input) | true source (npz, detector resolution) | true source
  (analytic, SR grid, display only) | **super-resolved source** | lensed with no
  PSF (supersampled) | + PSF + detector binning | prediction (= previous +
  background, exactly what χ² sees).

  On "prediction": with `--supersample 1` the binning step is a no-op, so
  columns 7 and 8 differ only by the additive background constant. Column 6 is
  rendered at supersample 3 purely for display, to show the sharp arc the
  super-resolved source implies before the instrument touches it.

- **`figS6_lr_vs_sr.png`** — just the observation and the reconstructed source,
  large. The slide figure.

Pick different examples with `--indices`.
