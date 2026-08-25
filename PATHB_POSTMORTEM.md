# Path B v1 post-mortem, and what v3 changes

The 40-epoch mu and uniform runs failed. This is the diagnosis, measured rather
than guessed, and the fixes now in `train_pathb.py`.

---

## 1. What the results said

| metric | Path A (per-image fit) | Path B mu | Path B uniform |
|---|---|---|---|
| theta_E rho | +0.956 | +0.956 | +0.960 |
| beta rho | +0.914 | **+0.228** | +0.258 |
| R_sersic rho | +0.924 | **+0.147** | +0.143 |
| n_sersic rho | +0.944 | **−0.014** | −0.011 |
| gamma rho | +0.571 | **+0.193** | +0.175 |
| \|e\| rho | +0.762 | **+0.313** | +0.277 |
| \|e\| median | 0.192 (true 0.214) | **0.048** | 0.054 |
| source corr | 0.9865 | 0.9132 | 0.9107 |
| source size_ratio | 1.0887 | 1.3421 | 1.3639 |
| source nmse | 0.0271 | 0.1677 | 0.1719 |
| chi2/dof | 3085 | **12,998** | 13,038 |

The blob you saw in figB1 is real: |e| = 0.048 against a true 0.216 means the
model lens is essentially **circular**, so every predicted image is a smooth
symmetric ring regardless of what the observation looks like. Row 3 of figB1 is
a clear pair of arcs and the model draws a perfect annulus.

---

## 2. The cause — one bug, everything else follows

**The chi^2 was never normalised.** `sigma` was the background standard deviation
alone, so a bright arc produced residuals of 10^3–10^4 sigma and a faint one ~10.

Measured on the 400 validation images:

```
per-image chi2:  p10 656   median 12,998   p90 206,917   max 21,953,076
  top  1% of images (4)   carry 50.8% of the total chi2
  top 10% of images (40)  carry 87.4%
expected share of the batch gradient taken by the single largest member
of a random batch of 16:   54.1%     (uniform would be 6.2%)
```

**The effective batch size was ~2.** Three consequences, all confirmed:

### (a) The magnification term never acted
`--lambda-corr 1.0` against a chi^2 of ~10^5 is a penalty six orders of
magnitude too small. Proof: the mu and uniform runs are the *same network* —

```
Pearson(mu prediction, uniform prediction):
  theta_E +0.9992   R_sersic +0.9937   n_sersic +0.9973   beta +0.9423
corr(|correction|, log mu):   mu run −0.034    uniform run −0.038
```

So the entire mu-vs-uniform ablation was void. **Magnification is not what broke
this** — it never got a chance to do anything, good or bad.

### (b) The tanh saturated
With no effective penalty the correction ran to its bound and stayed there:

```
correction rms: 28.86% of the Sersic amplitude   (cap 30%)
fraction of each map within 1% of the bound: 65–89%, median 75%
train_reg pinned at 0.0809 from epoch 7 to epoch 40
   (= corr_scale^2 x mean weight = 0.09 x 0.9, exactly saturation)
```

A saturated tanh has ~zero gradient. So the correction stopped learning **and**
starved the parametric head at the same time. You can see the square edge of the
saturated map in the "Sersic + correction" column of figB1, and the correction
maps in figB2 are pure two-level red/blue images.

### (c) Overfitting on top
Final train chi2 55,618 vs val 162,552 — **2.9x**. Val plateaued near epoch 20;
the remaining 20 epochs only widened the gap. **More epochs would not have
helped.**

One more symptom: `background` came out at a median of **−0.98** (Path A: +0.018,
and the network reached −3.6 on some images). The network dug a negative sky
pedestal so it could paint an over-large, over-bright source on top of it. That
is precisely the "blob" you noticed.

---

## 3. Two more problems the fix exposed

Once the loss was repaired, a second failure became visible: the network was
reproducing almost exactly the *one number it was handed as an input*, and
learning very little from the pixels.

```
                          predicts truth at rho      network v1 rho
ring theta_E (fed in)            +0.961                  +0.956   <- matched
ring |m1| dipole (NOT fed)       +0.875                  +0.228   <- lost
ring |m2| quadrupole (NOT fed)   +0.512                  +0.313   <- lost
```

Two causes:

**Input dynamic range.** The input was `image / sigma_bg`, whose peak varies
**260x** between images (p10 251, median 1,795, max 65,087). A shared
convolutional filter bank cannot absorb that; the same physical feature arrives
at wildly different activation scales. `arcsinh` compresses it to **1.9x** and is
the standard astronomical stretch — linear near zero so a 1-sigma fluctuation
still reads as 1, logarithmic in the wings.

**Scalar dilution.** The ring theta_E was supplied as a constant plane, then had
to survive four strided convolutions and a global average pool, competing with
~4,000 image pixels for room in a 128-dim bottleneck.

---

## 4. What v3 changes

| # | change | flag | why |
|---|---|---|---|
| 1 | fractional error floor, `sigma_eff^2 = sigma_bg^2 + (f·model)^2` | `--sigma-floor 0.02` | the PSF wings are measured wrong at the 1–3% level and that error scales with flux, so this is the correct noise model. On the per-image fits it flattens chi2 across SNR from 1411→8455 down to 329→229 and cuts p90/p10 from **95x to 8.7x** |
| 2 | lambda expressed as a **fraction of chi^2** (multiplied by `chi2.detach()`) | `--lambda-corr 3.0` | scale-free; cannot silently be six orders out again |
| 3 | tanh saturation measured and printed **every epoch**, warned at the end | — | it was invisible before |
| 4 | dihedral augmentation | `--augment 1` | free and exactly valid for a self-supervised loss — a rotated image is a legitimate member of the distribution and needs no relabelling |
| 5 | weight decay | `--weight-decay 1e-4` | the 2.9x gap |
| 6 | `arcsinh` input stretch | `--stretch 1` | 260x → 1.9x |
| 7 | four ring statistics as input, **bypassing the conv trunk** into the head | — | the scalars reach the parameter head undiluted |
| 8 | source offset **scaled by** the ring dipole | — | same conditioning trick that makes theta_E work |
| 9 | `background` bounded by a tanh | — | kills the −0.98 pedestal |
| 10 | free-form mode in the same file | `--source-mode b2` | makes B2-vs-B3 a controlled experiment, not an argument |
| 11 | Laplacian smoothness penalty | `--lambda-curv` | optional in b3, **essential** in b2 |
| 12 | best-val checkpoint saved separately | `*_best.pt` | val was best at epoch 38 of 40 in one run and 36 in the other |

Note on 8: an earlier attempt anchored beta as `r·(cos φ, sin φ)` with
`φ = atan2(z, z')`. That is singular at the origin in exactly the way `EPS_E2`
was, and `tests/test_pathb.py` check 5 caught it immediately (NaN gradients from
a zero-initialised head). Scaling has no singularity.

---

## 5. Measured effect of the fixes (small runs, 400–500 images, 12–14 epochs)

```
                                v1 (6000 img, 40 ep)     v3 (500 img, 14 ep)
chi^2 scale                          160,000                    1,700
tanh saturation                        75%                       0.0%
train/val gap                          2.9x                      0.6x
corr(|correction|, log mu)            −0.03                     +0.46
|e| spread ratio                       0.32                      0.09  (still bad)
beta rho                              +0.228                    +0.649
```

**The loss pathology is fixed.** chi^2 now converges in 3–4 epochs to a value
comparable with the per-image fit instead of plateauing 50x above it; the
correction no longer saturates; and the magnification weighting finally has an
effect — `corr(|correction|, log mu)` went from −0.03 to **+0.46**, which is the
first evidence in this project that the mu-as-information-map idea does anything
at all.

**Ellipticity is still not learned.** v3 gives |e| rho +0.08 against a ring-only
baseline of +0.52 and Path A's +0.76. This is the honest open problem. It may be
sample size (500 images, 14 epochs is tiny) — the full run is the test — but do
not assume it.

---

## 6. The control that now runs automatically

`eval_pathb.py` prints section **0b**: for each anchored quantity, the Spearman of
the *raw ring statistic* against truth, next to the network's own prediction.
Because the ring statistics are fed as inputs, a network that merely echoes them
would score well — 0b is what distinguishes echoing from learning.

```
   theta_E    ring-only rho +0.963   network rho +0.957   -> no gain (echoes input)
   beta       ring-only rho +0.832   network rho +0.649   -> network LOSES info
   e          ring-only rho +0.519   network rho +0.078   -> network LOSES info
```

Report this table in the paper. It is the difference between an honest
amortisation result and an inflated one.

---

## 7. What to run

```bash
# 0. gates
python superres/tests/test_pathb.py

# 1. B3 with the fixed loss  (~3 h CPU, minutes on a GPU)
python superres/train_pathb.py --root . --classes axion cdm wdm \
    --n-train 6000 --n-val 400 --epochs 40 --warmup-epochs 5 \
    --source-mode b3 --reg-mode mu --out superres/results/pb3_mu.pt

# 2. the ablation -- ONLY meaningful now that lambda is scale-free
python superres/train_pathb.py --root . --classes axion cdm wdm \
    --n-train 6000 --n-val 400 --epochs 40 --warmup-epochs 5 \
    --source-mode b3 --reg-mode uniform --out superres/results/pb3_uniform.pt

# 3. B2, free-form, same loss and same data -- the controlled comparison
python superres/train_pathb.py --root . --classes axion cdm wdm \
    --n-train 6000 --n-val 400 --epochs 40 --warmup-epochs 0 \
    --source-mode b2 --lambda-curv 3.0 --out superres/results/pb2.pt

# 4. score each, then compare with the SHARED scorer
python superres/eval_pathb.py --ckpt superres/results/pb3_mu_best.pt --root . \
    --out superres/results/fits_pb3_mu.json --out-npz superres/results/pb3_mu.npz
python superres/evaluate.py --fits superres/results/fits_pb3_mu.json --root .
python superres/make_figures_pathb.py --pathb superres/results/fits_pb3_mu.json \
    --npz superres/results/pb3_mu.npz --ckpt superres/results/pb3_mu_best.pt --root .
```

**Watch these three lines during training** — they are all that matter:

```
epoch  12  train  2838.86  val  1706.28  gap 0.60x  reg 0.0000  sat  0.0%
                                          ^^^^^^^^^            ^^^^^^^^^
                                     under ~1.5x = no          under ~10% = the
                                     overfitting               correction is free
```

and `val` should land within ~2x of **3085**, the per-image fit's chi2/dof.

---

## 8. Kill criteria — decide by 25 Aug, not later

Run step 1. If after 40 epochs `eval_pathb.py` section 0b still says
`e -> network LOSES info`, **stop developing Path B** and write the paper on
Path A + magnification, reporting Path B as a measured negative result with this
post-mortem as the explanation. That is a legitimate and publishable outcome, and
it is far better than an unfinished positive one four days before the deadline.
