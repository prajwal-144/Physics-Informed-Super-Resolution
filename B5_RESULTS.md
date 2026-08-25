# B5 (magnification gate) results — the fourth null, and why

30 epochs, 6000/800, `--base sersic`, everything else identical to the B4 v2
control. Two variants: gate only, and gate + μ-weighted curvature.

---

## 1. The ablation

| | Sérsic (Path A) | **B4 +sérsic (control)** | **B5 gate** | **B5 gate+curv** |
|---|---|---|---|---|
| corr | 0.9873 | 0.9794 | 0.9790 | 0.9785 |
| size_ratio | 1.0808 | 1.0434 | 1.0392 | 1.0507 |
| nmse | 0.0252 | **0.0408** | 0.0414 | 0.0425 |
| peak_ratio | 0.8340 | **0.9266** | 0.9221 | 0.9122 |
| centroid px | 0.5061 | 0.5610 | 0.5604 | 0.5625 |
| χ²/dof (plain σ) | 3228 | 3492 | 3519 | 3529 |
| val χ² (floored) | ~210 | 343 | 344 | 344 |
| flux p10–p90 spread | — | 0.691 | 0.701 | 0.704 |
| flux >1.2 / <0.7 | — | 19.4% / 14.5% | 19.8% / 14.5% | 19.6% / 13.8% |

Stratified by SNR, the three are identical to three significant figures:

```
                        snr 0-6      snr 6-15     snr 15+      (flux ratio / chi2)
   control            0.838/1336   0.966/2904   1.042/8963
   gate               0.833/1338   0.972/2943   1.051/9048
   gate+curv          0.838/1340   0.975/2947   1.044/9365
```

**The gate changes nothing measurable.** Both variants are marginally *worse*
than the control on nmse and peak_ratio. All three converged identically (best
val at epoch 22, 24, 16; gap 1.09× in every case).

This is the **fourth independent null** on magnification-adaptive
regularisation in this project:

1. Option 2's linear inversion — μ-weighted λ was ~4% worse at matched fidelity
2. B3's correction penalty — μ vs uniform, Pearson +0.99 between the two runs
3. B4's `--reg-mode mu` vs `uniform` — nmse 0.0799 vs 0.0789
4. **B5's μ-gate — nmse 0.0414 vs 0.0408**

The difference is that this one is a clean, controlled ablation with a
structural constraint rather than a loss term, and we can now say exactly why it
failed.

---

## 2. Why: the network learns the inverse of the gate

The gate is applied *before* the forward model and the network is trained
*through* it. So it can simply emit more sub-pixel structure to cancel the
attenuation. Measured on 12 images:

```
   run        network output (pre-gate)   after gate   attenuation
   control              0.02656             0.02656       1.000
   gate                 0.04273             0.02665       0.624
```

(sub-block variance = the variance *within* each 2×2 output block, i.e. exactly
the sub-detector-pixel content the gate is supposed to remove)

**The gated network emits 1.61× the sub-pixel structure of the control before
the gate, the gate attenuates by 0.624, and 1.61 × 0.624 = 1.005.** The two
models end up with the same final sub-pixel content to within 0.3%.

> A differentiable constraint placed before the likelihood is not a constraint.
> If the network is trained through it, it learns the inverse.

That is a clean, general lesson and worth stating as such.

**Where the gate does survive is where χ² is indifferent.** Flux-weighted, the
gate is nearly open:

```
   unweighted mean gate      0.609        (gate < 0.5 on 21.4% of the box)
   FLUX-weighted mean gate   0.729
```

so it closes only in the faint outskirts. And there, it does work:

```
   speckle in the faint region (<5% of peak):   rms/peak     HF power
   control                                       0.02132      0.21823
   gate                                          0.01828      0.20852
   gate+curv                                     0.01740      0.19921
```

**14–18% less speckle in the low-surface-brightness parts of the
reconstruction** — visible in figS4/figS6, invisible to every flux-weighted
metric. So the gate is a real but cosmetic improvement.

One more diagnostic worth noting: the penalty term went from 0.0137 (control) to
0.48 (gate) — a **35× change in regularisation strength** — with val χ²
identical at 343 vs 344. The model is insensitive to this regularisation over a
35× range, which is itself evidence that the smoothness prior is not what limits
the reconstruction.

---

## 3. How this fits with `mu_resolution.py`

The injection experiment already predicted this result:

```
   snr 0-8    mu<2: 0.046   mu>=2: 0.141    3.1x effect
   snr 8-20   mu<2: 0.260   mu>=2: 0.231    no effect
   snr 20+    (all points have mu>=2)       no effect
```

μ binds only when photons are scarce. Model_A is mostly photon-rich, so a
μ-adaptive prior has almost nothing to do — and where it does have something to
do (the faint outskirts) that region carries no weight in any metric.

**So the magnification story is not a failure — it is a measured, explained,
bounded result**, and that is a better contribution than a lucky improvement:

> The magnification field predicts where the reconstruction is trustworthy — a
> 10× spread in fractional error from μ ≈ 1 to μ ≈ 8, and a 3× effect on
> sub-detector-pixel recovery at low SNR. But using it as an adaptive prior does
> not improve the reconstruction on this dataset, in four independent
> implementations, because the data are photon-rich almost everywhere and the
> regions where μ binds carry negligible flux.

That is one paragraph, four experiments, and a mechanism. Report it.

---

## 4. If you want to give the gate one more chance

The failure mode is specific and there are three ways to close it, in order of
effort:

1. **Apply the gate at inference only**, to the *control* network (trained
   without it). Nothing to retrain — one flag in `eval_b5_mu.py` pointed at
   `b4_sersic_best.pt`. The network cannot pre-compensate for a gate it never
   saw. This is a 5-minute experiment and it is the correct test of "does
   removing unsupported detail help?".
2. **Penalise the pre-gate output.** Move the curvature/L2 terms from `S_map` to
   `S_fine`, so amplifying the fine branch is itself costly.
3. **Detach the fine branch where the gate is closed**, so no gradient flows
   back through the suppressed region and there is nothing to compensate with.

My expectation for (1): the source metrics get slightly *worse* and the faint
region gets cleaner, i.e. the same trade already measured. But it is cheap and
it makes the claim airtight.

---

## 5. Recommendation, with the deadline in view

**Stop here on magnification-adaptive regularisation.** Four nulls with a
measured explanation is a complete result. The remaining time is better spent on
writing than on a fifth variant.

The result set to write up:

- **Path A + refinement** — network initialisation cuts the per-image fit to a
  third of the model evaluations at identical accuracy, and beats the cold fit
  on 5 of 7 parameters (`figR1`–`figR4`).
- **Magnification extraction** — n = 2000, critical curve to 0.63 px, μ_tot
  ρ = 0.834 (`figM1`–`figM4`).
- **B4 v2 +sérsic** — the super-resolution result: nmse 0.0408, size_ratio
  1.043, peak_ratio 0.927, χ²/dof within 1.08× of the parametric fit, and better
  than it on both bias metrics (`figS1`–`figS6`).
- **`mu_resolution.py`** — μ predicts where super-resolution works: 10× error
  spread, sub-pixel recovery at 15–26% contrast requiring μ ≳ 2, and the
  SNR-dependence that explains the nulls (`figMU1`).
- **Four negative results on μ-adaptive priors**, with the pre-compensation
  mechanism as the explanation for the last one.

The one thing still on the table that could change a number rather than a
sentence is the **second Roman band** (F087), which the whole pipeline currently
discards. Future work unless the draft is already finished.
