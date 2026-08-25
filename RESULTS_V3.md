# v3 results: B3, B2, and the hybrid

50 epochs, 5 warm-up (0 for B2), 6000 train / 800 val, all three DM classes.

---

## 1. The scoreboard

Same 800 val images throughout. Path A = per-image Levenberg-Marquardt.

| | Path A | B3 v1 (broken) | **B3 mu** | **B3 uniform** | B2 |
|---|---|---|---|---|---|
| theta_E rho | +0.956 | +0.956 | **+0.960** | +0.960 | +0.949 |
| beta rho | +0.914 | +0.228 | **+0.826** | +0.835 | +0.641¹ |
| R_sersic rho | +0.924 | +0.147 | **+0.953** | +0.956 | n/a¹ |
| n_sersic rho | +0.944 | −0.014 | +0.475 | +0.476 | n/a¹ |
| gamma rho | +0.571 | +0.193 | +0.132 | +0.083 | +0.205 |
| **\|e\| rho** | **+0.762** | +0.313 | **+0.269** | +0.256 | +0.130 |
| \|g\| rho | +0.611 | +0.142 | +0.142 | +0.143 | +0.083 |
| source corr | 0.9865 | 0.9132 | 0.9566 | 0.9569 | 0.860² |
| source size_ratio | 1.0887 | 1.3421 | **0.9493** | 0.9699 | 0.682² |
| source peak_ratio | 0.8208 | 0.5175 | **0.9905** | 0.9526 | 1.814² |
| source nmse | **0.0271** | 0.1677 | 0.0845 | 0.0832 | 0.261² |
| chi2/dof, plain sigma | 3085 | 12,998 | 9430 | 9732 | 12,921 |
| chi2/dof, **2% floor** (the metric it optimised) | **210** | — | **337–346** | 337–346 | 402–417 |
| ms / image | 877 | 12 | 12 | 12 | 12 |

¹ B2 has no Sérsic, so its `beta`, `R_sersic`, `n_sersic` entries are unused
network outputs, not predictions. `beta` correlating at +0.641 is an artefact:
`sx = 2·|m1|·tanh(z)` and `|m1|` is the ring dipole, so even an untrained `z`
inherits the ring's correlation with the true offset.

² `evaluate.py` **cannot score B2** — it builds a Sérsic from those same unused
parameters. The real B2 source numbers are `source_truth_corrected` in
`fits_pb2.json`, quoted here. Ignore section 2 of the B2 `evaluate.py` output.

### What improved, and by how much

The v2 loss fixes worked. B3 went from a round blob to a genuine reconstruction:

- **beta +0.228 → +0.826**, **R_sersic +0.147 → +0.953** (better than Path A's
  +0.924), **n_sersic −0.014 → +0.475**
- source nmse **0.168 → 0.085** (2x better), corr 0.913 → 0.957
- **size_ratio 1.342 → 0.949 and peak_ratio 0.518 → 0.991** — both now *closer
  to 1 than Path A's* (1.089 and 0.821). The network's source is less biased in
  size and peak than the optimiser's.
- tanh saturation **75% → 0.0%**, train/val gap **2.9x → 0.97x**, val chi^2
  converged and flat from epoch ~42
- on the metric it actually optimised, B3 is **1.6x** Path A (346 vs 210), not
  the 3.1x the plain-sigma column suggests. The plain-sigma column is dominated
  by a handful of bright arcs.

---

## 2. What is still wrong: both ellipticities are shrunk ~4x

```
lens   |e|    network 0.049    Path A 0.192    truth 0.222     rho +0.269 vs +0.762
source |se|   network 0.065    Path A 0.245    truth 0.217     rho +0.326 vs +0.874
```

**It is not an orientation failure.** Decomposing into modulus and phase:

```
                      modulus rho     median phase error   (random = 45 deg)
B3 lens |e|              +0.269              19.0
B3 shear |g|             +0.142              24.9
Path A lens |e|          +0.776               3.9
B3 beta (a vector)       +0.826         9.3 deg direction error (random = 90)
```

The network gets the *direction* roughly right (19 deg vs 45 for random) and the
*magnitude* badly wrong. And the shrinkage is a flat factor of ~0.22 in **every**
phase-error bin:

```
   phase-err bin    n   median |e| pred   median |e| true   ratio
         0-5 deg  120           0.0576            0.2683    0.21
        5-15 deg  221           0.0536            0.2380    0.23
       15-30 deg  200           0.0503            0.2184    0.23
       30-90 deg  259           0.0393            0.1775    0.22
```

**This is what a squared loss is supposed to do.** A network trained on chi^2 is
a *conditional-mean* estimator. For a spin-2 quantity under orientation
uncertainty, a large ellipticity pointed the wrong way costs more than no
ellipticity at all, so the expected-error-minimising answer is to shrink the
modulus. The LM fit is a *maximum-likelihood* estimator on one image and has no
such incentive. Both are behaving exactly as their objectives specify.

More epochs will not fix this. It is a property of the estimator, not of the
optimisation.

---

## 3. The hybrid: this is the result to build on

Stop asking the network to *be* the estimator. Use it as the **starting point**
for the per-image fit. `refine_pathb.py`, measured on n = 25:

```
                       method  model evals  LM seconds  total ms/img   chi2/dof
                network alone            -           -          12.0       6817
           cold LM (4 stages)          362       0.842         842.5       2995
            network + warm LM          136       0.432         444.2       2995

   speedup 1.95x wall clock, 2.66x model evaluations,  chi2 ratio 1.000
```

Identical chi^2 — and **better parameters than the cold fit on five of seven**:

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
   network alone    0.9464      0.9580  0.1033       1.0517
   cold LM          0.9810      1.1214  0.0373       0.7560
   network + warm   0.9831      1.0939  0.0332       0.7556
```

The ellipticity shrinkage is completely undone (+0.075 → +0.849). The warm start
also needs **one** stage instead of four: the staged schedule in
`fit_per_image.py` exists only because a neutral start falls into local minima,
and from the network's start it is unnecessary. That is why warm beats cold on
`g`, `n_sersic` and `R_sersic` — a better basin, not just a faster route to the
same one.

**The claim this supports** (and the one to put in the paper):

> Amortised inference supplies an initialisation good enough to eliminate the
> staged-optimisation schedule and halve the cost of the per-image fit at
> identical final accuracy, while improving parameter recovery on five of seven
> parameters.

Not "the network replaces the fit". It does not, and the ellipticity numbers say
so plainly.

---

## 4. Magnification-adaptive regularisation: negative, second time

Now that lambda is scale-free, the ablation is finally valid:

```
                 correction rms   saturation   corr(|C|, log mu)   source nmse
   B3 mu             3.81%           0.0%           +0.575            0.0799
   B3 uniform        3.17%           0.0%           +0.459            0.0789
```

The mu weighting **does** work mechanically — it concentrates the correction
more strongly in high-magnification regions (+0.575 vs +0.459) — and it produces
**no measurable benefit** (nmse 0.0799 vs 0.0789, i.e. 1.3% *worse*). The two
networks remain nearly identical (Pearson +0.9999 on theta_E, +0.976 on e1).

This is the second independent negative result on the same idea: Option 2's
linear inversion found it ~4% worse at matched data fidelity. Report both.
A well-motivated idea that measurably does not pay is a legitimate contribution.

Note also that the correction itself now helps: nmse 0.0845 → 0.0799 (−5.4%,
mu) and 0.0832 → 0.0789 (−5.2%, uniform), at an amplitude of 3–4% of the Sérsic
and 0% saturation. That is the regime the design was aiming for.

---

## 5. Why the B2 source is blank

Three separate reasons, in order of importance:

**(a) The "Sérsic only" column is zero by construction.** In `--source-mode b2`
there is no Sérsic, so `eval_pathb.source_maps()` returns a zero map for `base`.
That panel is *supposed* to be blank. It is also why `evaluate.py` reports
nonsense for B2 (source corr 0.7793): it renders a Sérsic from parameters the
model never used.

**(b) The real source is 17x17 pixels inside a 127x127 frame.** The map spans
±0.8 arcsec = ±7.55 px, so it covers **1.8%** of the panel. The figure's
percentile stretch then computes p1 = p99.5 = 0 over a 98%-zero image and the
panel collapses to near-black with a tiny saturated square.

**(c) And it is genuinely truncated.** Measured on 200 images, the fraction of
true source flux inside a ±0.8 arcsec box:

```
   p10 0.426    median 0.722    p90 0.927
```

**28% of the source flux, at the median, cannot be represented at all.** In B3
that costs nothing because the Sérsic carries the wings and the map only adds a
core correction. In B2 the map *is* the source. That is why B2 gives
size_ratio 0.68 (too small) and peak_ratio 1.81 (too peaked).

**So the B2 run was mis-configured, and it is not a fair test.** It inherited
`--half-extent 0.8` from the B3 default. A fair B2 needs ~±2.5 arcsec, and there
is the rub: keeping the same 0.0516 arcsec/px resolution over ±2.5 arcsec needs
n_c ≈ 98, i.e. **9,604 unknowns against ~4,300 data pixels — 2.2:1
underdetermined**, straight back into the regime that produced size_ratio
10.6–12.8 in the original pipeline. B3 at ±0.8/32 is 4.1:1 *over*determined.

That trade-off is the actual argument for the parametric+correction split:
**free-form must choose between coverage and resolution; the hybrid does not
have to.**

---

## 6. Anirudh's SISR loop vs ours — and how to combine them

His pipeline (`sisr.py`, `train.py` in the repo root):

```
lr_image ──backward operator──▶ reconstructed_source (crude, source plane)
                                        │
        cat([reconstructed_source, lr_image]) ──▶ SISR (fully conv + PixelShuffle)
                                        │
                                   fine_source (HR)
                                        │
        forward operator ─▶ PSF ─▶ downsample ─▶ wMSE vs lr_image
```

**Yes, the self-supervision idea is the same** — no HR target, the loss is the
re-degraded prediction against the observed LR image. Two things are different,
one in his favour and one in ours:

| | Anirudh | ours (B2/B3) |
|---|---|---|
| network input | the image **already back-projected into the source plane** | the raw image |
| architecture | fully convolutional, residual blocks + PixelShuffle, **spatial correspondence preserved end to end** | conv trunk → **global average pool → 128-vector** → decode a 32x32 map |
| lens | fixed SIS, precomputed sparse operator bank, theta_E from the manifest | EPL + shear, **fitted per image, in-graph, analytic** |
| outcome | source size_ratio 10.6–12.8 | size_ratio 0.95 |

His architecture is the better *super-resolver* and ours is the better *lens
model*. His network never has to learn the geometry — it is applied analytically
before and after, so the SISR only has to deblur and upsample a spatially
registered image, which is exactly what SRResNet-style nets are good at. Ours
squeezes everything through a global pool and then hallucinates a spatial map
from a 128-dim summary, which is a much harder job and is the most likely reason
the correction map only reaches 3–4% and n_sersic/e stay weak.

**We can and should combine them.** The design (call it B4):

```
1. predict the lens (14 params) with the current trunk  -- OR take it from Path A
2. back-project the observed image through THAT lens into the source plane
     (this is exactly the L^T operator already written in Option2_failed/pixel_source.py)
3. feed cat([back_projection, image]) into sisr.SISR  -- fully convolutional,
     PixelShuffle x2, no global pool anywhere
4. forward-lens with our analytic differentiable EPL+shear -> PSF -> bin -> chi^2
     with the 2% floor
```

That keeps his spatial architecture and our correct lens, and every piece
already exists in the repo. It is roughly a day of work.

**But it is not the highest-value use of the next eight days.** See below.

---

## 7. Recommendation

Ranked by value per remaining day.

1. **Run `refine_pathb.py` at n = 800** (~15 min). If the n = 25 numbers hold,
   this is the paper's amortisation result and it is *already finished*.
2. **Freeze the story**: Path A (per-image, unsupervised, validated) +
   magnification extraction (n = 2000, critical curve to 0.63 px) + B3 as
   amortised initialisation + two honest negative results (mu-adaptive
   regularisation; free-form under-determination).
3. **Do not chase B2.** The run you have is mis-configured, and configuring it
   fairly puts it 2.2:1 underdetermined — the exact failure mode already
   documented twice in this project.
4. **B4 (Anirudh's SISR on our fitted lens) only if steps 1–2 are done and
   written.** It is the most interesting remaining idea and the most likely to
   improve the source map, but it is a new experiment eight days out.

Two smaller cleanups worth doing regardless:

- `figB1_mu_reconstruction.png` and `figB1_uniform_reconstruction.png` (15:07,
  15:28) are from the **failed v1 run**. The v3 ones are `*_mu3_*` and
  `*_uniform3_*`. Delete the stale pair so they cannot end up in the paper.
- Add a line to `eval_pathb.py`'s B2 branch, or to the paper, noting that
  `evaluate.py` section 2 is invalid for `--source-mode b2`.
