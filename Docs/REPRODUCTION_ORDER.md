# Reproduction order, from a clean checkout

23 August 2026. Every command is run from the **`Grid_Based_Experiment` root**, never from inside
`superres/`. Flags below were read off the `argparse` blocks today, so they are what the scripts
actually accept.

---

## Why the order is confusing (and it is not your fault)

The pipeline is not a straight line, because **B4 and B5 freeze a lens that Path B produced**.
The loop is:

```
Path A  ──▶  trains nothing, but defines the fit
   │
   ▼
Path B / B3  ──▶  a network that predicts a lens in 12 ms
   │
   ▼
refine_pathb  ──▶  network start + one LM stage = a good lens, cheaply
   │                         │
   │                         └──▶ 6,000 TRAIN lenses (this is the only reason
   │                              the 1.98 h refinement run exists)
   ▼
fits_refined_mu.json (800 VAL lenses)   +   fits_train.json (6,000 TRAIN lenses)
   │
   ▼
B4  ──▶  b4_sersic_best.pt
   │
   ├──▶ mu_resolution.py   (the injection experiment; no training)
   └──▶ B5               (the gate; b4_sersic is its control)
```

So the naming order (A, B3, B4, B5) **is** the run order, but B4 does not start from Path A
directly. It starts from the *product* of B3 plus the refinement. That is the piece that makes the
diagram look circular when it is not.

---

## Dependency graph, file by file

```
psf_empirical.npy
   └─▶ fits_img.json            (Path A, VAL n=2000)
          ├─▶ superres_metrics.json      (1x-8x flatness)
          ├─▶ magnification.json         (critical curve, mu_tot, stretch)
          └─▶ fig1..fig5, figM1..figM4

pb3_mu_best.pt / pb3_uniform_best.pt / pb2_best.pt     (B3 + ablation + control)
   ├─▶ fits_pb3_mu.json  (VAL n=800)  ─▶ figB1..figB4
   │      └─▶ fits_refined_mu.json    (VAL n=800 warm fit)  ─▶ figR1..figR4
   └─▶ fits_pathb_train.json (TRAIN n=6000)
          └─▶ fits_train.json         (TRAIN n=6000 warm fit)

fits_refined_mu.json + fits_train.json
   ├─▶ b4.pt / b4_best.pt              (B4 free-form,  H=1.6)
   ├─▶ b4_sersic.pt / b4_sersic_best.pt (B4 + Sersic,  H=1.2)  ◀── the headline SR result
   │      ├─▶ b4_sersic_metrics.json ─▶ figS1..figS6
   │      └─▶ mu_resolution.json     ─▶ figMU1
   ├─▶ b5_gate_best.pt
   └─▶ b5_gate_curv_best.pt
```

---

## Stage 0. Gates (~4 min). If anything fails, stop.

```bash
python superres/tests/run_all.py        # 5 numpy checks: lens, sersic, jacobian, raytrace, ring init
python superres/tests/test_pathb.py     # 5 torch gates. Run BEFORE any Path B training.
python superres/tests/test_b4.py        # 5 torch gates. Run BEFORE any B4 training.
```

`test_b4.py` check 1 is the one that matters: the back-projection must reproduce the source's
**value**, not just its shape, to a few per cent. If forward and backward disagree, everything from
B4 onward is meaningless and the loss will not tell you.

## Stage 1. Measure the instrument response (~3 min)

```bash
python superres/calibrate_psf.py --root . --n 40 \
    --save superres/results/psf_empirical.npy
```

**Trap:** `--save` defaults to `sie_pipeline/results/psf_empirical.npy`, and so does
`fit_per_image.py --psf-path`. Two copies exist in the repo. Pass `--psf-path` explicitly in every
downstream command so you know which kernel produced which number.

## Stage 2. Option 1 = Path A, validation split (~30 min at 0.88 s/image)

```bash
python superres/fit_per_image.py --root . --split val --classes axion --n 2000 \
    --target image --psf-mode empirical --psf-path superres/results/psf_empirical.npy \
    --supersample 1 --out superres/results/fits_img.json
```

`--target image`, never `image_nss`. The substructure-free arrays were rendered with a circular
deflector, so fitting them returns |e| ≈ 0.03 with Spearman ≈ 0.

## Stage 3. Score Path A and produce its figures (~15 min total)

```bash
python superres/evaluate.py            --fits superres/results/fits_img.json --root .
python superres/superresolve.py        --fits superres/results/fits_img.json --root . --factors 1 2 4 8 --n-save 6
python superres/magnification_extract.py --fits superres/results/fits_img.json --root .
python superres/make_figures.py               --root .
python superres/make_figures_magnification.py --root .
```

At this point you have: the seven-parameter recovery table, source corr 0.9865 / size ratio 1.0887,
the 1× to 8× flatness, the critical curve to 0.63 px, and μ_tot to about 1 per cent.

## Stage 4. Path B / B3, three training runs

```bash
# B3, magnification-weighted correction penalty
python superres/train_pathb.py --root . --classes axion \
    --n-train 6000 --n-val 800 --epochs 50 --warmup-epochs 5 \
    --sigma-floor 0.02 --lambda-corr 3.0 --augment 1 --reg-mode mu \
    --psf-path superres/results/psf_empirical.npy \
    --out superres/results/pb3_mu.pt

# the ablation: identical, uniform weighting
python superres/train_pathb.py ... --reg-mode uniform --out superres/results/pb3_uniform.pt

# the free-form control (mis-configured on purpose to expose the box-truncation failure)
python superres/train_pathb.py ... --source-mode b2 --lambda-curv 3.0 --warmup-epochs 0 \
    --out superres/results/pb2.pt
```

Then score each with the **same** scorer Path A used:

```bash
python superres/eval_pathb.py --ckpt superres/results/pb3_mu_best.pt --root . \
    --out superres/results/fits_pb3_mu.json --out-npz superres/results/pb3_mu.npz
python superres/evaluate.py --fits superres/results/fits_pb3_mu.json --root .
python superres/make_figures_pathb.py --pathb superres/results/fits_pb3_mu.json \
    --npz superres/results/pb3_mu.npz --ckpt superres/results/pb3_mu.pt --root .
```

**Note the two χ²/dof conventions.** `evaluate.py` divides by 6361 − 14; `eval_pathb.py` divides by
6361 − 14 − 1024, a factor 1.192. Decide which one you quote before you write anything.

## Stage 5. The refinement on the validation split (~15 min)

```bash
python superres/refine_pathb.py --pathb superres/results/fits_pb3_mu.json --root . --n 800 \
    --psf-path superres/results/psf_empirical.npy \
    --out superres/results/fits_refined_mu.json

python superres/evaluate.py --fits superres/results/fits_refined_mu.json --root .
python superres/make_figures_refine.py --root .
```

**Trap:** `--out` defaults to `fits_refined.json`. Every downstream default expects
`fits_refined_mu.json`. Pass it.

This is where the 480 → 151 evaluations, four stages → one, 3302 → 3228 result comes from, and it
also produces the 800 frozen validation lenses that B4 and B5 use.

## Stage 6. Lenses for the TRAIN split (the expensive step, ~2.5 h)

B4 and B5 freeze a lens per training image, so 6,000 of them are needed. Two routes.

**Route B, the one the recorded results used.** It is also the amortised-initialisation result being
used for real, which is the reason it is worth mentioning in the paper.

```bash
python superres/eval_pathb.py --ckpt superres/results/pb3_mu_best.pt --root . \
    --split train --n 6000 \
    --out superres/results/fits_pathb_train.json --out-npz superres/results/pathb_train.npz

python superres/refine_pathb.py --pathb superres/results/fits_pathb_train.json --root . --n 6000 \
    --psf-path superres/results/psf_empirical.npy \
    --out superres/results/fits_train.json
```

The refinement leg is **1.98 h measured**. `B4.md` estimates the pair at 50 min, which was
optimistic; use the measured number.

**Route A, if you want to skip Path B entirely.** Cold-fit the training split directly, roughly
1.4 h of CPU, and point `--lens-fits` at `fits_img.json` instead:

```bash
python superres/fit_per_image.py --root . --split train --classes axion --n 6000 \
    --psf-mode empirical --psf-path superres/results/psf_empirical.npy --supersample 1 \
    --out superres/results/fits_train.json
```

Route A gives you B4 without ever training B3. It also costs you the initialiser result and one row
of the representation table, so do not take it unless you only want the super-resolution number.

## Stage 7. B4, two runs

```bash
# B4 v2 free-form, the wider box
python superres/train_b4.py --root . --classes axion \
    --lens-fits superres/results/fits_refined_mu.json \
    --lens-fits-train superres/results/fits_train.json \
    --n-train 8000 --n-val 1000 --epochs 50 \
    --half-extent 1.6 --base none \
    --lambda-l2 0.5 --lambda-curv 3.0 --reg-mode mu \
    --out superres/results/b4.pt

# B4 v2 + Sersic base  <-- the headline super-resolution result
python superres/train_b4.py --root . --classes axion \
    --lens-fits superres/results/fits_refined_mu.json \
    --lens-fits-train superres/results/fits_train.json \
    --n-train 6000 --n-val 800 --epochs 30 \
    --half-extent 1.2 --base sersic \
    --lambda-l2 0.5 --lambda-curv 3.0 --reg-mode mu \
    --out superres/results/b4_sersic.pt
```

Score and plot:

```bash
python superres/eval_b4.py --ckpt superres/results/b4_sersic_best.pt --root . \
    --out superres/results/b4_sersic_metrics.json \
    --out-npz superres/results/b4_sersic_examples.npz
python superres/make_figures_b4.py --metrics superres/results/b4_sersic_metrics.json \
    --npz superres/results/b4_sersic_examples.npz
python superres/make_figures_b4_stages.py --ckpt superres/results/b4_sersic_best.pt \
    --root . --indices 0 2 5
```

Check the startup banner reads **2.07× finer than the detector** and **1.87:1**. Both runs bottom
out around epoch 22 to 24, so more epochs buy nothing.

## Stage 8. The injection measurement (~8 min, no training)

```bash
python superres/mu_resolution.py --ckpt superres/results/b4_sersic_best.pt --root . \
    --n-images 60 --n-pos 40
```

**Trap:** `MAGNIFICATION_SR.md` records this as `--n-pos 20`, which gives 1,200 rows. The JSON on
disk has **2,400 rows over 60 images**, so the run that produced the current numbers used
`--n-pos 40`. Use 40 to reproduce, and correct the markdown.

This is the highest value-per-minute command in the whole project. It produces `mu_resolution.json`
and `figMU1_resolution_vs_mu.png` and needs no GPU time at all.

## Stage 9. B5, the magnification gate

```bash
# gate only
python superres/train_b5_mu.py --root . --classes axion \
    --lens-fits superres/results/fits_refined_mu.json \
    --lens-fits-train superres/results/fits_train.json \
    --n-train 6000 --n-val 800 --epochs 30 --base sersic \
    --mu-gate 2.0 --curv-mu 0 --out superres/results/b5_gate.pt

# gate + magnification-weighted curvature
python superres/train_b5_mu.py ... --mu-gate 2.0 --curv-mu 1 \
    --out superres/results/b5_gate_curv.pt
```

```bash
python superres/eval_b5_mu.py --ckpt superres/results/b5_gate_best.pt --root . \
    --out superres/results/b5_gate_metrics.json \
    --out-npz superres/results/b5_gate_examples.npz
python superres/make_figures_b4.py --metrics superres/results/b5_gate_metrics.json \
    --npz superres/results/b5_gate_examples.npz
```

**Important, and it is not written down anywhere else.** `MAGNIFICATION_SR.md` §6 lists a
`b5_control.pt` run with `--mu-gate 0 --curv-mu 0`. **That run was never made.** There is no
`b5_control*` file in `results/`. The "control" column in `B5_RESULTS.md` is the B4 + Sérsic run
(`b4_sersic_best.pt`), which is a fair control because every other setting matches, but the
provenance should be stated as such rather than implying a dedicated control run.

---

## Shortest path to each headline number

| you want | run stages |
|---|---|
| the seven-parameter recovery table, size ratio 1.089, 1× to 8× flatness | 0, 1, 2, 3 |
| the critical curve and magnification validation | 0, 1, 2, 3 |
| the B3 row of the representation table, plus the μ-vs-uniform null | 0, 1, 2, 4 |
| the initialiser result, 480 → 151 | 0, 1, 2, 4, 5 |
| the headline super-resolution number, nmse 0.0408 | 0, 1, 2, 4, 5, 6, 7 (or 0, 1, 6-route-A, 7) |
| the injection measurement | everything up to 8 |
| the gate null | everything up to 9 |

Full cold reproduction, single GPU: roughly **10 to 12 hours of wall clock**, of which about 2.5 h
is Stage 6 and most of the rest is the five training runs.

---

## Five traps that will cost you a day each if you hit them

1. **`--classes axion cdm wdm` does not give you three classes.** `data_a.ModelADataset` sorts the
   *concatenated* path list before applying `limit`, and there are 10,000 axion train files and
   2,000 axion val files, so every run truncates inside the axion class. Either fix `limit` to apply
   per class, or pass `--classes axion` and say so honestly. Every number currently on disk is
   axion-only.
2. **Two `psf_empirical.npy` files exist**, one under `sie_pipeline/results/` and one under
   `superres/results/`, and different scripts default to different ones. Always pass `--psf-path`.
3. **`refine_pathb.py --out` defaults to the wrong name** for everything downstream. Pass
   `--out superres/results/fits_refined_mu.json`.
4. **Stale v1 figures are still on disk.** `figB1_mu_reconstruction.png` and
   `figB1_uniform_reconstruction.png` are from the failed v1 run; the v3 ones carry `mu3` and
   `uniform3` in the name. Delete the stale pair so they cannot end up in a paper.
5. **`evaluate.py` cannot score `--source-mode b2`.** It rebuilds a Sérsic from parameters the b2
   model never used. The real b2 source numbers are `source_truth_corrected` in `fits_pb2.json`.

---

## What is worth automating before the GSoC report

A single `reproduce.sh` that runs Stages 0 to 9 in order with explicit `--psf-path` and `--out`
everywhere, plus a `MANIFEST.md` mapping each figure to the run that produced it. The results
directory now holds four `figS1_*` variants and three `figS5_*` variants with nothing recording
which checkpoint each came from. That is a ten-minute fix and it is the difference between a
successor reproducing this in an afternoon and giving up.
