"""
train_b4.py -- B4: back-projection + fully-convolutional SISR + analytic
forward lensing, trained by physics self-supervision.

This is the original DeepLense grid-based idea (Anirudh Shankar) run on top of
this project's corrected lens.

    observed image
         |  ray-shoot through the FITTED lens (frozen, per image)
         v
    back-projection b = (L^T d)/(L^T 1)      + coverage map (= magnification)
         |
         v
    SourceSISR  (fully convolutional, residual blocks, PixelShuffle x2)
         |
         v
    source on a 2x-finer grid  ---- forward lens ---> PSF ---> bin ---> chi^2 vs the image

No high-resolution target anywhere. The only supervision is that the model,
once lensed, blurred and binned, must reproduce the observation.

WHAT IS THE SAME AS THE ORIGINAL, AND WHAT IS DIFFERENT
--------------------------------------------------------
Same: the self-supervision loop, the back-projected input, the SISR
architecture, PixelShuffle upsampling, the magnification-weighted regulariser.

Different, and this is the whole point:

  original                                  here
  --------                                  ----
  fixed SIS lens, one theta_E for all        EPL + external shear, FITTED PER
  images, theta_E read from the manifest     IMAGE and unsupervised (Path A, or
                                             the refined Path B)
  precomputed sparse operator bank, three    analytic in-graph deflection,
  of five directories had forward and        validated against lenstronomy to
  backward operators with the SAME sign      1e-15 (tests/test_lens_models.py)
  PSF assumed 0.10 arcsec Gaussian           empirical kernel, measured
  source scored against a broken metric      scored against the npz `unlensed`
                                             array, PSF-matched
  result: source size_ratio 10.6-12.8        target: size_ratio ~1

WHY THE LENS IS FROZEN
----------------------
The lens comes from --lens-fits, a JSON produced by fit_per_image.py or
refine_pathb.py. Both are unsupervised -- nothing is read from the manifest --
so freezing costs no scientific validity, and it isolates the question B4 exists
to answer: GIVEN a correct lens, does a fully-convolutional SISR on the
back-projection beat (a) a Sersic, (b) the B3 decoder, (c) the Option 2 linear
inversion? Those four all solve the same problem on the same frozen lens, which
makes the comparison clean. Learning the lens jointly is a natural extension and
is deliberately out of scope here.

BOX SIZE IS A MEASURED CHOICE, NOT A TASTE
-------------------------------------------
A boxed free-form source cannot represent flux outside its box. Measured on 300
val images, the fraction of true source flux inside +-H arcsec:

    H     n_out (x2)   out px scale   unknowns   vs ~4300 data px   median flux in box
   0.8        32         0.0516"        1024        4.20:1               0.715
   1.0        40         0.0513"        1600        2.69:1               0.827
   1.2        48         0.0511"        2304        1.87:1               0.891   <- default
   1.4        54         0.0528"        2916        1.47:1               0.935
   1.6        62         0.0525"        3844        1.12:1               0.958

H = 1.2 keeps 89% of the flux at 1.87:1 overdetermined, and 0.0511 arcsec/pixel
is 2.07x finer than the 0.10593 arcsec detector -- inside the 3.07x median
tangential stretch that magnification_extract.py measures, so the resolution is
supported by the lens rather than invented.

The B2 run failed partly because it inherited H = 0.8 from a B3 default, where
28% of the flux is outside the box. That is why this file prints the number.

Run from the Grid_Based_Experiment root:
    python superres/train_b4.py --root . --lens-fits superres/results/fits_refined_mu.json \\
        --n-train 6000 --n-val 800 --epochs 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:                                    # pragma: no cover
    raise SystemExit(f"train_b4.py needs torch (import failed: {exc})")

from backproject import backproject, ray_shoot_batch, sample_source
from data_a import ModelADataset
from raytrace import area_downsample, image_plane_grid, load_psf
from sisr_net import SourceSISR
from sources import SersicSource

LENS_KEYS = ("theta_E", "gamma", "e1", "e2", "g1", "g2")
SRC_KEYS = ("amp", "R_sersic", "n_sersic", "se1", "se2", "sx", "sy")


# ===========================================================================

def curvature(S):
    """Laplacian energy per pixel -- the standard source-plane regulariser
    (Suyu et al. 2006). Without it a free source grid reproduces the speckle of
    Option2_failed/results/fig6_pixel_source.png exactly."""
    lap = (S[:, :-2, 1:-1] + S[:, 2:, 1:-1] +
           S[:, 1:-1, :-2] + S[:, 1:-1, 2:] - 4.0 * S[:, 1:-1, 1:-1])
    return (lap ** 2).mean()


def mu_weights(cov, mode: str, power: float = 0.5, clip: float = 5.0):
    """w = (median coverage / coverage)^power, clipped. 'uniform' returns 1.

    Smooth less where the lens delivered many rays, more where it delivered few.
    power 0.5 rather than 1.0 because coverage spans ~1000x and the raw 1/mu
    weighting leaves the thin near-caustic strip effectively unregularised --
    measured in the Option 2 experiment as a drop in source correlation from
    0.96 to 0.74."""
    if mode == "uniform":
        return torch.ones_like(cov)
    B = cov.shape[0]
    flat = cov.reshape(B, -1)
    pos = torch.where(flat > 0, flat, torch.full_like(flat, float("nan")))
    med = torch.nan_to_num(torch.nanmedian(pos, dim=1, keepdim=True).values,
                           nan=1.0).clamp_min(1e-6)
    w = torch.pow(med / flat.clamp_min(1e-3 * med), power)
    return w.clamp(1.0 / clip, clip).view_as(cov)


def build_inputs(img, sig, bp, cov, base=None):
    """Network input, all arcsinh/log compressed.

    The raw signal-to-noise peak varies by 260x between Model_A images. A shared
    filter bank cannot absorb that -- the same physical feature arrives at wildly
    different activation scales. arcsinh is the standard astronomical stretch:
    linear near zero (a 1-sigma fluctuation still reads as 1) and logarithmic in
    the wings, compressing the across-image range to ~1.9x.
    """
    s = sig[:, None, None]
    ch = [torch.asinh(bp / s), torch.log10(1.0 + cov)]
    if base is not None:
        ch.append(torch.asinh(base / s))
    return torch.stack(ch, 1)


def pack(root, split, classes, n, lens_rows, pixel_scale):
    """Images, sigmas, and the frozen lens/source parameters, index-aligned."""
    ds = ModelADataset(root, split=split, classes=classes, limit=n)
    ds.assert_rows_match(lens_rows, f"lens fits for split={split}")
    keep = [i for i in range(len(ds)) if i in lens_rows]
    imgs = np.array([ds.image(i) for i in keep])
    sigs = np.array([ds.sigma(i, imgs[k]) for k, i in enumerate(keep)])
    lens = np.array([[lens_rows[i][k] for k in LENS_KEYS] for i in keep])
    src = np.array([[lens_rows[i][k] for k in SRC_KEYS] for i in keep])
    bg = np.array([lens_rows[i]["background"] for i in keep])
    return (torch.as_tensor(imgs, dtype=torch.float32),
            torch.as_tensor(sigs, dtype=torch.float32),
            torch.as_tensor(lens, dtype=torch.float32),
            torch.as_tensor(src, dtype=torch.float32),
            torch.as_tensor(bg, dtype=torch.float32), keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--classes", nargs="+", default=["axion", "cdm", "wdm"])
    ap.add_argument("--lens-fits", default="superres/results/fits_refined_mu.json",
                    help="JSON with the per-image fitted lens. Use the refined "
                         "Path B fits if you have them (best lens), otherwise "
                         "fits_img.json.")
    ap.add_argument("--lens-fits-train", default="superres/results/fits_train.json",
                    help="the same, for the TRAIN split. Produce it with "
                         "fit_per_image.py --split train.")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pixel-scale", type=float, default=0.10593)
    ap.add_argument("--psf-path", default="superres/results/psf_empirical.npy")
    ap.add_argument("--supersample", type=int, default=1)
    ap.add_argument("--fit-radius-px", type=float, default=45.0)
    ap.add_argument("--half-extent", type=float, default=1.2,
                    help="source box half-width in arcsec; see the table in the "
                         "header before changing it")
    ap.add_argument("--mag", type=int, default=2, help="PixelShuffle factor")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--base", default="none", choices=["none", "sersic"],
                    help="none = pure free-form (the original idea); sersic = "
                         "the fitted Sersic is added and the network predicts "
                         "only the residual, which cannot do worse than Path A")
    ap.add_argument("--reg-mode", default="mu", choices=["mu", "uniform", "none"])
    ap.add_argument("--reg-power", type=float, default=0.5)
    ap.add_argument("--lambda-curv", type=float, default=1.0,
                    help="Laplacian smoothness, as a FRACTION of chi^2")
    ap.add_argument("--lambda-l2", type=float, default=0.0,
                    help="mu-weighted L2 on the source (residual, if --base "
                         "sersic), as a fraction of chi^2")
    ap.add_argument("--sigma-floor", type=float, default=0.02,
                    help="sigma_eff^2 = sigma_bg^2 + (f*model)^2. Do not set to "
                         "0: the PSF wings are wrong at the 1-3%% level and that "
                         "error scales with flux, so without the floor one image "
                         "in a batch of 16 takes half the gradient.")
    ap.add_argument("--max-gain", type=float, default=3.07,
                    help="measured median tangential stretch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="superres/results/b4.pt")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = a.pixel_scale

    def load_rows(path):
        b = json.load(open(path))
        return {r["index"]: r for r in b["rows"]}, b["config"]

    val_rows, vcfg = load_rows(a.lens_fits)
    if not os.path.exists(a.lens_fits_train):
        raise SystemExit(
            f"missing {a.lens_fits_train}.\nB4 needs a fitted lens for the TRAIN "
            f"split too. Produce it once with:\n"
            f"  python superres/fit_per_image.py --root {a.root} --split train "
            f"--classes {' '.join(a.classes)} --n {a.n_train} "
            f"--out {a.lens_fits_train}\n"
            f"(about {a.n_train * 0.82 / 3600:.1f} h on CPU; it is a one-off.)")
    train_rows, _ = load_rows(a.lens_fits_train)

    psf_np = load_psf(a.psf_path)
    psf = torch.as_tensor(psf_np, dtype=torch.float32, device=dev)

    print("loading data ...", flush=True)
    Xtr, Str, Ltr, Ptr, Btr, _ = pack(a.root, "train", a.classes, a.n_train,
                                      train_rows, res)
    Xva, Sva, Lva, Pva, Bva, vidx = pack(a.root, "val", a.classes, a.n_val,
                                         val_rows, res)
    n_pix = Xtr.shape[-1]
    gx, gy = image_plane_grid(n_pix, res, a.supersample)
    GX = torch.as_tensor(gx, dtype=torch.float32, device=dev)
    GY = torch.as_tensor(gy, dtype=torch.float32, device=dev)
    c = (n_pix - 1) / 2.0
    yy, xx = np.indices((n_pix, n_pix))
    mask_np = np.hypot(yy - c, xx - c) <= a.fit_radius_px
    mask = torch.as_tensor(mask_np, device=dev)

    n_in = int(round(2 * a.half_extent / res)) + 1
    n_out = n_in * a.mag
    out_scale = 2.0 * a.half_extent / (n_out - 1)
    gain = res / out_scale

    net = SourceSISR(in_ch=2 + (a.base == "sersic"), width=a.width,
                     depth=a.depth, mag=a.mag, n_up=1).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    print(f"device {dev}   train {len(Xtr)}   val {len(Xva)}   "
          f"weights {sum(p.numel() for p in net.parameters()):,}")
    print(f"back-projection {n_in}^2 at {2*a.half_extent/(n_in-1):.4f} arcsec/px "
          f"(~detector)   ->  source {n_out}^2 at {out_scale:.4f} arcsec/px")
    print(f"super-resolution factor {gain:.2f}x  against a MEASURED median "
          f"tangential stretch of {a.max_gain:.2f}x")
    if gain > a.max_gain:
        print("  *** the requested grid is FINER than the lens can support;")
        print("  *** reduce --mag or raise --half-extent.")
    print(f"source unknowns {n_out**2} vs ~4300 data pixels "
          f"({4300/n_out**2:.2f}:1)  -- but they are produced by a SHARED "
          f"convolutional decoder, not solved for independently")
    print(f"base {a.base}   reg {a.reg_mode}   lambda_curv {a.lambda_curv} x chi2"
          f"   lambda_l2 {a.lambda_l2} x chi2   sigma_floor {a.sigma_floor}\n")

    def step(X, S, L, P, BG, train):
        X, S, L = X.to(dev), S.to(dev), L.to(dev)
        P, BG = P.to(dev), BG.to(dev)
        lens = {k: L[:, i] for i, k in enumerate(LENS_KEYS)}
        with torch.no_grad():
            bx, by = ray_shoot_batch(GX, GY, lens)
            bp, cov = backproject(X, bx, by, n_in, a.half_extent)

        src = {k: P[:, i].reshape(-1, 1, 1) for i, k in enumerate(SRC_KEYS)}
        base_in = base_out = None
        if a.base == "sersic":
            ax = torch.linspace(-a.half_extent, a.half_extent, n_in, device=dev)
            YY, XX = torch.meshgrid(ax, ax, indexing="ij")
            base_in = SersicSource(src).at(XX[None], YY[None])
            ao = torch.linspace(-a.half_extent, a.half_extent, n_out, device=dev)
            YO, XO = torch.meshgrid(ao, ao, indexing="ij")
            base_out = SersicSource(src).at(XO[None], YO[None])

        inp = build_inputs(X, S, bp, cov, base_in)
        y = net(inp)                                    # (B, n_out, n_out) > 0
        amp = P[:, SRC_KEYS.index("amp")].reshape(-1, 1, 1).clamp_min(1e-6)
        S_map = amp * y
        if base_out is not None:
            S_map = base_out + amp * (y - float(np.log(2.0)))   # softplus(0)=ln2

        sky = sample_source(S_map, bx, by, a.half_extent).unsqueeze(1)
        pad = psf.shape[-1] // 2
        sky = F.conv2d(F.pad(sky, (pad,) * 4, mode="replicate"), psf[None, None])
        pred = area_downsample(sky, a.supersample)[:, 0] + BG.reshape(-1, 1, 1)

        sig = S[:, None, None]
        if a.sigma_floor > 0:
            sig = torch.sqrt(sig ** 2 +
                             (a.sigma_floor * pred.detach().clamp_min(0.0)) ** 2)
        chi2 = ((((pred - X) / sig)[:, mask]) ** 2).mean()

        rel = (S_map - (base_out if base_out is not None else 0.0)) / amp
        pen = torch.zeros((), device=dev)
        if a.lambda_curv > 0:
            pen = pen + a.lambda_curv * curvature(rel)
        if a.lambda_l2 > 0 and a.reg_mode != "none":
            w = mu_weights(cov, a.reg_mode, a.reg_power)
            w = F.interpolate(w[:, None], size=rel.shape[-2:], mode="nearest")[:, 0]
            pen = pen + a.lambda_l2 * (w * rel ** 2).mean()

        loss = chi2 + chi2.detach().clamp_min(1e-12) * pen
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            bad = [n for n, q in net.named_parameters()
                   if q.grad is not None and not torch.isfinite(q.grad).all()]
            if bad or not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                return float("nan"), float("nan")
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        return float(chi2.detach()), float(pen.detach())

    hist, best = [], float("inf")
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        tc, tp, nb = 0.0, 0.0, 0
        for k in range(0, len(Xtr), a.batch_size):
            j = perm[k:k + a.batch_size]
            ch, pn = step(Xtr[j], Str[j], Ltr[j], Ptr[j], Btr[j], True)
            if np.isfinite(ch):
                tc += ch; tp += pn; nb += 1
        net.eval()
        vc, vn = 0.0, 0
        for k in range(0, len(Xva), a.batch_size):
            sl = slice(k, k + a.batch_size)
            with torch.enable_grad():
                ch, _ = step(Xva[sl], Sva[sl], Lva[sl], Pva[sl], Bva[sl], False)
            if np.isfinite(ch):
                vc += ch; vn += 1
        sched.step()
        row = {"epoch": ep + 1, "train_chi2": tc / max(nb, 1),
               "train_pen": tp / max(nb, 1), "val_chi2": vc / max(vn, 1),
               "lr": sched.get_last_lr()[0]}
        hist.append(row)
        print(f"epoch {ep+1:3d}  train {row['train_chi2']:10.2f}"
              f"  val {row['val_chi2']:10.2f}"
              f"  gap {row['val_chi2']/max(row['train_chi2'],1e-9):5.2f}x"
              f"  pen {row['train_pen']:9.5f}", flush=True)

        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        blob = {"model": net.state_dict(), "args": vars(a), "history": hist,
                "val_indices": vidx}
        torch.save(blob, a.out)
        if row["val_chi2"] < best:
            best = row["val_chi2"]
            torch.save(blob, a.out.replace(".pt", "_best.pt"))

    print(f"\nwrote {a.out}   best val chi2 {best:.2f} -> "
          f"{a.out.replace('.pt', '_best.pt')}")
    print("now run:  python superres/eval_b4.py --ckpt "
          + a.out.replace(".pt", "_best.pt") + " --root .")


if __name__ == "__main__":
    main()
