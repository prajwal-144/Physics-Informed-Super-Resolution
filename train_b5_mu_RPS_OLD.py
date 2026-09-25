"""
train_b5_mu.py -- B5: B4 with the magnification wired into the ARCHITECTURE,
not just the loss.

Nothing in train_b4.py / eval_b4.py / backproject.py / sisr_net.py is modified;
this file imports the shared physics from them so the two models cannot drift
apart.

WHY
---
mu_resolution.py measures, on a trained B4 model, that the fractional
reconstruction error depends strongly on the local magnification:

    local mu      1-2     2-4     4-8    8-16   16-32
    frac. error  0.282   0.083   0.037   0.040   0.031

A factor of nine between mu ~ 1 and mu ~ 16. Below mu ~ 2 the reconstruction is
essentially not constrained, yet B4 still emits a full-resolution source there --
which is where the speckle in figS4/figS6 comes from.

B5 adds two things, both switchable so each can be ablated:

1. THE MAGNIFICATION GATE  (--mu-gate)

       g      = sigmoid( (log mu - log mu_gate) / tau )        on the coarse grid
       S_out  = g * S_fine + (1 - g) * upsample(avgpool(S_fine))

   Where mu is large the network's full-resolution output passes through
   untouched. Where mu is small the sub-pixel detail is *removed* -- the output
   falls back to the coarse grid, which is the resolution the data actually
   support there. This is not a penalty that the optimiser can trade away; it is
   a structural constraint. The network cannot claim resolution the lens did not
   deliver.

   g is computed from the ray coverage of the FROZEN lens and is detached, so it
   is a fixed, physically determined mask per image, not something the network
   can learn its way around.

2. MAGNIFICATION-WEIGHTED CURVATURE  (--curv-mu)

   B4 weighted only the L2 term by 1/mu and left the Laplacian unweighted, so
   the smoothness prior was uniform across a source plane whose sampling varies
   by ~30x. Here the curvature penalty carries the same (median mu / mu)^p
   weight: smooth hard where the lens saw little, barely at all where it saw a
   lot.

WHAT A HONEST ABLATION LOOKS LIKE
---------------------------------
Three runs, identical otherwise:

    --mu-gate 0 --curv-mu 0      B4 reproduced (the control)
    --mu-gate 2.0 --curv-mu 0    gate only
    --mu-gate 2.0 --curv-mu 1    gate + weighted curvature

The magnification-adaptive idea has already come out neutral twice in this
project (Option 2's linear inversion, B3's correction penalty). If it comes out
neutral a third time, that is the result and it should be reported as such --
three independent negatives on a well-motivated idea is a genuine contribution.
The difference here is that the gate is the first version of the idea that
constrains the MODEL rather than nudging the loss.

Run from the Grid_Based_Experiment root:
    python superres/train_b5_mu.py --root . \\
        --lens-fits superres/results/fits_refined_mu.json \\
        --lens-fits-train superres/results/fits_train.json \\
        --n-train 6000 --n-val 800 --epochs 30 --base sersic \\
        --mu-gate 2.0 --curv-mu 1 --out superres/results/b5_gate.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:                                    # pragma: no cover
    raise SystemExit(f"train_b5_mu.py needs torch (import failed: {exc})")

from backproject import backproject, ray_shoot_batch, sample_source
from raytrace import area_downsample, image_plane_grid, load_psf
from sisr_net import SourceSISR
from sources import SersicSource
# shared, unmodified:
from train_b4 import LENS_KEYS, SRC_KEYS, build_inputs, curvature, mu_weights, pack

__all__ = ["mu_gate", "apply_gate"]


def mu_gate(cov, mu_gate_val: float, tau: float = 0.35):
    """Soft gate in log-mu, on the coarse (coverage) grid. Detached.

    tau is the softness in dex. 0.35 means the gate goes from 0.1 to 0.9 over
    about a factor of 5 in mu, which is gentle enough not to print a hard ring
    into the source at the threshold radius.
    """
    if mu_gate_val <= 0:
        return None
    with torch.no_grad():
        lm = torch.log10(cov.clamp_min(1e-3))
        return torch.sigmoid((lm - np.log10(mu_gate_val)) / tau)


def apply_gate(S_fine, g_coarse, mag: int):
    """S_out = g*S_fine + (1-g)*coarse(S_fine), with g upsampled from the coarse
    grid.

    coarse(S_fine) is an exact area average over each mag x mag block, put back
    at full size by nearest-neighbour repetition. It carries the same total flux
    as S_fine block by block, so the gate redistributes detail without creating
    or destroying flux -- which matters, because flux conservation is the one
    thing the forward model is most sensitive to.
    """
    if g_coarse is None:
        return S_fine
    B, n_out, _ = S_fine.shape
    coarse = F.avg_pool2d(S_fine[:, None], mag)
    coarse = F.interpolate(coarse, scale_factor=mag, mode="nearest")[:, 0]
    g = F.interpolate(g_coarse[:, None], size=(n_out, n_out),
                      mode="nearest")[:, 0]
    return g * S_fine + (1.0 - g) * coarse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--classes", nargs="+", default=["axion", "cdm", "wdm"])
    ap.add_argument("--lens-fits", default="superres/results/fits_refined_mu.json")
    ap.add_argument("--lens-fits-train", default="superres/results/fits_train.json")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pixel-scale", type=float, default=0.10593)
    ap.add_argument("--psf-path", default="superres/results/psf_empirical.npy")
    ap.add_argument("--supersample", type=int, default=1)
    ap.add_argument("--fit-radius-px", type=float, default=45.0)
    ap.add_argument("--half-extent", type=float, default=1.2)
    ap.add_argument("--mag", type=int, default=2)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--base", default="sersic", choices=["none", "sersic"])
    ap.add_argument("--reg-mode", default="mu", choices=["mu", "uniform", "none"])
    ap.add_argument("--reg-power", type=float, default=0.5)
    ap.add_argument("--lambda-curv", type=float, default=3.0)
    ap.add_argument("--lambda-l2", type=float, default=0.5)
    ap.add_argument("--sigma-floor", type=float, default=0.02)
    # ---- the two new things ------------------------------------------
    ap.add_argument("--mu-gate", type=float, default=2.0,
                    help="mu below which the super-resolution detail is removed. "
                         "0 disables the gate and reproduces B4. 2.0 is where "
                         "mu_resolution.py measures the error to jump.")
    ap.add_argument("--gate-tau", type=float, default=0.35,
                    help="gate softness in dex")
    ap.add_argument("--curv-mu", type=int, default=1,
                    help="1 = weight the Laplacian penalty by (median mu/mu)^p, "
                         "0 = uniform, as in B4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="superres/results/b5_gate.pt")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = a.pixel_scale

    val_rows = {r["index"]: r for r in json.load(open(a.lens_fits))["rows"]}
    if not os.path.exists(a.lens_fits_train):
        raise SystemExit(f"missing {a.lens_fits_train} -- see B4.md step 1")
    train_rows = {r["index"]: r for r in json.load(open(a.lens_fits_train))["rows"]}

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
    mask = torch.as_tensor(np.hypot(yy - c, xx - c) <= a.fit_radius_px, device=dev)

    n_in = int(round(2 * a.half_extent / res)) + 1
    n_out = n_in * a.mag
    out_scale = 2.0 * a.half_extent / (n_out - 1)

    net = SourceSISR(in_ch=2 + (a.base == "sersic"), width=a.width,
                     depth=a.depth, mag=a.mag, n_up=1).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    print(f"device {dev}   train {len(Xtr)}   val {len(Xva)}   "
          f"weights {sum(p.numel() for p in net.parameters()):,}")
    print(f"source {n_out}^2 at {out_scale:.4f} arcsec/px "
          f"({res/out_scale:.2f}x finer than the detector)   base {a.base}")
    if a.mu_gate > 0:
        print(f"MU GATE ON at mu = {a.mu_gate} (softness {a.gate_tau} dex): below "
              f"that the super-resolution\n  detail is removed and the output "
              f"falls back to {2*a.half_extent/(n_in-1):.4f} arcsec/px.")
    else:
        print("mu gate OFF -- this run reproduces B4")
    print(f"curvature weighting: {'mu-weighted' if a.curv_mu else 'uniform'}"
          f"   lambda_curv {a.lambda_curv}   lambda_l2 {a.lambda_l2}\n")

    def step(X, S, L, P, BG, train):
        X, S, L = X.to(dev), S.to(dev), L.to(dev)
        P, BG = P.to(dev), BG.to(dev)
        lens = {k: L[:, i] for i, k in enumerate(LENS_KEYS)}
        with torch.no_grad():
            bx, by = ray_shoot_batch(GX, GY, lens)
            bp, cov = backproject(X, bx, by, n_in, a.half_extent)
            g = mu_gate(cov, a.mu_gate, a.gate_tau)

        src = {k: P[:, i].reshape(-1, 1, 1) for i, k in enumerate(SRC_KEYS)}
        base_in = base_out = None
        if a.base == "sersic":
            ai = torch.linspace(-a.half_extent, a.half_extent, n_in, device=dev)
            YY, XX = torch.meshgrid(ai, ai, indexing="ij")
            base_in = SersicSource(src).at(XX[None], YY[None])
            ao = torch.linspace(-a.half_extent, a.half_extent, n_out, device=dev)
            YO, XO = torch.meshgrid(ao, ao, indexing="ij")
            base_out = SersicSource(src).at(XO[None], YO[None])

        y = net(build_inputs(X, S, bp, cov, base_in))
        amp = P[:, SRC_KEYS.index("amp")].reshape(-1, 1, 1).clamp_min(1e-6)
        S_map = amp * y
        if base_out is not None:
            S_map = base_out + amp * (y - float(np.log(2.0)))
        S_map = apply_gate(S_map, g, a.mag)          # <-- the structural constraint

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
        w = None
        if a.reg_mode != "none":
            w = mu_weights(cov, a.reg_mode, a.reg_power)
            w = F.interpolate(w[:, None], size=rel.shape[-2:], mode="nearest")[:, 0]
        pen = torch.zeros((), device=dev)
        if a.lambda_curv > 0:
            if a.curv_mu and w is not None:
                lap = (rel[:, :-2, 1:-1] + rel[:, 2:, 1:-1] +
                       rel[:, 1:-1, :-2] + rel[:, 1:-1, 2:] - 4.0 * rel[:, 1:-1, 1:-1])
                pen = pen + a.lambda_curv * (w[:, 1:-1, 1:-1] * lap ** 2).mean()
            else:
                pen = pen + a.lambda_curv * curvature(rel)
        if a.lambda_l2 > 0 and w is not None:
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
    print("now run:  python superres/eval_b5_mu.py --ckpt "
          + a.out.replace(".pt", "_best.pt") + " --root .")


if __name__ == "__main__":
    main()
