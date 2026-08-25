"""
fit_pixel_source.py -- free-form super-resolution on the frozen lens.

WHAT IT DOES
------------
For each image:
  1. read the lens parameters that fit_per_image.py already found (frozen),
  2. build the sparse lensing operator L for that lens,
  3. choose the regularisation strength by the L-curve corner,
  4. solve for a free grid of source pixels, finer than the detector,
  5. score the result against the npz `unlensed` truth.

By default it does this TWICE per image -- once with magnification-adaptive
regularisation and once with uniform -- so the value of the magnification
weighting is measured rather than asserted. That comparison is the point of the
script.

Nothing here refits the lens. The lens is an input, taken from
results/fits_img.json, and that is what makes the source problem linear and the
solve a single matrix inversion rather than an optimisation.

Run from the Grid_Based_Experiment root:
    python superres/fit_pixel_source.py --fits superres/results/fits_img.json \\
        --root . --n 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import metrics as M
from data_a import ModelADataset
from pixel_source import (PixelInversion, build_lensing_operator, lcurve_lambda,
                          source_grid)
from raytrace import (convolve, gaussian_psf, image_plane_grid, load_psf,
                      moffat_psf)
from sources import SersicSource

LENS_KEYS = ("theta_E", "gamma", "e1", "e2", "g1", "g2")
SRC_KEYS = ("amp", "R_sersic", "n_sersic", "se1", "se2", "sx", "sy")


def build_psf(cfg, pixel_scale, supersample):
    mode = cfg.get("psf_mode", "empirical")
    if mode == "gaussian":
        return gaussian_psf(cfg.get("psf_fwhm", 0.20), pixel_scale / supersample)
    if mode == "moffat":
        return moffat_psf(cfg.get("psf_fwhm", 0.20), pixel_scale / supersample,
                          beta=cfg.get("psf_beta", 2.5))
    p = cfg.get("psf_path", "superres/results/psf_empirical.npy")
    if not os.path.exists(p):
        p = "sie_pipeline/results/psf_empirical.npy"
    return load_psf(p)


def corr(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def score_against_truth(s_map, ds, idx, n_src, half, pixel_scale, psf_lr, n_pix):
    """Compare the pixellated source with the stored `unlensed` array.

    `unlensed` lives on the 127 px DETECTOR grid and is POST-PSF. The pixel
    source lives on a finer grid and is pre-PSF. To compare like with like we
    resample the pixel source onto the detector grid and convolve it, which is
    the same PSF-matching evaluate.py does for the parametric source.
    """
    from scipy.ndimage import map_coordinates
    n = n_src
    X, Y = image_plane_grid(n_pix, pixel_scale, 1)
    scale = 2.0 * half / (n - 1)
    u = (X + half) / scale
    v = (Y + half) / scale
    s_on_det = map_coordinates(s_map.reshape(n, n), [v, u], order=1,
                               mode="constant", cval=0.0)
    s_on_det = convolve(s_on_det, psf_lr)
    return M.source_truth(s_on_det, ds.unlensed(idx), pixel_scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fits", default="superres/results/fits_img.json")
    ap.add_argument("--root", default=".")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-src", type=int, default=64)
    ap.add_argument("--half-extent", type=float, default=1.2)
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--fit-radius-px", type=float, default=45.0)
    ap.add_argument("--reg-modes", nargs="+", default=["mu", "uniform"])
    ap.add_argument("--reg-power", type=float, default=0.5,
                    help="w = (median_coverage/coverage)^power. 0 = uniform, "
                         "1 = full 1/mu. See pixel_source._reg_weights.")
    ap.add_argument("--reg-clip", type=float, default=5.0)
    ap.add_argument("--n-lambda", type=int, default=13)
    ap.add_argument("--n-save", type=int, default=6)
    ap.add_argument("--out", default="superres/results/pixel_source.json")
    ap.add_argument("--out-npz", default="superres/results/pixel_examples.npz")
    a = ap.parse_args()

    blob = json.load(open(a.fits))
    cfg, rows = blob["config"], blob["rows"]
    rows = rows[:a.n]
    res = cfg.get("pixel_scale", 0.10593)
    ds = ModelADataset(a.root, split=cfg.get("split", "val"),
                       classes=cfg.get("classes", ["axion"]), limit=len(rows))
    n_pix = ds.image(0).shape[-1]
    S = a.supersample
    grid = image_plane_grid(n_pix, res, S)
    psf = build_psf(cfg, res, S)
    psf_lr = build_psf(cfg, res, 1)

    src_scale = 2.0 * a.half_extent / (a.n_src - 1)
    c = (n_pix - 1) / 2.0
    yy, xx = np.indices((n_pix, n_pix))
    mask = np.hypot(yy - c, xx - c) <= a.fit_radius_px

    print("=" * 78)
    print("FREE-FORM SOURCE RECONSTRUCTION ON THE FROZEN LENS")
    print("=" * 78)
    print(f"  lens from   : {a.fits}   (NOT refitted here)")
    print(f"  source grid : {a.n_src}^2 = {a.n_src**2} pixels at {src_scale:.4f} arcsec/px")
    print(f"  detector    : {n_pix}^2 at {res:.4f} arcsec/px"
          f"   -> {res/src_scale:.2f}x finer")
    print(f"  data pixels : {int(mask.sum())} inside r < {a.fit_radius_px:.0f} px")
    print(f"  ratio       : {a.n_src**2/int(mask.sum()):.2f} unknowns per datum")
    print(f"                (the old SIS pipeline ran at {64516/4300:.1f})")
    print(f"  reg modes   : {a.reg_modes}\n")

    out = {m: [] for m in a.reg_modes}
    examples = {}
    t0 = time.time()

    for i, r in enumerate(rows):
        lens = {k: r[k] for k in LENS_KEYS}
        lens["cx"] = lens["cy"] = 0.0
        img = ds.image(r["index"])
        sigma = r["sigma"]

        L, cov = build_lensing_operator(lens, n_pix, res, S, a.n_src,
                                        a.half_extent, grid=grid)
        for mode in a.reg_modes:
            inv = PixelInversion(L, cov, psf, n_pix, S, a.n_src, sigma,
                                 mask, reg_mode=mode,
                                 reg_power=a.reg_power, reg_clip=a.reg_clip)
            lams = np.logspace(-3, 3, a.n_lambda)
            lam, table = lcurve_lambda(inv, img, lambdas=lams)
            s, info = inv.solve(img, lam)
            sc = score_against_truth(s, ds, r["index"], a.n_src, a.half_extent,
                                     res, psf_lr, n_pix)
            # sweep for the matched-chi2 comparison: (chi2/dof, nmse, corr)
            sweep = []
            for lm in lams:
                ss, _ = inv.solve(img, lm)
                sq = score_against_truth(ss, ds, r["index"], a.n_src,
                                         a.half_extent, res, psf_lr, n_pix)
                sweep.append([lm, inv.chi2(ss, img) / max(inv.n_data() - 1, 1),
                              sq["nmse"], sq["corr"]])
            rec = {"index": r["index"], "lambda": lam,
                   "sweep": np.array(sweep).tolist(),
                   "chi2_per_dof": inv.chi2(s, img) / max(inv.n_data() - 1, 1),
                   "cg_info": int(info),
                   "mu_median": float(np.median(cov[cov > 0])),
                   "mu_max": float(cov.max())}
            rec.update({f"src_{k}": float(v) for k, v in sc.items()})
            out[mode].append(rec)

            if i < a.n_save:
                examples[f"src_{mode}_{i}"] = s.reshape(a.n_src, a.n_src).astype(np.float32)
                if mode == a.reg_modes[0]:
                    examples[f"obs_{i}"] = img.astype(np.float32)
                    examples[f"cov_{i}"] = cov.reshape(a.n_src, a.n_src).astype(np.float32)
                    examples[f"lcurve_{i}"] = table.astype(np.float32)
                    sx, sy = source_grid(a.n_src, a.half_extent)
                    t = ds.truth(r["index"])
                    examples[f"src_true_{i}"] = SersicSource(dict(
                        amp=1.0, R_sersic=t["source_R_sersic"],
                        n_sersic=t["source_n_sersic"], se1=t["source_e1"],
                        se2=t["source_e2"], sx=t["source_x"], sy=t["source_y"]
                    )).at(sx, sy).astype(np.float32)
                    examples[f"src_parametric_{i}"] = SersicSource(
                        {k: r[k] for k in SRC_KEYS}).at(sx, sy).astype(np.float32)

        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            el = time.time() - t0
            print(f"  {i+1}/{len(rows)}   {el/(i+1):.2f} s/image"
                  f"   eta {el/(i+1)*(len(rows)-i-1)/60:.1f} min", flush=True)

    # ---------------- summary ----------------
    print("\n" + "=" * 78)
    print("SOURCE RECOVERY vs the npz `unlensed` array")
    print("=" * 78)
    print(f"\n  {'reg mode':>12}{'corr':>10}{'size_ratio':>13}{'centroid px':>14}"
          f"{'nmse':>10}{'lambda':>11}")
    summary = {}
    for mode in a.reg_modes:
        R = out[mode]
        g = lambda k: np.array([x[k] for x in R], float)
        summary[mode] = {k: float(np.nanmedian(g(k)))
                         for k in ("src_corr", "src_size_ratio",
                                   "src_centroid_err_px", "src_nmse",
                                   "lambda", "chi2_per_dof")}
        print(f"  {mode:>12}{np.nanmedian(g('src_corr')):10.4f}"
              f"{np.nanmedian(g('src_size_ratio')):13.4f}"
              f"{np.nanmedian(g('src_centroid_err_px')):14.4f}"
              f"{np.nanmedian(g('src_nmse')):10.4f}"
              f"{np.nanmedian(g('lambda')):11.3g}")

    if len(a.reg_modes) == 2 and "mu" in summary and "uniform" in summary:
        print("\n  MAGNIFICATION-ADAPTIVE vs UNIFORM, matched on data fidelity")
        print("  Comparing each mode at its own L-curve lambda is not a fair test:")
        print("  the two modes have different H, so the same lambda means different")
        print("  amounts of smoothing. Instead, for every image we interpolate each")
        print("  mode's lambda sweep to a COMMON chi2/dof and compare the source")
        print("  error there -- i.e. at equal fit to the data, which source is")
        print("  closer to the truth?\n")
        # Targets are RELATIVE to each image's own best achievable chi2 (the
        # smallest lambda in the sweep). Absolute targets like chi2/dof = 1 are
        # unreachable here: the PSF-shape systematic and the unmodelled
        # substructure put chi2/dof in the thousands regardless of lambda, so a
        # fixed target has no root. What is meaningful is "at the same fraction
        # of the achievable fit, which source is closer to the truth?"
        factors = [1.02, 1.05, 1.10, 1.25]
        print(f"     {'chi2 / chi2_min':>16}{'nmse mu':>11}{'nmse unif':>12}{'change':>10}{'n':>5}")
        for fac in factors:
            am, au = [], []
            for rm, ru in zip(out["mu"], out["uniform"]):
                pair = []
                for rec in (rm, ru):
                    sw = np.array(rec["sweep"])
                    o = np.argsort(sw[:, 1])
                    ch, nm = sw[o, 1], sw[o, 2]
                    tgt = ch.min() * fac
                    if ch.min() <= tgt <= ch.max():
                        pair.append(np.interp(tgt, ch, nm))
                if len(pair) == 2 and all(np.isfinite(pair)):
                    am.append(pair[0]); au.append(pair[1])
            if len(am) < 3:
                print(f"     {fac:>16.2f}      (too few images span this)")
                continue
            vm, vu = float(np.median(am)), float(np.median(au))
            print(f"     {fac:>16.2f}{vm:>11.4f}{vu:>12.4f}"
                  f"{100*(vm-vu)/max(vu,1e-12):>9.1f}%{len(am):>5}")
        print("\n     A NEGATIVE change means the magnification weighting helps:")
        print("     lower source error at the same fit to the data.")

    print(f"\n  reference: the PARAMETRIC source from fit_per_image.py on the same")
    print(f"  images scores corr 0.987, size_ratio 1.089 (evaluate.py section 2).")
    print(f"  The pixellated source is free-form -- it can represent structure a")
    print(f"  Sersic cannot -- so matching or beating that is the bar.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"config": vars(a), "fits_config": cfg,
               "summary": summary, "rows": out}, open(a.out, "w"), indent=1)
    np.savez_compressed(a.out_npz, **examples)
    print(f"\n  wrote {a.out}")
    print(f"  wrote {a.out_npz}")


if __name__ == "__main__":
    main()
