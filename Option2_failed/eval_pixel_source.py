"""
eval_pixel_source.py -- score a fit_pixel_source.py run.

WHY A SEPARATE SCRIPT
---------------------
fit_pixel_source.py prints a summary as it goes, but that couples solving to
scoring: to re-examine the results you would have to re-run a 35-minute
inversion. This reads the JSON only. Same separation as Path A, where
fit_per_image.py solves and evaluate.py scores.

WHAT IT REPORTS
---------------
  1. source recovery per regularisation mode, against the npz `unlensed`
  2. the FAIR magnification comparison -- matched on data fidelity, not on each
     mode's own L-curve lambda (they have different H, so the same lambda means
     different smoothing)
  3. head-to-head against the PARAMETRIC source from the same frozen lens
  4. everything stratified by snr_max
  5. lambda and conditioning diagnostics

Run from the Grid_Based_Experiment root:
    python superres/eval_pixel_source.py --pixel superres/results/pixel_source.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import metrics as M
from data_a import ModelADataset
from fit_pixel_source import build_psf, score_against_truth
from raytrace import convolve, image_plane_grid
from sources import SersicSource

SRC_KEYS = ("amp", "R_sersic", "n_sersic", "se1", "se2", "sx", "sy")


def med(rows, key):
    v = np.array([r.get(key, np.nan) for r in rows], float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def strat(rows, key, snr, bins=((0, 6), (6, 15), (15, 1e9))):
    v = np.array([r.get(key, np.nan) for r in rows], float)
    out = []
    for lo, hi in bins:
        m = np.isfinite(v) & (snr >= lo) & (snr < hi)
        lbl = f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}"
        out.append((lbl, int(m.sum()),
                    float(np.median(v[m])) if m.sum() else np.nan))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pixel", default="superres/results/pixel_source.json")
    ap.add_argument("--fits", default="superres/results/fits_img.json",
                    help="the parametric fits, for the head-to-head")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    blob = json.load(open(a.pixel))
    cfg, fcfg, out = blob["config"], blob["fits_config"], blob["rows"]
    modes = list(out.keys())
    n = len(out[modes[0]])
    res = fcfg.get("pixel_scale", 0.10593)

    ds = ModelADataset(a.root, split=fcfg.get("split", "val"),
                       classes=fcfg.get("classes", ["axion"]),
                       limit=max(r["index"] for r in out[modes[0]]) + 1)
    n_pix = ds.image(0).shape[-1]
    snr = np.array([ds.truth(r["index"])["snr_max"] for r in out[modes[0]]])

    print("=" * 78)
    print(f"PIXEL-SOURCE EVALUATION   {a.pixel}")
    print("=" * 78)
    src_scale = 2.0 * cfg["half_extent"] / (cfg["n_src"] - 1)
    print(f"  n = {n}   source {cfg['n_src']}^2 at {src_scale:.4f} arcsec/px"
          f"   ({res/src_scale:.2f}x finer than the detector)")
    print(f"  reg power {cfg.get('reg_power')}   clip {cfg.get('reg_clip')}"
          f"   modes {modes}")
    print(f"  lens frozen from {cfg['fits']}\n")

    # ---------------- 1. source recovery ----------------
    print("1. SOURCE RECOVERY vs the npz `unlensed` array\n")
    print(f"   {'mode':>10}{'corr':>9}{'size_ratio':>12}{'centroid px':>13}"
          f"{'nmse':>9}{'peak_ratio':>12}{'lambda':>10}")
    for m in modes:
        R = out[m]
        print(f"   {m:>10}{med(R,'src_corr'):9.4f}{med(R,'src_size_ratio'):12.4f}"
              f"{med(R,'src_centroid_err_px'):13.4f}{med(R,'src_nmse'):9.4f}"
              f"{med(R,'src_peak_ratio'):12.4f}{med(R,'lambda'):10.3g}")

    # ---------------- 2. the fair magnification test ----------------
    if "mu" in out and "uniform" in out:
        print("\n2. MAGNIFICATION-ADAPTIVE vs UNIFORM, matched on data fidelity\n")
        print("   Each mode has a different H, so the same lambda means different")
        print("   smoothing -- comparing at each mode's own L-curve corner is not a")
        print("   test. Instead each image's lambda sweep is interpolated to a")
        print("   COMMON fraction of that image's best achievable chi2.\n")
        print(f"   {'chi2/chi2_min':>15}{'nmse mu':>11}{'nmse unif':>12}"
              f"{'change':>10}{'n':>5}{'mu better':>11}")
        for fac in (1.02, 1.05, 1.10, 1.25, 1.50):
            am, au = [], []
            for rm, ru in zip(out["mu"], out["uniform"]):
                pair = []
                for rec in (rm, ru):
                    sw = np.array(rec["sweep"])
                    o = np.argsort(sw[:, 1])
                    ch, nm = sw[o, 1], sw[o, 2]
                    t = ch.min() * fac
                    pair.append(np.interp(t, ch, nm) if ch.min() <= t <= ch.max()
                                else np.nan)
                if np.all(np.isfinite(pair)):
                    am.append(pair[0]); au.append(pair[1])
            if len(am) < 3:
                print(f"   {fac:>15.2f}      (too few images span this)")
                continue
            am, au = np.array(am), np.array(au)
            vm, vu = float(np.median(am)), float(np.median(au))
            frac_better = 100.0 * float((am < au).mean())
            print(f"   {fac:>15.2f}{vm:>11.4f}{vu:>12.4f}"
                  f"{100*(vm-vu)/max(vu,1e-12):>9.1f}%{len(am):>5}"
                  f"{frac_better:>10.0f}%")
        print("\n   NEGATIVE change = magnification weighting helps.")
        print("   'mu better' is the fraction of individual images on which it")
        print("   wins, which distinguishes a real effect from a median shifted")
        print("   by a few outliers.")

    # ---------------- 3. vs the parametric source ----------------
    if os.path.exists(a.fits):
        print("\n3. FREE-FORM vs PARAMETRIC, same frozen lens, same images\n")
        prows = {r["index"]: r for r in json.load(open(a.fits))["rows"]}
        psf_lr = build_psf(fcfg, res, 1)
        X, Y = image_plane_grid(n_pix, res, 1)
        par = []
        for r in out[modes[0]]:
            pr = prows.get(r["index"])
            if pr is None:
                continue
            s = convolve(SersicSource({k: pr[k] for k in SRC_KEYS}).at(X, Y), psf_lr)
            par.append(M.source_truth(s, ds.unlensed(r["index"]), res))
        gp = lambda k: float(np.nanmedian([x[k] for x in par]))
        print(f"   {'source model':>16}{'corr':>9}{'size_ratio':>12}{'nmse':>9}")
        print(f"   {'parametric (7 par)':>16}{gp('corr'):9.4f}"
              f"{gp('size_ratio'):12.4f}{gp('nmse'):9.4f}")
        for m in modes:
            print(f"   {'free-form ' + m:>16}{med(out[m],'src_corr'):9.4f}"
                  f"{med(out[m],'src_size_ratio'):12.4f}{med(out[m],'src_nmse'):9.4f}")
        print("\n   Model_A's sources ARE Sersics, so the 7-parameter model has")
        print("   exactly the right prior and 4,096 free pixels cannot beat it.")
        print("   That is expected and is not a failure of the free-form solve --")
        print("   the point of the free-form version is that it needs no such")
        print("   prior, which is what real galaxies require.")

    # ---------------- 4. stratified ----------------
    print("\n4. STRATIFIED BY snr_max\n")
    for key in ("src_nmse", "src_size_ratio"):
        print(f"   {key}")
        for m in modes:
            rowtxt = "   ".join(f"{lbl} n={c}: {v:.4f}"
                                for lbl, c, v in strat(out[m], key, snr))
            print(f"     {m:>8}   {rowtxt}")

    # ---------------- 5. diagnostics ----------------
    print("\n5. DIAGNOSTICS\n")
    for m in modes:
        R = out[m]
        cgfail = sum(1 for r in R if r.get("cg_info", 0) != 0)
        lams = np.array([r["lambda"] for r in R], float)
        print(f"   {m:>8}   chi2/dof median {med(R,'chi2_per_dof'):10.1f}"
              f"   lambda p10-p90 {np.percentile(lams,10):.3g}-{np.percentile(lams,90):.3g}"
              f"   CG non-converged {cgfail}/{len(R)}")
    print(f"   ray coverage (= magnification) per source pixel:"
          f" median {med(out[modes[0]],'mu_median'):.2f}"
          f"   max {med(out[modes[0]],'mu_max'):.1f}")
    print("\n   A high CG non-convergence count means the normal matrix is")
    print("   ill-conditioned -- reduce --n-src or raise --reg-clip.")

    if a.out:
        summ = {m: {k: med(out[m], k) for k in
                    ("src_corr", "src_size_ratio", "src_nmse",
                     "src_centroid_err_px", "lambda", "chi2_per_dof")}
                for m in modes}
        json.dump({"config": cfg, "summary": summ}, open(a.out, "w"), indent=1)
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
