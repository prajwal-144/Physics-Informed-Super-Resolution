"""
make_figures_option2.py -- figures for the free-form (pixellated) reconstruction.

Reads only files already on disk:
    results/pixel_source.json     scores + per-image lambda sweeps
    results/pixel_examples.npz    example source maps saved by fit_pixel_source.py
    results/fits_img.json         the parametric fits, for the side-by-side

Writes results/fig6..fig9. Recomputes nothing, so the figures cannot disagree
with eval_pixel_source.py.

FIGURES
-------
  fig6_pixel_source   observation | parametric source | free-form (uniform) |
                      free-form (mu-adaptive) | truth. The headline for Option 2.
  fig7_lcurve         the L-curve for a few images with the chosen corner marked.
                      Shows how lambda was selected, without hand tuning.
  fig8_magnification  ray coverage per source pixel -- which IS the magnification
                      -- beside the recovered source. Shows where the lens
                      delivered resolution.
  fig9_mu_vs_uniform  source error against data fidelity for both regularisation
                      modes. This is the figure that carries the negative result.

Run from the Grid_Based_Experiment root:
    python superres/make_figures_option2.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, AMBER, GOOD, BAD, GREY = "#1E2761", "#D98C1F", "#2E9E5B", "#C0392B", "#8891A5"


def _stretch(a, p=99.5):
    a = np.asarray(a, float)
    lo, hi = np.percentile(a, 1.0), np.percentile(a, p)
    a = np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1)
    return np.sqrt(a)


def fig6_pixel_source(ex, modes, out, n_show=3):
    have = [i for i in range(n_show) if f"obs_{i}" in ex]
    cols = ["observation", "parametric\n(7 parameters)"] + \
           [f"free-form\n({m})" for m in modes] + ["TRUE source"]
    fig, ax = plt.subplots(len(have), len(cols), figsize=(3.0 * len(cols), 3.1 * len(have)))
    if len(have) == 1:
        ax = ax[None, :]
    for k, i in enumerate(have):
        panels = [ex[f"obs_{i}"], ex.get(f"src_parametric_{i}")]
        panels += [ex.get(f"src_{m}_{i}") for m in modes]
        panels += [ex.get(f"src_true_{i}")]
        for c, im in enumerate(panels):
            if im is None:
                ax[k, c].axis("off"); continue
            ax[k, c].imshow(_stretch(im), origin="lower", cmap="magma")
            ax[k, c].set_xticks([]); ax[k, c].set_yticks([])
            if k == 0:
                ax[k, c].set_title(cols[c], fontsize=10, color=NAVY)
    fig.suptitle("Free-form source reconstruction on the frozen lens\n"
                 "the lens comes from the parametric fit and is NOT refitted here",
                 fontsize=12, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def fig7_lcurve(ex, rows, mode, out, n_show=4):
    have = [i for i in range(n_show) if f"lcurve_{i}" in ex]
    if not have:
        print("  (no lcurve arrays saved; skipping fig7)")
        return
    fig, ax = plt.subplots(1, len(have), figsize=(4.0 * len(have), 3.8))
    if len(have) == 1:
        ax = [ax]
    for k, i in enumerate(have):
        T = ex[f"lcurve_{i}"]          # lambda, chi2, reg, curvature
        x, y = np.log10(np.maximum(T[:, 1], 1e-30)), np.log10(np.maximum(T[:, 2], 1e-30))
        ax[k].plot(x, y, "-o", c=NAVY, ms=4, lw=1.4)
        j = int(np.argmax(T[:, 3]))
        ax[k].plot(x[j], y[j], "*", c=BAD, ms=18, zorder=5,
                   label=f"corner, $\\lambda$={T[j,0]:.3g}")
        ax[k].set_xlabel(r"$\log_{10}\ \chi^2$ (fit to data)")
        if k == 0:
            ax[k].set_ylabel(r"$\log_{10}\ \|$regularisation$\|$")
        ax[k].legend(fontsize=8); ax[k].grid(alpha=.25)
        ax[k].set_title(f"image {i}", fontsize=10, color=NAVY)
    fig.suptitle("Choosing the regularisation strength by the L-curve corner\n"
                 "left arm = over-smoothed, lower arm = fitting noise; "
                 "no hand tuning", fontsize=11, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def fig8_magnification(ex, modes, out, n_show=3):
    have = [i for i in range(n_show) if f"cov_{i}" in ex]
    if not have:
        print("  (no coverage arrays saved; skipping fig8)")
        return
    fig, ax = plt.subplots(len(have), 3, figsize=(10.5, 3.3 * len(have)))
    if len(have) == 1:
        ax = ax[None, :]
    for k, i in enumerate(have):
        cov = ex[f"cov_{i}"]
        im0 = ax[k, 0].imshow(np.log10(np.maximum(cov, 1e-2)), origin="lower",
                              cmap="viridis")
        plt.colorbar(im0, ax=ax[k, 0], fraction=0.046)
        ax[k, 1].imshow(_stretch(ex[f"src_{modes[0]}_{i}"]), origin="lower", cmap="magma")
        ax[k, 2].imshow(_stretch(ex[f"src_true_{i}"]), origin="lower", cmap="magma")
        for c in range(3):
            ax[k, c].set_xticks([]); ax[k, c].set_yticks([])
        if k == 0:
            ax[k, 0].set_title(r"$\log_{10}$ ray coverage per source pixel"
                               "\n(this IS the magnification)", fontsize=10, color=NAVY)
            ax[k, 1].set_title(f"recovered source ({modes[0]})", fontsize=10, color=NAVY)
            ax[k, 2].set_title("true source", fontsize=10, color=NAVY)
    fig.suptitle("Where the lens delivered resolution\n"
                 "coverage = column sums of the lensing operator, so the "
                 "magnification and the ray-tracer cannot disagree",
                 fontsize=11, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def fig9_mu_vs_uniform(out_rows, out, fac_grid=np.linspace(1.01, 1.6, 25)):
    if not ("mu" in out_rows and "uniform" in out_rows):
        print("  (need both mu and uniform runs; skipping fig9)")
        return
    curves = {}
    for m in ("mu", "uniform"):
        ys = []
        for fac in fac_grid:
            vals = []
            for rec in out_rows[m]:
                sw = np.array(rec["sweep"])
                o = np.argsort(sw[:, 1])
                ch, nm = sw[o, 1], sw[o, 2]
                t = ch.min() * fac
                if ch.min() <= t <= ch.max():
                    vals.append(np.interp(t, ch, nm))
            ys.append(np.median(vals) if len(vals) >= 3 else np.nan)
        curves[m] = np.array(ys)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot(fac_grid, curves["uniform"], "-o", c=NAVY, ms=4,
               label="uniform regularisation")
    ax[0].plot(fac_grid, curves["mu"], "-s", c=AMBER, ms=4,
               label=r"magnification-adaptive ($\lambda \propto 1/\sqrt{\mu}$)")
    ax[0].set_xlabel(r"$\chi^2\ /\ \chi^2_{\min}$   (worse fit $\rightarrow$)")
    ax[0].set_ylabel("median source nmse vs truth")
    ax[0].set_title("Matched on data fidelity", fontsize=11, color=NAVY)
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.25)

    d = 100.0 * (curves["mu"] - curves["uniform"]) / np.maximum(curves["uniform"], 1e-12)
    ax[1].axhline(0, c="k", lw=1.2)
    ax[1].plot(fac_grid, d, "-o", c=BAD, ms=4)
    ax[1].fill_between(fac_grid, 0, d, where=d > 0, color=BAD, alpha=.15)
    ax[1].fill_between(fac_grid, 0, d, where=d < 0, color=GOOD, alpha=.15)
    ax[1].set_xlabel(r"$\chi^2\ /\ \chi^2_{\min}$")
    ax[1].set_ylabel("change in source nmse  [%]")
    ax[1].set_title("Below zero = magnification weighting helps",
                    fontsize=11, color=NAVY)
    ax[1].grid(alpha=.25)
    fig.suptitle("Does magnification-adaptive regularisation help?",
                 fontsize=12, color=NAVY)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pixel", default="superres/results/pixel_source_small.json")
    ap.add_argument("--npz", default="superres/results/pixel_examples_small.npz")
    ap.add_argument("--outdir", default="superres/results")
    a = ap.parse_args()

    blob = json.load(open(a.pixel))
    rows = blob["rows"]
    modes = list(rows.keys())
    ex = dict(np.load(a.npz))
    os.makedirs(a.outdir, exist_ok=True)
    print(f"figures from {a.pixel}   modes {modes}\n")

    fig6_pixel_source(ex, modes, f"{a.outdir}/fig6_pixel_source.png")
    fig7_lcurve(ex, rows, modes[0], f"{a.outdir}/fig7_lcurve.png")
    fig8_magnification(ex, modes, f"{a.outdir}/fig8_magnification.png")
    fig9_mu_vs_uniform(rows, f"{a.outdir}/fig9_mu_vs_uniform.png")
    print("\ndone.")


if __name__ == "__main__":
    main()