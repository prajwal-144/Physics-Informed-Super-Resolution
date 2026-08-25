"""
make_figures_b4_stages.py -- the full B4 chain, panel by panel, plus a clean
two-panel "observation -> super-resolved source" figure.

Unlike make_figures_b4.py this one RE-RUNS the network on a few images, because
the intermediate stages (the intrinsic lensed image before the PSF, the image
after the PSF, the binned prediction) are not saved in the npz.

FIGURES
-------
  figS5_stages    eight columns per example:
                    1 LR observation                 what the telescope recorded
                    2 back-projection                 the network's INPUT
                    3 TRUE source, npz                the only truth that exists,
                                                      and it is at DETECTOR
                                                      resolution -- see the note
                    4 TRUE source, analytic           the manifest Sersic
                                                      re-rendered on the SR grid,
                                                      DISPLAY ONLY
                    5 SR source                       the B4 output, 2.07x finer
                    6 lensed, no PSF, supersampled    what the SR source implies
                                                      on the sky
                    7 lensed + PSF + detector binning
                    8 prediction = 7 + background     exactly what chi^2 sees

  figS6_lr_vs_sr  two big panels: the observation and the super-resolved source.

WHY THE "TRUE SOURCE" LOOKS PIXELATED -- THIS IS NOT A BUG
-----------------------------------------------------------
Model_A stores `unlensed` on the SAME 127x127 detector grid as the image, at
0.10593 arcsec/pixel, already PSF-convolved. Cropped to the +-1.2 arcsec source
box that is 23x23 pixels. B4's output is 48x48 over the same box.

So the reference is genuinely coarser than the reconstruction, and there is no
higher-resolution ground truth anywhere in the dataset. That is the whole point
of unsupervised super-resolution: if a high-resolution truth existed we could
train on it directly and none of this machinery would be needed.

Column 4 exists so the eye has a fair comparison: the true Sersic re-rendered
analytically from the manifest parameters on the 48x48 grid. It is used for
DISPLAY ONLY and never enters a metric -- every number in eval_b4.py is scored
against the npz array in column 3.

Run from the Grid_Based_Experiment root:
    python superres/make_figures_b4_stages.py --ckpt superres/results/b4_best.pt
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

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:                                    # pragma: no cover
    raise SystemExit(f"needs torch (import failed: {exc})")

from backproject import backproject, ray_shoot_batch, sample_source
from data_a import ModelADataset
from raytrace import area_downsample, image_plane_grid, load_psf
from sisr_net import SourceSISR
from sources import SersicSource
from train_b4 import LENS_KEYS, SRC_KEYS, build_inputs

NAVY, BAD = "#1E2761", "#C0392B"


def _st(a, p=99.5):
    a = np.asarray(a, float)
    lo, hi = np.percentile(a, 1.0), np.percentile(a, p)
    return np.sqrt(np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1))


def true_source_hr(t, n_out, H, flux_target):
    """The manifest Sersic re-rendered on the SR grid. DISPLAY ONLY.

    The manifest has no source amplitude, so the render is scaled to carry the
    same total flux as the npz array. Shape, size, ellipticity and position are
    all the true values; only the overall normalisation is borrowed.
    """
    ax = np.linspace(-H, H, n_out)
    X, Y = np.meshgrid(ax, ax, indexing="xy")[0], np.meshgrid(ax, ax, indexing="ij")[0]
    s = SersicSource(dict(amp=1.0, R_sersic=t["source_R_sersic"],
                          n_sersic=t["source_n_sersic"], se1=t["source_e1"],
                          se2=t["source_e2"], sx=t["source_x"],
                          sy=t["source_y"])).at(X, Y)
    return s * (flux_target / max(s.sum(), 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="superres/results/b4_best.pt")
    ap.add_argument("--root", default=".")
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 2, 5],
                    help="which val images to draw")
    ap.add_argument("--display-supersample", type=int, default=3,
                    help="only for the 'lensed, no PSF' column")
    ap.add_argument("--outdir", default="superres/results")
    a = ap.parse_args()

    try:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(a.ckpt, map_location="cpu")
    ta = ck["args"]
    res, H, S = ta["pixel_scale"], ta["half_extent"], ta["supersample"]
    n_in = int(round(2 * H / res)) + 1
    n_out = n_in * ta["mag"]

    lens_rows = {r["index"]: r for r in json.load(open(ta["lens_fits"]))["rows"]}
    ds = ModelADataset(a.root, split="val", classes=ta["classes"],
                       limit=max(a.indices) + 1)
    n_pix = ds.image(0).shape[-1]

    net = SourceSISR(in_ch=2 + (ta["base"] == "sersic"), width=ta["width"],
                     depth=ta["depth"], mag=ta["mag"], n_up=1)
    net.load_state_dict(ck["model"])
    net.eval()

    psf_np = load_psf(ta["psf_path"])
    psf = torch.as_tensor(psf_np, dtype=torch.float32)
    gx, gy = image_plane_grid(n_pix, res, S)
    GX, GY = torch.as_tensor(gx, dtype=torch.float32), torch.as_tensor(gy, dtype=torch.float32)
    hx, hy = image_plane_grid(n_pix, res, a.display_supersample)
    HX, HY = torch.as_tensor(hx, dtype=torch.float32), torch.as_tensor(hy, dtype=torch.float32)

    stages = []
    for i in a.indices:
        img = ds.image(i)
        sig = ds.sigma(i, img)
        r = lens_rows[i]
        X = torch.as_tensor(img[None], dtype=torch.float32)
        Sg = torch.as_tensor([sig], dtype=torch.float32)
        lens = {k: torch.tensor([r[k]], dtype=torch.float32) for k in LENS_KEYS}
        src = {k: torch.tensor([[[r[k]]]], dtype=torch.float32) for k in SRC_KEYS}

        with torch.no_grad():
            bx, by = ray_shoot_batch(GX, GY, lens)
            bp, cov = backproject(X, bx, by, n_in, H)
            base_in = base_out = None
            if ta["base"] == "sersic":
                ai = torch.linspace(-H, H, n_in)
                YY, XX = torch.meshgrid(ai, ai, indexing="ij")
                base_in = SersicSource(src).at(XX[None], YY[None])
                ao = torch.linspace(-H, H, n_out)
                YO, XO = torch.meshgrid(ao, ao, indexing="ij")
                base_out = SersicSource(src).at(XO[None], YO[None])
            y = net(build_inputs(X, Sg, bp, cov, base_in))
            amp = torch.tensor([[[r["amp"]]]], dtype=torch.float32).clamp_min(1e-6)
            S_map = amp * y
            if base_out is not None:
                S_map = base_out + amp * (y - float(np.log(2.0)))

            # --- stage 6: intrinsic lensed image, supersampled, NO PSF -------
            hbx, hby = ray_shoot_batch(HX, HY, lens)
            hi_sky = sample_source(S_map, hbx, hby, H)[0].numpy()

            # --- stages 7-8: exactly what the model does ---------------------
            sky = sample_source(S_map, bx, by, H).unsqueeze(1)
            pad = psf.shape[-1] // 2
            blur = F.conv2d(F.pad(sky, (pad,) * 4, mode="replicate"), psf[None, None])
            binned = area_downsample(blur, S)[0, 0].numpy()
            pred = binned + r["background"]

            det = sample_source(
                S_map,
                torch.as_tensor(image_plane_grid(n_pix, res, 1)[0], dtype=torch.float32)[None],
                torch.as_tensor(image_plane_grid(n_pix, res, 1)[1], dtype=torch.float32)[None],
                H)[0].numpy()

        t = ds.truth(i)
        c = (n_pix - 1) // 2
        h = int(round(H / res))
        tru_npz = ds.unlensed(i)[c - h:c + h + 1, c - h:c + h + 1]
        stages.append(dict(
            idx=i, obs=img, bp=bp[0].numpy(),
            tru_npz=tru_npz,
            tru_hr=true_source_hr(t, n_out, H, float(np.clip(tru_npz, 0, None).sum())),
            sr=S_map[0].numpy(), hi=hi_sky, binned=binned, pred=pred,
            snr=t["snr_max"], det=det))

    os.makedirs(a.outdir, exist_ok=True)

    # ---------------- figS5 --------------------------------------------
    cols = ["1. LR observation\n(the data)",
            "2. back-projection\n(network input)",
            f"3. TRUE source, npz\n{stages[0]['tru_npz'].shape[0]}$^2$ @ {res:.4f}\"",
            f"4. TRUE source, analytic\n{n_out}$^2$ @ {2*H/(n_out-1):.4f}\"  (display only)",
            f"5. SUPER-RESOLVED source\n{n_out}$^2$ @ {2*H/(n_out-1):.4f}\"",
            "6. lensed, no PSF\n(supersampled)",
            "7. + PSF + detector binning",
            "8. prediction\n(= 7 + background)"]
    keys = ["obs", "bp", "tru_npz", "tru_hr", "sr", "hi", "binned", "pred"]
    n = len(stages)
    fig, ax = plt.subplots(n, 8, figsize=(25, 3.4 * n))
    if n == 1:
        ax = ax[None, :]
    for k, S_ in enumerate(stages):
        for c2, key in enumerate(keys):
            ax[k, c2].imshow(_st(S_[key]), origin="lower", cmap="magma")
            ax[k, c2].set_xticks([]); ax[k, c2].set_yticks([])
            if k == 0:
                ax[k, c2].set_title(cols[c2], fontsize=9.5, color=NAVY)
        ax[k, 0].set_ylabel(f"image {S_['idx']}\nSNR {S_['snr']:.0f}",
                            fontsize=9, color=NAVY)
    for c2 in (4,):
        for k in range(n):
            for sp in ax[k, c2].spines.values():
                sp.set_edgecolor(BAD); sp.set_linewidth(2.0)
    fig.suptitle(
        "B4, end to end. Columns 1-2 and 6-8 are the actual pipeline; 5 is the "
        "answer.\n"
        "Column 3 is the ONLY ground truth the dataset contains and it is stored "
        "at detector resolution -- there is no high-resolution truth to train on, "
        "which is why this is unsupervised.\n"
        "Column 4 re-renders the true Sersic analytically on the output grid so "
        "the eye can compare like with like; it is never used in a metric.",
        fontsize=12, color=NAVY)
    fig.tight_layout()
    out = f"{a.outdir}/figS5_stages.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")

    # ---------------- figS6 --------------------------------------------
    fig, ax = plt.subplots(n, 2, figsize=(9, 4.4 * n))
    if n == 1:
        ax = ax[None, :]
    for k, S_ in enumerate(stages):
        ax[k, 0].imshow(_st(S_["obs"]), origin="lower", cmap="magma")
        ax[k, 1].imshow(_st(S_["sr"]), origin="lower", cmap="magma")
        for c2 in range(2):
            ax[k, c2].set_xticks([]); ax[k, c2].set_yticks([])
        ax[k, 0].set_ylabel(f"image {S_['idx']}", fontsize=9, color=NAVY)
        if k == 0:
            ax[k, 0].set_title(f"observed\n{S_['obs'].shape[0]}$^2$ @ {res:.4f}\"/px",
                               fontsize=11, color=NAVY)
            ax[k, 1].set_title(f"reconstructed source\n{n_out}$^2$ @ "
                               f"{2*H/(n_out-1):.4f}\"/px  "
                               f"({res/(2*H/(n_out-1)):.2f}x finer)",
                               fontsize=11, color=NAVY)
    fig.suptitle("Unsupervised super-resolution: lensed observation in, "
                 "source out\nno high-resolution image was used in training",
                 fontsize=12, color=NAVY)
    fig.tight_layout()
    out = f"{a.outdir}/figS6_lr_vs_sr.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()