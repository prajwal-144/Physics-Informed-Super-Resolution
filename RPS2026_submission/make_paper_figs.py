"""Publication figures for the RPS 2026 submission.

Everything here replots or re-renders results that already exist in
superres/results/. The forward-model stages (lensed sky, + PSF + binning) are
re-rendered with the project's own numpy modules from the frozen lens fit and
the network's stored native source map, and are validated against the stored
prediction array before being drawn.
"""
import json, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

UP = "/mnt/user-data/uploads/Grid_Based_Experiment/superres"
R = f"{UP}/results"
OUT = "/home/claude/rps2026/figs"
MAN = "/root/.claude/uploads/ff05c4ab-0730-58da-8f88-db5c381e7d59/75ce89d4-manifest.csv"
sys.path.insert(0, UP)

from raytrace import image_plane_grid, convolve, area_downsample, load_psf  # noqa: E402
from lens_models import ray_shoot                                          # noqa: E402
from sources import SersicSource                                           # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "figure.dpi": 400,
})
NAVY, AMBER = "#22315c", "#d98b28"
RES = 0.10593
HALF = 1.2          # b4_sersic half-extent, arcsec
NPIX = 127

man = {r["path"].replace("/", "\\"): r for r in csv.DictReader(open(MAN))}
LENS = {r["index"]: r for r in json.load(open(f"{R}/fits_refined_mu.json"))["rows"]}


def stretch(a, lo=1.0, hi=99.5):
    v0, v1 = np.percentile(a, lo), np.percentile(a, hi)
    if not np.isfinite(v0) or not np.isfinite(v1) or v1 <= v0:
        v0, v1 = float(np.min(a)), float(np.max(a)) + 1e-12
    return v0, v1


class BoxSource:
    """Bilinear sampler over a square source map spanning +-half arcsec."""

    def __init__(self, grid, half=HALF):
        self.g = np.asarray(grid, float)
        self.half = float(half)

    def at(self, bx, by):
        n = self.g.shape[0]
        u = (np.asarray(bx) + self.half) / (2 * self.half) * (n - 1)
        v = (np.asarray(by) + self.half) / (2 * self.half) * (n - 1)
        inside = (u >= 0) & (u <= n - 1) & (v >= 0) & (v <= n - 1)
        u = np.clip(u, 0, n - 1 - 1e-6)
        v = np.clip(v, 0, n - 1 - 1e-6)
        i0, j0 = np.floor(v).astype(int), np.floor(u).astype(int)
        fy, fx = v - i0, u - j0
        g = self.g
        out = (g[i0, j0] * (1 - fy) * (1 - fx) + g[i0 + 1, j0] * fy * (1 - fx)
               + g[i0, j0 + 1] * (1 - fy) * fx + g[i0 + 1, j0 + 1] * fy * fx)
        return np.where(inside, out, 0.0)


def true_sersic_on_box(path, n=48, half=HALF):
    """Render the generating Sersic on the super-resolved grid. Display only."""
    r = man[path.replace("/", "\\")]
    p = dict(amp=1.0,
             R_sersic=float(r["source_R_sersic"]), n_sersic=float(r["source_n_sersic"]),
             se1=float(r["source_e1"]), se2=float(r["source_e2"]),
             sx=float(r["source_x"]), sy=float(r["source_y"]))
    ax = (np.arange(n) - (n - 1) / 2.0) * (2 * half / n)
    X, Y = np.meshgrid(ax, ax, indexing="xy")
    Yg = np.meshgrid(ax, ax, indexing="ij")[0]
    return SersicSource(p).at(X, Yg)


def crop_center(a, half_px):
    c = (a.shape[0] - 1) // 2
    return a[c - half_px:c + half_px + 1, c - half_px:c + half_px + 1]


def stages(idx, npz, display_supersample=3):
    """Re-render the forward-model stages for one saved example.

    The prediction is rendered at supersample 1, which is what eval_b4.py used,
    so it can be checked against the stored array. The unblurred sky panel is
    rendered at supersample 3 for display only.
    """
    src = npz[f"src_b4_native_{idx}"].astype(float)
    row = LENS[idx]
    lp = {k: row[k] for k in ("theta_E", "gamma", "e1", "e2", "g1", "g2")}
    psf = load_psf(f"{R}/psf_empirical.npy")
    S = BoxSource(src)
    X, Y = image_plane_grid(NPIX, RES, 1)
    bx, by = ray_shoot(X, Y, lp)
    sky1 = S.at(bx, by)
    pred = area_downsample(convolve(sky1, psf), 1) + float(row["background"])
    Xd, Yd = image_plane_grid(NPIX, RES, display_supersample)
    bxd, byd = ray_shoot(Xd, Yd, lp)
    sky = S.at(bxd, byd)
    return dict(src=src, sky=sky, pred=pred, path=row["path"])


def figure_pipeline(indices=(1, 5), out=f"{OUT}/fig1_pipeline.png", check=True,
                    drop=()):
    npz = np.load(f"{R}/b4_sersic_examples.npz")
    cols = ["observation", "back-projection", "super-resolved\nsource, 2$\\times$",
            "true source,\nanalytic 2$\\times$", "stored truth,\ndetector 1$\\times$",
            "lensed sky,\nno PSF", "$+$ PSF $+$ binning\n$=$ model",
            "data $-$ model"]
    cols = [c for k, c in enumerate(cols) if k not in drop]
    nr, nc = len(indices), len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(5.5, 5.5 / nc * nr + 0.30))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(indices):
        st = stages(i, npz)
        obs = npz[f"obs_{i}"].astype(float)
        stored_pred = npz[f"pred_{i}"].astype(float)
        if check:
            num = np.abs(st["pred"] - stored_pred).max() / max(stored_pred.max(), 1e-12)
            print(f"  example {i}: re-rendered prediction matches stored to "
                  f"{num:.3f} of peak")
        tru2 = true_sersic_on_box(st["path"])
        tru1 = crop_center(npz[f"src_true_{i}"].astype(float), 11)
        resid = obs - st["pred"]
        sv = stretch(st["src"])
        ov = stretch(obs)
        rv = np.percentile(np.abs(resid), 99.0)
        panels = [(obs, "magma", ov), (npz[f"bp_{i}"].astype(float), "magma", None),
                  (st["src"], "magma", sv),
                  (tru2, "magma", None),
                  (tru1, "magma", stretch(tru1)),
                  (st["sky"], "magma", None), (st["pred"], "magma", None),
                  (resid, "RdBu_r", (-rv, rv))]
        panels = [q for c, q in enumerate(panels) if c not in drop]
        for c, (a, cm, vv) in enumerate(panels):
            ax = axes[r, c]
            v = vv if vv is not None else stretch(a)
            ax.imshow(a, cmap=cm, vmin=v[0], vmax=v[1], origin="lower",
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("0.55")
            if r == 0:
                ax.set_title(cols[c], fontsize=4.9, color=NAVY, pad=2.0,
                             linespacing=1.15)
        axes[r, 0].set_ylabel(f"system {i}", fontsize=5.2, color=NAVY, labelpad=2)
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.032, right=0.996,
                        top=0.845, bottom=0.008)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("wrote", out)


def figure_validation(curve_idx=(0, 1, 3, 4), out=f"{OUT}/fig2_validation.png"):
    z = np.load(f"{R}/magnification_examples.npz")
    d = json.load(open(f"{R}/mu_resolution.json"))

    fig = plt.figure(figsize=(4.25, 1.66))
    gs = GridSpec(2, 24, figure=fig, height_ratios=[0.85, 1.0],
                  left=0.02, right=0.98, top=0.88, bottom=0.20, hspace=0.60)

    spans = [(0, 6), (6, 12), (12, 18), (18, 24)]
    for k, i in enumerate(curve_idx):
        ax = fig.add_subplot(gs[0, spans[k][0]:spans[k][1]], projection="polar")
        T = z[f"rcrit_{i}"]
        phi, rf, rt = T[:, 0], T[:, 1], T[:, 2]
        cl = lambda p, r: (np.append(p, p[0]), np.append(r, r[0]))
        ax.plot(*cl(phi, rt), c=NAVY, lw=0.95)
        ax.plot(*cl(phi, rf), c=AMBER, lw=0.85, ls="--")
        ax.set_yticklabels([]); ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels([]); ax.grid(lw=0.3, color="0.85"); ax.tick_params(pad=0)
        if k == 0:
            ax.set_title("tangential critical curve: true (solid) against recovered "
                         "(dashed), four systems", fontsize=5.2, color=NAVY,
                         pad=2, loc="left", x=0.0)

    bins, ea = d["bins"], d["experiment_a"]
    xs, ys = [], []
    for k in sorted(ea, key=int):
        n, err = ea[k]
        if n < 20:
            continue
        lo, hi = bins[int(k)], bins[int(k) + 1]
        xs.append(np.sqrt(lo * hi)); ys.append(err)
    rows = d["experiment_b"]
    mu = np.array([r["mu"] for r in rows])
    ct = np.array([r["contrast"] for r in rows])
    sn = np.array([r["snr"] for r in rows])

    axA = fig.add_subplot(gs[1, 0:10])
    axA.scatter(sn, ct, s=0.5, c=NAVY, alpha=0.11, linewidths=0)
    edges = np.unique(np.percentile(sn, np.linspace(0, 100, 9)))
    cen, med, q1, q3 = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (sn >= a) & (sn < b)
        if m.sum() < 15:
            continue
        cen.append(np.sqrt(a * b)); med.append(np.median(ct[m]))
        q1.append(np.percentile(ct[m], 25)); q3.append(np.percentile(ct[m], 75))
    axA.fill_between(cen, q1, q3, color=AMBER, alpha=0.22, lw=0)
    axA.plot(cen, med, "o-", c=AMBER, lw=1.0, ms=2.8)
    axA.set_xscale("log"); axA.set_ylim(0, 0.85)
    axA.set_xlabel("signal-to-noise of the image", fontsize=5.4, labelpad=1)
    axA.set_ylabel("recovered / injected", fontsize=5.4, labelpad=1)
    axA.set_title("A. clump recovery against photon count", fontsize=5.6,
                  color=NAVY, pad=2)

    c0 = {r["index"]: r["chi2_per_dof"]
          for r in json.load(open(f"{R}/b4_sersic_metrics.json"))["rows"]}
    g0 = {r["index"]: r["chi2_per_dof"]
          for r in json.load(open(f"{R}/b5_gate_metrics.json"))["rows"]}
    keys = sorted(set(c0) & set(g0))
    dl = np.array([np.log10(g0[i] / c0[i]) for i in keys])
    frac = float((dl < 0).mean())
    axB = fig.add_subplot(gs[1, 14:24])
    axB.hist(np.clip(dl, -0.06, 0.06), bins=61, range=(-0.06, 0.06),
             color=NAVY, alpha=0.85, lw=0)
    axB.set_xlim(-0.06, 0.06)
    axB.axvline(0, c="0.35", lw=0.7, ls="--")
    axB.set_xlabel(r"$\log_{10}(\chi^2_{\rm gate} / \chi^2_{\rm control})$",
                   fontsize=5.4, labelpad=1)
    axB.set_ylabel("images", fontsize=5.4, labelpad=1)
    axB.set_title(f"B. magnification gate: better on {frac*100:.0f}% of images",
                  fontsize=5.6, color=NAVY, pad=2)
    print(f"  gate better on {frac*100:.1f} per cent of images")

    for a in (axA, axB):
        a.tick_params(labelsize=4.8, length=1.6, pad=1.0)
        a.grid(lw=0.3, color="0.9"); a.set_axisbelow(True)
        for s in a.spines.values():
            s.set_linewidth(0.45); s.set_color("0.6")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    figure_pipeline(indices=(1,), out=f"{OUT}/fig1_pipeline.png")
    figure_pipeline(indices=(1, 5, 0, 2), out=f"{OUT}/figA_pipeline_extra.png",
                    check=False, drop=(5, 7))
    figure_validation()
