"""
calibrate_psf.py -- measure Model_A's PSF instead of assuming it.

WHY
---
Every SIS run used `--psf-fwhm-arcsec 0.10`. That number was never measured. It
can be, because the npz `unlensed` array is the source ALREADY CONVOLVED with
the instrument PSF, and the manifest gives the analytic Sersic parameters of
that source. So the PSF is the ratio of the two, and it is over-determined.

Two independent estimates are produced:

  A. PARAMETRIC SWEEP. Render the analytic Sersic, convolve with a trial
     Gaussian, and minimise the residual against `unlensed`. Result: a clean
     single minimum at FWHM = 0.18-0.20 arcsec, residual nmse 0.0029, against
     0.0071 at 0.10 arcsec -- 2.4x worse. Reported in
     diagnostics_2026_08_12/PROJECT_REPORT.md section 7.4.

  B. EMPIRICAL FOURIER EXTRACTION. Since unlensed = sersic * PSF, in Fourier
     space PSF_hat = FFT(unlensed) / FFT(sersic), regularised as a Wiener filter
     and stacked over many images to beat down the noise. This assumes NOTHING
     about the kernel shape.

WHY IT MATTERS MORE THAN IT LOOKS
---------------------------------
The residual floor of estimate A is nmse 0.003 rather than ~0, which already
says the true kernel is not exactly Gaussian. Model_A's arcs reach
peak / sigma_background of 10^3 - 10^4 (sigma measured from the r > 50 px
annulus), so a 1% error in the PSF SHAPE is a ~50 sigma per-pixel residual. That
is why fit_per_image.py reaches chi^2/dof ~ 3000 with a Gaussian PSF even
though it recovers theta_E to 0.048 arcsec and R_sersic to 0.018 arcsec: the
parameters are right and the kernel shape is the systematic floor.

Run from the Grid_Based_Experiment root:
    python sie_pipeline/calibrate_psf.py
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from data_a import ModelADataset
from raytrace import area_downsample, convolve, gaussian_psf, image_plane_grid
from sources import SersicSource

RES_DEFAULT = 0.10593


def _sersic_hr(t, n_pix, pixel_scale, supersample):
    X, Y = image_plane_grid(n_pix, pixel_scale, supersample)
    sp = dict(amp=1.0, R_sersic=t["source_R_sersic"], n_sersic=t["source_n_sersic"],
              se1=t["source_e1"], se2=t["source_e2"],
              sx=t["source_x"], sy=t["source_y"])
    return SersicSource(sp).at(X, Y)


def nmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    s = (a * b).sum() / max((a * a).sum(), 1e-30)
    return float(((s * a - b) ** 2).sum() / max((b * b).sum(), 1e-30))


def sweep_gaussian(ds, n_pix, pixel_scale, supersample=5, n=25,
                   grid=np.arange(0.0, 0.34, 0.02), max_R=0.8):
    """Estimate A -- best-fit Gaussian width."""
    print("A. Gaussian FWHM sweep against the npz `unlensed` array")
    curves, used = [], 0
    for i in range(len(ds)):
        t = ds.truth(i)
        if t["source_R_sersic"] > max_R:      # PSF barely matters for huge sources
            continue
        hr = _sersic_hr(t, n_pix, pixel_scale, supersample)
        u = ds.unlensed(i)
        row = []
        for fw in grid:
            k = gaussian_psf(fw, pixel_scale / supersample) if fw > 0 else None
            row.append(nmse(area_downsample(convolve(hr, k), supersample), u))
        curves.append(row)
        used += 1
        if used >= n:
            break
    C = np.array(curves)
    med = np.median(C, axis=0)
    print("   n = %d compact sources (R_sersic < %.2f arcsec)" % (used, max_R))
    print("   %9s%10s" % ("FWHM(as)", "nmse"))
    for fw, v in zip(grid, med):
        star = "   <-- minimum" if v == med.min() else ""
        print(f"   {fw:9.2f}{v:10.5f}{star}")
    best = float(grid[int(np.argmin(med))])
    print("\n   best-fit Gaussian FWHM = %.2f arcsec" % best)
    print(f"   residual nmse at the minimum = {med.min():.5f}")
    print("   A floor well above 0 means the true kernel is NOT Gaussian;")
    print("   this number is therefore an EQUIVALENT GAUSSIAN WIDTH.\n")
    return best, med.min()


def extract_empirical(ds, n_pix, pixel_scale, n=40, supersample=1,
                      snr_reg=1e-3, max_R=0.8):
    """Estimate B -- Wiener deconvolution, stacked, assuming no kernel shape.

    unlensed = sersic (*) psf  =>  PSF_hat = U_hat * conj(S_hat) / (|S_hat|^2 + eps)

    eps is a Wiener regularisation: without it, frequencies where the Sersic has
    no power divide by ~0 and the estimate explodes. Stacking over images with
    different source sizes fills in different parts of the frequency plane.
    """
    print("B. Empirical PSF by regularised Fourier division (no shape assumed)")
    acc = None
    used = 0
    for i in range(len(ds)):
        t = ds.truth(i)
        if t["source_R_sersic"] > max_R:
            continue
        s = _sersic_hr(t, n_pix, pixel_scale, supersample)
        if supersample > 1:
            s = area_downsample(s, supersample)
        u = ds.unlensed(i)
        s = s / max(s.sum(), 1e-30)
        u = u / max(u.sum(), 1e-30)
        S = np.fft.rfft2(np.fft.ifftshift(s))
        U = np.fft.rfft2(np.fft.ifftshift(u))
        P = U * np.conj(S) / (np.abs(S) ** 2 + snr_reg * np.abs(S).max() ** 2)
        acc = P if acc is None else acc + P
        used += 1
        if used >= n:
            break
    psf = np.fft.fftshift(np.fft.irfft2(acc / used, s=(n_pix, n_pix)))
    psf = np.clip(psf, 0, None)
    psf /= psf.sum()

    c = (n_pix - 1) // 2
    yy, xx = np.indices(psf.shape)
    r = np.hypot(yy - c, xx - c) * pixel_scale
    tot = psf.sum()
    # half-light radius -> FWHM equivalent for a Gaussian is 2*r_half*1.1774/1.1774
    order = np.argsort(r.ravel())
    cum = np.cumsum(psf.ravel()[order]) / tot
    r_half = float(r.ravel()[order][np.searchsorted(cum, 0.5)])
    fwhm_equiv = 2.0 * r_half
    # second moment
    sig = float(np.sqrt((psf * r ** 2).sum() / tot / 2.0))
    print(f"   n = {used} images stacked")
    print("   half-light radius        = %.3f arcsec  (FWHM-equivalent %.3f)" % (r_half, fwhm_equiv))
    print("   rms width  sqrt(<r^2>/2) = %.3f arcsec  (Gaussian FWHM equivalent %.3f)"
          % (sig, sig * 2.3548))
    prof = []
    for rr in np.arange(0, 0.55, 0.05):
        m = np.abs(r - rr) < 0.03
        if m.sum():
            prof.append((rr, float(psf[m].mean() / psf.max())))
    print("   radial profile, normalised to the peak:")
    g_sig = (0.18 / 2.3548)
    for rr, v in prof:
        g = np.exp(-0.5 * (rr / g_sig) ** 2)
        print("     r = %.2f   empirical %8.4f   gaussian(0.18) %8.4f   ratio %6.2f"
              % (rr, v, g, v / max(g, 1e-9)))
    print("   A ratio rising with radius means the true kernel has broader wings")
    print("   than a Gaussian -- i.e. it is Moffat-like.\n")
    return psf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="val")
    ap.add_argument("--classes", nargs="+", default=["axion"])
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--pixel-scale", type=float, default=RES_DEFAULT)
    ap.add_argument("--save", default="sie_pipeline/results/psf_empirical.npy")
    a = ap.parse_args()

    ds = ModelADataset(a.root, split=a.split, classes=a.classes, limit=2000)
    n_pix = ds.image(0).shape[-1]
    print("=" * 74)
    print(f"PSF CALIBRATION -- {ds.summary()}")
    print("=" * 74 + "\n")
    sweep_gaussian(ds, n_pix, a.pixel_scale, n=a.n)
    psf = extract_empirical(ds, n_pix, a.pixel_scale, n=max(a.n, 40))
    os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
    np.save(a.save, psf)
    print(f"wrote {a.save}")


if __name__ == "__main__":
    main()
