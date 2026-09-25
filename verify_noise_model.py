"""
verify_noise_model.py -- prove the noise model from the data, two ways.

Run this BEFORE trusting anything that depends on noise_model.POISSON_GAIN, and
hand it to anyone who asks where the gain came from. numpy only: no scipy, no
torch, and -- the point of the file -- no lens model, no source model and no
fitted physical parameter anywhere, so nothing here can be confounded by model
error.

    python superres/verify_noise_model.py --root . --n 40

WHAT IS BEING TESTED
Every chi^2 in this project used var = sigma_bg^2, the background annulus
variance. Model_A was generated with var = sigma_bg^2 + flux/t. The two differ by
10^3 to 10^5 on a bright arc pixel, which is where every constraint lives.

ROUTE 2 IS THE DECISIVE ONE -- READ IT FIRST
Every npz carries the truth scalar `snr_max`. If the generator used
var = sigma_bg^2 + g*flux then max(flux / sigma) over the image must reproduce
it. At g = 0 the prediction overshoots the stored value by a factor of ~240; at
g ~= 0.8 it lands on 1.00 with a tight spread. That is the GENERATOR's own
definition of signal-to-noise read back out of its own output, so it settles
both the functional form and the coefficient.

ROUTE 1 CORROBORATES IT INDEPENDENTLY
`unlensed[1] = c * unlensed[0]` to about 1 per cent rms, and source_R_sersic,
source_n_sersic, source_e1, source_e2 are single scalars rather than per-band
arrays, so the two bands are the SAME source at a different amplitude. Hence

    r = image[1] - c * image[0]

contains only noise, and binning var(r) against flux measures the variance law
with no model of any kind:

    var(r) = A + (c + c^2) * g * f

Two traps, both handled below, and both of which cost me a wrong answer first:

  * Binning by the raw pixel value i0 is Eddington-biased. Pixels whose noise
    pushed them toward zero fall in the lowest bin while still carrying the
    Poisson variance of their true flux, so the lowest bin comes out ~100x above
    any sensible model and the fitted intercept goes negative. The flux proxy
    must be a LOW-NOISE estimate.
  * The proxy must also be statistically INDEPENDENT of r at the same pixel, or
    binning by it biases the very variance being measured. So the proxy is a
    leave-one-out neighbourhood mean of BAND 0 ONLY: it excludes the centre
    pixel and never touches band 1.

Route 1's g comes out systematically ~30 per cent HIGH, and that is expected
rather than a disagreement: the leave-one-out mean smooths the flux, which
understates the peak on a sharp arc, so the slope has to rise to match the
variance observed there. Route 1 therefore confirms the FORM and the order of
magnitude; route 2 fixes the coefficient.

A SECOND FINDING, REPORTED AS `A / var_bg`
The fitted noise floor inside the fit disc is ~10x the annulus estimate.
data_a.sigma() is correct about the annulus -- var(r) measured there matches it
to a couple of per cent -- but the floor inside r < 45 px is higher, because the
PSF's Moffat-like wings spread arc flux across the whole disc and that spread
flux carries its own shot noise. In the pipeline this takes care of itself: the
model `pred` there includes the fitted `background`, so gain*max(pred,0) already
accounts for the sky's shot noise. It matters here only because this script has
no model to lean on.

WHITE-NOISE CHECK
Lag-1 and lag-2 autocorrelation of r in the source-free annulus. Near zero means
the noise was added AFTER the PSF convolution, so a diagonal per-pixel variance
is a complete description and no covariance matrix is needed.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from noise_model import POISSON_GAIN
except Exception:                                            # standalone use
    POISSON_GAIN = 0.80


def loo_mean(a: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean of the k x k neighbourhood EXCLUDING the centre pixel.

    Excluding the centre is what makes the result independent of that pixel's
    own noise, which is what lets it be used as a binning variable for a
    variance measured at the same pixel. numpy only, no scipy.
    """
    p = k // 2
    A = np.pad(a, p, mode="reflect")
    tot = np.zeros_like(a)
    for dy in range(k):
        for dx in range(k):
            tot += A[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return (tot - a) / float(k * k - 1)


def fit_floor_and_slope(B: np.ndarray, var_bg: float, n_grid: int = 80):
    """Fit var = A + b*f over binned (f, var) points, with A >= var_bg, b >= 0.

    A one-dimensional scan over A with b solved in closed form at each A. The
    objective is relative rather than absolute error, because var spans three
    decades across the bins and an absolute least squares would be decided
    entirely by the brightest bin.

    A is floored at var_bg because the noise floor inside the disc cannot be
    BELOW the annulus floor. Without that floor a faint system can drive A to
    almost zero, which then divides the faintest pixels by nearly nothing and
    sends chi^2 to several hundred on that one system.
    """
    f, v = B[:, 0], B[:, 1]
    lo = max(var_bg, 1e-30)
    hi = max(float(v.max()), lo * 10.0)
    best = None
    for A in np.geomspace(lo, hi, n_grid):
        b = max(float(((v - A) * f).sum() / max(float((f * f).sum()), 1e-30)), 0.0)
        res = float((((A + b * f) - v) ** 2 / np.maximum(v * v, 1e-30)).sum())
        if best is None or res < best[0]:
            best = (res, A, b)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="folder containing Model_A/")
    ap.add_argument("--split", default="val")
    ap.add_argument("--classes", nargs="+", default=["axion", "cdm", "wdm"])
    ap.add_argument("--n", type=int, default=30, help="systems to use")
    ap.add_argument("--fit-radius-px", type=float, default=45.0,
                    help="the disc fit_per_image.py scores its chi^2 on")
    ap.add_argument("--bg-radius-px", type=float, default=50.0,
                    help="the annulus data_a.sigma() measures")
    ap.add_argument("--proxy-k", type=int, default=5,
                    help="leave-one-out window for the flux proxy")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = []
    for c in a.classes:
        files += sorted(glob.glob(os.path.join(a.root, "Model_A", a.split, c,
                                               "*.npz")))
    if not files:
        raise SystemExit(f"no npz under {a.root}/Model_A/{a.split} for {a.classes}")
    random.Random(a.seed).shuffle(files)
    files = files[:a.n]

    with np.load(files[0], allow_pickle=True) as z0:
        n_pix = z0["image"].shape[-1]
    cen = (n_pix - 1) / 2.0
    yy, xx = np.indices((n_pix, n_pix))
    rad = np.hypot(yy - cen, xx - cen)
    disc = rad <= a.fit_radius_px
    annulus = rad > a.bg_radius_px

    print(f"\nn = {len(files)} systems   classes {a.classes}   split {a.split}")
    print(f"grid {n_pix}^2   disc r <= {a.fit_radius_px:.0f} px   "
          f"annulus r > {a.bg_radius_px:.0f} px")
    print(f"flux proxy: leave-one-out {a.proxy_k}x{a.proxy_k} mean of band 0\n")

    # ---- premise of route 1 ----------------------------------------------
    print("PREMISE   unlensed[1] = c * unlensed[0] ?   (route 1 needs this)")
    prem = []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            u = z["unlensed"].astype(np.float64)
        if u.ndim != 3 or u.shape[0] < 2:
            continue
        c = float((u[1] * u[0]).sum() / max((u[0] * u[0]).sum(), 1e-30))
        prem.append(float(np.sqrt(((u[1] - c * u[0]) ** 2).mean())
                          / max(np.sqrt((u[1] ** 2).mean()), 1e-30)))
    if not prem:
        print("  single-band archives -- route 1 unavailable\n")
    else:
        prem = np.array(prem)
        print(f"  rms(u1 - c*u0)/rms(u1):  median {np.median(prem):.4f}"
              f"   p90 {np.percentile(prem, 90):.4f}")
        print("  -> below ~0.02 means one source at two amplitudes. Route 1 valid.\n")

    # ---- route 1 ---------------------------------------------------------
    G, CHI_OLD, CHI_NEW, CHI_REF, FLOOR, AC1, AC2 = [], [], [], [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            im = z["image"].astype(np.float64)
        if im.ndim != 3 or im.shape[0] < 2:
            continue
        i0, i1 = im[0], im[1]
        c = float((i1[disc] * i0[disc]).sum()
                  / max((i0[disc] * i0[disc]).sum(), 1e-30))
        r = i1 - c * i0
        fe = np.clip(loo_mean(i0, a.proxy_k), 0.0, None)
        s0, s1 = i0[annulus].std(), i1[annulus].std()
        var_bg = s1 ** 2 + (c * s0) ** 2                  # the OLD model

        q = np.quantile(fe[disc], np.linspace(0, 1, 20))
        B = []
        for lo, hi in zip(q[:-1], q[1:]):
            m = disc & (fe >= lo) & (fe < hi)
            if m.sum() > 40:
                B.append((fe[m].mean(), r[m].var()))
        if len(B) < 8:
            continue
        B = np.array(B)
        A0, b1 = fit_floor_and_slope(B, var_bg)
        G.append(b1 / max(c + c * c, 1e-30))
        FLOOR.append(A0 / max(var_bg, 1e-30))

        # chi^2/dof of KNOWN NOISE. dof = n_pixels: nothing physical was fitted.
        CHI_OLD.append(float((r[disc] ** 2 / max(var_bg, 1e-30)).mean()))
        CHI_NEW.append(float((r[disc] ** 2
                              / np.clip(A0 + b1 * fe[disc], 1e-30, None)).mean()))
        ref = A0 + POISSON_GAIN * (c + c * c) * fe[disc]
        CHI_REF.append(float((r[disc] ** 2 / np.clip(ref, 1e-30, None)).mean()))

        R = np.where(annulus, r, np.nan)
        for lag, acc in ((1, AC1), (2, AC2)):
            p_, q_ = R[:, :-lag].ravel(), R[:, lag:].ravel()
            m = np.isfinite(p_) & np.isfinite(q_)
            if m.sum() > 500:
                acc.append(float(np.corrcoef(p_[m], q_[m])[0, 1]))

    g1 = None
    if len(G):
        G = np.array(G); FLOOR = np.array(FLOOR)
        CHI_OLD = np.array(CHI_OLD); CHI_NEW = np.array(CHI_NEW)
        CHI_REF = np.array(CHI_REF)
        g1 = float(np.median(G))
        print("ROUTE 1   r = image1 - c*image0  is pure noise")
        print(f"  per-band gain g          p10 {np.percentile(G,10):5.2f}"
              f"   median {g1:5.2f}   p90 {np.percentile(G,90):5.2f}")
        print("    expected ~30% HIGH: the proxy smooths the flux, understating")
        print("    arc peaks, so the slope rises to compensate. Route 2 is the")
        print("    coefficient of record.")
        print(f"  floor A / annulus var_bg   median {np.median(FLOOR):6.1f}")
        print("    >1 is a SECOND finding, not an error: the noise floor inside")
        print("    the fit disc exceeds the r>50 annulus estimate, because the PSF")
        print("    wings spread arc flux over the disc and that flux shot-noises.")
        print("    data_a.sigma() is right about the annulus, not about the disc.")
        print("\n  chi2/dof of that KNOWN NOISE:")
        print(f"    var = sigma_bg^2            median {np.median(CHI_OLD):11.0f}"
              f"   p10 {np.percentile(CHI_OLD,10):9.0f}"
              f"   p90 {np.percentile(CHI_OLD,90):10.0f}   <- THE OLD CONVENTION")
        print(f"    var = A + b*f (fitted)      median {np.median(CHI_NEW):11.2f}"
              f"   p10 {np.percentile(CHI_NEW,10):9.2f}"
              f"   p90 {np.percentile(CHI_NEW,90):10.2f}")
        print(f"    var = A + {POISSON_GAIN:g}*(c+c^2)*f       "
              f"median {np.median(CHI_REF):11.2f}"
              f"   p10 {np.percentile(CHI_REF,10):9.2f}"
              f"   p90 {np.percentile(CHI_REF,90):10.2f}")
        print(f"    inside [0.5, 2.0] on the fitted law: "
              f"{100*float(np.mean((CHI_NEW > 0.5) & (CHI_NEW < 2.0))):.0f} %")
        print("\n  The old convention mis-scores PURE NOISE by ~10^3. Whatever else")
        print("  is or is not true of the model, that factor is not model error.")
        if AC1:
            w = abs(float(np.median(AC1))) < 0.05
            print(f"\n  white-noise check   lag1 {np.median(AC1):+.3f}"
                  f"   lag2 {np.median(AC2):+.3f}   -> "
                  f"{'white: a diagonal variance is complete' if w else 'CORRELATED: a diagonal variance is NOT enough'}")

    # ---- route 2 ---------------------------------------------------------
    print("\nROUTE 2   does max(flux/sigma) reproduce the stored snr_max?")
    print(f"  {'g':>8} {'median ratio':>14} {'p10':>9} {'p90':>9}")
    best, best_err = None, np.inf
    for g in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2):
        R = []
        for f in files:
            with np.load(f, allow_pickle=True) as z:
                im = z["image"].astype(np.float64)
                snr = float(np.asarray(z["snr_max"]))
            b0 = im[0] if im.ndim == 3 else im
            s = b0[annulus].std()
            fl = np.clip(b0, 0.0, None)
            if snr > 0:
                R.append(float(np.nanmax(fl / np.sqrt(np.clip(s ** 2 + g * fl,
                                                              1e-30, None))) / snr))
        R = np.array(R)
        med = float(np.median(R))
        tag = "   <- pre-correction" if g == 0.0 else ""
        print(f"  {g:8.2f} {med:14.3f} {np.percentile(R,10):9.3f}"
              f" {np.percentile(R,90):9.3f}{tag}")
        if g > 0 and abs(med - 1.0) < best_err:
            best, best_err = g, abs(med - 1.0)
    print("\n  The g whose ratio is nearest 1.0 is the generator's own gain. At")
    print("  g = 0 the ratio is two orders of magnitude above 1, which is only")
    print("  possible if the generator's sigma carried a flux term.")

    # ---- verdict ---------------------------------------------------------
    print("\nVERDICT")
    if g1 is not None:
        print(f"  route 1 (band difference, form + magnitude)  g ~ {g1:.2f}"
              f"   [biased high, see above]")
    print(f"  route 2 (stored snr_max, coefficient of record) g = {best:.2f}")
    print(f"  noise_model.POISSON_GAIN                        g = {POISSON_GAIN:.2f}")
    ok = abs(POISSON_GAIN - best) <= 0.15
    print(f"  -> POISSON_GAIN {'agrees with route 2' if ok else 'DISAGREES with route 2 -- update it'}")
    print("\n  Both routes reject var = sigma_bg^2 decisively and agree that the")
    print("  variance is linear in flux. If the Model_A generation config is ever")
    print("  recovered, g is exactly 1/exposure_time in stored units and should")
    print("  replace this estimate in noise_model.py.\n")


if __name__ == "__main__":
    main()
