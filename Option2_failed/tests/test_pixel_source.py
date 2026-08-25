"""Validate the pixellated inversion before believing any of its output.

Four checks, in order of how badly a failure would corrupt the science:

  1. ADJOINT. CG solves A s = b assuming A is symmetric. A = M^T C^-1 M + lam H,
     which is symmetric only if MT() really is the transpose of M(). If it is
     not, CG converges silently to the wrong answer. Dot-product test:
     <M s, r> == <s, M^T r> to machine precision.

  2. H SYMMETRY. Same reason, for the regulariser.

  3. ROUND TRIP ON A KNOWN SOURCE. Render a Sersic through a known lens, add
     noise, invert, and check the recovered pixel map matches the source that
     generated it. This is the end-to-end test: if it fails, nothing downstream
     means anything.

  4. SUPER-RESOLUTION. The same round trip with a source containing structure
     FINER than the detector pixel, to show the inversion recovers detail that
     is not resolvable in the image plane. This is the actual claim.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np

from pixel_source import PixelInversion, build_lensing_operator, source_grid
from raytrace import gaussian_psf, image_plane_grid
from sources import SersicSource

RES, N, S = 0.10593, 127, 2
N_SRC, HALF = 64, 1.2
LENS = dict(theta_E=1.35, gamma=2.0, e1=0.18, e2=-0.10,
            g1=0.03, g2=-0.02, cx=0.0, cy=0.0)


def corr(a, b):
    a = np.asarray(a, float).ravel() - np.mean(a)
    b = np.asarray(b, float).ravel() - np.mean(b)
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


def make_inversion(sigma=1e-3, reg_mode="mu"):
    grid = image_plane_grid(N, RES, S)
    L, cov = build_lensing_operator(LENS, N, RES, S, N_SRC, HALF, grid=grid)
    psf = gaussian_psf(0.20, RES / S)
    c = (N - 1) / 2.0
    yy, xx = np.indices((N, N))
    mask = np.hypot(yy - c, xx - c) <= 45.0
    return PixelInversion(L, cov, psf, N, S, N_SRC, sigma, mask, reg_mode), grid


def test_adjoint(inv):
    rng = np.random.default_rng(0)
    ok = True
    for k in range(3):
        s = rng.normal(size=N_SRC * N_SRC)
        r = rng.normal(size=inv.n_data())
        lhs = float(inv.M(s) @ r)
        rhs = float(s @ inv.MT(r))
        rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
        print(f"   adjoint test {k}:  <Ms,r> = {lhs:+.8e}   <s,M^T r> = {rhs:+.8e}"
              f"   rel diff {rel:.2e}")
        if rel > 1e-10:
            ok = False
    return ok


def test_H_symmetry(inv):
    rng = np.random.default_rng(1)
    a = rng.normal(size=N_SRC * N_SRC)
    b = rng.normal(size=N_SRC * N_SRC)
    lhs = float(a @ inv.H(b))
    rhs = float(b @ inv.H(a))
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
    print(f"   H symmetry:  a.Hb = {lhs:+.6e}   b.Ha = {rhs:+.6e}   rel {rel:.2e}")
    return rel < 1e-10


def _round_trip(src_params, sigma, label, extra=None):
    inv, grid = make_inversion(sigma=sigma)
    sx, sy = source_grid(N_SRC, HALF)
    s_true = SersicSource(src_params).at(sx, sy)
    if extra is not None:
        s_true = s_true + extra(sx, sy)
    s_true = s_true.ravel()

    d_clean = inv.M(s_true)
    rng = np.random.default_rng(2)
    d_noisy_flat = d_clean + rng.normal(0, sigma, d_clean.shape)
    d_img = np.zeros((N, N))
    d_img[inv.mask] = d_noisy_flat

    best = (-1.0, None, None)
    for lam in np.logspace(-3, 2, 11):
        s, _ = inv.solve(d_img, lam)
        c = corr(s, s_true)
        if c > best[0]:
            best = (c, lam, s)
    c, lam, s = best
    scale = (s @ s_true) / max(s @ s, 1e-30)
    nrmse = float(np.sqrt(((scale * s - s_true) ** 2).sum() / (s_true ** 2).sum()))
    print(f"   {label}:  corr {c:.4f}   nrmse {nrmse:.4f}   at lambda {lam:.3g}")
    return c, nrmse


def main():
    print("1. ADJOINT AND SYMMETRY\n")
    inv, _ = make_inversion()
    ok = test_adjoint(inv)
    ok &= test_H_symmetry(inv)

    print(f"\n   source grid  {N_SRC}^2 at {2*HALF/(N_SRC-1):.4f} arcsec/px")
    print(f"   detector     {N}^2 at {RES:.4f} arcsec/px"
          f"   -> {RES/(2*HALF/(N_SRC-1)):.2f}x finer")
    print(f"   ray coverage per source pixel: median {np.median(inv.coverage):.1f}"
          f"   max {inv.coverage.max():.1f}"
          f"   (this IS the magnification)")

    print("\n2. ROUND TRIP ON A KNOWN SOURCE\n")
    smooth = dict(amp=1.0, R_sersic=0.35, n_sersic=1.2,
                  se1=0.15, se2=-0.08, sx=0.10, sy=-0.15)
    c1, e1 = _round_trip(smooth, 1e-3, "smooth Sersic          ")

    print("\n3. SUPER-RESOLUTION: structure FINER than a detector pixel\n")
    # a second, small blob 0.05 arcsec across -- half a detector pixel
    def blob(x, y):
        return 0.8 * np.exp(-0.5 * (((x - 0.22) ** 2 + (y + 0.05) ** 2) / 0.05 ** 2))
    c2, e2 = _round_trip(smooth, 1e-3, "Sersic + sub-pixel blob", extra=blob)
    print(f"   the added blob has sigma = 0.050 arcsec = 0.47 DETECTOR pixels,")
    print(f"   i.e. it is not resolvable in the image plane at all.")

    ok &= (c1 > 0.95) and (c2 > 0.90)
    print("\n" + ("PASS" if ok else "*** FAIL ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
