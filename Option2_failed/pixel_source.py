"""
pixel_source.py -- free-form (pixellated) source reconstruction on a FROZEN lens.

WHAT THIS IS
------------
fit_per_image.py gives a 7-parameter Sersic source. That is a strong baseline and
it validates the whole forward model, but a Sersic can only produce
Sersic-shaped things. This module drops the parametric form entirely: the source
becomes a free grid of pixels, and we solve for all of them.

THE KEY FACT THAT MAKES IT TRACTABLE
------------------------------------
With the lens parameters FROZEN (taken from fit_per_image.py), the predicted
image is LINEAR in the source pixels:

    d = B L s + n

  s : the source pixels we want            (n_src^2 unknowns)
  L : the lensing operator -- which source pixels each image subpixel reads from,
      with bilinear weights. Built once from the frozen lens.
  B : PSF convolution followed by detector binning. Also linear.
  d : the observed image
  n : noise, standard deviation sigma

So the maximum-a-posteriori source has a closed form:

    s = (M^T C^-1 M + lambda H)^-1 M^T C^-1 d ,      M = B L

This is the semilinear inversion of Warren & Dye (2003), the standard technique
in the lens-modelling literature. No training, no learning rate, no collapse
mode -- it either solves or it does not.

WHY IT IS WELL POSED NOW AND WAS NOT BEFORE
-------------------------------------------
The old pipeline solved for 254 x 254 = 64,516 free source pixels against
~4,300 informative data pixels, with the lens WRONG and theta_E leaked from the
manifest. That is 15:1 underdetermined with a misspecified operator; the result
was a source 10.6-12.8x too large across a 100x sweep of the TV weight.

Here: n_src = 64 gives 4,096 unknowns against ~4,300 informative pixels -- about
1:1 -- and the lens is correct, fitted per image, recovering ellipticity at
rho = 0.76 and shear at 0.61. The regulariser H then supplies the rest.

MAGNIFICATION-ADAPTIVE REGULARISATION -- THE PHYSICS
-----------------------------------------------------
The lens does not deliver the same resolution everywhere in the source plane.
Where the magnification |mu| is large, a small source patch is spread over many
detector pixels, so the sky has already oversampled it and fine source pixels
there are well constrained. Where |mu| ~ 1, they are not, and fine pixels will
simply fill with noise.

So the smoothing should not be uniform:

    lambda_eff(beta) = lambda_0 / |mu(beta)|

We get |mu| for free and self-consistently: the COLUMN SUMS of L count how many
image subpixels map into each source pixel, and that count IS the magnification
of that source pixel (up to the constant subpixel area). No separate mu
computation, no possibility of the mu map and the ray-tracer describing
different lenses -- which was exactly the defect in
`physics_losses.fixed_sis_magnification`, where mu came from a hard-coded
circular formula unconnected to the sparse operator in use.

This is the first place in the project where magnification is USED rather than
reported. `--reg-mode uniform` runs the same solve without it, so the difference
is measurable rather than asserted.

WHY MATRIX-FREE
---------------
Forming M explicitly means one PSF convolution per source pixel -- 4,096 of them,
each producing a dense-ish column. Instead we never build M: conjugate gradients
only needs the ACTION of M and of M^T, and both are a sparse matvec plus one FFT
convolution. Memory is a few MB and a solve takes well under a second.

The adjoint M^T must be exact or CG converges to the wrong answer silently.
tests/test_pixel_source.py checks it with a dot-product test,
<M s, r> == <s, M^T r>, to machine precision. Do not modify the operators
without re-running it.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.signal import fftconvolve
from scipy.sparse.linalg import LinearOperator, cg

from lens_models import ray_shoot
from raytrace import image_plane_grid

__all__ = ["source_grid", "build_lensing_operator", "PixelInversion",
           "lcurve_lambda"]

EPS = 1e-12


# ---------------------------------------------------------------------------
# grids and the lensing operator
# ---------------------------------------------------------------------------

def source_grid(n_src: int, half_extent: float):
    """Source-plane coordinates (arcsec). Square, centred on the origin.

    half_extent is measured to PIXEL CENTRES, so the grid spans
    [-half_extent, +half_extent] inclusive and the pixel scale is
    2*half_extent/(n_src-1).

    Model_A sources have |beta| <= 0.58 arcsec and R_sersic <= 1.69 arcsec, so
    half_extent = 1.2 arcsec covers the population with room to spare while
    keeping the source pixel scale finer than the detector: at n_src = 64 that
    is 0.038 arcsec/px against the detector's 0.106, i.e. a 2.8x finer grid.
    That factor IS the super-resolution being attempted.
    """
    a = np.linspace(-half_extent, half_extent, n_src)
    sy, sx = np.meshgrid(a, a, indexing="ij")
    return sx, sy


def build_lensing_operator(lens_params: Dict, n_pix: int, pixel_scale: float,
                           supersample: int, n_src: int, half_extent: float,
                           grid=None) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Sparse L mapping source pixels -> image-plane SUBpixels, bilinear.

    Returns (L, coverage) where coverage[j] is the number of image subpixels
    that read from source pixel j -- the magnification of that source pixel.

    Rays that land outside the source grid contribute nothing (their rows are
    empty). That is correct: the source is defined to be zero out there.
    """
    if grid is None:
        X, Y = image_plane_grid(n_pix, pixel_scale, supersample)
    else:
        X, Y = grid
    bx, by = ray_shoot(X, Y, lens_params)

    scale = 2.0 * half_extent / (n_src - 1)
    u = (bx.ravel() + half_extent) / scale        # continuous column index
    v = (by.ravel() + half_extent) / scale        # continuous row index

    i0 = np.floor(v).astype(np.int64)
    j0 = np.floor(u).astype(np.int64)
    fy = v - i0
    fx = u - j0

    n_rows = u.size
    rows, cols, vals = [], [], []
    for di in (0, 1):
        for dj in (0, 1):
            ii = i0 + di
            jj = j0 + dj
            w = ((1 - fy) if di == 0 else fy) * ((1 - fx) if dj == 0 else fx)
            ok = (ii >= 0) & (ii < n_src) & (jj >= 0) & (jj < n_src) & (w > 0)
            rows.append(np.nonzero(ok)[0])
            cols.append(ii[ok] * n_src + jj[ok])
            vals.append(w[ok])

    L = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n_rows, n_src * n_src))
    coverage = np.asarray(L.sum(axis=0)).ravel()
    return L, coverage


# ---------------------------------------------------------------------------
# the inversion
# ---------------------------------------------------------------------------

class PixelInversion:
    """Solve (M^T C^-1 M + lambda H) s = M^T C^-1 d, matrix-free, by CG."""

    def __init__(self, L: sp.csr_matrix, coverage: np.ndarray, psf: np.ndarray,
                 n_pix: int, supersample: int, n_src: int, sigma: float,
                 mask: Optional[np.ndarray] = None, reg_mode: str = "mu",
                 reg_power: float = 0.5, reg_clip: float = 5.0):
        self.L = L
        self.psf = psf
        self.n_pix = n_pix
        self.S = supersample
        self.n_src = n_src
        self.sigma = float(sigma)
        self.mask = mask                                   # (n_pix, n_pix) bool
        self.coverage = coverage
        self.reg_mode = reg_mode
        self.reg_power = float(reg_power)
        self.reg_clip = float(reg_clip)
        self.w = self._reg_weights(coverage, reg_mode, reg_power, reg_clip)

    # -- forward and adjoint ------------------------------------------------
    def M(self, s: np.ndarray) -> np.ndarray:
        """source pixels -> detector image (flat, masked)."""
        sub = (self.L @ s).reshape(self.n_pix * self.S, self.n_pix * self.S)
        if self.psf is not None:
            sub = fftconvolve(sub, self.psf, mode="same")
        img = sub.reshape(self.n_pix, self.S, self.n_pix, self.S).mean(axis=(1, 3))
        return img[self.mask] if self.mask is not None else img.ravel()

    def MT(self, r: np.ndarray) -> np.ndarray:
        """detector residual -> source pixels. The exact adjoint of M.

        Each step is reversed in order and replaced by its transpose:
          mask   -> scatter back into a full image
          mean   -> replicate / S^2   (adjoint of averaging is spreading)
          conv   -> correlate, i.e. convolve with the 180-degree-rotated kernel
          L      -> L^T
        """
        if self.mask is not None:
            img = np.zeros((self.n_pix, self.n_pix))
            img[self.mask] = r
        else:
            img = r.reshape(self.n_pix, self.n_pix)
        sub = np.repeat(np.repeat(img, self.S, axis=0), self.S, axis=1) / (self.S ** 2)
        if self.psf is not None:
            sub = fftconvolve(sub, self.psf[::-1, ::-1], mode="same")
        return self.L.T @ sub.ravel()

    # -- regularisation -----------------------------------------------------
    @staticmethod
    def _reg_weights(coverage: np.ndarray, mode: str, power: float = 0.5,
                     clip: float = 5.0) -> np.ndarray:
        """Per-source-pixel penalty weight.

        mode='uniform' : w = 1 everywhere. The control.
        mode='mu'      : w = (median_coverage / coverage)^power, clipped to
                         [1/clip, clip].

        `coverage` is the column sum of L -- how many image subpixels read from
        each source pixel -- which IS the magnification of that source pixel, up
        to the constant subpixel area. Where many rays sample a pixel the data
        constrain it, so it is penalised lightly.

        WHY power = 0.5 AND NOT 1.0. The raw coverage spans ~1000x across the
        source plane (median 1.4 rays per pixel, max ~24, and many pixels get
        essentially none). Taking w = 1/coverage directly makes the smoothing
        vary by the same 1000x, which leaves the high-magnification strip next
        to the caustic effectively unregularised and it fills with noise --
        measured: corr 0.74 against 0.96 for uniform. The square root keeps the
        physical ordering (less smoothing where the lens delivered more
        resolution) while keeping the dynamic range sane. `clip` bounds it
        further. Both are exposed so the choice can be swept rather than
        asserted.
        """
        if mode == "uniform":
            return np.ones_like(coverage)
        c = np.maximum(coverage, 0.0)
        med = np.median(c[c > 0]) if np.any(c > 0) else 1.0
        w = np.power(med / np.maximum(c, 1e-3 * med), power)
        return np.clip(w, 1.0 / clip, clip)

    def _laplacian(self, s2: np.ndarray) -> np.ndarray:
        """5-point discrete Laplacian with replicate edges."""
        p = np.pad(s2, 1, mode="edge")
        return (-4.0 * s2 + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])

    def H(self, s: np.ndarray) -> np.ndarray:
        """H s = D^T W D s -- curvature regularisation, weighted per pixel.

        Curvature rather than total variation, for the reason the earlier TV
        sweep demonstrated: TV's unconstrained minimiser is a flat field, so
        raising its weight drives the source toward a wash (measured: size_ratio
        stayed at 10.6-12.8 over a 100x sweep). Curvature is a quadratic form, so
        the whole problem stays convex and has one solution.
        """
        n = self.n_src
        d = self._laplacian(s.reshape(n, n))
        return self._laplacian((self.w.reshape(n, n) * d)).ravel()

    # -- solve --------------------------------------------------------------
    def solve(self, d_obs: np.ndarray, lam: float, tol: float = 1e-8,
              maxiter: int = 400, s0: Optional[np.ndarray] = None):
        """Return (s, info) for one value of lambda."""
        n = self.n_src * self.n_src
        inv_var = 1.0 / max(self.sigma ** 2, EPS)
        d = d_obs[self.mask] if self.mask is not None else d_obs.ravel()

        def A(s):
            return self.MT(self.M(s)) * inv_var + lam * self.H(s)

        rhs = self.MT(d) * inv_var
        op = LinearOperator((n, n), matvec=A, dtype=np.float64)
        try:
            s, info = cg(op, rhs, rtol=tol, maxiter=maxiter,
                         x0=s0 if s0 is not None else np.zeros(n))
        except TypeError:                       # scipy < 1.12 uses `tol`
            s, info = cg(op, rhs, tol=tol, maxiter=maxiter,
                         x0=s0 if s0 is not None else np.zeros(n))
        return s, info

    # -- diagnostics --------------------------------------------------------
    def chi2(self, s: np.ndarray, d_obs: np.ndarray) -> float:
        d = d_obs[self.mask] if self.mask is not None else d_obs.ravel()
        r = (self.M(s) - d) / max(self.sigma, EPS)
        return float((r ** 2).sum())

    def reg_norm(self, s: np.ndarray) -> float:
        n = self.n_src
        d = self._laplacian(s.reshape(n, n))
        return float((self.w.reshape(n, n) * d * d).sum())

    def n_data(self) -> int:
        return int(self.mask.sum()) if self.mask is not None else self.n_pix ** 2


# ---------------------------------------------------------------------------

def lcurve_lambda(inv: "PixelInversion", d_obs: np.ndarray,
                  lambdas=np.logspace(-4, 3, 15), verbose: bool = False):
    """Choose lambda by the corner of the L-curve (Hansen 1992).

    Plotting log||residual|| against log||regularisation|| as lambda varies
    traces an L. Too little smoothing sits on the steep arm (noise is being
    fitted); too much sits on the flat arm (signal is being smoothed away). The
    corner -- the point of maximum curvature -- is the compromise, and it needs
    no knowledge of the true noise level.

    The discrepancy principle (choose lambda so chi^2/dof = 1) would be the more
    usual choice, but it is unusable here: the PSF-shape systematic alone puts
    chi^2/dof in the thousands (see evaluate.py section 3), so no lambda reaches
    1 and the criterion has no root.

    Returns (lambda_best, table) where table has one row per lambda:
    (lambda, chi2, reg_norm, curvature).
    """
    rows, s_prev = [], None
    for lam in lambdas:
        s, _ = inv.solve(d_obs, lam, s0=s_prev)
        s_prev = s
        rows.append((lam, inv.chi2(s, d_obs), inv.reg_norm(s)))
        if verbose:
            print(f"    lambda {lam:9.4g}   chi2 {rows[-1][1]:12.4g}   "
                  f"reg {rows[-1][2]:12.4g}")
    T = np.array(rows)
    x = np.log10(np.maximum(T[:, 1], EPS))          # log residual
    y = np.log10(np.maximum(T[:, 2], EPS))          # log regularisation
    # discrete curvature of the (x, y) path
    curv = np.zeros(len(x))
    if len(x) >= 3:
        dx, dy = np.gradient(x), np.gradient(y)
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        denom = np.power(dx ** 2 + dy ** 2, 1.5) + EPS
        curv = np.abs(dx * ddy - dy * ddx) / denom
        curv[0] = curv[-1] = 0.0                    # endpoints are not corners
    k = int(np.argmax(curv))
    return float(T[k, 0]), np.column_stack([T, curv])
