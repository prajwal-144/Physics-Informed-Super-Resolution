"""
raytrace.py -- the observation operator.

    source (sky) --lens--> lensed sky --PSF--> blurred sky --pixels--> data

The order is physical and not negotiable: lensing remaps the sky, the PSF is an
optical convolution that happens on the sky, and pixelation is an integration
performed by the detector. Blurring after binning would be a different (and
wrong) instrument. `train_sis_bank.py` already had this right
(`F.interpolate(apply_psf(intrinsic, psf), ..., mode="area")`, line 388) and it
is preserved.

WHAT CHANGES RELATIVE TO THE OPERATOR BANK
------------------------------------------
The bank applied lensing as `torch.sparse.mm(M, source_flat)` with M built
offline for a single Einstein radius. Here it is

    beta = theta - alpha(theta ; theta_E, gamma, e1, e2, g1, g2)

evaluated inside the graph, so the lens parameters carry gradients and can be
fitted from the image rather than read from a manifest.

FLUX CONSERVATION
-----------------
The sparse operators conserved flux exactly because they accumulated area
fractions. Bilinear sampling does not, and neither does point-sampling a
continuous profile. The mitigation is supersampling: evaluate the image plane at
S x S subpixels, then area-average. The residual error falls as O(1/S^2) and is
MEASURED in tests/test_raytrace.py against an S = 9 reference -- do not change
SUPERSAMPLE without re-running that test.

S = 3 is the default: it puts the sampling error well below the photon noise of
Model_A (median snr_max ~ 10) at a 127 -> 381 grid.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from backend import get_backend
from lens_models import ray_shoot

__all__ = ["image_plane_grid", "gaussian_psf", "moffat_psf", "load_psf",
           "convolve", "area_downsample", "render", "SUPERSAMPLE"]

SUPERSAMPLE = 3


# ---------------------------------------------------------------------------

def image_plane_grid(n: int, pixel_scale: float, supersample: int = 1):
    """Angular coordinates (arcsec) of every (sub)pixel centre, as numpy arrays.

    Detector pixel i spans [i - 0.5, i + 0.5] in pixel units about its centre, so
    its S subpixel centres sit at i + (k + 0.5)/S - 0.5 for k = 0..S-1. The grid
    origin is (n - 1)/2 -- the convention used throughout this repository
    (data_manifest.ManifestDataset._crop, evaluate_sis.source_metrics,
    build_sis_mappings.build_scatter_matrix).

    x varies along axis 1 and y along axis 0, matching the .npz image arrays.
    """
    S = int(supersample)
    c = (n - 1) / 2.0
    idx = np.arange(n, dtype=np.float64)
    if S > 1:
        off = (np.arange(S, dtype=np.float64) + 0.5) / S - 0.5
        pix = (idx[:, None] + off[None, :]).reshape(-1)
    else:
        pix = idx
    ang = (pix - c) * pixel_scale
    Y, X = np.meshgrid(ang, ang, indexing="ij")
    return X, Y


def gaussian_psf(fwhm_arcsec: float, pixel_scale: float, truncate: float = 4.0):
    """Normalised Gaussian kernel on a grid of the given pixel scale.

    fwhm = 2 sqrt(2 ln 2) sigma. Pass the SUPERSAMPLED scale -- the convolution
    happens before binning.

    Model_A's effective PSF was measured, not assumed: fitting the analytic
    Sersic (manifest parameters) against the npz `unlensed` array over a sweep of
    trial widths gives a clean single minimum at FWHM 0.18-0.20 arcsec (residual
    nmse 0.0029) against 0.0071 at the 0.10 arcsec the old runs used -- 2.4x
    worse. See calibrate_psf.py and PROJECT_REPORT.md section 7.4. Caveat
    recorded there: this is the best-fit GAUSSIAN, and the residual floor of
    0.003 rather than ~0 says the true kernel is not exactly Gaussian, so 0.18
    is an equivalent Gaussian width.
    """
    if fwhm_arcsec is None or fwhm_arcsec <= 0:
        return None
    sigma = (fwhm_arcsec / 2.3548200450309493) / pixel_scale
    rad = max(1, int(math.ceil(truncate * sigma)))
    t = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    k /= k.sum()
    return np.outer(k, k)


def moffat_psf(fwhm_arcsec: float, pixel_scale: float, beta: float = 2.5,
               truncate: float = 6.0):
    """Moffat kernel: I(r) = (1 + (r/alpha)^2)^(-beta), alpha = fwhm/(2 sqrt(2^(1/beta)-1)).

    calibrate_psf.py extracts Model_A's PSF empirically (Wiener deconvolution of
    `unlensed` by the analytic Sersic, stacked over 40 images) and finds wings far
    broader than a Gaussian: at r = 0.25 arcsec the empirical profile is 0.139 of
    the peak against 0.005 for a 0.18 arcsec Gaussian, a factor of 29, rising
    further outward. That is the signature of a Moffat, and it is why a Gaussian
    forward model leaves chi^2/dof ~ 3000 on arcs whose peak/sigma reaches 10^4
    even when every parameter is recovered correctly.

    beta = 2.5 is a starting value; beta -> infinity recovers a Gaussian. Fit it,
    or use `load_psf` with the empirical kernel.
    """
    if fwhm_arcsec is None or fwhm_arcsec <= 0:
        return None
    alpha = (fwhm_arcsec / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))) / pixel_scale
    rad = max(1, int(math.ceil(truncate * alpha)))
    t = np.arange(-rad, rad + 1, dtype=np.float64)
    xx, yy = np.meshgrid(t, t, indexing="xy")
    k = (1.0 + (xx * xx + yy * yy) / (alpha * alpha)) ** (-beta)
    return k / k.sum()


def load_psf(path: str, crop: int = 25):
    """Load the empirical kernel written by calibrate_psf.py and centre-crop it.

    Cropping matters: the extracted kernel is defined on the full 127 px frame,
    and convolving with a 127 px kernel every model evaluation is ~25x slower
    than with a 25 px one while adding nothing -- the profile is below 1e-3 of
    the peak beyond r ~ 0.5 arcsec (5 px).
    """
    k = np.load(path)
    n = k.shape[-1]
    c = n // 2
    h = crop // 2
    k = k[c - h:c + h + 1, c - h:c + h + 1]
    return k / k.sum()


def convolve(img, kernel):
    """2-D convolution with edge replication.

    'replicate' rather than zero padding: the arcs sit well inside the frame, but
    a zero pad darkens the border and would bias the background level that the
    chi^2 partly measures.
    """
    if kernel is None:
        return img
    xp = get_backend(img)
    if xp.name == "torch":                              # pragma: no cover
        import torch
        import torch.nn.functional as F
        k = kernel if torch.is_tensor(kernel) else torch.as_tensor(kernel)
        k = k.to(img.dtype).to(img.device)[None, None]
        pad = k.shape[-1] // 2
        return F.conv2d(F.pad(img, (pad, pad, pad, pad), mode="replicate"), k)
    from scipy.signal import fftconvolve
    pad = kernel.shape[-1] // 2
    padded = np.pad(np.asarray(img), pad, mode="edge")
    return fftconvolve(padded, kernel, mode="valid")


def area_downsample(img, factor: int):
    """Exact area average of S x S blocks -- the detector's pixel integration.

    The explicit reshape (rather than F.interpolate(mode='area')) makes the
    assumption visible: equal-area subpixels on aligned blocks. That is what
    tests/test_raytrace.py checks flux against.
    """
    if factor == 1:
        return img
    S = int(factor)
    xp = get_backend(img)
    if xp.name == "torch":                              # pragma: no cover
        b, c, h, w = img.shape
        assert h % S == 0 and w % S == 0
        return img.reshape(b, c, h // S, S, w // S, S).mean(dim=(3, 5))
    a = np.asarray(img)
    h, w = a.shape[-2:]
    assert h % S == 0 and w % S == 0, f"{h}x{w} not divisible by {S}"
    return a.reshape(*a.shape[:-2], h // S, S, w // S, S).mean(axis=(-3, -1))


# ---------------------------------------------------------------------------

def render(source, lens_params: Dict, n_pix: int, pixel_scale: float,
           psf=None, supersample: int = SUPERSAMPLE, grid=None,
           background: float = 0.0, return_sky: bool = False):
    """Full forward model -> predicted detector image.

    `source` must expose `.at(bx, by)`. `grid` may be a precomputed (X, Y) pair
    from image_plane_grid() to avoid rebuilding it every call; it must match
    n_pix, pixel_scale and supersample.

    `background` is a free additive constant. It exists because the images are
    used in NATIVE units (see data_a.py) rather than min-max normalised, so any
    residual sky level has to live somewhere fittable rather than being absorbed
    silently into the normalisation.
    """
    if grid is None:
        X, Y = image_plane_grid(n_pix, pixel_scale, supersample)
    else:
        X, Y = grid
    bx, by = ray_shoot(X, Y, lens_params)
    sky = source.at(bx, by)
    blurred = convolve(sky, psf)
    pred = area_downsample(blurred, supersample) + background
    if return_sky:
        return pred, sky
    return pred
