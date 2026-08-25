"""
theta_e_init.py -- unsupervised starting values, measured from the image alone.

WHY THIS CLOSES THE LEAK
------------------------
`train_sis_bank.py` line 344 reads the Einstein radius straight from the
manifest:

    t_px = ds.theta_E / args.resolution      # manifest column
    bins = assign_bins(t_px, alphas)

and that integer then (a) selects the backward operator, (b) is broadcast into
the network as a third input channel, and (c) selects the forward operator.
`evaluate_sis.py` lines 586-588 repeat it. So the true Einstein radius shapes the
network's input, is told to it numerically, and picks the operator that grades
its output. That is why Spearman(predicted ring radius, true theta_E) = 0.95 in
every run: the number is imposed, not learned, and the project's unsupervised
claim does not survive it.

In the new pipeline theta_E is a FREE PARAMETER of the forward model, fitted
against the image. This module only supplies a starting point, and it does so
from the image itself.

THE MEASUREMENT
---------------
For each of 72 azimuths, walk outward and record the radius of peak brightness:
r(phi). Fit, weighted by the ring brightness (dark azimuths carry a meaningless
radius),

    r(phi) = r0 + a1 cos phi + b1 sin phi + a2 cos 2phi + b2 sin 2phi

  m = 0   r0                  -> theta_E                    LENS property
  m = 1   (a1, b1)            -> the source offset beta      SOURCE property
                                 AND its direction: for a circular lens the
                                 brighter image lies on the same side as the
                                 source, so r(phi) peaks toward it
  m = 2   hypot(a2, b2)       -> quadrupole. Zero for a circular lens at ANY
                                 source position, so it is a model-free
                                 ellipticity/substructure indicator.

Routing on the brightest annulus instead would conflate lens and source: the
brightest ring sits at r ~ theta_E + beta, and beta has median 2.90 px against a
1 px operator bin, so a typical image would land three bins away. That is the
correction `ring_analysis.py cmd_index` already documents.

ACCURACY, MEASURED AGAINST TRUTH
--------------------------------
Run on 400 Model_A val images and compared with the manifest (evaluation only,
never used in the fit): median |theta_E_est - theta_E_true| = 0.48 px,
p90 = 1.49 px, on a ring of median radius 12.5 px. Model_A rings have median
completeness 0.90, versus 0.25 on Model_4, which is why the estimator is far
more reliable here than the 12.5%-of-images subset it could be checked on
before. tests/test_theta_e_init.py reproduces those numbers.

This is a numpy module on purpose: it produces an initialisation, so it sits
outside the gradient path entirely.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = ["polar_ridge", "fourier_rphi", "completeness", "initial_guess"]

EPS = 1e-12


def polar_ridge(img: np.ndarray, n_ang: int = 72, r_min: float = 1.5,
                r_max: float = None, dr: float = 0.5):
    """Trace the ring. Returns (phi, r_of_phi, intensity_of_phi).

    Intensity is returned as a WEIGHT, not discarded: azimuths where the image is
    dark have a meaningless peak radius and must not be given equal say in the
    fit below.
    """
    a = np.clip(np.squeeze(np.asarray(img, dtype=np.float64)), 0, None)
    h, w = a.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r_max = r_max or (min(h, w) / 2.0 - 1.5)

    phi = np.linspace(-np.pi, np.pi, n_ang, endpoint=False)
    radii = np.arange(r_min, r_max, dr)
    yy = cy + radii[None, :] * np.sin(phi[:, None])
    xx = cx + radii[None, :] * np.cos(phi[:, None])

    y0 = np.clip(np.floor(yy).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(xx).astype(int), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    fy, fx = yy - y0, xx - x0
    vals = (a[y0, x0] * (1 - fy) * (1 - fx) + a[y1, x0] * fy * (1 - fx)
            + a[y0, x1] * (1 - fy) * fx + a[y1, x1] * fy * fx)

    j = np.argmax(vals, axis=1)
    return phi, radii[j], vals[np.arange(n_ang), j]


def fourier_rphi(phi, r, weight):
    """Weighted least squares r(phi) = r0 + m1 + m2.

    Returns (r0, beta_x_dir, beta_y_dir, m2_amplitude) where the m=1 vector
    (a1, b1) points from the lens centre toward the source: for a circular lens
    the same-side image is the further one, so r(phi) is largest in the source
    direction and the fitted (a1, b1) IS beta in pixels, as a vector.
    """
    w = np.clip(weight, 0, None)
    if w.sum() <= EPS:
        return (np.nan,) * 4
    w = w / w.sum()
    A = np.stack([np.ones_like(phi), np.cos(phi), np.sin(phi),
                  np.cos(2 * phi), np.sin(2 * phi)], axis=1)
    W = np.sqrt(w)[:, None]
    try:
        coef, *_ = np.linalg.lstsq(A * W, r * W[:, 0], rcond=None)
    except np.linalg.LinAlgError:
        return (np.nan,) * 4
    return (float(coef[0]), float(coef[1]), float(coef[2]),
            float(np.hypot(coef[3], coef[4])))


def completeness(intensity, frac: float = 0.35) -> float:
    """Fraction of azimuths brighter than `frac` of the peak. 1.0 = full ring."""
    peak = float(np.max(intensity))
    if peak <= EPS:
        return 0.0
    return float((intensity >= frac * peak).mean())


def initial_guess(img: np.ndarray, pixel_scale: float) -> Dict[str, float]:
    """Starting values for the fit, from the image alone. Arcsec.

    Everything not measurable from the ring geometry starts at a
    population-neutral value, NOT at a manifest value. `m2` is reported for
    diagnostics but is deliberately not converted into an e1/e2 guess: the
    ring quadrupole responds to substructure as well as to macro ellipticity
    (Model_A's axion class has kappa_sub reaching +/-0.66 against a macro kappa
    of ~0.6), so turning it into a lens ellipticity would import an
    interpretation the fit is supposed to make for itself.
    """
    phi, r, ival = polar_ridge(img)
    r0, a1, b1, m2 = fourier_rphi(phi, r, ival)
    comp = completeness(ival)
    if not np.isfinite(r0) or r0 <= 0:
        r0, a1, b1, m2 = 12.5, 0.0, 0.0, 0.0     # Model_A median, last resort
    return {
        "theta_E": float(r0 * pixel_scale),
        "sx": float(a1 * pixel_scale),
        "sy": float(b1 * pixel_scale),
        "m2_over_theta_E": float(m2 / max(r0, EPS)),
        "ring_completeness": float(comp),
        "ring_peak": float(np.max(ival)),
    }
