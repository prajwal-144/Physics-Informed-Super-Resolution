"""
metrics.py -- scoring that survives contact with low-SNR data.

WHAT WAS WRONG WITH `skill`
---------------------------
`evaluate_sis.py` reports  skill = 1 - MSE(pred, obs) / mean(obs^2)  against the
min-max-normalised NOISY observation. Measured on Model_A
(diagnostics_2026_08_12/PROJECT_REPORT.md section 7.3):

    Spearman(skill, snr_max)                 = -0.85
    flat constant image                      ->  0.348
    2-px Gaussian blur OF THE INPUT ITSELF   ->  0.951
    3-px blur of the input                   ->  0.931
    the trained models                       ->  0.85 - 0.91
    the EXACT physical model, noiseless      ->  0.630

A physical model cannot beat the truth. Anything above 0.63 there is fitting the
noise realisation, and the metric rewards smoothing most exactly where there is
least signal. Two causes: the target is noisy, and the denominator is
mean(obs^2) rather than a variance, so the pedestal that min-max normalisation
induces inflates it for free.

WHAT REPLACES IT
----------------
    chi2_per_dof   the standard goodness of fit. sigma is measured from the
                   background annulus of each image (data_a.ModelADataset.sigma);
                   dof = n_fitted_pixels - n_free_parameters. A correct model on
                   correctly estimated noise gives 1.0. Unlike `skill` this has
                   an absolute meaning, so "did it fit" stops being relative to
                   an arbitrary baseline.

    source_truth   the model source, CONVOLVED WITH THE PSF, against the npz
                   `unlensed` array. evaluate_sis.py compared the intrinsic
                   pre-PSF source with the post-PSF truth -- an apples-to-oranges
                   difference of one PSF width before any model error is counted.
                   Small next to a factor of 11, but it has to go before
                   size_ratio can serve as the calibration criterion its own
                   docstring proposes.

    parameter_recovery   recovered minus true, per parameter. This is the metric
                   the old pipeline could not have: Model_4 had no truth, so only
                   plausibility ("does this look like a galaxy?") could be
                   scored. Model_A shows the difference is not academic --
                   at tv = 0.3 the plausibility metrics read fill factor 0.66 and
                   main-flux-fraction 0.998, both healthy, while the source was
                   11x too large with zero positional information.

    stratify       every summary split by snr_max. A single median hides a
                   correlation of -0.85.

Everything here is numpy: scoring happens outside the gradient path.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

__all__ = ["chi2_per_dof", "source_truth", "centroid_size", "parameter_errors",
           "stratify", "null_baselines"]

EPS = 1e-12


# ---------------------------------------------------------------------------
# goodness of fit
# ---------------------------------------------------------------------------

def chi2_per_dof(pred: np.ndarray, data: np.ndarray, sigma: float,
                 mask: Optional[np.ndarray] = None, n_params: int = 0) -> float:
    """sum(((pred - data)/sigma)^2) / (n_pixels - n_params).

    sigma is a scalar per image (background rms). Model_A ships no noise map;
    the arcs sit at r < 30 px on a 127 px grid so the r > 50 px annulus is
    source-free, and `image` vs `image_nss` differ there by sqrt(2) times that
    value, confirming both carry independent detector noise and the estimate is
    not picking up residual signal.
    """
    r = (pred - data) / max(sigma, EPS)
    if mask is not None:
        r = r[mask]
    n = r.size
    return float((r ** 2).sum() / max(n - n_params, 1))


# ---------------------------------------------------------------------------
# source-plane truth
# ---------------------------------------------------------------------------

def centroid_size(a: np.ndarray, thresh_frac: float = 0.05):
    """(centroid_y, centroid_x, rms size) about the source's OWN centroid.

    About its own centroid, not the grid centre: a correct reconstruction of a
    source at offset beta sits off-centre BY CONSTRUCTION, and beta is a
    recovered physical quantity rather than an error. Measuring the second
    moment about the grid centre penalises a correct answer -- the reason
    `evaluate_sis.source_metrics` moved away from `compactness`.
    """
    a = np.clip(np.squeeze(np.asarray(a, dtype=np.float64)), 0, None)
    peak = a.max()
    if peak <= EPS:
        return np.nan, np.nan, np.nan
    f = np.where(a >= thresh_frac * peak, a, 0.0)
    tot = f.sum()
    if tot <= EPS:
        return np.nan, np.nan, np.nan
    yy, xx = np.indices(a.shape)
    cy = float((f * yy).sum() / tot)
    cx = float((f * xx).sum() / tot)
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return cy, cx, float(np.sqrt((f * r2).sum() / tot))


def source_truth(model_source_psf: np.ndarray, truth_unlensed: np.ndarray,
                 pixel_scale: float = 1.0) -> Dict[str, float]:
    """Score a PSF-CONVOLVED model source against the npz `unlensed` array.

    `model_source_psf` must already have the PSF applied and be on the same grid
    as the truth. Passing the intrinsic source here reproduces the evaluate_sis.py
    bug.

    size_ratio is the headline: it was 10.6-12.8 for the old pipeline across a
    100x sweep of the TV weight, against a target of 1.
    """
    p = np.clip(np.squeeze(np.asarray(model_source_psf, float)), 0, None)
    t = np.clip(np.squeeze(np.asarray(truth_unlensed, float)), 0, None)
    out: Dict[str, float] = {}

    pf, tf = p.ravel(), t.ravel()
    a = pf - pf.mean()
    b = tf - tf.mean()
    out["corr"] = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + EPS))

    s = (pf * tf).sum() / max((pf * pf).sum(), EPS)          # optimal amplitude
    out["nmse"] = float(((s * pf - tf) ** 2).sum() / max((tf * tf).sum(), EPS))

    cy_p, cx_p, sz_p = centroid_size(p)
    cy_t, cx_t, sz_t = centroid_size(t)
    out["pred_size_px"] = sz_p
    out["true_size_px"] = sz_t
    out["size_ratio"] = float(sz_p / sz_t) if sz_t and np.isfinite(sz_t) else np.nan
    out["centroid_err_px"] = float(np.hypot(cy_p - cy_t, cx_p - cx_t))
    out["centroid_err_arcsec"] = out["centroid_err_px"] * pixel_scale

    ps = p / max(p.sum(), EPS)
    ts = t / max(t.sum(), EPS)
    out["peak_ratio"] = float(ps.max() / max(ts.max(), EPS))
    return out


# ---------------------------------------------------------------------------
# parameter recovery
# ---------------------------------------------------------------------------

_CIRCULAR = {"e": "vector", "g": "vector"}


def parameter_errors(fit: Dict[str, float], truth: Dict[str, float],
                     keys: Iterable[str]) -> Dict[str, float]:
    """Signed recovered-minus-true for each key, plus vector errors for (e1,e2)
    and (g1,g2), whose two components are not independently meaningful -- a
    spin-2 quantity has a modulus and an orientation, and reporting e1 alone
    would call a pure rotation a failure."""
    out = {}
    for k in keys:
        if k in fit and k in truth and np.isfinite(truth[k]):
            out[f"d_{k}"] = float(fit[k] - truth[k])
    for a, b, name in (("e1", "e2", "e"), ("g1", "g2", "g")):
        if a in fit and a in truth:
            out[f"{name}_fit"] = float(np.hypot(fit[a], fit[b]))
            out[f"{name}_true"] = float(np.hypot(truth[a], truth[b]))
            out[f"d_{name}_vec"] = float(np.hypot(fit[a] - truth[a], fit[b] - truth[b]))
    return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def stratify(rows, key: str, by: str = "snr_max",
             bins=((0, 6), (6, 15), (15, 1e9))) -> str:
    """Median of `key` in bins of `by`. A single median hid Spearman -0.85."""
    lines = [f"  {key} stratified by {by}"]
    v = np.array([r.get(key, np.nan) for r in rows], float)
    s = np.array([r.get(by, np.nan) for r in rows], float)
    for lo, hi in bins:
        m = np.isfinite(v) & np.isfinite(s) & (s >= lo) & (s < hi)
        lbl = f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}"
        if m.sum() == 0:
            lines.append(f"    {lbl:>10}  n=  0")
            continue
        lines.append(f"    {lbl:>10}  n={m.sum():3d}   median {np.median(v[m]):9.4f}"
                     f"   p25 {np.percentile(v[m],25):9.4f}"
                     f"   p75 {np.percentile(v[m],75):9.4f}")
    ok = np.isfinite(v)
    lines.append(f"    {'ALL':>10}  n={ok.sum():3d}   median {np.median(v[ok]):9.4f}")
    return "\n".join(lines)


def null_baselines(images, sigmas) -> Dict[str, float]:
    """chi2/dof of models that contain no physics, as a sanity floor.

    Reported alongside the fit so a reviewer can see what "1.0" is worth on this
    data. The equivalent exercise on `skill` is what exposed that metric.
    """
    from scipy.ndimage import gaussian_filter
    out = {"constant": [], "blur2": [], "blur3": []}
    for img, sig in zip(images, sigmas):
        out["constant"].append(chi2_per_dof(np.full_like(img, img.mean()), img, sig))
        out["blur2"].append(chi2_per_dof(gaussian_filter(img, 2.0), img, sig))
        out["blur3"].append(chi2_per_dof(gaussian_filter(img, 3.0), img, sig))
    return {k: float(np.median(v)) for k, v in out.items()}
