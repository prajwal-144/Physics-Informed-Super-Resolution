"""
noise_model.py -- the per-pixel variance of a Model_A image. ONE definition.

WHY THIS FILE EXISTS
--------------------
Every chi^2 in this project used

    sigma^2 = sigma_bg^2                                    (background only)

where sigma_bg is the standard deviation of the r > 50 px annulus
(data_a.ModelADataset.sigma). Model_A was generated with

    sigma^2 = sigma_bg^2 + flux / t                         (background + shot)

so the variance on a bright arc pixel was understated by three to five orders of
magnitude, and every pixel-weighted quantity -- the Levenberg-Marquardt fit, the
training gradient, the reported chi^2/dof, the measured chi^2 floor -- inherited
that error. This module holds the corrected variance so it cannot drift between
the nine files that need it, for the same reason backend.py holds the array shim.

THE EVIDENCE, TWO INDEPENDENT ROUTES
------------------------------------
Both are reproduced by verify_noise_model.py, which prints the numbers quoted
here. Neither uses a lens model, a source model or a fitted physical parameter,
so neither can be confounded by model error.

Route 2 is decisive and is the coefficient of record -- the stored truth
scalar. Every npz carries `snr_max`. If the generator used
var = sigma_bg^2 + g*flux then max(flux / sigma) over the image must reproduce
it. Scanning g:

    g       median predicted / stored snr_max      p10     p90
    0.00              241.901                   177.133  316.632
    0.70                1.077                     0.826    1.126
    0.80                1.008                     0.772    1.054
    1.00                0.901                     0.691    0.942

At g = 0 the prediction overshoots by a factor of ~240. At g = 0.80 it lands on
1.008. This is the GENERATOR's own definition of signal-to-noise read back out
of its own output, so it settles the functional form and the coefficient
together. POISSON_GAIN is set from it.

Route 1 corroborates it independently -- the two bands.
`unlensed[1] = c * unlensed[0]` to about 1 per cent rms (correlation 0.9999 to
1.0000 on every system tested), and source_R_sersic, source_n_sersic,
source_e1, source_e2 are single scalars rather than per-band arrays, so the two
bands are the SAME source at a different amplitude. Hence

    r = image[1] - c * image[0]

contains only noise, and binning var(r) against a low-noise flux proxy measures
the variance law with no model at all:

    var(r) = A + (c + c^2) * g * f

Scoring that same pure-noise residual on the fit disc:

    var = sigma_bg^2         median chi2/dof = 3297   (p10 2290, p90 7941)
    var = A + b*f  (fitted)  median chi2/dof =  1.36  (p10 1.04)
                             83 per cent of systems inside [0.5, 2.0]

Route 1's g comes out ~1.10 against route 2's 0.80. That is a known bias, not a
disagreement: the flux proxy is a smoothed estimate, which understates the peak
on a sharp arc, so the fitted slope rises to match the variance observed there.
Route 1 establishes the FORM and the order of magnitude; route 2 fixes the
number. Both reject var = sigma_bg^2 by three orders of magnitude.

A SECOND FINDING -- THE DISC FLOOR IS NOT THE ANNULUS FLOOR
Route 1's fitted floor A comes out ~11x the annulus variance. data_a.sigma() is
correct about the annulus (var(r) measured there matches it to a couple of per
cent), but the floor inside r < 45 px is higher, because the PSF's Moffat-like
wings spread arc flux across the whole disc and that spread flux carries its own
shot noise. Inside the pipeline this needs no special handling: `pred` there
includes the fitted `background` parameter, so gain*max(pred,0) already picks up
the sky's shot noise. It matters only to a model-free measurement like
verify_noise_model.py, which has no `pred` to lean on.

THE NOISE IS WHITE
------------------
Lag-1 autocorrelation of r in the background annulus is +0.002 and lag-2 is
-0.002, so the noise was added AFTER the PSF convolution. The covariance is
therefore diagonal and a per-pixel variance is the correct and complete
description -- no covariance matrix, no correlated-noise machinery.

WHAT REMAINS FOR THE FRACTIONAL FLOOR
-------------------------------------
`--sigma-floor f` adds (f * model)^2, i.e. variance proportional to flux SQUARED.
That is the right shape for a multiplicative model error (a PSF-shape error is a
fixed fraction of the model) and the wrong shape for photon noise (variance
proportional to flux, not its square). It was introduced to fix a symptom that
the Poisson term is the actual cause of, so the two now mean different things and
must not both be tuned to absorb the same effect:

    poisson_gain   photon noise -- a property of the DATA.    MEASURED, not free.
    sigma_floor    model error  -- a property of the MODEL.   Free, and now
                                   expected to be much smaller than 0.02.

Default sigma_floor is therefore 0.0 in the training scripts. Anything it still
buys on top of a correct photon-noise model is a measurement of residual model
misspecification, which is a result worth having rather than a knob.

REPRODUCING THE R/PS SUBMISSION NUMBERS
---------------------------------------
Every script takes `--poisson-gain`. Passing

    --poisson-gain 0 --sigma-floor 0.02

restores the exact arithmetic of the submitted run. `gain = 0` short-circuits to
the scalar path, so it is bit-identical and not merely close.

RE-MEASURING THE GAIN
---------------------
    python superres/verify_noise_model.py --root . --n 40

prints both routes and a recommended POISSON_GAIN. If the Model_A generation
configuration is ever recovered, g is exactly 1 / exposure_time in stored units
and should replace the measured value here.
"""
from __future__ import annotations

__all__ = ["POISSON_GAIN", "variance", "sigma_eff", "describe"]

# Per-band variance per unit flux, in Model_A's native stored units.
# Route 1 (band difference): 0.88, p10 0.75, p90 1.01.
# Route 2 (stored snr_max):  0.75 to 0.80.
# 0.80 sits inside both intervals. Treat 0.75-0.90 as the uncertainty.
POISSON_GAIN = 0.80


def variance(model, sigma_bg, gain: float = POISSON_GAIN, frac: float = 0.0):
    """Per-pixel variance of the residual (data - model).

        var = sigma_bg^2  +  gain * max(model, 0)  +  (frac * max(model, 0))^2
              background      photon noise            model error

    `model` may be a numpy array, a torch tensor or a scalar; `sigma_bg` may be a
    scalar or an array broadcastable against it. Works under both backends
    without importing either: only clipping and arithmetic are used, and each is
    reached through duck typing exactly as backend.py does.

    IMPORTANT -- pass the MODEL, not the data. Weighting by the observed counts
    biases the fit low on faint pixels (the standard Poisson-weighting bias);
    weighting by the model does not. Callers inside a gradient graph must pass a
    detached model, so that the weights are held constant across the step rather
    than being something the optimiser can lower by shrinking the model.

    gain = 0.0 and frac = 0.0 returns sigma_bg^2 unchanged, which is the
    pre-correction behaviour, bit for bit.
    """
    var = sigma_bg * sigma_bg
    if gain <= 0.0 and frac <= 0.0:
        return var
    m = _clip0(model)
    if gain > 0.0:
        var = var + gain * m
    if frac > 0.0:
        var = var + (frac * m) ** 2
    return var


def sigma_eff(model, sigma_bg, gain: float = POISSON_GAIN, frac: float = 0.0):
    """sqrt(variance(...)), floored away from zero. Backend-agnostic."""
    var = variance(model, sigma_bg, gain=gain, frac=frac)
    try:                                   # torch
        return var.clamp_min(1e-24).sqrt()
    except AttributeError:
        pass
    try:                                   # numpy
        import numpy as np
        return np.sqrt(np.clip(var, 1e-24, None))
    except Exception:
        return max(float(var), 1e-24) ** 0.5


def _clip0(x):
    """max(x, 0) for torch tensors, numpy arrays and plain scalars."""
    try:
        return x.clamp_min(0.0)            # torch
    except AttributeError:
        pass
    try:
        return x.clip(min=0.0)             # numpy >= 1.17
    except (AttributeError, TypeError):
        pass
    try:
        import numpy as np
        return np.clip(x, 0.0, None)
    except Exception:
        return x if x > 0.0 else 0.0 * x


def describe(gain: float = POISSON_GAIN, frac: float = 0.0) -> str:
    """One line for a script banner, so every run records its noise model."""
    if gain <= 0.0 and frac <= 0.0:
        return ("noise model: sigma_bg^2 only  [PRE-CORRECTION, reproduces the "
                "R/PS submission]")
    parts = ["sigma_bg^2"]
    if gain > 0.0:
        parts.append(f"{gain:g}*model  (photon, measured)")
    if frac > 0.0:
        parts.append(f"({frac:g}*model)^2  (model error)")
    return "noise model: var = " + " + ".join(parts)
