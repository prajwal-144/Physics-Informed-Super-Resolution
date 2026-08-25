"""
lens_models.py -- differentiable analytic deflection fields (SIS / SIE / EPL + shear).

WHY THIS FILE REPLACES THE OPERATOR BANK
----------------------------------------
The old pipeline stored the lens as precomputed sparse matrices, one set per
Einstein radius (`build_sis_mappings.py` -> `bank_manifest/alpha_*`). That
representation blocks three things we now need, for reasons that are structural
rather than incidental:

  1. alpha never enters the autograd graph. In
     `torch.sparse.mm(M, source_flat)` the only tensor carrying a gradient is
     the source, so d(loss)/d(alpha) does not exist.
  2. alpha determines WHICH entries of M are non-zero, not merely their values.
     That is a discrete change to a data structure, not a smooth reweighting.
  3. the construction itself is non-differentiable:
     `build_sis_mappings.build_scatter_matrix` finds the landing pixel with
     `torch.round(Xo + c_out)`, and round has zero derivative almost everywhere.

So a bank indexed by one scalar is the only lens family that representation can
express. diagnostics_2026_08_12/PROJECT_REPORT.md section 7.1 measured what that
costs on Model_A: with a perfect source, a circular lens with theta_E refit
per image reaches correlation 0.888 with the true lensed image, while SIE plus
external shear reaches 0.997. Refitting theta_E buys +0.019; adding ellipticity
buys +0.104.

Here the deflection is a closed-form function evaluated inside the graph, so
every lens parameter is an ordinary leaf tensor.

CONVENTIONS -- verified numerically against lenstronomy 1.14.2
--------------------------------------------------------------
  lens equation     beta = theta - alpha(theta)
  coordinates       x -> array axis 1, y -> array axis 0; arcsec; origin at the
                    grid centre, (N-1)/2 convention (the same one used by
                    data_manifest.ManifestDataset._crop, evaluate_sis.py and
                    build_sis_mappings.py)
  ellipticity       phi = arctan2(e2, e1)/2 ; c = hypot(e1, e2) ;
                    q = (1-c)/(1+c)   [minor/major]
  EPL normalisation b = theta_E * sqrt(q) ,  t = gamma - 1 ,
                    kappa = (2-t)/2 * (b / sqrt(q^2 x'^2 + y'^2))^t
                    in the major-axis frame  (Tessore & Metcalf 2015, eq. 2)
  frame rotation    lenstronomy Util.util.rotate(x, y, phi):
                    x' =  x cos phi + y sin phi
                    y' = -x sin phi + y cos phi

  gamma = 2, e = 0  ->  SIS  (|alpha| = theta_E everywhere, purely radial)
  gamma = 2         ->  SIE
  free gamma        ->  EPL

THE HYPERGEOMETRIC, AND WHY IT IS A SERIES HERE
-----------------------------------------------
Tessore & Metcalf (2015) eq. 22-23 give

    alpha = 2/(1+q) (b/R)^t  Z  2F1(1, t/2; 2 - t/2; w),
    Z = q x' + i y',  R = |Z|,  w = -(1-q)/(1+q) * Z / conj(Z).

lenstronomy calls `scipy.special.hyp2f1`, which is neither differentiable nor
available in torch. Because (1)_n / n! = 1 the series collapses to

    2F1(1, t/2; 2 - t/2; w) = sum_n c_n w^n ,
    c_0 = 1 ,   c_{n+1}/c_n = (t/2 + n) / (2 - t/2 + n) ,

and |w| = (1-q)/(1+q) < 1 always, so it converges geometrically. Model_A's most
elliptical lens has |e| = 0.393 -> q = 0.436 -> |w| = 0.393, giving 0.393^40 ~
1e-17. tests/test_lens_models.py MEASURES the truncation error against
scipy.special.hyp2f1 rather than assuming it.

Written with explicit real and imaginary parts, not complex tensors, so the
autograd path has no Wirtinger-derivative subtleties to reason about.

MAGNIFICATION
-------------
`magnification()` differentiates the deflection field that is ACTUALLY applied.
It therefore cannot disagree with the ray-tracer. That was the defect in
`physics_losses.fixed_sis_magnification`, which hard-codes
det A = 1 - theta_E/r: correct only for a circular SIS (lambda_r = 1 requires
kappa = |gamma| at every radius) and with no link to the sparse matrices in use.
`hessian_analytic()` is the independent closed form, used to CHECK the autograd
result in tests/test_jacobian.py and to provide magnification under numpy where
there is no autograd.
"""
from __future__ import annotations

from typing import Dict, Tuple

from backend import get_backend

__all__ = ["ellipticity_to_phi_q", "deflection_epl", "deflection_shear",
           "deflection", "ray_shoot", "hessian_analytic", "magnification",
           "magnification_autograd", "LENS_PARAM_NAMES"]

N_HYP_TERMS = 50
R_MIN_ARCSEC = 1e-4        # ~1/1000 of a Model_A pixel; see deflection_epl
E_MAX = 0.9                # cap on |e| during optimisation; Model_A max is 0.393
EPS_E2 = 1e-24             # guards the 0/0 gradient of |e| at the origin
E_FLOOR = 1e-10            # floor on |e|; see ellipticity_to_phi_q

LENS_PARAM_NAMES = ("theta_E", "gamma", "e1", "e2", "g1", "g2", "cx", "cy")


# ---------------------------------------------------------------------------

def ellipticity_to_phi_q(e1, e2, max_c: float = E_MAX, floor: float = E_FLOOR):
    """(e1, e2) -> (position angle, axis ratio q = minor/major).

    lenstronomy caps |e| at 0.9999 (q = 5e-5). We cap at 0.9 (q = 0.053): far
    outside the Model_A range (max |e| = 0.393 -> q = 0.436) but far enough from
    q = 0 that (b/R)^t and the 1/(1+q) prefactor stay finite if a free e1/e2
    transiently wanders during optimisation.

    WHY THERE IS A FLOOR AS WELL AS A CAP
    -------------------------------------
    e1 = e2 = 0 is a removable singularity in the VALUE and a hard one in the
    GRADIENT. Both operations below are 0/0 there under reverse-mode autodiff:

        d/de1 sqrt(e1^2 + e2^2) = e1 / sqrt(e1^2 + e2^2)   -> 0/0
        d/de2 atan2(e2, e1)     = e1 / (e1^2 + e2^2)       -> 0/0

    The forward pass is perfectly well behaved (q = 1, the circular fast path in
    deflection_epl fires), which is what makes this so easy to miss: nothing is
    wrong until something calls .backward().

    This is not a hypothetical. train_amortised.Encoder zero-initialises the
    final Linear's weight AND bias so the network starts at "neutral"
    parameters, which means e1 = 0.6 tanh(0) = 0 and e2 = 0 EXACTLY on step one.
    The first backward pass therefore produced NaN gradients on the ellipticity
    head; clip_grad_norm_ does not filter NaN, so it scaled every parameter by a
    NaN coefficient and Adam wrote NaN into the whole head; on step two the
    network emitted NaN, q was NaN, and _n_terms_for died on math.ceil(nan) with
    "ValueError: cannot convert float NaN to integer" -- a message pointing at
    the hypergeometric term count, six frames away from the actual cause.

    And it is not merely an initialisation accident that a nudge would cure:
    audit_dataset.py shows Model_A's macro lens IS circular, so e = 0 is where
    the optimum lives for this dataset. The fit is expected to sit on the
    singular point, not merely to pass through it.

    The fix pushes a degenerate (e1, e2) onto the +e1 axis at |e| = `floor`
    BEFORE either singular operation, using `where`, so the degenerate branch
    contributes exactly zero gradient instead of NaN. phi is arbitrary there
    because q = 1 makes the frame rotation a no-op.

    floor = 1e-10 perturbs q by 2e-10 and the deflection by ~theta_E * 2e-10
    ~ 3e-10, which is two orders below the 1e-8 agreement that
    tests/test_lens_models.py requires against lenstronomy.
    """
    xp = get_backend(e1)
    u = e1 * e1 + e2 * e2
    degenerate = u < floor * floor
    e1 = xp.where(degenerate, xp.ones_like(e1) * floor, e1)
    e2 = xp.where(degenerate, xp.zeros_like(e2), e2)
    phi = 0.5 * xp.atan2(e2, e1)
    # +EPS_E2 inside the sqrt: at exactly e1 = e2 = 0 the derivative of
    # sqrt(e1^2+e2^2) is 0/0 = inf, which produces a NaN gradient on the
    # very first step of any optimiser whose ellipticity head starts at
    # zero. With the offset the derivative there is 0 -- finite and
    # correct, since |e| genuinely has a minimum at the origin. The shift
    # shift in |e| is at most 1e-12 -- invisible in float32 and far below
    # Model_A's smallest recorded |e| of 1e-3. tests/test_lens_models.py
    # still agrees with lenstronomy to 1e-12 with this in place.
    c = xp.clip(xp.sqrt(e1 * e1 + e2 * e2 + EPS_E2), None, max_c)
    q = (1.0 - c) / (1.0 + c)
    return phi, q


def _rotate(x, y, phi):
    """lenstronomy Util.util.rotate -- rotates the FRAME by phi."""
    xp = get_backend(x)
    c, s = xp.cos(phi), xp.sin(phi)
    return x * c + y * s, -x * s + y * c


def _n_terms_for(f, tol: float = 1e-12, cap: int = N_HYP_TERMS) -> int:
    """How many series terms are needed for |w| = f to reach `tol`.

    The terms fall off like f^n (the coefficient ratio tends to 1 from below), so
    n = log(tol)/log(f). Model_A's most elliptical lens gives f = 0.393 -> n = 30;
    a near-circular lens gives f ~ 0.02 -> n = 8. Adapting rather than always
    running N_HYP_TERMS is worth a factor of 2-5 inside an optimiser loop, and
    tests/test_lens_models.py measures the resulting error against scipy.

    NOTE ON NaN: the clamp below must test for NaN explicitly. Python's min/max
    propagate NaN silently -- min(max(nan, 1e-12), 0.999) is nan, not 0.999 --
    so a NaN f used to sail past the clamp and die four lines later inside
    math.ceil with "cannot convert float NaN to integer", which reads like a
    bug in the series and is not. A NaN here always means the CALLER handed us a
    NaN ellipticity, so say that instead.
    """
    import math
    try:
        fmax = float(f.max()) if hasattr(f, "max") else float(f)
    except Exception:
        return cap
    if not math.isfinite(fmax):
        raise FloatingPointError(
            f"_n_terms_for got a non-finite f = {fmax}. f = (1-q)/(1+q), so this "
            "means e1/e2 reaching ellipticity_to_phi_q were NaN or inf -- the "
            "lens parameters have already diverged upstream. In a training loop "
            "the usual cause is a NaN gradient poisoning the weights; check the "
            "first step where the loss stops being finite, not this function.")
    fmax = min(max(fmax, 1e-12), 0.999)
    if fmax < 1e-9:
        return 1
    return int(min(cap, max(2, math.ceil(math.log(tol) / math.log(fmax)))))


def _hyp2f1_series(wr, wi, t, n_terms: int = N_HYP_TERMS):
    """2F1(1, t/2; 2 - t/2; w) with w = wr + i wi. Real arithmetic only."""
    xp = get_backend(wr)
    half_t = 0.5 * t
    sr = xp.ones_like(wr)
    si = xp.zeros_like(wr)
    tr = xp.ones_like(wr)
    ti = xp.zeros_like(wr)
    for n in range(n_terms):
        c = (half_t + n) / (2.0 - half_t + n)
        nr = c * (tr * wr - ti * wi)
        ni = c * (tr * wi + ti * wr)
        tr, ti = nr, ni
        sr = sr + tr
        si = si + ti
    return sr, si


def deflection_epl(x, y, theta_E, gamma, e1, e2, cx=0.0, cy=0.0,
                   r_min: float = R_MIN_ARCSEC, n_terms: int = N_HYP_TERMS):
    """Elliptical power-law deflection.

    x, y and the parameters must broadcast against each other. In the torch
    training loop the parameters are (B, 1, 1) and the grids are (B, H, W); in
    the numpy fitter everything is a scalar or an (H, W) array.

    r_min clamps the elliptical radius. The profile is genuinely singular at the
    centre, the central pixel is unresolved anyway, and an unclamped 1/R there
    produces gradients that destabilise a fit for no physical gain.
    """
    xp = get_backend(x)
    x = x - cx
    y = y - cy

    phi, q = ellipticity_to_phi_q(e1, e2)
    b = theta_E * xp.sqrt(q)
    t = gamma - 1.0

    xr, yr = _rotate(x, y, phi)
    Zr = q * xr
    Zi = yr
    R2 = xp.clip(Zr * Zr + Zi * Zi, r_min * r_min, None)
    R = xp.sqrt(R2)

    f = (1.0 - q) / (1.0 + q)

    # CIRCULAR FAST PATH. For q = 1 the argument w is identically zero and
    # 2F1(...; 0) = 1 exactly, so the series is pure overhead. This matters
    # because audit_dataset.py shows Model_A's macro lens IS circular, and
    # because the staged fit in fit_per_image.py holds e1 = e2 = 0 for its first
    # three stages. Threshold 1e-9 on f, i.e. |e| < 5e-10 -- far below anything
    # the optimiser will sit at by accident.
    if _n_terms_for(f) <= 1:
        ror, roi = Zr, Zi
    else:
        inv = 1.0 / R2
        wr = -f * (Zr * Zr - Zi * Zi) * inv
        wi = -f * (2.0 * Zr * Zi) * inv
        nt = min(n_terms, _n_terms_for(f))
        sr, si = _hyp2f1_series(wr, wi, t, nt)
        ror = Zr * sr - Zi * si
        roi = Zr * si + Zi * sr

    pref = (2.0 / (1.0 + q)) * xp.power(b / R, t)
    return _rotate(pref * ror, pref * roi, -phi)


def deflection_shear(x, y, g1, g2):
    """External shear: alpha = [[g1, g2], [g2, -g1]] theta.  kappa = 0."""
    return g1 * x + g2 * y, g2 * x - g1 * y


def deflection(x, y, p: Dict, n_terms: int = N_HYP_TERMS):
    """Total deflection of EPL + external shear.

    The shear acts about the GRID origin, not the lens centre, matching
    lenstronomy's SHEAR with ra_0 = dec_0 = 0 (which is how Model_A was
    generated). A shear about a different point differs by a constant
    deflection, which is exactly degenerate with the source position, so the
    choice costs no generality.
    """
    ax, ay = deflection_epl(x, y, p["theta_E"], p.get("gamma", 2.0),
                            p.get("e1", 0.0), p.get("e2", 0.0),
                            p.get("cx", 0.0), p.get("cy", 0.0), n_terms=n_terms)
    if "g1" in p or "g2" in p:
        sx, sy = deflection_shear(x, y, p.get("g1", 0.0), p.get("g2", 0.0))
        ax, ay = ax + sx, ay + sy
    return ax, ay


def ray_shoot(x, y, p: Dict, **kw):
    """beta = theta - alpha(theta)."""
    ax, ay = deflection(x, y, p, **kw)
    return x - ax, y - ay


# ---------------------------------------------------------------------------
# convergence, shear, magnification
# ---------------------------------------------------------------------------

def hessian_analytic(x, y, p: Dict, r_min: float = R_MIN_ARCSEC):
    """Closed-form (kappa, gamma1, gamma2) in the IMAGE frame, EPL + shear.

    Tessore & Metcalf (2015) eq. 17 in the corrigendum form used by
    lenstronomy's EPLMajorAxis.hessian, then rotated out of the major-axis
    frame. gamma is spin-2, so rotating the frame by phi maps

        gamma1' = gamma1 cos(2 phi) - gamma2 sin(2 phi)
        gamma2' = gamma1 sin(2 phi) + gamma2 cos(2 phi)

    kappa is a scalar and is unchanged. External shear adds directly to
    (gamma1, gamma2) and contributes nothing to kappa. The sign of the rotation
    is fixed by tests/test_lens_models.py against lenstronomy's own hessian, not
    by argument.
    """
    xp = get_backend(x)
    x = x - p.get("cx", 0.0)
    y = y - p.get("cy", 0.0)
    theta_E, gamma = p["theta_E"], p.get("gamma", 2.0)
    e1, e2 = p.get("e1", 0.0), p.get("e2", 0.0)

    phi, q = ellipticity_to_phi_q(e1, e2)
    b = theta_E * xp.sqrt(q)
    t = gamma - 1.0

    xr, yr = _rotate(x, y, phi)
    R = xp.clip(xp.sqrt((q * xr) ** 2 + yr * yr), r_min, None)
    r = xp.clip(xp.sqrt(xr * xr + yr * yr), r_min, None)

    Zr, Zi = q * xr, yr
    R2 = xp.clip(Zr * Zr + Zi * Zi, r_min * r_min, None)
    f = (1.0 - q) / (1.0 + q)
    wr = -f * (Zr * Zr - Zi * Zi) / R2
    wi = -f * (2.0 * Zr * Zi) / R2
    sr, si = _hyp2f1_series(wr, wi, t)
    ror, roi = Zr * sr - Zi * si, Zr * si + Zi * sr
    pref = (2.0 / (1.0 + q)) * xp.power(b / R, t)
    ax_, ay_ = pref * ror, pref * roi                # major-axis frame

    cos, sin = xr / r, yr / r
    cos2, sin2 = cos * cos * 2.0 - 1.0, sin * cos * 2.0
    kappa = (2.0 - t) / 2.0 * xp.power(b / R, t)
    g1m = (1.0 - t) * (ax_ * cos - ay_ * sin) / r - kappa * cos2
    g2m = (1.0 - t) * (ay_ * cos + ax_ * sin) / r - kappa * sin2

    c2, s2 = xp.cos(2.0 * phi), xp.sin(2.0 * phi)
    g1 = g1m * c2 - g2m * s2
    g2 = g1m * s2 + g2m * c2

    if "g1" in p or "g2" in p:
        g1 = g1 + p.get("g1", 0.0)
        g2 = g2 + p.get("g2", 0.0)
    return kappa, g1, g2


def magnification(x, y, p: Dict, mu_clip: float = 50.0, signed: bool = False):
    """mu = 1 / det A = 1 / ((1-kappa)^2 - |gamma|^2), from the analytic Hessian.

    SURFACE BRIGHTNESS IS CONSERVED BY LENSING. mu must never multiply an
    intensity: it is a change of solid angle. Its legitimate uses are telling a
    source-plane prior where the lens has actually delivered resolution, and
    diagnostics. This is the same convention already documented in
    STAGE_D_E_FIXED_SIS.md and implemented in physics_losses.py -- what changes
    here is only that mu is now computed from the lens being fitted rather than
    from a hard-coded circular formula unconnected to the operator in use.
    """
    xp = get_backend(x)
    kappa, g1, g2 = hessian_analytic(x, y, p)
    det = (1.0 - kappa) ** 2 - (g1 * g1 + g2 * g2)
    sign = xp.where(det >= 0, xp.ones_like(det), -xp.ones_like(det))
    safe = sign * xp.clip(xp.abs(det), 1.0 / mu_clip, None)
    mu = 1.0 / safe
    return mu if signed else xp.clip(xp.abs(mu), None, mu_clip)


def magnification_autograd(x, y, p: Dict, mu_clip: float = 50.0):   # pragma: no cover
    """Same quantity, obtained by differentiating the applied deflection.

    torch only. This is the production path inside the training loop: because it
    differentiates `deflection()` itself, it is structurally incapable of
    describing a different lens from the one being ray-traced.

    Two vector-Jacobian products give all four entries of d(alpha)/d(theta) at
    every pixel at once. That works because the map is pointwise -- alpha at one
    pixel does not depend on theta at another -- so grad(sum(alpha_x)) w.r.t. x
    is d(alpha_x)/dx everywhere, with no cross-pixel contamination. That
    assumption is checked against `hessian_analytic` in tests/test_jacobian.py.
    """
    import torch
    x = x.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)
    ax, ay = deflection(x, y, p)
    axx, axy = torch.autograd.grad(ax.sum(), (x, y), create_graph=True)
    ayx, ayy = torch.autograd.grad(ay.sum(), (x, y), create_graph=True)
    kappa = 0.5 * (axx + ayy)
    g1 = 0.5 * (axx - ayy)
    g2 = 0.5 * (axy + ayx)
    det = (1.0 - kappa) ** 2 - (g1 * g1 + g2 * g2)
    sign = torch.where(det >= 0, torch.ones_like(det), -torch.ones_like(det))
    return (1.0 / (sign * det.abs().clamp_min(1.0 / mu_clip))).abs().clamp(max=mu_clip)
