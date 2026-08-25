"""
sources.py -- source-plane models.

WHY TWO OF THEM
---------------
diagnostics_2026_08_12/PROJECT_REPORT.md section 5.2 measured
`src_truth_size_ratio` = 10.6-12.8 across a 100x sweep of the TV weight
(tv = 0, 0.03, 0.1, 0.3, 3.0). That is not a regularisation-strength failure.
The old pipeline gave the network 254 x 254 = 64,516 free source pixels against
~4,300 data pixels above 3 sigma -- a 15:1 underdetermined inverse problem held
down only by a hand-tuned total-variation prior whose unconstrained minimiser is
a flat field. Raising TV moves toward the wash, not away from it; the most
compact result in the whole sweep was at tv = 0.03, and it was still 10.6x too
large.

Two honest ways out, both here:

  SersicSource  7 parameters. With the 6 lens parameters that is 13 unknowns
                against ~4,300 informative pixels -- overdetermined ~330:1. The
                problem is well posed, and every parameter can be scored against
                the manifest. This must work before anything free-form is
                believable.

  PixelSource   a free source grid with CURVATURE regularisation whose strength
                is set by the discrepancy principle (chi^2/dof -> 1) rather than
                chosen by hand. This is the semi-parametric route of
                Warren & Dye (2003) and Suyu et al. (2006). Two differences from
                the old TV term: curvature is a quadratic form, so with a
                Gaussian likelihood the source sub-problem is convex; and lambda
                is determined by the data instead of swept.

Both expose `.at(bx, by)`, so raytrace.render() does not care which is in use.

WHY THE PARAMETRIC SOURCE IS EXACT
----------------------------------
SersicSource.at() evaluates a closed form directly at the ray-shot coordinates.
There is no source grid and hence no interpolation error: the forward model is
exact up to the image-plane supersampling. PixelSource.at() must interpolate
(bilinear), which is differentiable in beta -- and therefore in the lens
parameters -- but introduces a resampling error that tests/test_raytrace.py
quantifies.

CONVENTION
----------
Matched to lenstronomy's SERSIC_ELLIPSE with its default
`sersic_major_axis=False`, because that is what generated Model_A:

    norm = sqrt(|1 - e1^2 - e2^2|)
    x' = ((1 - e1) dx - e2 dy) / norm
    y' = (-e2 dx + (1 + e1) dy) / norm
    R  = hypot(x', y')                       "product average" ellipse
    I  = amp * exp(-b_n * ((R/R_s)^(1/n) - 1))
    b_n = 1.9992 n - 0.3271

Do NOT substitute the more accurate Ciotti & Bertin (1999) expansion for b_n.
The goal is to reproduce the simulator, not to be independently more correct; a
different b_n shows up as a systematic bias in recovered R_sersic. Verified to
machine precision against lenstronomy in tests/test_sources.py.
"""
from __future__ import annotations

from typing import Dict

from backend import get_backend

__all__ = ["SersicSource", "PixelSource", "SOURCE_PARAM_NAMES", "b_n",
           "sersic_defaults"]

SOURCE_PARAM_NAMES = ("amp", "R_sersic", "n_sersic", "se1", "se2", "sx", "sy")

_SMOOTHING = 1e-4          # lenstronomy SersicUtil default
_N_MIN, _N_MAX = 0.3, 6.0  # Model_A range is 0.50-2.86; a cushion for optimisation
_R_MIN = 1e-3              # arcsec


def b_n(n):
    """lenstronomy's b(n), floored exactly as lenstronomy floors it."""
    xp = get_backend(n)
    return xp.clip(1.9992 * n - 0.3271, 1e-5, None)


class SersicSource:
    """Elliptical Sersic. `params` values must broadcast against (bx, by)."""

    def __init__(self, params: Dict):
        self.p = params

    def at(self, bx, by):
        xp = get_backend(bx)
        p = self.p
        Rs = xp.clip(p["R_sersic"], _R_MIN, None)
        n = xp.clip(p["n_sersic"], _N_MIN, _N_MAX)
        e1, e2 = p.get("se1", 0.0), p.get("se2", 0.0)
        dx = bx - p.get("sx", 0.0)
        dy = by - p.get("sy", 0.0)

        norm = xp.sqrt(xp.clip(xp.abs(1.0 - e1 * e1 - e2 * e2), 1e-6, None))
        xpr = ((1.0 - e1) * dx - e2 * dy) / norm
        ypr = (-e2 * dx + (1.0 + e1) * dy) / norm
        R = xp.clip(xp.sqrt(xpr * xpr + ypr * ypr), _SMOOTHING, None)

        return p["amp"] * xp.exp(-b_n(n) * (xp.power(R / Rs, 1.0 / n) - 1.0))


def sersic_defaults() -> Dict[str, float]:
    """Population-neutral start: a round, exponential source at the origin.

    Deliberately NOT drawn from the manifest. The initialisation must not smuggle
    truth in through the back door -- that is exactly the defect this rewrite
    removes (train_sis_bank.py line 344 read ds.theta_E from the manifest, which
    then chose the backward operator, became an input channel, and chose the
    forward operator). theta_E and the source offset come from the image itself
    via theta_e_init.py; everything else starts here.
    """
    return {"amp": 1.0, "R_sersic": 0.45, "n_sersic": 1.0,
            "se1": 0.0, "se2": 0.0, "sx": 0.0, "sy": 0.0}


class PixelSource:                                    # pragma: no cover - torch only
    """Free source grid, bilinearly sampled, with curvature regularisation.

    The grid covers +/- `half_extent` arcsec at `n_src` pixels a side. Model_A's
    true sources have |beta| <= 0.58 arcsec and R_sersic <= 1.69 arcsec, so
    half_extent = 1.6 at n_src = 48 (0.067 arcsec/px, finer than the 0.106
    arcsec detector) covers the population with room to spare while keeping the
    free parameters to 2,304 -- about half the ~4,300 informative data pixels,
    rather than fifteen times as many. That ratio is the whole point.
    """

    def __init__(self, pixels, half_extent: float = 1.6):
        self.pix = pixels                             # (B, 1, n, n)
        self.half = float(half_extent)

    @classmethod
    def zeros(cls, batch: int, n_src: int = 48, half_extent: float = 1.6,
              device=None, dtype=None, init: float = 1e-3):
        import torch
        t = torch.full((batch, 1, n_src, n_src), init, device=device,
                       dtype=dtype or torch.float32)
        return cls(t.requires_grad_(True), half_extent)

    def at(self, bx, by):
        """Bilinear sample at (bx, by) arcsec.

        grid_sample with align_corners=False treats -1 and +1 as the OUTER EDGES
        of the border pixels, while our grid spans [-half, +half] to pixel
        CENTRES. The `edge` factor converts between the two so a source at
        +half lands on the last pixel centre rather than half a pixel outside it.
        """
        import torch
        import torch.nn.functional as F
        n = self.pix.shape[-1]
        edge = self.half * (1.0 + 1.0 / (n - 1))
        grid = torch.stack([bx / edge, by / edge], dim=-1)     # (B, H, W, 2)
        return F.grid_sample(self.pix, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)[:, 0]

    def curvature(self):
        """Sum of squared discrete Laplacians -- Suyu et al. (2006) 'curvature'.

        Preferred over total variation for the reason PROJECT_REPORT.md section
        5.2 documents: TV's unconstrained minimiser is a constant field, so
        raising its weight drives the source toward a flat wash, and the measured
        size_ratio was flat at 10.6-12.8 over a 100x sweep. Curvature is a
        quadratic form, so with a Gaussian likelihood the source sub-problem is
        convex and lambda has a determinate value set by chi^2/dof -> 1 rather
        than by preference.
        """
        s = self.pix
        lap = (-4.0 * s[..., 1:-1, 1:-1]
               + s[..., :-2, 1:-1] + s[..., 2:, 1:-1]
               + s[..., 1:-1, :-2] + s[..., 1:-1, 2:])
        return (lap ** 2).sum(dim=(1, 2, 3))

    def nonneg_penalty(self):
        """Surface brightness cannot be negative. A soft floor, not a clamp --
        clamping would zero the gradient exactly where it is needed."""
        return (self.pix.clamp(max=0.0) ** 2).sum(dim=(1, 2, 3))

    def parameters(self):
        return [self.pix]
