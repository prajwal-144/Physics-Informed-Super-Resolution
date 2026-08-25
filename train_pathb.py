"""
train_pathb.py -- Path B (design B3): a CNN that does physics-constrained
super-resolution, with magnification controlling where it is allowed to add
detail.

THE MODEL, IN ONE PICTURE
------------------------

    image ──CNN──┬──▶ 14 lens+source parameters ──▶ Sersic(beta)      ┐
                 │                                                     ├─▶ source(beta)
                 └──▶ n_c x n_c correction map C ──▶ C(beta)          ┘
                                                            │
                              beta = theta - alpha(theta; predicted lens)
                                                            │
                            PSF convolve ─▶ pixel bin ─▶ chi^2 vs the image

There is NO high-resolution target anywhere. The only supervision is the
physics: whatever the network emits must, after being lensed, blurred and
binned, reproduce the observed image.

WHY A SERSIC *PLUS* A CORRECTION, RATHER THAN FREE PIXELS
----------------------------------------------------------
Two earlier attempts bracket this design.

  * The original pipeline let a network emit 254^2 = 64,516 free source pixels
    against ~4,300 informative data pixels -- 15:1 underdetermined -- and the
    reconstructed source came out 10.6-12.8x too large across a 100x sweep of
    the regularisation weight.

  * Option2_failed/ solved for 64^2 = 4,096 free pixels by linear inversion on
    a CORRECT lens. Better posed, but the median ray coverage is only 1.44 rays
    per source pixel, so most of the grid is unconstrained; conjugate gradients
    failed to converge on ~30% of images and the reconstructions came out as
    speckle (see Option2_failed/results/fig6_pixel_source.png).

The lesson from both is that the source plane simply does not contain enough
independent information to support thousands of free parameters per image.

B3 therefore splits the source into a part the physics can predict and a part it
cannot:

    source = Sersic(7 params)  +  correction(n_c^2, bounded and penalised)

With n_c = 24 that is 14 + 576 = 590 numbers against ~4,300 data pixels -- still
overdetermined ~7:1. And it DEGRADES GRACEFULLY: if the correction learns
nothing it collapses to the parametric solution, which is already known to work
(source correlation 0.987, size_ratio 1.089 at n = 2000). The network cannot do
worse than Path A; it can only add what the Sersic misses.

WHY A NETWORK AT ALL, WHEN THE LINEAR SOLVE FAILED
---------------------------------------------------
A per-image solve has only that image's ~4,300 pixels to constrain 4,096
unknowns. A network shares one set of weights across the whole training set, so
it carries a LEARNED PRIOR over what galaxies look like. That is a genuinely
different resource, and it is the one thing the linear inversion lacked. It is
also the reason this is worth trying even though Option 2 failed: the failures
have different causes.

MAGNIFICATION IN THE LOSS -- THE PHYSICS
-----------------------------------------
The lens does not deliver the same resolution everywhere in the source plane.
Where the magnification |mu| is large, a small source patch is spread over many
detector pixels, so the data genuinely constrain fine structure there. Where
|mu| ~ 1 they do not, and any detail the network paints in is invention.

So the correction is penalised as

    R = sum_j  w_j * C_j^2  ,      w_j = (median coverage / coverage_j)^p

where coverage_j counts how many image sub-pixels ray-trace into source pixel j.
That count IS the magnification of that source pixel, up to the constant
sub-pixel area -- so mu comes from the SAME ray-shooting that renders the image
and cannot describe a different lens. That was precisely the defect in
`physics_losses.fixed_sis_magnification`, which hard-coded
det A = 1 - theta_E/r with no link to the sparse operator actually in use.

This is the magnification-as-information-map idea from the original pipeline,
working for the first time because the lens underneath it is finally correct and
fitted per image. `--reg-mode uniform` runs the identical model without it, so
the contribution is measured rather than asserted.

WHAT WENT WRONG LAST TIME, AND WHAT IS DIFFERENT HERE
------------------------------------------------------
An earlier amortised run (sie_pipeline/train_amortised.py) collapsed: every
parameter came out constant across all 200 validation images, Spearman
undefined, chi2/dof 25x worse than the per-image fit. Two causes, both fixed:

  1. The output head was initialised with `nn.init.zeros_`, so the network
     emitted exactly the neutral parameters and, if gradients failed on step
     one, stayed there for ever. Here the head uses small random weights.
  2. At exactly e1 = e2 = 0 the derivative of sqrt(e1^2 + e2^2) is 0/0 = inf,
     which poisons every gradient in the batch. Fixed at source in
     lens_models.ellipticity_to_phi_q via EPS_E2.

A NaN guard is kept anyway, and it reports rather than silently skipping.

WHAT WENT WRONG IN THE FIRST 40-EPOCH RUN, AND WHAT v2 CHANGES
---------------------------------------------------------------
The first full run (6000/400, 40 epochs, both reg modes) produced a round blob:
|e| = 0.048 against a true 0.216, n_sersic Spearman -0.01, chi2/dof 12,998
against the per-image fit's 3,085. A single cause explains all of it, and it is
NOT the architecture and NOT the number of epochs.

    THE chi^2 WAS NEVER NORMALISED.

sigma was the background std alone, so a bright arc produced residuals of
10^3-10^4 sigma and a faint one ~10. Measured consequences:

  * per-image chi^2 spanned 656 to 21,953,076. Four images out of 400 carried
    51% of the total loss; in a batch of 16 the single largest member took 54%
    of the gradient, so the EFFECTIVE BATCH SIZE WAS ~2.
  * --lambda-corr 1.0 against a chi^2 of ~10^5 made the correction penalty six
    orders of magnitude too small to do anything. The mu and uniform runs came
    out identical (Pearson +0.99 on every parameter) and
    corr(|correction|, log mu) was -0.03: the magnification term never acted.
  * with no effective penalty the tanh saturated. 75% of every correction map
    sat within 1% of the bound, where the gradient is ~0, so the correction
    stopped learning and starved the parametric head at the same time. You can
    see the square edge of the saturated map in figB1.

v2 fixes exactly that:

  1. --sigma-floor 0.02: sigma_eff^2 = sigma_bg^2 + (0.02 * model)^2. The PSF
     wings are measured to be wrong at the 1-3% level (calibrate_psf.py) and
     that error scales with flux, so this is the correct noise model, not a
     fudge. On the per-image fits it flattens chi^2 across SNR from
     1411 -> 8455 down to 329 -> 229 and cuts the p90/p10 spread from 95x to 8.7x.
  2. lambda is now a FRACTION OF chi^2 (multiplied by chi2.detach()), so it is
     scale-free and cannot be silently six orders out again.
  3. tanh saturation is measured and printed EVERY epoch, and warned about at
     the end. It was invisible before.
  4. --augment: random 90-degree rotations and flips. Free and exactly valid
     for a self-supervised objective. The first run overfitted 2.9x.
  5. --source-mode b2 runs the free-form model (no Sersic) through the same
     code, so B2-vs-B3 is a controlled experiment rather than an argument.

Run from the Grid_Based_Experiment root:
    python superres/train_pathb.py --root . --classes axion cdm wdm \\
        --n-train 6000 --n-val 400 --epochs 40 --source-mode b3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:                                    # pragma: no cover
    raise SystemExit(
        "train_pathb.py needs torch.\n"
        f"  import failed: {exc}\n"
        "Everything else in superres/ runs on numpy/scipy alone.")

from data_a import ModelADataset
from lens_models import deflection
from raytrace import area_downsample, image_plane_grid, load_psf
from sources import SersicSource
from theta_e_init import initial_guess

PARAMS = ("theta_E", "gamma", "e1", "e2", "g1", "g2",
          "amp", "R_sersic", "n_sersic", "se1", "se2", "sx", "sy", "background")
LENS_KEYS = ("theta_E", "gamma", "e1", "e2", "g1", "g2", "cx", "cy")
SRC_KEYS = ("amp", "R_sersic", "n_sersic", "se1", "se2", "sx", "sy")


# ===========================================================================
# network
# ===========================================================================

class PathBNet(nn.Module):
    """Shared conv trunk -> (14 parameters, n_c x n_c correction map).

    The trunk is deliberately small (~0.5M weights). The task is not visual
    recognition; it is reading a handful of geometric quantities off an arc, and
    a larger network mostly buys overfitting on 4,000 images.

    Input, from build_input(), all measured from the image -- nothing from the
    manifest:
      channel 0 (image):   arcsinh(image / sigma_bg), range-compressed
      channels 1..4:       the four ring statistics, as constant planes

    THE RING SCALARS BYPASS THE CONV TRUNK. They are fed as constant planes for
    the trunk to use spatially, AND concatenated straight onto the pooled
    feature vector. Without the bypass they have to survive four strided
    convolutions and a global average pool, competing with ~4,000 image pixels
    for room in a 128-dimensional bottleneck -- and measurably they do not: a
    v2 network handed |m1| at Spearman +0.832 emitted a source offset at +0.049,
    i.e. it DESTROYED a signal it had been given. The bypass makes the scalars
    available to the parameter head undiluted.
    """

    def __init__(self, n_c: int = 24, width: int = 32, n_par: int = 14,
                 in_ch: int = 5, n_ring: int = 4):
        super().__init__()
        c = width
        self.n_c = n_c
        self.n_ring = n_ring
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, c, 5, stride=2, padding=2), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.GroupNorm(8, 2 * c), nn.SiLU(),
            nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1), nn.GroupNorm(8, 4 * c), nn.SiLU(),
            nn.Conv2d(4 * c, 4 * c, 3, stride=2, padding=1), nn.GroupNorm(8, 4 * c), nn.SiLU(),
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        feat = 4 * c + n_ring

        # --- head 1: the 14 physical parameters --------------------------
        self.par_head = nn.Sequential(nn.Linear(feat, 128), nn.SiLU(),
                                      nn.Linear(128, n_par))

        # --- head 2: the source-plane correction map ----------------------
        # A small decoder rather than one big Linear: the correction is a
        # spatial field, and transposed convolutions impose locality, which is a
        # far better prior than an unstructured 128 -> 576 dense map.
        # base = n_c // 4 so the two stride-2 transposed convolutions land on
        # exactly n_c and nothing is bilinearly resized. A resize would make the
        # effective number of degrees of freedom smaller than n_c^2, which would
        # quietly contradict the conditioning arithmetic printed at startup.
        self.base = max(2, n_c // 4)
        self.dec_fc = nn.Linear(feat, 64 * self.base * self.base)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

        # SMALL RANDOM init, NOT zeros. Zero-init was what let the previous
        # amortised run sit at its neutral output for every epoch.
        for m in (self.par_head[-1], self.dec[-1]):
            nn.init.normal_(m.weight, std=1e-3)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # the ring scalars are constant over the plane, so [:, 1:, 0, 0] reads
        # them back exactly; concatenating them re-injects them past the trunk
        ring = x[:, 1:1 + self.n_ring, 0, 0]
        h = torch.cat([self.pool(self.trunk(x)), ring], dim=1)
        z = self.par_head(h)
        c = self.dec(self.dec_fc(h).view(-1, 64, self.base, self.base))
        if c.shape[-1] != self.n_c:
            c = F.interpolate(c, size=(self.n_c, self.n_c), mode="bilinear",
                              align_corners=False)
        return z, c[:, 0]


def to_params(z, ring):
    """Unconstrained reals -> physical parameters, smoothly and invertibly.

    `ring` is the (B, 4) ring-summary vector: [theta_E, |m1|, |m2|/theta_E,
    completeness], all measured from the image by theta_e_init.initial_guess().

    Every quantity is mapped through tanh/sigmoid/softplus into its physical
    range, so the network can never emit a negative Einstein radius or a Sersic
    index of zero -- either of which produces NaNs inside (b/R)^t or R^(1/n)
    before any gradient exists to correct it.

    TWO QUANTITIES ARE ANCHORED ON THE RING GEOMETRY rather than predicted from
    scratch, because both estimators are good and the residual is far better
    conditioned than the absolute value:

        theta_E  = ring_theta_E * exp(0.3 tanh(.))      ring rho +0.963
        |beta|   = ring_|m1|    * exp(0.5 tanh(.))      ring rho +0.832

    The anchoring is why theta_E works and, in v2, why beta did not: theta_E was
    anchored and came out at +0.955, while beta was free and came out at +0.049
    despite |m1| being available. The ANGLE of beta is still learned from the
    pixels -- only the magnitude is anchored -- so the network still has real
    work to do, and eval_pathb.py prints the ring-only baseline as the control.
    """
    s = lambda i: z[:, i]
    tE, m1 = ring[:, 0], ring[:, 1]
    # The source offset is SCALED by the ring dipole rather than forced equal to
    # it. Forcing the magnitude (r*cos(phi), r*sin(phi) with phi = atan2(z, z'))
    # reintroduces exactly the EPS_E2 disease: atan2 is singular at the origin
    # and a zero-initialised head lands there, so tests/test_pathb.py check 5
    # went straight back to NaN. Scaling has no singularity, still gives the
    # network a dimensionless residual to learn, and still lets it move beta by
    # up to a factor ~2.8 either way.
    b_scale = 2.0 * m1.clamp_min(1e-3)
    return {
        "theta_E": tE * torch.exp(0.3 * torch.tanh(s(0))),
        "gamma": 2.0 + 0.5 * torch.tanh(s(1)),
        "e1": 0.6 * torch.tanh(s(2)),
        "e2": 0.6 * torch.tanh(s(3)),
        "g1": 0.3 * torch.tanh(s(4)),
        "g2": 0.3 * torch.tanh(s(5)),
        "amp": F.softplus(s(6) + 1.0),
        "R_sersic": 0.05 + 2.0 * torch.sigmoid(s(7) - 1.0),
        "n_sersic": 0.3 + 5.0 * torch.sigmoid(s(8) - 1.0),
        "se1": 0.6 * torch.tanh(s(9)),
        "se2": 0.6 * torch.tanh(s(10)),
        "sx": b_scale * torch.tanh(s(11)),
        "sy": b_scale * torch.tanh(s(12)),
        # bounded: v2 left this as a free 0.1*z and it drifted to a median of
        # -0.98 (Path A gives +0.018), i.e. the network dug a negative sky
        # pedestal so it could paint an over-bright over-large source on top.
        "background": torch.tanh(s(13)),
        "cx": torch.zeros_like(s(0)),
        "cy": torch.zeros_like(s(0)),
    }


# ===========================================================================
# forward model
# ===========================================================================

def sample_correction(C, bx, by, half_extent):
    """Bilinearly read the correction map at the ray-shot positions beta.

    grid_sample with align_corners=False treats -1 and +1 as the OUTER EDGES of
    the border pixels, while our map spans [-half, +half] to pixel CENTRES, so
    the coordinates are divided by half*(1 + 1/(n-1)). Getting this wrong shifts
    the correction by half a pixel relative to the Sersic.

    padding_mode='zeros': outside the map the correction is zero, i.e. the source
    reverts to the pure Sersic. That is the intended behaviour.
    """
    n = C.shape[-1]
    edge = half_extent * (1.0 + 1.0 / (n - 1))
    grid = torch.stack([bx / edge, by / edge], dim=-1)        # (B,H,W,2)
    return F.grid_sample(C.unsqueeze(1), grid, mode="bilinear",
                         padding_mode="zeros", align_corners=False)[:, 0]


def ray_coverage(bx, by, n_c, half_extent, smooth: int = 3):
    """How many image sub-pixels land in each source pixel = the magnification.

    Nearest-pixel histogram of the ray landing points, per image in the batch.
    DETACHED on purpose: this is a specification of the prior (where is the
    source well sampled?), not a quantity we want gradients to flow through.
    Letting it be differentiable would let the network lower its own penalty by
    moving the lens, which is not a physical incentive.

    RAYS THAT LAND OUTSIDE THE MAP ARE DISCARDED, NOT CLAMPED. Only about a
    third of the image-plane rays land inside a 0.8 arcsec half-width source
    map; clamping the rest onto the border pixels made the EDGE of the map the
    highest-coverage region, which inverted the whole point of the weighting --
    the network would have been least penalised exactly where the data say
    nothing. tests/test_pathb.py check 3 is what caught this.
    """
    B = bx.shape[0]
    scale = 2.0 * half_extent / (n_c - 1)
    with torch.no_grad():
        fj = (bx.detach() + half_extent) / scale
        fi = (by.detach() + half_extent) / scale
        inside = ((fj >= -0.5) & (fj <= n_c - 0.5) &
                  (fi >= -0.5) & (fi <= n_c - 0.5))
        j = torch.round(fj).long().clamp(0, n_c - 1)
        i = torch.round(fi).long().clamp(0, n_c - 1)
        flat = (i * n_c + j).reshape(B, -1)
        cov = torch.zeros(B, n_c * n_c, device=bx.device, dtype=bx.dtype)
        cov.scatter_add_(1, flat, inside.reshape(B, -1).to(bx.dtype))
        cov = cov.view(B, 1, n_c, n_c)

        # VARIANCE REDUCTION. The histogram is a Monte-Carlo estimate of mu from
        # a finite number of rays. At supersample = 1 there are ~1-2 rays per
        # source pixel, so the raw count is mostly shot noise -- a checkerboard
        # of 0s and 1s that has nothing to do with the smooth underlying
        # magnification field. Weighting by that noise would randomise the
        # penalty pixel to pixel. A 3x3 box average keeps the large-scale
        # structure (which is what mu actually is) and removes the aliasing.
        # It does NOT create information: the fix for genuinely low coverage is
        # --supersample 2, and the startup banner reports whether you need it.
        if smooth and smooth > 1:
            pad = smooth // 2
            cov = F.avg_pool2d(F.pad(cov, (pad,) * 4, mode="replicate"),
                               smooth, stride=1)
    return cov[:, 0]


def reg_weights(cov, mode: str, power: float = 0.5, clip: float = 5.0):
    """w = (median coverage / coverage)^power, clipped. Uniform mode returns 1.

    power = 0.5 rather than 1.0 because the raw coverage spans ~1000x across the
    source plane; the full 1/mu weighting leaves the thin high-magnification
    strip beside the caustic effectively unpenalised, and in the Option 2
    experiment that dropped source correlation from 0.96 to 0.74.
    """
    if mode == "uniform":
        return torch.ones_like(cov)
    B = cov.shape[0]
    flat = cov.view(B, -1)
    pos = torch.where(flat > 0, flat, torch.full_like(flat, float("nan")))
    med = torch.nanmedian(pos, dim=1, keepdim=True).values
    med = torch.nan_to_num(med, nan=1.0).clamp_min(1e-6)
    w = torch.pow(med / flat.clamp_min(1e-3 * med), power)
    return w.clamp(1.0 / clip, clip).view_as(cov)


def render_batch(p, C, grid, psf, n_pix, supersample, half_extent,
                 corr_scale: float, source_mode: str = "b3"):
    """Full forward model. Returns (pred, the correction actually applied, betas).

    TWO SOURCE MODELS, one code path, so B2 and B3 are compared on exactly the
    same physics, optimiser and data:

    b3  source = Sersic(7 params) + corr_scale * amp * tanh(C)
        The correction is bounded by construction, so the network cannot delete
        the Sersic and rebuild the source out of free pixels. It scales with the
        source brightness, which makes corr_scale dimensionless and transferable
        between images of very different flux.

    b2  source = amp * softplus(C)          -- NO Sersic at all
        The free-form / "Anirudh-style" model: the network emits the whole
        source as an n_c x n_c image. softplus enforces non-negativity (surface
        brightness cannot be negative) and there is no bound, so this mode needs
        the smoothness penalty in `curvature()` to be well posed. The 7 Sersic
        parameters are still predicted and still returned -- they are simply not
        used to build the source -- so the parameter-recovery table stays
        comparable. In b2 the source-shape parameters have no gradient and will
        read as collapsed; that is expected, not a bug.
    """
    X, Y = grid
    B = p["theta_E"].shape[0]
    b = lambda t: t.reshape(B, 1, 1)
    lens = {k: b(p[k]) for k in LENS_KEYS}

    ax, ay = deflection(X[None], Y[None], lens)
    bx, by = X[None] - ax, Y[None] - ay                       # ray shooting

    if source_mode == "b2":
        C_eff = b(p["amp"]) * F.softplus(C)
        sky = sample_correction(C_eff, bx, by, half_extent)
    else:
        src = {k: b(p[k]) for k in SRC_KEYS}
        sky = SersicSource(src).at(bx, by)
        C_eff = corr_scale * b(p["amp"]) * torch.tanh(C)
        sky = sky + sample_correction(C_eff, bx, by, half_extent)

    sky = sky.unsqueeze(1)
    pad = psf.shape[-1] // 2
    sky = F.conv2d(F.pad(sky, (pad, pad, pad, pad), mode="replicate"),
                   psf[None, None])
    pred = area_downsample(sky, supersample)[:, 0] + b(p["background"])
    return pred, C_eff, (bx, by)


def curvature(C):
    """Discrete Laplacian energy of the source map, per pixel.

    The standard source-plane regulariser in lens modelling (Suyu et al. 2006):
    it penalises second differences, so a smooth source is free and a speckled
    one is not. In b3 the Sersic already supplies smoothness and this is a minor
    term; in b2 it is the ONLY thing keeping the problem well posed, and without
    it a free source grid reproduces the speckle of Option2_failed exactly.
    """
    lap = (C[:, :-2, 1:-1] + C[:, 2:, 1:-1] +
           C[:, 1:-1, :-2] + C[:, 1:-1, 2:] - 4.0 * C[:, 1:-1, 1:-1])
    return (lap ** 2).mean()


# ===========================================================================

RING_FEATURES = ("theta_E", "m1_abs", "m2_abs", "ring_completeness")


def ring_summary(img, pixel_scale):
    """Four numbers read off the ring geometry of the image itself.

    ALL FOUR ARE MEASURED FROM THE PIXELS. Nothing here touches the manifest --
    this is theta_e_init.initial_guess(), the same function fit_per_image.py uses
    for its starting point. Feeding them is feature engineering, not leakage.

    Only ROTATION-INVARIANT quantities are used (magnitudes, not vectors), so
    they survive the dihedral augmentation unchanged and do not have to be
    transformed alongside the image.

    Why bother, when a CNN 'should' learn these itself: measured on 400 val
    images, the ring dipole |m1| predicts the true source offset at Spearman
    +0.875 and the quadrupole |m2| predicts the true |e| at +0.512, while the
    v1 network -- which was handed only theta_E -- returned +0.228 and +0.313.
    It reproduced almost exactly the one number it was given and learned little
    else from the pixels. eval_pathb.py now prints this ring-only baseline next
    to the network, so 'did the network add anything?' is answerable.
    """
    q = initial_guess(img, pixel_scale)
    return [q["theta_E"], float(np.hypot(q["sx"], q["sy"])),
            abs(q["m2_over_theta_E"]), q["ring_completeness"]]


def pack_split(ds_root, split, classes, n, target, pixel_scale):
    ds = ModelADataset(ds_root, split=split, classes=classes, limit=n)
    imgs, sigs, feat = [], [], []
    for i in range(len(ds)):
        im = ds.image(i) if target == "image" else ds.image_nss(i)
        imgs.append(im)
        sigs.append(ds.sigma(i, im))
        feat.append(ring_summary(im, pixel_scale))
    return (torch.as_tensor(np.array(imgs), dtype=torch.float32),
            torch.as_tensor(np.array(sigs), dtype=torch.float32),
            torch.as_tensor(np.array(feat), dtype=torch.float32))


def build_input(X, S, R, stretch: bool = True):
    """(B,H,W) image + (B,) sigma + (B,4) ring features -> (B,5,H,W).

    Channel 0 is arcsinh(X / sigma_bg), NOT X / sigma_bg.

    The raw signal-to-noise peak varies by 260x between images (p10 251, max
    65,087). A convolutional filter bank is shared across all of them, so the
    same physical feature -- an arc edge, say -- arrives at wildly different
    activation scales and the network cannot use one filter for both. arcsinh is
    the standard astronomical stretch: linear near zero, so the noise scale is
    preserved and a 1-sigma fluctuation still reads as 1, and logarithmic in the
    wings. It compresses the across-image range from 260x to 1.9x.

    Channels 1-4 are the ring features, broadcast as constant planes. Constant
    planes are a crude but effective way to give a fully-convolutional trunk
    access to a global scalar.
    """
    x = X / S[:, None, None]
    if stretch:
        x = torch.asinh(x)
    ones = torch.ones_like(x)
    ch = [x] + [R[:, k, None, None] * ones for k in range(R.shape[1])]
    return torch.stack(ch, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--classes", nargs="+", default=["axion", "cdm", "wdm"])
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--target", default="image", choices=["image", "image_nss"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pixel-scale", type=float, default=0.10593)
    ap.add_argument("--psf-path", default="superres/results/psf_empirical.npy")
    ap.add_argument("--supersample", type=int, default=1)
    ap.add_argument("--fit-radius-px", type=float, default=45.0)
    ap.add_argument("--n-c", type=int, default=32, help="correction map side")
    ap.add_argument("--half-extent", type=float, default=0.8,
                    help="source-plane half-width in arcsec. Together with --n-c "
                         "this sets the super-resolution factor; see the note "
                         "printed at startup.")
    ap.add_argument("--max-gain", type=float, default=3.1,
                    help="measured median tangential stretch from "
                         "magnification_extract.py. The source pixel should not "
                         "be finer than the detector pixel divided by this.")
    ap.add_argument("--corr-scale", type=float, default=0.30,
                    help="max correction as a fraction of the Sersic amplitude "
                         "(b3 only). This is a SAFETY BOUND, not an operating "
                         "point -- if the reported saturation is high, the "
                         "penalty is too weak, not the bound too tight.")
    ap.add_argument("--source-mode", default="b3", choices=["b3", "b2"],
                    help="b3 = Sersic + bounded correction; "
                         "b2 = free-form source only, no Sersic")
    ap.add_argument("--reg-mode", default="mu", choices=["mu", "uniform", "none"])
    ap.add_argument("--reg-power", type=float, default=0.5)
    ap.add_argument("--lambda-corr", type=float, default=3.0,
                    help="correction penalty, as a FRACTION OF chi^2 (the code "
                         "multiplies by chi2.detach(), so this is scale-free). "
                         "At full saturation the b3 penalty is "
                         "lambda_corr * corr_scale^2 = 3 * 0.09 = 27% of chi^2.")
    ap.add_argument("--lambda-curv", type=float, default=0.0,
                    help="Laplacian smoothness penalty on the source map, also "
                         "as a fraction of chi^2. Optional in b3; ESSENTIAL in "
                         "b2, where 3.0 is a sane starting value.")
    ap.add_argument("--sigma-floor", type=float, default=0.02,
                    help="fractional systematic error floor: "
                         "sigma_eff^2 = sigma_bg^2 + (f * model)^2. "
                         "THE SINGLE MOST IMPORTANT FLAG IN THIS FILE -- see the "
                         "note in the header. 0 reproduces the broken v1 loss.")
    ap.add_argument("--augment", type=int, default=1,
                    help="random 90-degree rotations and flips of the training "
                         "images. Free and exactly valid here, because the loss "
                         "is self-supervised: a rotated image is a legitimate "
                         "member of the data distribution and needs no label "
                         "transformation.")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--stretch", type=int, default=1,
                    help="arcsinh-stretch the input image channel. The raw "
                         "S/N peak varies 260x between images, which a shared "
                         "filter bank cannot absorb; arcsinh cuts that to 1.9x.")
    ap.add_argument("--warmup-epochs", type=int, default=5,
                    help="epochs with the correction switched OFF, so the "
                         "parametric part converges first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="superres/results/pathb.pt")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    psf_np = load_psf(a.psf_path) if os.path.exists(a.psf_path) \
        else load_psf("sie_pipeline/results/psf_empirical.npy")
    psf = torch.as_tensor(psf_np, dtype=torch.float32, device=dev)

    print("loading data ...", flush=True)
    Xtr, Str, Rtr = pack_split(a.root, "train", a.classes, a.n_train,
                               a.target, a.pixel_scale)
    Xva, Sva, Rva = pack_split(a.root, "val", a.classes, a.n_val,
                               a.target, a.pixel_scale)
    n_pix = Xtr.shape[-1]
    gx, gy = image_plane_grid(n_pix, a.pixel_scale, a.supersample)
    grid = (torch.as_tensor(gx, dtype=torch.float32, device=dev),
            torch.as_tensor(gy, dtype=torch.float32, device=dev))
    c = (n_pix - 1) / 2.0
    yy, xx = np.indices((n_pix, n_pix))
    mask = torch.as_tensor(np.hypot(yy - c, xx - c) <= a.fit_radius_px, device=dev)

    net = PathBNet(n_c=a.n_c, in_ch=1 + len(RING_FEATURES)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr,
                           weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    n_par = sum(q.numel() for q in net.parameters())

    src_scale = 2.0 * a.half_extent / (a.n_c - 1)
    gain = a.pixel_scale / src_scale
    print(f"device {dev}   train {len(Xtr)}   val {len(Xva)}   weights {n_par:,}")
    print(f"correction {a.n_c}^2 = {a.n_c**2} px at {src_scale:.4f} arcsec/px "
          f"({gain:.2f}x finer than the detector)")
    print(f"unknowns per image: 14 + {a.n_c**2} = {14+a.n_c**2} "
          f"vs ~4300 informative data pixels "
          f"({4300/(14+a.n_c**2):.1f}:1 overdetermined)")
    # The super-resolution factor is not a free choice: the lens can only
    # deliver detail up to its own stretch factor. magnification_extract.py
    # measures that (median tangential 1/|1-kappa-|gamma||). Asking for a finer
    # source grid than the lens supports is interpolation dressed up as physics.
    print(f"super-resolution factor {gain:.2f}x against a MEASURED median "
          f"tangential stretch of {a.max_gain:.2f}x")
    if gain > a.max_gain:
        print("  *** WARNING: the requested source grid is FINER than the lens can")
        print("  *** support. Reduce --n-c or raise --half-extent, or the network")
        print("  *** will be inventing structure the data cannot constrain.")

    # Rays per source pixel. Roughly a third of image-plane rays land inside a
    # 0.8 arcsec source map (measured), and that fraction is what decides
    # whether the magnification weights are a signal or shot noise.
    rays = (n_pix * a.supersample) ** 2
    per_px = 0.34 * rays / (a.n_c ** 2)
    print(f"rays per source pixel ~ {per_px:.1f} "
          f"({rays:,} rays, ~34% land inside the map, {a.n_c**2} source pixels)")
    if per_px < 4 and a.reg_mode == "mu":
        print("  *** Below ~4 rays/pixel the coverage histogram is mostly shot")
        print("  *** noise. ray_coverage() 3x3-smooths it, which helps, but")
        print("  *** --supersample 2 is the real fix (4x the rays, ~4x the cost).")
    print(f"source mode {a.source_mode}   reg {a.reg_mode}   "
          f"lambda_corr {a.lambda_corr} x chi2   lambda_curv {a.lambda_curv} x chi2")
    print(f"sigma floor {a.sigma_floor:.3f} (fractional systematic)   "
          f"augment {bool(a.augment)}   weight decay {a.weight_decay}   "
          f"warmup {a.warmup_epochs} epochs\n")
    if a.sigma_floor <= 0:
        print("  *** sigma-floor = 0 reproduces the v1 loss, in which one image")
        print("  *** in a batch of 16 took 54% of the gradient. Do not use it")
        print("  *** except to reproduce the failure.\n")

    n_skip = [0]
    hist = []

    def augment_batch(X):
        """Random dihedral transform of the images. Valid because the objective
        is self-supervised -- the network predicts the lens of whatever image it
        is shown, and chi^2 is taken against that same image. Nothing has to be
        relabelled. n_pix is odd, so the grid centre is a pixel centre and the
        rotation is exact."""
        k = int(torch.randint(0, 4, (1,)))
        if k:
            X = torch.rot90(X, k, dims=(-2, -1))
        if int(torch.randint(0, 2, (1,))):
            X = torch.flip(X, dims=(-1,))
        return X.contiguous()

    def step(X, S, R, train, use_corr):
        X, S, R = X.to(dev), S.to(dev), R.to(dev)
        if train and a.augment:
            X = augment_batch(X)          # ring features are rotation-invariant
        inp = build_input(X, S, R, stretch=bool(a.stretch))
        z, Craw = net(inp)
        p = to_params(z, R)               # ring vector: tE, |m1|, |m2|, comp
        if not use_corr:
            Craw = torch.zeros_like(Craw)
        pred, C_eff, (bx, by) = render_batch(
            p, Craw, grid, psf, n_pix, a.supersample, a.half_extent,
            a.corr_scale, a.source_mode)

        # ---- the error model -------------------------------------------
        # sigma_bg alone says the ONLY uncertainty is background noise. It is
        # not: calibrate_psf.py measures the PSF wings to be wrong at the
        # ~1-3% level, and that error scales with the model flux. Without the
        # floor, a bright arc gets residuals of 10^3-10^4 sigma while a faint
        # one gets ~10, per-image chi^2 spans 300x, and the batch gradient is
        # owned by whichever image happens to be brightest -- measured: one
        # image out of 16 took 54% of the gradient, so the effective batch size
        # was 2. Adding a fractional floor is the standard way to admit a
        # multiplicative systematic, and on the per-image fits it flattens the
        # chi^2-vs-SNR trend from 1411 -> 8455 down to 329 -> 229 and cuts the
        # p90/p10 spread from 95x to 8.7x.
        sig = S[:, None, None]
        if a.sigma_floor > 0:
            sig = torch.sqrt(sig ** 2 +
                             (a.sigma_floor * pred.detach().clamp_min(0.0)) ** 2)
        r = ((pred - X) / sig)[:, mask]
        chi2 = (r ** 2).mean()

        reg = torch.zeros((), device=dev)
        curv = torch.zeros((), device=dev)
        sat = torch.zeros((), device=dev)
        if use_corr:
            amp = p["amp"].reshape(-1, 1, 1).clamp_min(1e-6)
            rel = C_eff / amp
            if a.reg_mode != "none":
                cov = ray_coverage(bx, by, a.n_c, a.half_extent)
                w = reg_weights(cov, a.reg_mode, a.reg_power)
                reg = (w * rel ** 2).mean()
            if a.lambda_curv > 0:
                curv = curvature(rel)
            if a.source_mode == "b3":
                # fraction of the map sitting within 1% of the tanh bound. If
                # this is not small the penalty is too weak and the correction
                # has stopped receiving gradient -- the exact failure of the
                # first 40-epoch run (75% saturated, corr(|C|,log mu) = -0.03).
                sat = (rel.abs() > 0.99 * a.corr_scale).to(rel.dtype).mean()

        # lambda is expressed as a FRACTION OF chi^2 and multiplied by a
        # detached chi^2, so it is scale-free. In v1, lambda = 1.0 against a
        # chi^2 of ~10^5 meant the penalty was six orders of magnitude too small
        # to do anything, which is why the mu and uniform runs came out
        # identical (Pearson +0.99 on every parameter).
        scale = chi2.detach().clamp_min(1e-12)
        loss = chi2 + scale * (a.lambda_corr * reg + a.lambda_curv * curv)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            bad = [n for n, q in net.named_parameters()
                   if q.grad is not None and not torch.isfinite(q.grad).all()]
            if bad or not torch.isfinite(loss):
                n_skip[0] += 1
                if n_skip[0] <= 5:
                    print(f"   [skip] non-finite loss/grad; loss={float(loss)}; "
                          f"first bad: {bad[:3]}", flush=True)
                opt.zero_grad(set_to_none=True)
                return float("nan"), float("nan")
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        return (float(chi2.detach()), float(reg.detach()), float(sat.detach()))

    best = float("inf")
    for ep in range(a.epochs):
        use_corr = ep >= a.warmup_epochs
        net.train()
        perm = torch.randperm(len(Xtr))
        tc, tr_, ts, nb = 0.0, 0.0, 0.0, 0
        for k in range(0, len(Xtr), a.batch_size):
            j = perm[k:k + a.batch_size]
            ch, rg, st = step(Xtr[j], Str[j], Rtr[j], True, use_corr)
            if np.isfinite(ch):
                tc += ch; tr_ += rg; ts += st; nb += 1
        net.eval()
        vc, vn = 0.0, 0
        for k in range(0, len(Xva), a.batch_size):
            sl = slice(k, k + a.batch_size)
            with torch.enable_grad():          # deflection needs grad-capable coords
                ch, _, _ = step(Xva[sl], Sva[sl], Rva[sl], False, use_corr)
            if np.isfinite(ch):
                vc += ch; vn += 1
        sched.step()
        row = {"epoch": ep + 1, "train_chi2": tc / max(nb, 1),
               "train_reg": tr_ / max(nb, 1), "val_chi2": vc / max(vn, 1),
               "saturation": ts / max(nb, 1),
               "correction_on": bool(use_corr), "lr": sched.get_last_lr()[0]}
        hist.append(row)
        tag = "  (correction ON)" if use_corr and ep == a.warmup_epochs else ""
        # train/val gap and saturation are printed every epoch because they are
        # the two numbers that told us the first run had failed, and neither was
        # visible until the run was over.
        print(f"epoch {ep+1:3d}  train {row['train_chi2']:10.2f}"
              f"  val {row['val_chi2']:10.2f}"
              f"  gap {row['val_chi2']/max(row['train_chi2'],1e-9):5.2f}x"
              f"  reg {row['train_reg']:8.4f}"
              f"  sat {100*row['saturation']:5.1f}%{tag}", flush=True)

        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        blob = {"model": net.state_dict(), "args": vars(a),
                "history": hist, "n_skipped": n_skip[0]}
        torch.save(blob, a.out)
        if row["val_chi2"] < best:            # keep the best-val checkpoint too
            best = row["val_chi2"]
            torch.save(blob, a.out.replace(".pt", "_best.pt"))

    print(f"\nwrote {a.out}   (skipped batches: {n_skip[0]})")
    print(f"best val chi2 {best:.2f} -> {a.out.replace('.pt', '_best.pt')}")
    fin = hist[-1]
    if a.source_mode == "b3" and fin["saturation"] > 0.25:
        print(f"\n  *** {100*fin['saturation']:.0f}% of the correction map is at the")
        print("  *** tanh bound. A saturated tanh has ~zero gradient, so the")
        print("  *** correction has stopped learning AND the penalty can no longer")
        print("  *** shape it -- the magnification weighting is inert. Raise")
        print("  *** --lambda-corr (try 10) or lengthen --warmup-epochs so the")
        print("  *** parametric part converges before the correction is released.")
    if fin["val_chi2"] > 2.0 * fin["train_chi2"]:
        print(f"\n  *** val/train = {fin['val_chi2']/fin['train_chi2']:.1f}x: overfitting.")
        print("  *** Raise --weight-decay, keep --augment 1, or use fewer epochs.")
    print("\nnow run:  python superres/eval_pathb.py --ckpt "
          + a.out.replace(".pt", "_best.pt") + " --root .")


if __name__ == "__main__":
    main()
