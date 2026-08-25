"""
backproject.py -- the source-plane back-projection operator, and the source
sampler that inverts it. Shared by train_b4.py / eval_b4.py.

WHAT BACK-PROJECTION IS
-----------------------
Ray-shoot every image pixel through the lens, beta = theta - alpha(theta), and
deposit that pixel's brightness at beta. Averaging the deposits per source pixel
gives

    b = (L^T d) / (L^T 1)

where L is the lensing operator and 1 is a vector of ones, so L^T 1 counts how
many image pixels landed in each source pixel. This is NOT a reconstruction --
it is a crude, noisy, unevenly-sampled *estimate* of the source, exactly the
`cross_grid_fill(lr_image, backward_mapping)` step in the original pipeline's
train.py.

Its value is that it puts the data into the source plane, in spatial register,
before a convolutional network ever sees it. A conv net given a back-projection
only has to deblur and sharpen a picture that is already roughly correct. A conv
net given the raw lensed image (B2/B3) has to infer the geometry, throw the
image through a global pool, and then hallucinate a spatial map back out of a
128-dimensional summary. The first task is what SRResNet-style architectures are
built for; the second is not.

L^T 1 IS THE MAGNIFICATION
--------------------------
The same count that normalises the back-projection is the ray coverage, i.e. the
magnification of each source pixel up to the constant sub-pixel area. It comes
from the same ray-shooting that renders the image, so it cannot describe a
different lens -- the defect in physics_losses.fixed_sis_magnification. It is
returned alongside the back-projection and used both as a network input channel
and as the weight in the regularisation.

NO GRADIENT FLOWS THROUGH THE BACK-PROJECTION
---------------------------------------------
It is an input feature, built under no_grad. The gradient reaches the network
through the FORWARD model only. That is also what the original pipeline did (its
backward operator was a fixed precomputed sparse matrix), and it means the
nearest-pixel rounding here costs nothing.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from lens_models import deflection

__all__ = ["source_grid", "ray_shoot_batch", "backproject", "sample_source"]

LENS_KEYS = ("theta_E", "gamma", "e1", "e2", "g1", "g2", "cx", "cy")


def source_grid(n: int, half_extent: float):
    """Pixel-centre coordinates of an n x n source grid spanning [-H, +H]."""
    ax = np.linspace(-half_extent, half_extent, n)
    return np.meshgrid(ax, ax, indexing="xy")[0], np.meshgrid(ax, ax, indexing="ij")[0]


def ray_shoot_batch(X, Y, lens: dict):
    """beta = theta - alpha(theta) for a batch of lenses on a shared grid.

    X, Y are (H, W) tensors; lens values are (B,) tensors. Returns (B, H, W).
    """
    B = lens["theta_E"].shape[0]
    p = {k: lens[k].reshape(B, 1, 1) for k in LENS_KEYS if k in lens}
    p.setdefault("cx", torch.zeros(B, 1, 1, dtype=X.dtype, device=X.device))
    p.setdefault("cy", torch.zeros(B, 1, 1, dtype=X.dtype, device=X.device))
    ax, ay = deflection(X[None], Y[None], p)
    return X[None] - ax, Y[None] - ay


def backproject(img, bx, by, n_src: int, half_extent: float, smooth: int = 3):
    """(B,H,W) image + ray landing points -> (B,n_src,n_src) back-projection and
    coverage.

    Rays that land outside the box are DISCARDED, not clamped. Clamping piles
    every far-field ray onto the border and makes the map edge the
    highest-coverage region, which is the opposite of the truth; that bug was
    caught by tests/test_pathb.py check 3 and is not repeated here.

    `smooth` box-averages the coverage only. The coverage is a Monte-Carlo count
    from a finite number of rays and is shot-noise dominated at low
    supersampling; the underlying magnification field is smooth. The
    back-projection itself is NOT smoothed -- that would throw away the
    sub-pixel information the whole method is trying to recover.
    """
    B = img.shape[0]
    scale = 2.0 * half_extent / (n_src - 1)
    with torch.no_grad():
        fj = (bx + half_extent) / scale
        fi = (by + half_extent) / scale
        inside = ((fj >= -0.5) & (fj <= n_src - 0.5) &
                  (fi >= -0.5) & (fi <= n_src - 0.5))
        j = torch.round(fj).long().clamp(0, n_src - 1)
        i = torch.round(fi).long().clamp(0, n_src - 1)
        flat = (i * n_src + j).reshape(B, -1)
        w = inside.reshape(B, -1).to(img.dtype)

        num = torch.zeros(B, n_src * n_src, dtype=img.dtype, device=img.device)
        cov = torch.zeros_like(num)
        num.scatter_add_(1, flat, img.reshape(B, -1) * w)
        cov.scatter_add_(1, flat, w)

        bp = (num / cov.clamp_min(1.0)).view(B, 1, n_src, n_src)
        cv = cov.view(B, 1, n_src, n_src)
        if smooth and smooth > 1:
            pad = smooth // 2
            cv = F.avg_pool2d(F.pad(cv, (pad,) * 4, mode="replicate"),
                              smooth, stride=1)
    return bp[:, 0], cv[:, 0]


def sample_source(S, bx, by, half_extent: float):
    """Bilinearly read a (B,n,n) source map at the ray landing points.

    grid_sample with align_corners=False puts -1 and +1 at the OUTER EDGES of the
    border pixels while our map spans [-H, +H] to pixel CENTRES, so the divisor
    is H*(1 + 1/(n-1)). Getting this wrong shifts the whole source by half a
    pixel, which is 25% of the super-resolution being claimed.

    padding_mode='zeros': outside the box the source is zero. That is a real
    limitation of any boxed free-form source and is why the box size is chosen
    from a measured flux fraction rather than by taste -- see B4.md.
    """
    n = S.shape[-1]
    edge = half_extent * (1.0 + 1.0 / (n - 1))
    grid = torch.stack([bx / edge, by / edge], dim=-1)
    return F.grid_sample(S.unsqueeze(1), grid, mode="bilinear",
                         padding_mode="zeros", align_corners=False)[:, 0]
