"""
sisr_net.py -- a fully-convolutional super-resolution network for the source
plane.

ARCHITECTURE
------------
SRResNet / ESPCN in shape: a stem convolution, R residual blocks at constant
resolution, a long skip connection, then sub-pixel (PixelShuffle) upsampling by
`mag`, then a projection to one channel through softplus.

This mirrors the architecture in the original DeepLense grid-based work
(`sisr.py` in the repository root, Anirudh Shankar), and is used here for the
same reason it works there: it is fully convolutional, so spatial correspondence
between input and output is preserved end to end.

WHY THAT MATTERS HERE -- THE POINT OF B4
-----------------------------------------
train_pathb.py (B2/B3) pushes the image through

    conv trunk -> AdaptiveAvgPool2d(1) -> 128-vector -> decode a 32x32 map

A global average pool discards position. Everything downstream has to
reconstruct a spatial map from a global summary, which is a hard generative
problem, and measurably the correction it produces only ever reaches 3-4% of the
Sersic. There is no bottleneck anywhere in this file: a source pixel is produced
by a receptive field centred on the corresponding input pixel.

TWO DELIBERATE DEPARTURES FROM THE ORIGINAL
-------------------------------------------
1. GroupNorm instead of BatchNorm. Model_A arc brightness varies by 260x between
   images; BatchNorm mixes those statistics across the batch and makes the
   normalisation of one image depend on which others happen to share its batch.
   GroupNorm is per-sample.
2. softplus instead of ReLU on the output. Surface brightness must be
   non-negative, but ReLU has exactly zero gradient on the negative side, so a
   pixel that starts negative can never recover. softplus is strictly positive
   with a non-zero gradient everywhere.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SourceSISR"]


def _gn(c):
    return nn.GroupNorm(min(8, c), c)


class _Residual(nn.Module):
    """conv-norm-act-conv-norm with an identity skip, resolution preserved."""

    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), _gn(c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1), _gn(c),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.body(x))


class SourceSISR(nn.Module):
    """(B, in_ch, n, n) -> (B, n*mag**n_up, n*mag**n_up), strictly positive.

    in_ch is 2 or 3:
      0  arcsinh(back-projection / sigma)   -- the data, in the source plane
      1  log10(1 + ray coverage)            -- the magnification map
      2  arcsinh(Sersic base / sigma)       -- only when --base sersic
    """

    def __init__(self, in_ch: int = 2, width: int = 64, depth: int = 6,
                 mag: int = 2, n_up: int = 1):
        super().__init__()
        c = width
        self.stem = nn.Sequential(nn.Conv2d(in_ch, c, 3, padding=1), _gn(c), nn.SiLU())
        self.body = nn.Sequential(*[_Residual(c) for _ in range(depth)])
        self.merge = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), _gn(c))
        up = []
        for _ in range(n_up):
            up += [nn.Conv2d(c, c * mag * mag, 3, padding=1),
                   nn.PixelShuffle(mag), _gn(c), nn.SiLU()]
        self.up = nn.Sequential(*up)
        self.head = nn.Conv2d(c, 1, 3, padding=1)

        # small random init on the head, never zeros: a zero-initialised head is
        # what let the first amortised network sit at its neutral output for
        # every epoch (sie_pipeline/train_amortised.py).
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.stem(x)
        h = h + self.merge(self.body(h))          # long skip, as in SRResNet
        return torch.nn.functional.softplus(self.head(self.up(h)))[:, 0]
