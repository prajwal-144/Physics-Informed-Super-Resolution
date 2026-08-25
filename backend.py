"""
backend.py -- a minimal array-namespace shim so the physics runs under BOTH
numpy and torch, from one source file.

WHY
---
The lens and source models must be (a) validated against lenstronomy and against
the Model_A truth arrays, which is easiest in numpy/scipy, and (b) usable inside
a torch training loop, where the lens parameters need gradients. Writing them
twice guarantees the two copies drift apart -- and a silent drift in a deflection
field is precisely the class of bug the operator sign audit already caught once
(diagnostics_2026_08_12/PROJECT_REPORT.md section 3.1: three of five mapping
directories had forward and backward operators with the same sign, invisible in
the loss).

So the physics is written once against the eleven operations below. The numpy
path is the one exercised by tests/ on the machine where these were developed;
the torch path is the one the training loop uses. tests/test_backend_parity.py
runs the identical function under both and requires agreement to 1e-6, so a
divergence fails loudly instead of quietly changing the lens.

USAGE
-----
    from backend import get_backend
    xp = get_backend(x)          # dispatches on the type of x
    r  = xp.sqrt(x * x + y * y)

Only elementwise real arithmetic is abstracted. Anything backend-specific --
autograd, grid_sample, optimisers -- lives in the module that needs it and is
guarded by an explicit import.
"""
from __future__ import annotations

import numpy as np

__all__ = ["get_backend", "NumpyBackend", "TorchBackend", "HAS_TORCH"]

try:                                            # pragma: no cover - env dependent
    import torch as _torch
    HAS_TORCH = True
except Exception:                               # pragma: no cover
    _torch = None
    HAS_TORCH = False


class NumpyBackend:
    name = "numpy"
    sqrt = staticmethod(np.sqrt)
    exp = staticmethod(np.exp)
    cos = staticmethod(np.cos)
    sin = staticmethod(np.sin)
    atan2 = staticmethod(np.arctan2)
    abs = staticmethod(np.abs)
    where = staticmethod(np.where)
    ones_like = staticmethod(np.ones_like)
    zeros_like = staticmethod(np.zeros_like)

    @staticmethod
    def power(x, p):
        return np.power(x, p)

    @staticmethod
    def clip(x, lo=None, hi=None):
        return np.clip(x, lo, hi)

    @staticmethod
    def ndim(x):
        return np.ndim(x)


class TorchBackend:                             # pragma: no cover - needs torch
    name = "torch"

    @staticmethod
    def sqrt(x):
        return _torch.sqrt(x)

    @staticmethod
    def exp(x):
        return _torch.exp(x)

    @staticmethod
    def cos(x):
        return _torch.cos(x)

    @staticmethod
    def sin(x):
        return _torch.sin(x)

    @staticmethod
    def atan2(a, b):
        return _torch.atan2(a, b)

    @staticmethod
    def abs(x):
        return _torch.abs(x)

    @staticmethod
    def where(c, a, b):
        return _torch.where(c, a, b)

    @staticmethod
    def ones_like(x):
        return _torch.ones_like(x)

    @staticmethod
    def zeros_like(x):
        return _torch.zeros_like(x)

    @staticmethod
    def power(x, p):
        return _torch.pow(x, p)

    @staticmethod
    def clip(x, lo=None, hi=None):
        return _torch.clamp(x, min=lo, max=hi)

    @staticmethod
    def ndim(x):
        return x.dim()


def get_backend(x):
    """Dispatch on the array type. Python floats and numpy arrays -> numpy."""
    if HAS_TORCH and isinstance(x, _torch.Tensor):
        return TorchBackend
    return NumpyBackend
