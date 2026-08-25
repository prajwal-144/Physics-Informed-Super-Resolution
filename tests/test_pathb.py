"""
test_pathb.py -- the gate to run BEFORE spending hours on train_pathb.py.

Five checks, each aimed at a specific way this can be silently wrong.

  1. TORCH/NUMPY PARITY. With the correction set to zero, render_batch must
     reproduce raytrace.render() -- the numpy forward model that produced
     fits_img.json and was itself validated against lenstronomy. If these two
     disagree the network is fitting a different lens from the one the paper
     describes, and no metric would show it.

  2. CORRECTION ALIGNMENT. Put a single spike in the correction map at a known
     source-plane position and check it is read back at that position. A half-
     pixel error here (grid_sample's align_corners convention) would shift the
     correction relative to the Sersic and show up only as a slightly worse fit.

  3. COVERAGE CONSERVATION. The ray-coverage histogram must account for every
     ray, and must be larger inside the caustic than outside. Coverage IS the
     magnification in this code, so if it is wrong the mu-weighted penalty is
     weighting the wrong thing.

  4. REGULARISATION SIGN. High coverage must give a LOW weight. The sign was
     backwards once already in the Option 2 print-out; a test is cheaper than
     re-reading the code.

  5. THE NaN-AT-ZERO-ELLIPTICITY BUG. Build the graph at exactly e1 = e2 = 0
     and require every gradient to be finite. This is the failure that made the
     previous amortised network emit constants for every image, and it is
     invisible in the forward pass.

Run from the Grid_Based_Experiment root:
    python superres/tests/test_pathb.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

try:
    import torch
except Exception as exc:                                    # pragma: no cover
    raise SystemExit(f"test_pathb.py needs torch (import failed: {exc})")

from raytrace import gaussian_psf, image_plane_grid, render
from sources import SersicSource
from train_pathb import (RING_FEATURES, PathBNet, build_input, ray_coverage,
                         reg_weights, render_batch, sample_correction,
                         to_params)

N_PIX, RES, HALF, N_C = 63, 0.10593, 0.8, 32
LENS = dict(theta_E=1.4, gamma=2.05, e1=0.15, e2=-0.08, g1=0.03, g2=0.01,
            cx=0.0, cy=0.0)
SRC = dict(amp=1.7, R_sersic=0.42, n_sersic=1.6, se1=0.1, se2=0.05,
           sx=0.18, sy=-0.11)
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {detail}")


# --------------------------------------------------------------------------
print("\n1. TORCH/NUMPY PARITY OF THE FORWARD MODEL")
S = 2
gx, gy = image_plane_grid(N_PIX, RES, S)
psf_np = gaussian_psf(0.18, RES / S)
ref = render(SersicSource(SRC), LENS, N_PIX, RES, psf=psf_np, supersample=S,
             grid=(gx, gy), background=0.03)

p = {k: torch.tensor([v], dtype=torch.float64) for k, v in {**LENS, **SRC}.items()}
p["background"] = torch.tensor([0.03], dtype=torch.float64)
grid = (torch.as_tensor(gx), torch.as_tensor(gy))
psf_t = torch.as_tensor(psf_np)
got, _, _ = render_batch(p, torch.zeros(1, N_C, N_C, dtype=torch.float64),
                         grid, psf_t, N_PIX, S, HALF, corr_scale=0.0)
rel = float(np.abs(got[0].numpy() - ref).max() / max(np.abs(ref).max(), 1e-30))
check("render_batch == raytrace.render with zero correction", rel < 1e-9,
      f"max rel diff {rel:.2e}")

# --------------------------------------------------------------------------
print("\n2. CORRECTION-MAP ALIGNMENT")
C = torch.zeros(1, N_C, N_C, dtype=torch.float64)
i0, j0 = 11, 24
C[0, i0, j0] = 1.0
scale = 2.0 * HALF / (N_C - 1)
xq = torch.tensor([[[-HALF + j0 * scale]]], dtype=torch.float64)
yq = torch.tensor([[[-HALF + i0 * scale]]], dtype=torch.float64)
v_on = float(sample_correction(C, xq, yq, HALF)[0, 0, 0])
v_off = float(sample_correction(C, xq + scale, yq, HALF)[0, 0, 0])
check("a spike reads back at its own pixel centre", abs(v_on - 1.0) < 1e-6,
      f"value {v_on:.6f} (expected 1.0)")
check("and is zero one pixel away", abs(v_off) < 1e-6,
      f"value {v_off:.2e}")

# --------------------------------------------------------------------------
print("\n3. RAY COVERAGE = MAGNIFICATION")
bx = torch.as_tensor(gx)[None] - 0.0
by = torch.as_tensor(gy)[None] - 0.0
lens_b = {k: torch.tensor([[[v]]], dtype=torch.float64) for k, v in LENS.items()}
from lens_models import deflection
ax, ay = deflection(torch.as_tensor(gx)[None], torch.as_tensor(gy)[None], lens_b)
bx, by = torch.as_tensor(gx)[None] - ax, torch.as_tensor(gy)[None] - ay
cov = ray_coverage(bx, by, N_C, HALF)
# the histogram accepts a ray up to half a pixel beyond the outermost pixel
# CENTRE, so the reference box is HALF + scale/2, not HALF
edge = HALF + (2.0 * HALF / (N_C - 1)) / 2.0
n_in = int(((bx.abs() <= edge) & (by.abs() <= edge)).sum())
check("only in-range rays are counted, and each exactly once",
      abs(float(cov.sum()) - n_in) < 0.5 * N_C,
      f"cov.sum() {float(cov.sum()):.0f}   in-range {n_in}   total rays {bx.numel()}")
c = cov[0].numpy()
mid = N_C // 2
inner = c[mid - 4:mid + 4, mid - 4:mid + 4].mean()
outer = np.r_[c[:4].ravel(), c[-4:].ravel()].mean()
check("coverage is higher near the caustic than at the map edge", inner > outer,
      f"inner {inner:.1f}  vs  edge {outer:.1f}")

# --------------------------------------------------------------------------
print("\n4. REGULARISATION WEIGHT SIGN")
w = reg_weights(cov, "mu", power=0.5)[0].numpy()
m = c > 0
r = float(np.corrcoef(np.log10(c[m]), w[m])[0, 1])
check("weight DEcreases with coverage (smooth less where mu is high)", r < -0.5,
      f"corr(log coverage, weight) = {r:+.3f}")
wu = reg_weights(cov, "uniform")[0].numpy()
check("uniform mode is exactly 1 everywhere", np.allclose(wu, 1.0))

# --------------------------------------------------------------------------
print("\n5. GRADIENT AT EXACTLY e1 = e2 = 0   (the bug that killed the last run)")
net = PathBNet(n_c=N_C, in_ch=1 + len(RING_FEATURES))
torch.nn.init.zeros_(net.par_head[-1].weight)      # force the old failure mode
torch.nn.init.zeros_(net.par_head[-1].bias)
x = build_input(torch.randn(2, N_PIX, N_PIX).abs(),
                torch.ones(2), torch.rand(2, len(RING_FEATURES)))
z, Craw = net(x)
pp = to_params(z, torch.tensor([[1.3, 0.2, 0.05, 0.9],
                                [1.5, 0.3, 0.02, 0.7]]))
check("ellipticity is EXACTLY zero, i.e. the singular point",
      float(pp["e1"].abs().max()) == 0.0 and float(pp["e2"].abs().max()) == 0.0)
pred, _, _ = render_batch(pp, Craw, (torch.as_tensor(gx).float(),
                                     torch.as_tensor(gy).float()),
                          torch.as_tensor(psf_np).float(), N_PIX, S, HALF, 0.3)
pred.square().mean().backward()
bad = [n for n, q in net.named_parameters()
       if q.grad is not None and not torch.isfinite(q.grad).all()]
check("all gradients finite", not bad, f"non-finite in: {bad[:3]}")
gn = float(torch.sqrt(sum((q.grad ** 2).sum() for q in net.parameters()
                          if q.grad is not None)))
check("gradient is non-zero (the network can actually move)", gn > 0,
      f"||grad|| = {gn:.4e}")

print("\n" + ("PASS -- safe to run train_pathb.py" if ok else "FAIL"))
sys.exit(0 if ok else 1)
