"""Does the forward model reproduce the simulator?

This is the test that matters. Everything downstream -- parameter recovery,
metrics, the amortised network -- is meaningless if render() does not reproduce
image_nss when handed the true parameters. It also measures the supersampling
error rather than assuming O(1/S^2), and checks flux conservation.

WHY THE MACRO LENS IS RENDERED CIRCULAR
---------------------------------------
"The true parameters" is not the same thing as "the recorded parameters".
audit_dataset.py establishes that Model_A's smooth deflector is CIRCULAR: the
stored kappa_nss has quadrupole amplitude m2 = 0.0000 in every split and class,
and is invariant under a 90 degree rotation to 2.4e-7 against a peak of 3.45,
while the substructure-bearing kappa carries 16 per cent azimuthal scatter at
fixed radius. The host_e1 / host_e2 columns were sampled and recorded but were
never applied to the mass model. The external shear WAS applied.

So this test renders with e1 = e2 = 0 and keeps gamma1_ext / gamma2_ext. Feeding
the recorded ellipticity instead does not test render() -- it tests render()
against a lens that does not exist in this dataset, and it is what made the
earlier version of this file report a spurious failure:

    lens ellipticity fed to render()     median   p10      min
    host_e1 / host_e2 (recorded)         0.7793   0.5516   0.3809
    e1 = e2 = 0                          0.9251   0.7766   0.6092
    deflector_e1 / deflector_e2          0.7765   0.5788   0.4973
    e1 = e2 = 0, shear also zeroed       0.8627   0.7301   0.5902

The gain from zeroing e tracks |host_e| (correlation +0.65) and vanishes for
already-round lenses -- 0.947 -> 0.952 at |e| < 0.1 against 0.632 -> 0.899 at
|e| > 0.3 -- which is the signature of an ellipticity that is not in the data
rather than one applied under a different convention. Ruled out separately:
all seven global flip/rotation conventions (identity wins outright), missing
deflector light (a free-amplitude deflector Sersic buys +0.002), and the PSF
(median is flat at 0.80-0.82 across FWHM 0.05-0.45" and Moffat beta 1.2-6).

The recorded-ellipticity arm is therefore kept as a NON-ASSERTING regression
probe. If Model_A is ever regenerated with the ellipticity actually applied,
that arm will overtake the circular one and say so loudly.

KNOWN OPEN GAP
--------------
The circular arm reaches median 0.9251 but p10 0.7766, and even the |e| < 0.1
subset tops out near 0.95 rather than ~0.99, so a second-order term is still
unmodelled. It is NOT detector noise: injecting ds.sigma(i) into image_nss and
correlating it against itself under this weighting gives median 1.0000 and p10
0.9996, because the background noise is negligible inside the arc mask. The
threshold below is set to 0.9 to catch a real regression in render(), not to
certify that the forward model is complete. Do not raise it until that gap is
explained.
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np
from data_a import ModelADataset
from sources import SersicSource
from raytrace import render, image_plane_grid, gaussian_psf, area_downsample

# Repo root = two levels up from tests/ (tests -> sie_pipeline -> Grid_Based_Experiment).
# Anchored to __file__ so the tests do not depend on the working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.environ.get("MODELA_ROOT", _REPO_ROOT)
RES, N = 0.10593, 127
PSF_FWHM = 0.18

def corr(a, b, w=None):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    w = np.ones_like(a) if w is None else np.asarray(w, float).ravel()
    W = w.sum(); a = a - (w*a).sum()/W; b = b - (w*b).sum()/W
    return float((w*a*b).sum()/np.sqrt((w*a*a).sum()*(w*b*b).sum()+1e-30))

def truth_params(t, circular=True):
    """Manifest row -> (lens, source) kwargs.

    `circular=True` is the physical dataset: e1 = e2 = 0, shear retained. See the
    module docstring and audit_dataset.py. `circular=False` reproduces the
    recorded ellipticity and exists only for the regression probe.
    """
    lens = dict(theta_E=t["theta_E"], gamma=t["host_slope"],
                e1=0.0 if circular else t["host_e1"],
                e2=0.0 if circular else t["host_e2"],
                g1=t["gamma1_ext"], g2=t["gamma2_ext"], cx=0.0, cy=0.0)
    src = dict(amp=1.0, R_sersic=t["source_R_sersic"], n_sersic=t["source_n_sersic"],
               se1=t["source_e1"], se2=t["source_e2"], sx=t["source_x"], sy=t["source_y"])
    return lens, src

def main():
    ds = ModelADataset(ROOT, split="val", classes=["axion"], limit=40)
    print(ds.summary())
    grid = image_plane_grid(N, RES, 3)
    psf = gaussian_psf(PSF_FWHM, RES/3)
    cs, cs_amp, cs_rec, host_e = [], [], [], []
    for i in range(len(ds)):
        t = ds.truth(i)
        lens, sp = truth_params(t, circular=True)
        pred = render(SersicSource(sp), lens, N, RES, psf=psf, supersample=3, grid=grid)
        ref = ds.image_nss(i)
        w = (ref > 0.1*ref.max()).astype(float) + 0.02
        cs.append(corr(pred, ref, w))
        # amplitude-matched residual, the honest version
        s = (pred*ref).sum()/max((pred*pred).sum(), 1e-30)
        cs_amp.append(float(np.sqrt(((s*pred-ref)**2).sum()/ (ref**2).sum())))
        # ---- regression probe: the recorded ellipticity, NOT asserted on ----
        lens_r, _ = truth_params(t, circular=False)
        pred_r = render(SersicSource(sp), lens_r, N, RES, psf=psf, supersample=3,
                        grid=grid)
        cs_rec.append(corr(pred_r, ref, w))
        host_e.append(float(np.hypot(t["host_e1"], t["host_e2"])))
    cs = np.array(cs); cs_amp = np.array(cs_amp)
    cs_rec = np.array(cs_rec); host_e = np.array(host_e)
    print(f"\nrender(TRUE params, CIRCULAR macro lens) vs image_nss, n={len(cs)}")
    print(f"   arc-weighted correlation : median {np.median(cs):.4f}   p10 {np.percentile(cs,10):.4f}   min {cs.min():.4f}")
    print(f"   amplitude-matched nrmse  : median {np.median(cs_amp):.4f}")
    print("   (a second-order term is still unmodelled -- see the module")
    print("    docstring; it is not the detector noise floor)")

    # ---- arm B: is the recorded ellipticity still inert? ----------------
    print(f"\nregression probe -- same images rendered with the RECORDED host_e1/e2:")
    print(f"   arc-weighted correlation : median {np.median(cs_rec):.4f}   p10 {np.percentile(cs_rec,10):.4f}   min {cs_rec.min():.4f}")
    for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 1.0)):
        m = (host_e >= lo) & (host_e < hi)
        if m.sum():
            print(f"     |host_e| in [{lo:.1f},{hi:.1f}): n={m.sum():2d}   "
                  f"circular {np.median(cs[m]):.4f}   recorded {np.median(cs_rec[m]):.4f}")
    if np.median(cs_rec) > np.median(cs):
        print("   *** the recorded ellipticity now BEATS the circular lens. Model_A")
        print("       appears to have been regenerated with host_e1/e2 applied to the")
        print("       mass model. audit_dataset.py's conclusion, and the circular")
        print("       assumption in this file and in metrics.py, must be revisited. ***")
    else:
        print("   host_e1/e2 remain inert (circular still wins) -- as audit_dataset.py found")

    # ---- supersampling convergence, measured ---------------------------
    print("\nsupersampling error vs an S=9 reference (same params, noiseless):")
    t = ds.truth(0); lens, sp = truth_params(t)
    ref9 = render(SersicSource(sp), lens, N, RES, psf=gaussian_psf(PSF_FWHM, RES/9),
                  supersample=9)
    for S in (1, 2, 3, 5):
        p = render(SersicSource(sp), lens, N, RES, psf=gaussian_psf(PSF_FWHM, RES/S),
                   supersample=S)
        rel = np.abs(p - ref9).max()/ref9.max()
        dflux = abs(p.sum() - ref9.sum())/ref9.sum()
        print(f"   S={S}   max |dI|/Imax = {rel:.2e}    total flux error = {dflux:.2e}")

    # ---- area_downsample conserves the mean exactly --------------------
    a = np.random.default_rng(0).random((1, 1, 12, 12))
    d = area_downsample(a, 3)
    print(f"\narea_downsample mean preserved: |d.mean - a.mean| = {abs(d.mean()-a.mean()):.2e}")

    ok = np.median(cs) > 0.9 and abs(d.mean()-a.mean()) < 1e-12
    print("\n" + ("PASS" if ok else "*** FAIL ***"))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
