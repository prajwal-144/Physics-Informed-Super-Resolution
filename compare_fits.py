"""
compare_fits.py -- paired comparison of two fit_per_image.py runs.

    python superres/compare_fits.py OLD.json NEW.json [labelA labelB]

Joins the two runs on the npz FILENAME, never on `index` -- data_a.py's own
docstring warns that index means different things across the class-interleave
fix and that a mismatch "will not raise; it will produce plausible and wrong
numbers". Truth is read straight from each npz, so no dataset object, no
filters, and no ordering assumptions are involved.

chi2 values are NOT compared across runs: each is computed on its own noise
model and the two are different scales. Compare |error| against truth instead,
which is noise-model independent.
"""
import json, os, sys, glob
from pathlib import Path
import warnings
import numpy as np
from scipy.stats import spearmanr, wilcoxon

warnings.filterwarnings("ignore")


def fisher_sigma(s1, s2, n):
    """Significance of a Spearman change, via Fisher z. Clipped because a
    correlation of exactly 1 sends arctanh to infinity, which happens on tiny
    samples and never at n = 2000."""
    c = lambda r: float(np.clip(r, -0.99999, 0.99999))
    return (np.arctanh(c(s2)) - np.arctanh(c(s1))) / np.sqrt(2.0 / max(n - 3, 1))

if len(sys.argv) < 3:
    raise SystemExit(__doc__)
A, B = sys.argv[1], sys.argv[2]
LA = sys.argv[3] if len(sys.argv) > 3 else "A"
LB = sys.argv[4] if len(sys.argv) > 4 else "B"

def load(p):
    d = json.load(open(p))
    return {Path(str(r["path"]).replace("\\", "/")).name: r for r in d["rows"]}, d["config"]

a, ca = load(A); b, cb = load(B)
keys = sorted(set(a) & set(b))
print(f"\n{LA}: {len(a)} rows    {LB}: {len(b)} rows    matched on filename: {len(keys)}")
if not keys:
    raise SystemExit("no overlap -- are both runs on the same split and classes?")
diffs = [k for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k)]
print("config differences (anything besides the noise flags and --out "
      "invalidates the comparison):")
for k in diffs:
    print(f"   {k:<18} {LA}={ca.get(k)}   {LB}={cb.get(k)}")

TRUTH = {"theta_E":  lambda z: float(z["theta_E"]),
         "beta":     lambda z: float(np.hypot(z["source_x"], z["source_y"])),
         "R_sersic": lambda z: float(z["source_R_sersic"]),
         "n_sersic": lambda z: float(z["source_n_sersic"]),
         "gamma":    lambda z: float(z["host_slope"]),
         "e":        lambda z: float(np.hypot(z["host_e1"], z["host_e2"])),
         "g":        lambda z: float(np.hypot(z["gamma1_ext"], z["gamma2_ext"]))}
P = list(TRUTH)

_cache = {}
def find(row, name):
    """Resolve the npz, tolerating Windows/POSIX separators."""
    p = str(row["path"]).replace("\\", "/")
    if os.path.exists(p):
        return p
    if not _cache:
        for f in glob.glob(os.path.join("Model_A", "*", "*", "*.npz")):
            _cache[os.path.basename(f)] = f
    return _cache.get(name)

t = {p: [] for p in P}; ea = {p: [] for p in P}; eb = {p: [] for p in P}
fa = {p: [] for p in P}; fb = {p: [] for p in P}
snr, ok, miss = [], [], 0
for k in keys:
    ra, rb = a[k], b[k]
    f = find(ra, k)
    if f is None:
        miss += 1; continue
    try:
        with np.load(f, allow_pickle=True) as z:
            tv = {p: fn(z) for p, fn in TRUTH.items()}
            s = float(z["snr_max"])
    except Exception:
        miss += 1; continue
    for p in P:
        t[p].append(tv[p])
        fa[p].append(ra[p]); fb[p].append(rb[p])
        ea[p].append(abs(ra[p] - tv[p])); eb[p].append(abs(rb[p] - tv[p]))
    snr.append(s); ok.append(k)
n = len(ok)
if miss:
    print(f"\n   {miss} npz could not be located -- run this from the "
          f"Grid_Based_Experiment root")
print(f"   usable pairs: {n}")
if n < 20:
    raise SystemExit("too few pairs to test")
snr = np.array(snr)

print(f"\nPAIRED |error| AGAINST TRUTH        {LA}  ->  {LB}")
print(f"{'param':<10}{'med '+LA:>12}{'med '+LB:>12}{'change':>9}"
      f"{'p90 '+LB:>12}{LB+' closer':>12}{'Wilcoxon p':>13}  verdict")
for p in P:
    x, y = np.array(ea[p]), np.array(eb[p])
    d = y - x; nz = d[d != 0]
    pv = float(wilcoxon(nz).pvalue) if len(nz) > 10 else np.nan
    frac = 100.0 * np.mean(y < x)
    mx, my = np.median(x), np.median(y)
    v = ("BETTER" if frac > 50 else "WORSE") if pv < 1e-3 else "ns"
    print(f"{p:<10}{mx:>12.4f}{my:>12.4f}{100*(my/mx-1):>+8.0f}%"
          f"{np.percentile(y,90):>12.4f}{frac:>11.0f}%{pv:>13.2e}  {v}")

print(f"\nSPEARMAN AGAINST TRUTH")
print(f"{'param':<10}{LA:>10}{LB:>10}{'delta':>10}{'sigma':>9}")
for p in P:
    s1 = spearmanr(fa[p], t[p]).correlation
    s2 = spearmanr(fb[p], t[p]).correlation
    print(f"{p:<10}{s1:>10.3f}{s2:>10.3f}{s2-s1:>+10.3f}"
          f"{fisher_sigma(s1, s2, n):>+9.1f}")

print(f"\nMEDIAN BIAS  (fit - true)/true, per cent")
print(f"{'param':<10}{LA:>10}{LB:>10}")
for p in P:
    tt = np.median(t[p])
    print(f"{p:<10}{100*(np.median(fa[p])/tt-1):>+9.1f}%{100*(np.median(fb[p])/tt-1):>+9.1f}%")

print(f"\nFRACTION OF IMAGES WHERE {LB} IS CLOSER TO TRUTH, BY snr_max")
print(f"{'param':<10}{'0-6':>9}{'6-15':>9}{'>15':>9}   n per bin")
cnt = [int(((snr >= lo) & (snr < hi)).sum()) for lo, hi in ((0,6),(6,15),(15,np.inf))]
for p in P:
    x, y = np.array(ea[p]), np.array(eb[p]); cells = []
    for lo, hi in ((0,6),(6,15),(15,np.inf)):
        m = (snr >= lo) & (snr < hi)
        cells.append(100.0*np.mean(y[m] < x[m]) if m.sum() > 10 else np.nan)
    print(f"{p:<10}" + "".join(f"{c:>8.0f}%" for c in cells) +
          ("   " + str(cnt) if p == P[0] else ""))
print()
