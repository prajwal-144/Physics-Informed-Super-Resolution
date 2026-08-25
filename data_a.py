"""
data_a.py -- self-contained Model_A reader.

Deliberately does NOT import data_manifest.py. That file is used by every
existing SIS script and must not change; this one adds three things it does not
have (an ellipticity filter for the control experiment, a per-image noise
estimate, and raw rather than min-max-normalised images) and keeping them
separate means no existing run can be perturbed.

WHY RAW IMAGES INSTEAD OF MIN-MAX NORMALISED
--------------------------------------------
`data_manifest.ManifestDataset.__getitem__` returns `(a - min) / (max - min)`.
That was fine while the objective was a weighted MSE, but it breaks a chi^2:
min-max normalisation is a per-image affine map with a scale set by the single
brightest and single faintest pixel -- both noise-dominated at Model_A's median
snr_max of 10 -- so the noise level after normalisation is itself a random
variable. PROJECT_REPORT.md section 7.3 shows the damage: because
`mean(obs^2)` includes the induced pedestal, `skill` ends up anti-correlated
with SNR at Spearman -0.85, and a 3-px blur of the input scores 0.93 while the
exact physical model scores 0.63.

Here the image is kept in native units and the model carries a free amplitude
and a free background, so the affine freedom lives in the model where it can be
fitted, not in the data where it corrupts the noise scale.

NOISE ESTIMATE
--------------
sigma is the standard deviation of the background annulus r > `bg_radius_px`.
Model_A ships `snr_max` but no noise map. The arcs sit at r < 30 px on a 127 px
grid, so r > 50 px is source-free; measured this way the background sigma is
~0.032 in native units, and `image` vs `image_nss` differ there by ~sqrt(2)
times that, confirming the two carry independent noise realisations and that the
estimate is picking up real detector noise rather than residual signal.

TRUTH IS FOR EVALUATION ONLY
----------------------------
`truth()` and `unlensed()` exist so results can be scored. Nothing in
fit_per_image.py or train_amortised.py may call them. The one place the old
pipeline violated this -- `train_sis_bank.py` line 344,
`t_px = ds.theta_E / args.resolution` used to select the operator -- is exactly
what this rewrite removes.
"""
from __future__ import annotations

import csv
import glob
import itertools
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

__all__ = ["ModelADataset", "SCALAR_KEYS"]

SCALAR_KEYS = ("theta_E", "gamma1_ext", "gamma2_ext", "host_e1", "host_e2",
               "host_slope", "source_x", "source_y", "source_R_sersic",
               "source_n_sersic", "source_e1", "source_e2", "snr_max",
               "z_lens", "z_source", "num_subhalos")


def _scalar(z, key, default=np.nan):
    try:
        return float(np.asarray(z[key]))
    except Exception:
        return default


class ModelADataset:
    """Indexable list of Model_A .npz files with filters and truth access.

    Files are discovered by globbing `<root>/Model_A/<split>/<class>/*.npz` and
    the per-image scalars are read from the archives themselves, so no
    manifest.csv is required. Pass `manifest=` to restrict to the rows of an
    existing manifest instead (the paths must be relative to `root`).

    ORDERING -- CHANGED, AND WHY IT HAD TO BE
    -----------------------------------------
    Files are sorted WITHIN each class and then interleaved round-robin ACROSS
    classes, so any prefix of the list is class-balanced.

    It used to concatenate the per-class globs and `sorted()` the whole thing.
    Path strings sort with the class name as the only differing component, so
    every `.../axion/...` sorted before every `.../cdm/...`, and since there are
    10,000 axion train files and 2,000 axion val files, ANY realistic `limit`
    was exhausted inside the axion block before cdm or wdm was reached. Every
    run that passed `--classes axion cdm wdm` silently received axion only.
    Verified by reading the `path` field of every row of fits_img.json (2000 of
    2000 axion), fits_pb3_mu.json (800 of 800) and fits_pb2.json (800 of 800).

    Round-robin rather than a per-class `limit` because it is balanced at EVERY
    prefix length rather than only at multiples of the class count, and because
    it leaves the filter loop below untouched. Round-robin rather than a seeded
    shuffle because balance is then guaranteed rather than expected: a 60-image
    smoke test cannot come out 25/18/17 without anyone noticing.

    A consequence to respect: this changes the meaning of an index. Result JSONs
    written before this change carry `index` values that refer to the OLD
    ordering, so re-scoring an old JSON under this ordering pairs each fit with
    a different image's truth. It will not raise; it will produce plausible and
    wrong numbers. Every row also stores `path`, so the guard is one assertion:

        assert Path(row["path"]) == ds.paths[row["index"]]

    The old ordering also matched `data_manifest.ManifestDataset` index for
    index, which was a validation bridge during the rewrite (`zero_mse`
    recomputed from the npz files matched `eval_man_tv00/metrics.json` to 2e-7
    relative, against 47.7 for a shuffled control). That correspondence is now
    broken deliberately. Join on `path` instead; it was always the stronger
    identity and it does not depend on two lists being built the same way.
    """

    def __init__(self, root: str = ".", split: str = "val",
                 classes: Optional[List[str]] = None, band: int = 0,
                 limit: Optional[int] = None,
                 max_host_e: Optional[float] = None,
                 min_host_e: Optional[float] = None,
                 min_snr: Optional[float] = None,
                 max_snr: Optional[float] = None,
                 bg_radius_px: float = 50.0,
                 manifest: Optional[str] = None,
                 model: str = "Model_A"):
        self.root = Path(root)
        self.band = int(band)
        self.bg_radius_px = float(bg_radius_px)
        classes = classes or ["axion", "cdm", "wdm"]

        if manifest:
            paths = []
            with open(manifest, newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("model") != model or r.get("split") != split:
                        continue
                    if r.get("class") not in classes:
                        continue
                    paths.append(self.root / r["path"])
            paths.sort()
        else:
            per_class = [sorted(glob.glob(str(self.root / model / split / c / "*.npz")))
                         for c in classes]
            paths = [Path(p) for p in
                     itertools.chain.from_iterable(itertools.zip_longest(*per_class))
                     if p is not None]
        if not paths:
            raise FileNotFoundError(
                f"no .npz under {self.root/model/split} for classes={classes}")

        # ---- read scalars once, then filter ------------------------------
        keep, meta = [], []
        for p in paths:
            with np.load(p, allow_pickle=True) as z:
                m = {k: _scalar(z, k) for k in SCALAR_KEYS}
            m["host_e"] = float(np.hypot(m["host_e1"], m["host_e2"]))
            m["gamma_ext"] = float(np.hypot(m["gamma1_ext"], m["gamma2_ext"]))
            m["beta"] = float(np.hypot(m["source_x"], m["source_y"]))
            if max_host_e is not None and not (m["host_e"] <= max_host_e):
                continue
            if min_host_e is not None and not (m["host_e"] >= min_host_e):
                continue
            if min_snr is not None and not (m["snr_max"] >= min_snr):
                continue
            if max_snr is not None and not (m["snr_max"] <= max_snr):
                continue
            keep.append(p)
            meta.append(m)
            if limit and len(keep) >= limit:
                break
        self.paths, self.meta = keep, meta
        if not self.paths:
            raise ValueError("every image was filtered out; loosen the cuts")

        # Realised per-class counts. Recorded rather than assumed, because the
        # filters below can reject unevenly across classes even when the path
        # list going in was balanced, and because "we used three classes" is a
        # claim that has already been wrong once in this project.
        self.classes = list(classes)
        self.class_counts = Counter(p.parent.name for p in self.paths)

    def __len__(self):
        return len(self.paths)

    # ---- index guard -----------------------------------------------------
    def assert_rows_match(self, rows, what: str = "rows") -> int:
        """Refuse to use a results JSON whose `index` no longer names its file.

        `index` is a position in whatever list this class built when the fit was
        written. The ordering changed (see the class docstring: it used to sort
        the concatenated per-class globs, which silently truncated inside the
        axion class), so any JSON written before that change carries indices
        that now point at different images.

        Nothing downstream would raise on that. evaluate.py would pair each fit
        with another image's truth, train_b4.py would freeze the wrong lens on
        every training image, and both would return plausible, wrong numbers.
        This turns that into a crash.

        Rows are matched on class directory plus filename rather than on the
        full path string, so a JSON written with a different `--root` prefix
        still validates. Separators are normalised first, because these JSONs
        are written on Windows and carry backslashes, which a POSIX `Path` would
        swallow into a single filename component. Rows without both keys are
        skipped and counted.

        Returns the number of rows actually checked.
        """
        def tail(p) -> tuple:
            parts = [q for q in str(p).replace("\\", "/").split("/") if q]
            return tuple(parts[-2:])

        checked, bad = 0, []
        for r in (rows.values() if isinstance(rows, dict) else rows):
            i, p = r.get("index"), r.get("path")
            if i is None or p is None:
                continue
            checked += 1
            if not (0 <= int(i) < len(self.paths)):
                bad.append((i, p, "<index out of range>"))
                continue
            have = self.paths[int(i)]
            if tail(p) != tail(have):
                bad.append((i, str(p), str(have)))
            if len(bad) >= 5:
                break
        if bad:
            lines = "\n".join(f"    index {i}: json says {w}, dataset has {h}"
                              for i, w, h in bad)
            raise RuntimeError(
                f"{what}: index/path mismatch on {len(bad)}+ of {checked} rows.\n"
                f"{lines}\n"
                "  The JSON was written under a different dataset ordering. Re-run\n"
                "  the step that produced it rather than re-scoring it, or the\n"
                "  numbers will be silently wrong. See ModelADataset's docstring.")
        return checked

    # ---- pixel data ------------------------------------------------------
    def _band(self, a: np.ndarray) -> np.ndarray:
        a = np.squeeze(np.asarray(a))
        if a.ndim == 3:
            a = a[min(self.band, a.shape[0] - 1)]
        return a.astype(np.float64)

    def image(self, i: int) -> np.ndarray:
        """The observation, in NATIVE units. No normalisation."""
        with np.load(self.paths[i], allow_pickle=True) as z:
            return self._band(z["image"])

    def image_nss(self, i: int) -> np.ndarray:
        """Same lens and source, substructure removed. For substructure work."""
        with np.load(self.paths[i], allow_pickle=True) as z:
            return self._band(z["image_nss"])

    def sigma(self, i: int, img: Optional[np.ndarray] = None) -> float:
        """Background standard deviation, from an annulus outside the arcs."""
        a = self.image(i) if img is None else img
        n = a.shape[-1]
        c = (n - 1) / 2.0
        yy, xx = np.indices(a.shape[-2:])
        bg = np.hypot(yy - c, xx - c) > self.bg_radius_px
        return float(a[bg].std())

    # ---- truth: EVALUATION ONLY -----------------------------------------
    def truth(self, i: int) -> Dict[str, float]:
        return dict(self.meta[i])

    def unlensed(self, i: int) -> np.ndarray:
        """The true source plane, ALREADY CONVOLVED WITH THE PSF.

        This matters and was a bug in evaluate_sis.py: `source_truth_metrics`
        compared the network's intrinsic (pre-PSF) source against this array,
        which is post-PSF, so the two differed by one PSF width before any model
        error was counted. metrics.source_truth() convolves the model source
        first. Established by fitting the analytic Sersic (manifest parameters)
        against this array over a sweep of trial PSF widths: the residual has a
        clean minimum at FWHM 0.18-0.20 arcsec with nmse 0.003, versus 0.0106
        with no PSF at all.
        """
        with np.load(self.paths[i], allow_pickle=True) as z:
            return self._band(z["unlensed"])

    def kappa_sub(self, i: int) -> np.ndarray:
        with np.load(self.paths[i], allow_pickle=True) as z:
            return self._band(z["kappa_sub"])

    def summary(self) -> str:
        m = self.meta
        g = lambda k: np.array([x[k] for x in m], dtype=float)
        by_class = " ".join(f"{c}={self.class_counts.get(c, 0)}"
                            for c in self.classes)
        return (f"{len(m)} images   [{by_class}]   "
                f"theta_E {np.median(g('theta_E')):.2f}\"   "
                f"|e| {np.median(g('host_e')):.3f}   "
                f"snr_max {np.median(g('snr_max')):.1f}   "
                f"beta {np.median(g('beta')):.3f}\"")
