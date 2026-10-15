"""Aggregating very large samples onto a hexagonal lattice, so the browser draws (and
recolors) one hexagon per bin instead of one marker per cell.

Gene / pathway values are averaged over the cells in each bin; categorical fields show
each bin's most common (selected) category.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

#: samples with more cells than this are shown hex-binned by default
BINNING_THRESHOLD = 1_000_000
#: the default bin width aims for roughly this many bins per sample
TARGET_BINS = 25_000


@dataclass(frozen=True)
class HexBinning:
    """Pointy-top hexagonal bins covering one sample's cells."""
    centers: np.ndarray  # (n_bins, 2) bin centers, in obsm['spatial'] units
    cell_bin: np.ndarray  # bin index of each of the sample's cells
    counts: np.ndarray  # number of cells in each bin
    width: float  # flat-to-flat width of a bin
    mean_matrix: sp.csr_matrix  # (n_bins, n_cells): mean_matrix @ values = per-bin mean

    @property
    def n_bins(self) -> int:
        return len(self.counts)

    def mean(self, values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Per-bin mean of `values` (one per cell). With a boolean `mask`, only the
        masked-in cells are averaged, and bins without any are NaN."""
        values = np.nan_to_num(np.asarray(values, dtype=float))
        if mask is None:
            return self.mean_matrix @ values
        n_in = np.bincount(self.cell_bin, weights=mask.astype(float), minlength=self.n_bins)
        total = np.bincount(self.cell_bin, weights=np.where(mask, values, 0.0), minlength=self.n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n_in > 0, total / n_in, np.nan)

    def majority(self, codes: np.ndarray) -> np.ndarray:
        """Most common non-negative code in each bin (-1 for bins with none)."""
        valid = codes >= 0
        if not valid.any():
            return np.full(self.n_bins, -1, dtype=np.int32)
        tally = sp.csr_matrix(
            (np.ones(int(valid.sum())), (self.cell_bin[valid], codes[valid])),
            shape=(self.n_bins, int(codes.max()) + 1),
        )
        result = np.asarray(tally.argmax(axis=1)).ravel().astype(np.int32)
        result[tally.getnnz(axis=1) == 0] = -1
        return result


def default_width_factor(n_cells: int) -> float:
    """Bin width, in multiples of the cell spacing, giving about TARGET_BINS bins
    (a bin `k` spacings wide holds roughly k^2 cells)."""
    return round(max(1.5, float(np.sqrt(n_cells / TARGET_BINS))), 1)


def make_hex_binning(coords: np.ndarray, width: float) -> HexBinning:
    """Assigns every cell to the pointy-top hexagon (flat-to-flat `width`) containing it."""
    size = width / np.sqrt(3)  # circumradius
    x, y = coords[:, 0].astype(float), coords[:, 1].astype(float)
    # Fractional axial coordinates, then cube rounding to the nearest hexagon.
    q = (np.sqrt(3) / 3 * x - y / 3) / size
    r = (2 / 3 * y) / size
    s = -q - r
    rq, rr, rs = np.round(q), np.round(r), np.round(s)
    dq, dr, ds = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    rq[fix_q] = -rr[fix_q] - rs[fix_q]
    rr[fix_r] = -rq[fix_r] - rs[fix_r]
    axial = np.stack([rq, rr], axis=1).astype(np.int64)

    unique, cell_bin = np.unique(axial, axis=0, return_inverse=True)
    cell_bin = cell_bin.ravel()
    counts = np.bincount(cell_bin, minlength=len(unique))
    centers = np.stack([
        size * np.sqrt(3) * (unique[:, 0] + unique[:, 1] / 2),
        size * 1.5 * unique[:, 1],
    ], axis=1)
    mean_matrix = sp.csr_matrix(
        (1.0 / counts[cell_bin], (cell_bin, np.arange(len(cell_bin)))),
        shape=(len(unique), len(cell_bin)),
    )
    return HexBinning(centers, cell_bin, counts, width, mean_matrix)
