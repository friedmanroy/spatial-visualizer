"""Aggregating very large samples onto a hexagonal lattice, so the browser draws (and
recolors) one hexagon per bin instead of one marker per cell.

A gene's value in a bin comes from its cells' summed counts, log1p'd (normalized first if
chosen) per bin, when the app kept the raw counts (Dataset.bin_gene_values); otherwise the
cells' values are averaged. Pathway scores are computed on the bins' pooled profiles when the
load page pooled the spots into hexagons (Dataset.bin_gene_set_scores), else averaged too. Categorical fields show each bin's most common (selected)
category.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

#: samples with more cells than this are shown hex-binned by default
BINNING_THRESHOLD = 1_000_000
#: the automatic hexagon width tiles the largest sample's bounding rectangle with about this many hexagons
DEFAULT_HEXAGONS = 50_000
#: hexagons with fewer spots than this fraction of their sample's median have too little data
#: to analyse (a tile's rim, fiducial marks); low counts alone can be real tissue (white matter)
DEFAULT_MIN_COVERAGE = 0.5
#: hexagons with fewer summed counts than this have too little data at any hexagon size
DEFAULT_MIN_COUNTS = 15
#: genes must be detected in at least this fraction of hexagons to count as variable: scaling
#: blows up genes seen in a handful of hexagons, and clusters would follow those stray counts
DEFAULT_MIN_GENE_DETECTION = 0.02
#: hexagon widths stay within this many cell spacings
_MIN_WIDTH_FACTOR, _MAX_WIDTH_FACTOR = 1.0, 100.0
#: what the hexagon edge length inputs describe them with
EDGE_LABEL = "Hexagon edge length (pts)"
EDGE_DESCRIPTION = ("How long each of a hexagon's 6 sides is, in pts: the units of the coordinates in "
                    "obsm['spatial']. A hexagon holds more spots / cells the longer they are.")


@dataclass(frozen=True)
class HexFilter:
    """Which hexagons are analysed, and how their summed counts become profiles."""
    min_coverage: float = DEFAULT_MIN_COVERAGE  # fraction of the sample's median spots per hexagon
    min_counts: int = DEFAULT_MIN_COUNTS  # summed counts per hexagon
    normalize: bool = False  # normalize_total before the log1p, which always applies
    min_gene_detection: float = DEFAULT_MIN_GENE_DETECTION  # of the analysed hexagons, for variable genes

    def keep(self, binning: "HexBinning", totals: np.ndarray | None = None) -> np.ndarray:
        """Boolean mask over `binning`'s bins; `totals` (their summed counts) is None when
        the data holds no counts to test."""
        if binning.n_bins == 0:
            return np.zeros(0, dtype=bool)
        keep = binning.counts >= self.min_coverage * np.median(binning.counts)
        return keep if totals is None else keep & (totals >= self.min_counts)


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

    @property
    def sum_matrix(self) -> sp.csr_matrix:
        """(n_bins, n_cells): sum_matrix @ values = per-bin sum."""
        return sp.csr_matrix(sp.diags(self.counts.astype(np.float32)) @ self.mean_matrix)

    def subset(self, keep: np.ndarray) -> tuple[np.ndarray, "HexBinning"]:
        """(mask over the cells of the bins in `keep`, a binning of only those bins and cells)."""
        cells = keep[self.cell_bin]
        new_id = np.cumsum(keep) - 1
        return cells, HexBinning(self.centers[keep], new_id[self.cell_bin[cells]], self.counts[keep], self.width,
                                 sp.csr_matrix(self.mean_matrix[keep][:, np.flatnonzero(cells)]))

    def sum(self, values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Per-bin sum of `values` (one per cell), over only the masked-in cells if given."""
        values = np.nan_to_num(np.asarray(values, dtype=float))
        weights = values if mask is None else np.where(mask, values, 0.0)
        return np.bincount(self.cell_bin, weights=weights, minlength=self.n_bins)

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
        return self.majority_share(codes)[0]

    def majority_share(self, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(most common non-negative code per bin or -1, and the fraction of the bin's
        cells that have it - NaN for bins without any)."""
        valid = codes >= 0
        if not valid.any():
            return np.full(self.n_bins, -1, dtype=np.int32), np.full(self.n_bins, np.nan)
        tally = sp.csr_matrix(
            (np.ones(int(valid.sum())), (self.cell_bin[valid], codes[valid])),
            shape=(self.n_bins, int(codes.max()) + 1),
        )
        result = np.asarray(tally.argmax(axis=1)).ravel().astype(np.int32)
        share = np.asarray(tally.max(axis=1).todense()).ravel() / self.counts
        empty = tally.count_nonzero(axis=1) == 0
        result[empty] = -1
        share[empty] = np.nan
        return result, share


def edge_length(factor: float, spacing: float) -> float:
    """Side length of a hexagon `factor` cell spacings wide (flat to flat), in the units
    of `spacing`; to 4 significant digits, as coordinates can be on any scale."""
    return float(f"{factor * spacing / np.sqrt(3):.4g}")


def width_factor(edge: float, spacing: float) -> float:
    """Hexagon width, in cell spacings (as HexBinning and clusterings take it), giving
    sides `edge` long when cells are `spacing` apart (Lattice.spacing)."""
    factor = edge * np.sqrt(3) / spacing
    return round(float(np.clip(factor, _MIN_WIDTH_FACTOR, _MAX_WIDTH_FACTOR)), 3)


def hexagon_count(extent: tuple[float, float], factor: float, spacing: float) -> float:
    """About how many hexagons `factor` cell spacings wide tile a rectangle of `extent`
    (width, height), each covering sqrt(3)/2 * width^2."""
    return extent[0] * extent[1] / (np.sqrt(3) / 2 * (factor * spacing) ** 2)


def default_width_factor(extent: tuple[float, float], spacing: float) -> float:
    """Hexagon width, in cell spacings, tiling a sample of `extent` with about
    DEFAULT_HEXAGONS hexagons; as near as width_factor's limits allow."""
    area = extent[0] * extent[1]
    if area <= 0:
        return _MIN_WIDTH_FACTOR
    return width_factor(np.sqrt(area / DEFAULT_HEXAGONS / (3 * np.sqrt(3) / 2)), spacing)


def edge_input_range(spacing: float) -> dict:
    """The start, end and step of a hexagon edge length input."""
    start = edge_length(_MIN_WIDTH_FACTOR, spacing)
    return {"start": start, "end": edge_length(_MAX_WIDTH_FACTOR, spacing), "step": float(f"{start:.1g}")}


def _rounded(count: float) -> str:
    """An estimated count to 3 significant digits."""
    return f"{float(f'{count:.3g}'):,.0f}"


def width_note(extent: tuple[float, float], spacing: float, factor: float | None, subject: str = "This sample") -> str:
    """One line for under a hexagon edge length input: the sample's extent, about how many
    hexagons `factor` cell spacings wide tile it, and the automatic edge length."""
    if extent[0] * extent[1] <= 0:
        return ""
    text = f"{subject} spans about {extent[0]:,.4g} x {extent[1]:,.4g} pts"
    if factor:
        text += (f": {edge_length(factor, spacing):g} pts edges make "
                 f"~{_rounded(hexagon_count(extent, factor, spacing))} hexagons")
    auto = default_width_factor(extent, spacing)
    return text + (f" (automatic: {edge_length(auto, spacing):g} pts, ~{_rounded(hexagon_count(extent, auto, spacing))} "
                   "hexagons; estimated as if the tissue filled that rectangle).")


def make_hex_binning(coords: np.ndarray, width: float) -> HexBinning:
    """Assigns every cell to the pointy-top hexagon (flat-to-flat `width`) containing it."""
    cell_bin, counts, centers = hex_cells(coords, width)
    mean_matrix = sp.csr_matrix(
        (1.0 / counts[cell_bin], (cell_bin, np.arange(len(cell_bin)))),
        shape=(len(counts), len(cell_bin)),
    )
    return HexBinning(centers, cell_bin, counts, width, mean_matrix)


def hex_cells(coords: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(each cell's hexagon, the number of cells in each, their centers) - the pointy-top
    hexagons (flat-to-flat `width`) holding at least one cell."""
    size = width / np.sqrt(3)  # circumradius
    x, y = coords[:, 0].astype(float, copy=False), coords[:, 1].astype(float, copy=False)
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
    del x, y, q, r, s, rs, dq, dr, ds, fix_q, fix_r  # each as big as the coordinates: not kept for the sort
    rq, rr = rq.astype(np.int64), rr.astype(np.int64)

    # One sortable number per hexagon, ordered like (rq, rr): sorting pairs is far slower.
    q_min, r_min = rq.min(initial=0), rr.min(initial=0)
    n_r = rr.max(initial=0) - r_min + 1
    keys, cell_bin = np.unique((rq - q_min) * n_r + (rr - r_min), return_inverse=True)
    del rq, rr
    cell_bin = cell_bin.ravel()
    counts = np.bincount(cell_bin, minlength=len(keys))
    unique_q, unique_r = keys // n_r + q_min, keys % n_r + r_min
    centers = np.stack([
        size * np.sqrt(3) * (unique_q + unique_r / 2),
        size * 1.5 * unique_r,
    ], axis=1)
    return cell_bin, counts, centers
