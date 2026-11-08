"""Aggregating very large samples onto a hexagonal lattice, so the browser draws (and
recolors) one hexagon per bin instead of one marker per cell.

A gene's value in a bin comes from its cells' summed counts, normalized per bin, when the
app normalized the data (Dataset.bin_gene_values); otherwise, and for pathway scores, the
cells' values are averaged. Categorical fields show each bin's most common (selected)
category.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

#: samples with more cells than this are shown hex-binned by default
BINNING_THRESHOLD = 1_000_000
#: the automatic hexagon width gives hexagons sides this long, in obsm['spatial'] units
#: (taken to be microns)
DEFAULT_EDGE_MICRONS = 7.0
#: hexagon widths stay within this many cell spacings
_MIN_WIDTH_FACTOR, _MAX_WIDTH_FACTOR = 1.0, 100.0
#: what the hexagon edge length inputs describe them with
EDGE_LABEL = "Hexagon edge length (\u00b5m)"
EDGE_DESCRIPTION = ("How long each of a hexagon's 6 sides is, taking the coordinates to be in \u00b5m. "
                    "A hexagon holds more spots / cells the longer they are.")


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
    of `spacing`."""
    return round(float(factor * spacing / np.sqrt(3)), 2)


def width_factor(edge: float, spacing: float) -> float:
    """Hexagon width, in cell spacings (as HexBinning and clusterings take it), giving
    sides `edge` long when cells are `spacing` apart (Lattice.spacing)."""
    factor = edge * np.sqrt(3) / spacing
    return round(float(np.clip(factor, _MIN_WIDTH_FACTOR, _MAX_WIDTH_FACTOR)), 3)


def default_width_factor(spacing: float) -> float:
    """Hexagon width, in cell spacings, giving sides DEFAULT_EDGE_MICRONS long; never
    under one cell spacing."""
    return width_factor(DEFAULT_EDGE_MICRONS, spacing)


def edge_bounds(spacing: float) -> dict:
    """The start and end of a hexagon edge length input."""
    return {"start": edge_length(_MIN_WIDTH_FACTOR, spacing), "end": edge_length(_MAX_WIDTH_FACTOR, spacing)}


def width_note(width_spots: float, spacing: float, factor: float | None, subject: str = "This sample") -> str:
    """One line for under a hexagon edge length input: how wide the sample is in spots,
    the side length of and how many hexagons `factor` spots wide fit across it, and the
    automatic side length."""
    if width_spots <= 0:
        return ""
    auto = edge_length(default_width_factor(spacing), spacing)
    text = f"{subject} is about {width_spots:,.0f} spots / cells wide"
    if factor:
        text += (f": {edge_length(factor, spacing):.3g} \u00b5m edges make hexagons {factor:g} spots wide, "
                 f"~{width_spots / factor:,.0f} across it")
    return text + f" (automatic: {auto:.3g} \u00b5m, if the coordinates are in \u00b5m)."


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
