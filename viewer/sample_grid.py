"""Laying a file's samples out side by side, for the Spatial tab's "all samples" view.

Each sample's cells are moved (in a copy - obsm['spatial'] is left alone) into a grid whose
number of columns makes it about as wide for its height as the plot it's shown in. Each
column is as wide as its widest sample and each row as tall as its tallest, with
PAD_FRACTION of the widest sample between them. Above every row is a band (BAND_FRACTION of
the widest sample) for the samples' names, so a name never covers cells.

Moves are whole steps of the hexagonal lattice when the hexagon width is known, so the
hexagons over the grid hold the same cells as each sample's own hexagons of that width.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: the gap between samples is this fraction of the widest sample
PAD_FRACTION = 1 / 20
#: the band above each row for the samples' names is this fraction of the widest sample
BAND_FRACTION = 1 / 8
#: the plot's width / height when it isn't known yet (before it's first drawn)
DEFAULT_ASPECT = 1.6


@dataclass(frozen=True)
class SampleGrid:
    coords: np.ndarray  # (n_cells, 2): every sample's cells, moved, in the order they're given
    offsets: dict[str, np.ndarray]  # what was added to each sample's coordinates
    # Per sample: (name, x of its middle, y of the middle of the band above it, width of its column)
    labels: list[tuple[str, float, float, float]]
    pad: float  # the gap between samples
    band: float  # the height of the band above each row for the samples' names
    n_cols: int


def extents(samples: list[tuple[str, np.ndarray]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(lowest x, y; width, height) of each sample's cells."""
    lows, sizes = [], []
    for _, coords in samples:
        if len(coords):
            low, high = coords[:, :2].min(axis=0), coords[:, :2].max(axis=0)
        else:
            low = high = np.zeros(2)
        lows.append(low.astype(float))
        sizes.append((high - low).astype(float))
    return lows, sizes


def _spans(sizes, n_cols: int, pad: float, band: float) -> tuple[list[float], list[float]]:
    n = len(sizes)
    col_w = [max((sizes[i][0] for i in range(c, n, n_cols)), default=0.0) for c in range(n_cols)]
    row_h = [max((sizes[i][1] for i in range(r * n_cols, min(n, (r + 1) * n_cols))), default=0.0)
             for r in range(-(-n // n_cols))]
    return col_w, row_h


def best_columns(sizes, aspect: float | None = None) -> int:
    """The number of columns whose grid's width / height is closest to `aspect` (the plot's)."""
    n = len(sizes)
    if n <= 1:
        return 1
    aspect = aspect if aspect and aspect > 0 else DEFAULT_ASPECT
    widest = max(s[0] for s in sizes)
    pad, band = (widest * PAD_FRACTION or 1.0), (widest * BAND_FRACTION or 1.0)

    def misfit(n_cols):
        col_w, row_h = _spans(sizes, n_cols, pad, band)
        width = sum(col_w) + pad * (len(col_w) - 1)
        height = sum(row_h) + (band + pad) * len(row_h) - pad
        return abs(np.log(max(width, 1e-9) / max(height, 1e-9) / aspect))

    return min(range(1, n + 1), key=misfit)


def layout(samples: list[tuple[str, np.ndarray]], hex_width: float | None = None,
           n_cols: int | None = None) -> SampleGrid:
    """The grid of `samples` ((name, (n, 2) coordinates) each, in reading order), in
    `n_cols` columns (best_columns if not given). With `hex_width`, the flat-to-flat width
    of pointy-top hexagons on hexbin's lattice, every move is a whole lattice step
    (multiples of hex_width across, sqrt(3) * hex_width down)."""
    lows, sizes = extents(samples)
    n = len(samples)
    n_cols = max(1, min(n_cols or best_columns(sizes), n))
    widest = max((s[0] for s in sizes), default=0.0)
    pad, band = widest * PAD_FRACTION or 1.0, widest * BAND_FRACTION or 1.0
    col_w, row_h = _spans(sizes, n_cols, pad, band)
    col_x = np.concatenate([[0.0], np.cumsum([w + pad for w in col_w])])
    # Each row starts below its name band (the figure's y axis points down).
    row_y = band + np.concatenate([[0.0], np.cumsum([h + pad + band for h in row_h])])
    # Lattice steps, if small next to the gaps: rounding up stays clear of the name band.
    step = None if not hex_width or hex_width >= pad / 2 else np.array([hex_width, np.sqrt(3) * hex_width])

    offsets, labels, parts = {}, [], []
    for i, (name, coords) in enumerate(samples):
        r, c = divmod(i, n_cols)
        offset = np.array([col_x[c], row_y[r]]) - lows[i]
        if step is not None:
            offset = np.ceil(offset / step) * step
        offsets[name] = offset
        left, top = lows[i] + offset
        labels.append((name, float(left + sizes[i][0] / 2), float(top - band / 2), float(col_w[c])))
        parts.append(coords[:, :2] + offset)
    coords = np.concatenate(parts) if parts else np.zeros((0, 2))
    return SampleGrid(coords, offsets, labels, float(pad), float(band), n_cols)
