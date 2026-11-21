"""Small gene/pathway-score distribution histograms, rendered as embedded SVGs (vector, so the axis numbers stay sharp at any screen
resolution or zoom, unlike a small PNG stretched to the sidebar's width) -
mirrors the TNBC viewer's per-channel "current view vs whole dataset" histogram.
"""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

WIDTH_PX = 280
HEIGHT_PX = 46
#: the value axis runs from this far (px) from the histogram's left edge to as far from its right
AXIS_INSET_PX = 8.0
_TOTALS_HEIGHT_PX = 70


def _as_img(fig, height: int) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", transparent=True)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%;height:{height}px;display:block" />'


def render_totals_histogram(totals: np.ndarray, cutoff: float) -> str:
    """Hexagons' total counts on a log axis, with a dashed line at `cutoff`."""
    fig, ax = plt.subplots(figsize=(WIDTH_PX / 100, _TOTALS_HEIGHT_PX / 100), dpi=100)
    edges = np.logspace(0, np.log10(max(totals.max(), 10) + 1), 41)
    ax.hist(np.maximum(totals, 1), bins=edges, color="#9e9e9e")
    ax.axvline(max(cutoff, 1), color="#d32f2f", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=10)  # or the low-count tail this exists to show can't be seen
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=7, length=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.2)
    return _as_img(fig, _TOTALS_HEIGHT_PX)


def render_histogram(local_values: np.ndarray, global_values: np.ndarray | None, color: str) -> str:
    """`local_values`' distribution (in `color`), over `global_values`' (gray) if given,
    as density bars on a shared x axis. Drawn as SVG directly rather than with
    matplotlib: these are redrawn on every recolor, and a matplotlib figure takes ~0.1 s."""
    local_values = np.asarray(local_values, dtype=float)
    local_values = local_values[np.isfinite(local_values)]
    if local_values.size == 0:
        return ""
    if global_values is not None:
        global_values = np.asarray(global_values, dtype=float)
        global_values = global_values[np.isfinite(global_values)]
    lo, hi = value_bounds(local_values, global_values)
    bin_edges = np.linspace(lo, hi, 31)
    series = []
    if global_values is not None and global_values.size:
        series.append((np.histogram(global_values, bins=bin_edges, density=True)[0], "#9e9e9e", 0.5))
    series.append((np.histogram(local_values, bins=bin_edges, density=True)[0], color, 0.8))
    top = max(float(h.max()) for h, _, _ in series) or 1.0

    w, h = WIDTH_PX, HEIGHT_PX
    x0, x1, y_top, y_axis = AXIS_INSET_PX, w - AXIS_INSET_PX, 2.0, h - 15.0
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    for heights, fill, alpha in series:
        path = []
        for i in np.flatnonzero(heights > 0):
            left, right = sx(bin_edges[i]), sx(bin_edges[i + 1])
            bar = heights[i] / top * (y_axis - y_top)
            path.append(f"M{left:.2f} {y_axis:.2f}V{y_axis - bar:.2f}H{right:.2f}V{y_axis:.2f}Z")
        parts.append(f'<path d="{"".join(path)}" fill="{fill}" fill-opacity="{alpha}"/>')
    parts.append(f'<path d="M{x0:.2f} {y_axis:.2f}H{x1:.2f}" stroke="#000" stroke-width="0.8"/>')
    ticks = [t for t in MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
             if lo - 1e-9 * (hi - lo) <= t <= hi + 1e-9 * (hi - lo)]
    decimals = _decimals(ticks[1] - ticks[0]) if len(ticks) > 1 else 2
    for t in ticks:
        x = sx(t)
        label = f"{t:.{decimals}f}"
        if label.startswith("-") and float(label) == 0:
            label = label[1:]
        label = label.replace("-", "\u2212")  # a typographic minus, as matplotlib draws it
        parts.append(f'<path d="M{x:.2f} {y_axis:.2f}v3" stroke="#000" stroke-width="0.8"/>'
                     f'<text x="{x:.2f}" y="{h - 3}" font-size="9.7" font-family="DejaVu Sans, Arial, sans-serif" '
                     f'text-anchor="middle">{label}</text>')
    parts.append("</svg>")
    b64 = base64.b64encode("".join(parts).encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%;height:{h}px;display:block" />'


def value_bounds(local_values: np.ndarray, global_values: np.ndarray | None = None) -> tuple[float, float]:
    """(lowest, highest) value on render_histogram's axis: the finite values' range."""
    local_values = np.asarray(local_values, dtype=float)
    parts = [local_values[np.isfinite(local_values)]]
    if global_values is not None:
        global_values = np.asarray(global_values, dtype=float)
        parts.append(global_values[np.isfinite(global_values)])
    values = np.concatenate(parts)
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = float(values.min()), float(values.max())
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def _decimals(step: float) -> int:
    """How many decimals tick labels `step` apart need to show the step exactly."""
    for d in range(7):
        if abs(round(step, d) - step) <= 1e-9 * max(abs(step), 1e-12):
            return d
    return 6
