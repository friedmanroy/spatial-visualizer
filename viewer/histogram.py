"""Small gene/pathway-score distribution histograms, rendered as embedded PNGs -
mirrors the TNBC viewer's per-channel "current view vs whole dataset" histogram.
"""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WIDTH_PX = 280
HEIGHT_PX = 46


def render_histogram(local_values: np.ndarray, global_values: np.ndarray | None, color: str) -> str:
    local_values = np.asarray(local_values, dtype=float)
    local_values = local_values[np.isfinite(local_values)]
    if local_values.size == 0:
        return ""

    fig, ax = plt.subplots(figsize=(WIDTH_PX / 100, HEIGHT_PX / 100), dpi=100)
    if global_values is not None:
        global_values = np.asarray(global_values, dtype=float)
        global_values = global_values[np.isfinite(global_values)]
        lo = min(local_values.min(), global_values.min())
        hi = max(local_values.max(), global_values.max())
    else:
        lo, hi = local_values.min(), local_values.max()
    if hi <= lo:
        hi = lo + 1.0
    bin_edges = np.linspace(lo, hi, 31)

    if global_values is not None and global_values.size:
        ax.hist(global_values, bins=bin_edges, color="#9e9e9e", alpha=0.5, density=True, label="all cells")
    ax.hist(local_values, bins=bin_edges, color=color, alpha=0.8, density=True, label="shown")

    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=6, length=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;height:{HEIGHT_PX}px;display:block" />'
