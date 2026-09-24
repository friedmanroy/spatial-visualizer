"""Color palette assignment for arbitrary obs fields and gene/pathway RGB blending.

No obs/uns column names are hardcoded here beyond the scanpy convention of looking up
``adata.uns[f"{colname}_colors"]`` for a pre-existing categorical color scheme.
"""
from __future__ import annotations

import numpy as np
from bokeh.palettes import Category20_20, Category20b_20, Turbo256, viridis

GRAY = "#BEBEBE"


def default_categorical_palette(n: int) -> list[str]:
    base = list(Category20_20) + list(Category20b_20)
    if n <= len(base):
        return base[:n]
    idxs = np.linspace(0, 255, n).astype(int)
    return [Turbo256[i] for i in idxs]


def default_sequential_palette(n: int) -> list[str]:
    return list(viridis(max(n, 2)))[:n]


def existing_uns_scheme(adata, colname: str) -> dict[str, str] | None:
    """Look up a pre-existing scanpy-convention color scheme for a categorical column.

    Scanpy/squidpy store this as ``adata.uns[f"{colname}_colors"]``, a list of hex colors
    aligned positionally with ``adata.obs[colname].cat.categories``.
    """
    uns_key = f"{colname}_colors"
    if uns_key not in adata.uns:
        return None
    raw = list(adata.uns[uns_key])
    col = adata.obs[colname]
    if not hasattr(col, "cat"):
        return None
    cat_order = [str(c) for c in col.cat.categories]
    if len(raw) != len(cat_order):
        return None
    return dict(zip(cat_order, raw))


def assign_categorical_colors(adata, colname: str, categories: list[str]) -> dict[str, str]:
    scheme = existing_uns_scheme(adata, colname)
    palette = default_categorical_palette(len(categories))
    colors = {}
    for i, cat in enumerate(categories):
        if scheme is not None and cat in scheme:
            colors[cat] = scheme[cat]
        else:
            colors[cat] = palette[i % len(palette)]
    return colors


def assign_sequential_colors(labels: list[str]) -> dict[str, str]:
    palette = default_sequential_palette(len(labels))
    return dict(zip(labels, palette))


def field_fill_colors(field_spec, active_labels: set[str]) -> np.ndarray:
    """Vectorized per-cell hex color array for a categorical/binned FieldSpec.

    Rows whose category/bin is not in ``active_labels`` (or missing) are gray.
    """
    categories = field_spec.categories
    lut = [field_spec.colors.get(c, GRAY) if c in active_labels else GRAY for c in categories]
    lut.append(GRAY)  # code == -1 (missing) maps here
    lut_arr = np.array(lut, dtype=object)
    codes = field_spec.codes
    idx = np.where(codes < 0, len(categories), codes)
    return lut_arr[idx]


# ---- RGB gene / pathway channel blending (ported from the TNBC viewer's approach) ----

CHANNEL_VECTORS = {
    "R": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 1.0])),  # red <-> cyan
    "G": (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 1.0])),  # green <-> magenta
    "B": (np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.0])),  # blue <-> yellow
}


def compute_rgb_colors(n: int, channels: dict[str, tuple[np.ndarray | None, bool]]) -> np.ndarray:
    """Blend up to three channels ('R', 'G', 'B') of per-cell values into hex colors.

    ``channels`` maps a channel label to (values, bipolar). ``values`` is a length-n
    array or None (channel unused). If ``bipolar`` the values are scaled symmetrically
    around zero (e.g. pathway scores); otherwise linearly between their 1st/99th
    percentiles (e.g. raw gene expression).
    """
    rgb = np.zeros((n, 3))
    for label, (values, bipolar) in channels.items():
        if values is None:
            continue
        primary, complement = CHANNEL_VECTORS[label]
        values = np.nan_to_num(values.astype(float))
        if bipolar:
            absmax = max(float(np.max(np.abs(values))), 1e-9)
            t = values / absmax
            pos = np.clip(t, 0, 1)
            neg = np.clip(-t, 0, 1)
            rgb += pos[:, None] * primary[None, :] + neg[:, None] * complement[None, :]
        else:
            vmin, vmax = np.percentile(values, 1), np.percentile(values, 99)
            if vmax > vmin:
                t = np.clip((values - vmin) / (vmax - vmin), 0, 1)
            else:
                t = np.zeros_like(values)
            rgb += t[:, None] * primary[None, :]
    rgb = np.clip(rgb, 0, 1)
    r = (rgb[:, 0] * 255).astype(int)
    g = (rgb[:, 1] * 255).astype(int)
    b = (rgb[:, 2] * 255).astype(int)
    hexcolors = np.array([f"#{ri:02x}{gi:02x}{bi:02x}" for ri, gi, bi in zip(r, g, b)], dtype=object)
    return hexcolors
