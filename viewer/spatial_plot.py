"""Dense per-cell spatial rendering, in the same visual spirit as the TNBC viewer's
hex-binned look (solid, densely-packed blobs rather than sparse dots).

Purely geometric (driven by the point cloud's own spacing), so it carries over to any
dataset with obsm['spatial'] coordinates without assuming anything about obs schema.

Cells are drawn as data-space circles (a single scalar radius per sample, from the
same nearest-neighbor spacing estimate a true hex-bin would use) rather than
individually-computed hexagon polygons: Bokeh's `Patches` glyph requires each
polygon's vertices as a *ragged* array, which only supports slow per-row JSON
serialization - resending that for every ~10-100k-cell sample on every switch is
the main thing that made switching sample/FOV or color field feel slow. A `circle`
glyph's x/y/color columns are plain 1D arrays, which get Bokeh's fast binary-buffer
transport, and the (per-sample-constant) radius is a single cheap property update.
"""
from __future__ import annotations

import numpy as np
from bokeh.plotting import figure
from sklearn.neighbors import NearestNeighbors


def estimate_hex_radius(coords: np.ndarray) -> float:
    n = len(coords)
    if n < 2:
        return 1.0
    k = min(6, n)
    nn = NearestNeighbors(n_neighbors=k).fit(coords)
    dist, _ = nn.kneighbors(coords)
    spacing = np.percentile(dist[:, 1], 5)
    return max(float(spacing) * 0.58, 1e-6)


#: background for Field (categorical/binned obs) coloring - light, so category colors read clearly
BG_FIELD = "#ffffff"
#: background for Gene & Pathway RGB coloring - neutral gray, so low/zero expression
#: (near-black in the RGB blend) still stands out instead of vanishing into the background
BG_EXPRESSION = "#808080"


def make_figure(title: str = "Spatial view") -> figure:
    fig = figure(
        title=title, match_aspect=True,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        sizing_mode="stretch_both",
        background_fill_color=BG_FIELD,
    )
    fig.y_range.flipped = True
    fig.xgrid.visible = False
    fig.ygrid.visible = False
    fig.axis.visible = False
    return fig


def try_background_image(fig: figure, adata, sample_value: str | None):
    """Best-effort: draw a squidpy/Visium-style tissue image from uns['spatial'] behind
    the hex patches, if present and matched to the current sample. Silently does
    nothing when uns['spatial'] is absent or no library matches - most datasets
    (including the bundled TNBC/open_st example) simply won't have this."""
    spatial_uns = adata.uns.get("spatial")
    if not spatial_uns:
        return
    library_id = None
    if sample_value is not None and sample_value in spatial_uns:
        library_id = sample_value
    elif len(spatial_uns) == 1:
        library_id = next(iter(spatial_uns))
    if library_id is None:
        return
    try:
        entry = spatial_uns[library_id]
        images = entry.get("images", {})
        img = images.get("hires", images.get("lowres"))
        if img is None:
            return
        scalefactors = entry.get("scalefactors", {})
        scalef = scalefactors.get("tissue_hires_scalef", scalefactors.get("tissue_lowres_scalef", 1.0))
        h, w = img.shape[0], img.shape[1]
        if img.ndim == 3:
            rgba = np.dstack([img[..., :3], np.full(img.shape[:2], 255, dtype=np.uint8)]) if img.shape[-1] == 3 else img
            packed = rgba.astype(np.uint8).view(np.uint32).reshape(h, w)
        else:
            return
        fig.image_rgba(
            image=[np.flipud(packed)], x=0, y=0, dw=w / scalef, dh=h / scalef,
            level="image",
        )
    except Exception:
        return
