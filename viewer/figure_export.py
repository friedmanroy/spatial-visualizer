"""Saving the Spatial tab's current view as a publication-style matplotlib figure: the
same cells, colors, regions and background image, cropped to exactly what the canvas
shows, plus a legend for what the colors mean.

The Spatial tab collects what is drawn into a FigureSpec (spatial_page.figure_spec) and
render() turns it into a file. Every figure uses the same layout (FIGURE_SHORT_SIDE,
FONT_SIZE...): no axes, ticks, grid or frame; the map shows the canvas's view - cropped
closer to the cells where the view has empty space around them (zoomed out) - in that
area's aspect ratio; legends, the colorbar and channel names sit outside the map.

Uses matplotlib's object API (matplotlib.figure.Figure) rather than pyplot, since it
runs in a worker thread.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

#: the map's shorter side (inches); legends etc. are added outside it
FIGURE_SHORT_SIDE = 5.0
FONT_SIZE = 18
#: resolution of PNG / JPEG files, and of the (rasterized) cells inside a PDF
DPI = 300
#: file formats offered (label -> (matplotlib format, extension))
FORMATS = {"PNG": ("png", ".png"), "PDF": ("pdf", ".pdf"), "JPEG": ("jpeg", ".jpg")}
#: channel name colors above a Gene & Pathway figure (as the viewer's channel labels)
CHANNEL_COLORS = {"R": "#ff4444", "G": "#44cc44", "B": "#4488ff"}
#: region outline width, and the contrasting line drawn just under it (points)
REGION_LINE_WIDTH = 2.0
REGION_HALO_WIDTH = 4.5
#: vertices used to draw a circle marker
_CIRCLE_VERTICES = 24
#: empty space around the cells is cropped to this margin (a fraction of the cells'
#: larger extent); a view tighter than that is kept as it is
CROP_MARGIN = 0.03


@dataclass
class Layer:
    """Markers drawn with one glyph: centers, packed colors (colors.pack_rgb) and the
    glyph's shape ("Hexagons" / "Squares" / "Circles") and properties (data units)."""
    x: np.ndarray
    y: np.ndarray
    colors: np.ndarray
    shape: str
    props: dict


@dataclass
class FigureSpec:
    box: tuple[float, float, float, float]  # x0, x1, y0, y1 in view (obsm['spatial'] units)
    layers: list[Layer]
    alpha: float = 1.0  # cell opacity
    expression: bool = False  # Gene & Pathway view: black background, channel names on top
    channels: dict[str, str] = field(default_factory=dict)  # "R"/"G"/"B" -> gene / pathway name
    legend: list[tuple[str, str]] = field(default_factory=list)  # (category, color) of a categorical field
    colorbar: tuple[str, list[str], list[float]] | None = None  # (field, bin colors, bin boundaries)
    regions: list[tuple[str, str, list]] = field(default_factory=list)  # (name, color, [(xs, ys), ...])
    image: tuple[np.ndarray, float, float, float, float] | None = None  # (RGBA uint8, x, y, dw, dh)


def unpack_colors(packed: np.ndarray) -> np.ndarray:
    """Packed 0xRRGGBBAA colors (colors.pack_rgb) -> (n, 3) RGB floats in 0-1."""
    b = np.ascontiguousarray(packed, dtype="<u4").view(np.uint8).reshape(-1, 4)  # bytes: A, B, G, R
    return b[:, [3, 2, 1]].astype(float) / 255.0


def _marker_vertices(layer: Layer) -> np.ndarray:
    """(n, k, 2) polygon vertices, in data coordinates, of the layer's markers as Bokeh
    draws them. Bokeh angles turn counter-clockwise on screen, and the viewer's y axis
    points down, so a screen offset (dx, dy) is (dx, -dy) in data coordinates."""
    p = layer.props
    angle = float(p.get("angle", 0.0) or 0.0)
    if layer.shape == "Squares":
        hw, hh = float(p["width"]) / 2, float(p["height"]) / 2
        local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    else:
        n = 6 if layer.shape == "Hexagons" else _CIRCLE_VERTICES
        theta = np.pi / 2 + 2 * np.pi * np.arange(n) / n  # Bokeh's ngon: pointy-top at angle 0
        local = float(p["radius"]) * np.c_[np.cos(theta), np.sin(theta)]
    c, s = np.cos(angle), np.sin(angle)
    rotated = local @ np.array([[c, s], [-s, c]])  # counter-clockwise rotation (screen)
    offsets = np.c_[rotated[:, 0], -rotated[:, 1]]  # screen y up -> data y down
    return np.stack([layer.x, layer.y], axis=1)[:, None, :] + offsets[None, :, :]


def cropped_box(box, layers: list[Layer]) -> tuple[float, float, float, float]:
    """`box` (x0, x1, y0, y1) shrunk to the markers' extent plus CROP_MARGIN, never grown
    past it - so a zoomed-out view loses its empty borders, a zoomed-in one is unchanged."""
    shown = [layer for layer in layers if len(layer.x)]
    if not shown:
        return box
    extents = []
    for layer in shown:
        verts = _marker_vertices(layer)
        extents.append((verts[..., 0].min(), verts[..., 0].max(), verts[..., 1].min(), verts[..., 1].max()))
    dx0, dx1 = min(e[0] for e in extents), max(e[1] for e in extents)
    dy0, dy1 = min(e[2] for e in extents), max(e[3] for e in extents)
    margin = CROP_MARGIN * max(dx1 - dx0, dy1 - dy0)
    x0, x1, y0, y1 = box
    return (max(x0, dx0 - margin), min(x1, dx1 + margin), max(y0, dy0 - margin), min(y1, dy1 + margin))


def _draw_regions(ax, regions, halo: str):
    for name, color, polygons in regions:
        for xs, ys in polygons:
            xs, ys = list(xs) + [xs[0]], list(ys) + [ys[0]]
            ax.plot(xs, ys, color=halo, linewidth=REGION_HALO_WIDTH, solid_joinstyle="round", zorder=3)
            ax.plot(xs, ys, color=color, linewidth=REGION_LINE_WIDTH, solid_joinstyle="round", zorder=4)
        # Name just above the region's largest polygon (y points down: its smallest y is its top).
        xs, ys = max(polygons, key=lambda p: np.ptp(p[0]) * np.ptp(p[1]))
        ax.annotate(name, (float(np.mean([min(xs), max(xs)])), float(min(ys))), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", color=color,
                    fontsize=FONT_SIZE, zorder=5, annotation_clip=True,
                    path_effects=[patheffects.withStroke(linewidth=3, foreground=halo)])


def render(spec: FigureSpec, path: str, fmt: str):
    """Draws `spec` and saves it to `path` as `fmt` ("png", "pdf" or "jpeg")."""
    x0, x1, y0, y1 = cropped_box(spec.box, spec.layers)
    w, h = x1 - x0, y1 - y0
    size = ((FIGURE_SHORT_SIDE * w / h, FIGURE_SHORT_SIDE) if w >= h
            else (FIGURE_SHORT_SIDE, FIGURE_SHORT_SIDE * h / w))
    background = "black" if spec.expression else "white"
    text_color = "white" if spec.expression else "black"
    with matplotlib.rc_context({"font.size": FONT_SIZE, "axes.grid": False}):
        fig = Figure(figsize=size, facecolor=background)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_facecolor(background)
        if spec.image is not None:
            rgba, ix, iy, dw, dh = spec.image
            # The viewer draws image row 0 at y (the top, as y points down).
            ax.imshow(rgba, extent=(ix, ix + dw, iy, iy + dh), origin="lower", interpolation="nearest", zorder=0)
        for layer in spec.layers:
            if not len(layer.x):
                continue
            rgba = np.c_[unpack_colors(layer.colors), np.full(len(layer.x), spec.alpha)]
            # Cells are rasterized in a PDF (vector text and lines stay sharp, the file stays small).
            ax.add_collection(PolyCollection(_marker_vertices(layer), facecolors=rgba, edgecolors="none",
                                             linewidths=0, antialiaseds=False, zorder=1, rasterized=True))
        _draw_regions(ax, spec.regions, halo="white" if spec.expression else "black")

        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)  # y points down, as in the viewer
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if spec.expression:
            for label, (x, ha) in {"R": (0.0, "left"), "G": (0.5, "center"), "B": (1.0, "right")}.items():
                if spec.channels.get(label):
                    ax.text(x, 1.01, spec.channels[label], transform=ax.transAxes, ha=ha, va="bottom",
                            color=CHANNEL_COLORS[label], fontsize=FONT_SIZE)
        elif spec.legend:
            handles = [Patch(facecolor=color, edgecolor="none", label=label) for label, color in spec.legend]
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
                      fontsize=FONT_SIZE, labelcolor=text_color, borderaxespad=0,
                      ncol=int(np.ceil(len(handles) / 18)))
        elif spec.colorbar is not None:
            name, colors, bounds = spec.colorbar
            cax = fig.add_axes((1.04, 0.1, 0.04, 0.8))
            mappable = ScalarMappable(norm=BoundaryNorm(bounds, len(colors)), cmap=ListedColormap(colors))
            bar = fig.colorbar(mappable, cax=cax, spacing="uniform", ticks=bounds)
            bar.set_ticklabels([f"{b:.3g}" for b in bounds])
            bar.ax.tick_params(labelsize=FONT_SIZE, colors=text_color)
            bar.outline.set_visible(False)
            bar.set_label(name, fontsize=FONT_SIZE, color=text_color)
        fig.savefig(path, format=fmt, dpi=DPI, bbox_inches="tight", pad_inches=0.1, facecolor=background)
