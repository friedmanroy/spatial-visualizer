"""Dense per-cell spatial rendering, in the same visual spirit as the TNBC viewer's
hex-binned look (solid, densely-packed blobs rather than sparse dots).

Purely geometric (driven by the point cloud's own spacing), so it carries over to any
dataset with obsm['spatial'] coordinates without assuming anything about obs schema.

Cells are drawn as data-space hexagons, squares or circles, each sized (and, for
hexagons/squares, rotated) from the sample's own nearest-neighbor spacing so that at
scale 1 they would tile the plane if every lattice position were observed. They use
Bokeh's `ngon` / `rect` / `circle` glyphs with one scalar size and angle per sample,
rather than individually-computed polygons: Bokeh's `Patches` glyph requires each
polygon's vertices as a *ragged* array, which only supports slow per-row JSON
serialization - resending that for every ~10-100k-cell sample on every switch is
the main thing that made switching sample/FOV or color field feel slow. These
glyphs' x/y/color columns are plain 1D arrays, which get Bokeh's fast binary-buffer
transport, and the (per-sample-constant) size/angle is a single cheap property update.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from bokeh.plotting import figure
from sklearn.neighbors import NearestNeighbors


MARKER_SHAPES = ("Hexagons", "Squares", "Circles")
#: dropdown value meaning "use the shape detected from the sample's own lattice"
AUTO_SHAPE = "Auto"

#: neighbors closer than this multiple of the spacing count as the lattice's first ring;
#: the second ring is at sqrt(2) (square) / sqrt(3) (hex) times the spacing.
_FIRST_RING = 1.2
#: how close to perfect (1) the 6-fold / 4-fold bond order must be to call a lattice
#: hexagonal / square; lower values tolerate more positional jitter.
_ORDER_THRESHOLD = 0.7
#: spacing/orientation are estimated from at most this many cells' neighborhoods (each
#: still looked up among *all* cells), which keeps million-cell samples fast
_MAX_LATTICE_QUERIES = 50_000


@dataclass(frozen=True)
class Lattice:
    """Per-sample geometry the markers are sized/rotated from."""
    spacing: float  # typical center-to-center distance between neighboring cells
    hex_angle: float  # rotation (radians) that lines hexagon sides up with the neighbors
    square_angle: float  # same, for squares
    shape: str  # detected marker shape: one of MARKER_SHAPES


def _bond_order(phi: np.ndarray, fold: int) -> tuple[float, float]:
    """(order, angle) of neighbor-bond directions `phi` under `fold`-fold symmetry:
    order is |mean(exp(i*fold*phi))|, ~1 when all bonds sit on a fold-fold star (6 for a
    hex lattice, 4 for square) and ~0 otherwise; angle is that star's rotation."""
    if len(phi) == 0:
        return 0.0, 0.0
    z = np.mean(np.exp(1j * fold * phi))
    return float(np.abs(z)), float(np.angle(z) / fold)


def estimate_lattice(coords: np.ndarray) -> Lattice:
    """Spacing, orientation and shape of the lattice the cells sit on (if any).

    The shape comes from the directions to each cell's first-ring neighbors: on a hex
    lattice they're multiples of 60 deg apart, on a square one multiples of 90 deg, and any
    3 first-ring neighbors are enough to tell the two apart. Pooled over all cells (to
    tolerate jitter) that's the 6-fold vs 4-fold bond order. No cell with 3+ neighbors
    in the first ring, or neither order high enough -> not a lattice -> circles.
    """
    n = len(coords)
    if n < 2:
        return Lattice(1.0, 0.0, 0.0, "Circles")
    nn = NearestNeighbors(n_neighbors=min(7, n)).fit(coords)  # self + up to 6 (hex) neighbors
    if n > _MAX_LATTICE_QUERIES:
        query = np.random.default_rng(0).choice(n, _MAX_LATTICE_QUERIES, replace=False)
    else:
        query = np.arange(n)
    dist, ind = nn.kneighbors(coords[query])
    dist, ind = dist[:, 1:], ind[:, 1:]
    d = dist[:, 0]
    positive = d[d > 0]  # duplicate positions would otherwise give a zero spacing
    if len(positive) == 0:
        return Lattice(1.0, 0.0, 0.0, "Circles")
    # Each cell's distance to its closest neighbor, then the median over cells: the global
    # minimum (or a low percentile) is dominated by a few unusually close / jittered pairs
    # and would leave visible gaps between the markers.
    spacing = max(float(np.median(positive)), 1e-6)

    ring = (dist > 0) & (dist < _FIRST_RING * spacing)
    rows, cols = np.nonzero(ring)
    vec = coords[ind[rows, cols]] - coords[query[rows]]
    # Bokeh angles are counter-clockwise on screen, and the figure's y axis is flipped.
    phi = np.arctan2(-vec[:, 1], vec[:, 0])
    hex_order, hex_angle = _bond_order(phi, 6)
    square_order, square_angle = _bond_order(phi, 4)

    if ring.sum(axis=1).max(initial=0) < 3:
        shape = "Circles"
    elif hex_order >= _ORDER_THRESHOLD and hex_order > square_order:
        shape = "Hexagons"
    elif square_order >= _ORDER_THRESHOLD:
        shape = "Squares"
    else:
        shape = "Circles"
    return Lattice(spacing, hex_angle, square_angle, shape)


def marker_props(shape: str, lattice: Lattice, scale: float = 1.0) -> dict:
    """Glyph properties for `shape` so that, at scale 1, markers tile the plane on the
    sample's lattice: hexagons of flat-to-flat width = spacing (circumradius spacing/sqrt(3)),
    squares of side = spacing, and circles with the hexagon's circumradius (they can't
    tile exactly, so they overlap just enough to leave no gaps on a hex lattice)."""
    if shape == "Hexagons":
        return {"radius": lattice.spacing / np.sqrt(3) * scale, "angle": lattice.hex_angle}
    if shape == "Squares":
        side = lattice.spacing * scale
        return {"width": side, "height": side, "angle": lattice.square_angle}
    return {"radius": lattice.spacing / np.sqrt(3) * scale}


#: background for Field (categorical/binned obs) coloring - light, so category colors read clearly
BG_FIELD = "#ffffff"
#: background for Gene & Pathway RGB coloring - neutral gray, so low/zero expression
#: (near-black in the RGB blend) still stands out instead of vanishing into the background
BG_EXPRESSION = "#999999"


def make_figure(title: str = "Spatial view") -> figure:
    fig = figure(
        title=title, match_aspect=True,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        sizing_mode="stretch_both",
        background_fill_color=BG_FIELD,
        # GPU drawing: needed for panning smoothly over tens of thousands of markers.
        output_backend="webgl",
    )
    fig.y_range.flipped = True
    fig.xgrid.visible = False
    fig.ygrid.visible = False
    fig.axis.visible = False
    return fig


#: images larger than this (px, longest side) are strided down before being sent to the
#: browser - full-res tissue scans can be tens of thousands of px across.
_MAX_IMAGE_SIDE = 4000
#: arrays smaller than this on either side aren't treated as images (e.g. small
#: coordinate/metadata arrays that also live under uns['spatial'])
_MIN_IMAGE_SIDE = 32


def _is_image_array(value) -> bool:
    if not isinstance(value, np.ndarray) or value.ndim not in (2, 3):
        return False
    if value.ndim == 3 and value.shape[-1] not in (1, 3, 4):
        return False
    if not (np.issubdtype(value.dtype, np.number) or value.dtype == bool):
        return False
    return min(value.shape[0], value.shape[1]) >= _MIN_IMAGE_SIDE


def _find_scalef(name: str, containers: list) -> float | None:
    """Looks for a squidpy/Visium-style scale factor for image `name` in a
    'scalefactors' dict next to the image or one level up."""
    for container in containers:
        sf = container.get("scalefactors") if isinstance(container, Mapping) else None
        if not isinstance(sf, Mapping):
            continue
        for key in (f"tissue_{name}_scalef", f"{name}_scalef"):
            if key in sf:
                try:
                    return float(np.asarray(sf[key]).ravel()[0])
                except (TypeError, ValueError, IndexError):
                    pass
    return None


def find_background_images(adata) -> dict[str, tuple[np.ndarray, float]]:
    """Every image-shaped array anywhere under adata.uns['spatial'], keyed by a readable
    path (e.g. 'lib1/images/hires', or just 'hires' when the image sits directly in
    uns['spatial']), mapped to (image, scale factor from image px to obsm['spatial']
    units). Makes no assumption about a library_id / 'images' layout; the scale factor
    falls back to 1.0 (image px == spatial coordinates) when none is found."""
    spatial_uns = adata.uns.get("spatial")
    found: dict[str, tuple[np.ndarray, float]] = {}

    def walk(value, path: list[str], parents: list):
        if len(path) > 5:
            return
        if _is_image_array(value):
            name = path[-1] if path else "spatial"
            scalef = _find_scalef(name, list(reversed(parents[-2:])))
            label = "/".join(path) or "spatial"
            if scalef is None:
                scalef = 1.0
                label += " (no scale factor found, assuming 1)"
            found[label] = (value, scalef)
        elif isinstance(value, Mapping):
            for key, child in value.items():
                walk(child, path + [str(key)], parents + [value])

    if spatial_uns is not None:
        walk(spatial_uns, [], [])
    return found


def image_to_rgba(img: np.ndarray) -> np.ndarray:
    """Any grayscale/RGB/RGBA array (uint8, other ints, 0-1 or 0-255 floats) -> the
    packed (h, w) uint32 RGBA array Bokeh's image_rgba expects."""
    img = np.asarray(img)
    step = max(1, int(np.ceil(max(img.shape[0], img.shape[1]) / _MAX_IMAGE_SIDE)))
    img = img[::step, ::step]
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.dtype != np.uint8:
        img = np.nan_to_num(img.astype(np.float32))
        mx = float(img.max()) if img.size else 0.0
        if mx <= 1.0:
            img = img * 255.0
        elif mx > 255.0:
            img = img * (255.0 / mx)
        img = img.clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 3:
        img = np.dstack([img, np.full(img.shape[:2], 255, dtype=np.uint8)])
    h, w = img.shape[:2]
    return np.ascontiguousarray(img).view(np.uint32).reshape(h, w)


#: How to reorient a background image before drawing it, since datasets disagree on
#: whether obsm['spatial'] is (x, y) or (row, col) relative to the stored image.
#: Applied to the (row, col[, channel]) array; row 0 is drawn at the top.
IMAGE_ORIENTATIONS = {
    "As stored": lambda a: a,
    "Transposed (swap x/y)": lambda a: np.swapaxes(a, 0, 1),
    "Flip horizontal": lambda a: a[:, ::-1],
    "Flip vertical": lambda a: a[::-1],
    "Rotate 180°": lambda a: a[::-1, ::-1],
    "Rotate 90° clockwise": lambda a: np.rot90(a, k=-1),
    "Rotate 90° counter-clockwise": lambda a: np.rot90(a, k=1),
    "Anti-transposed": lambda a: np.swapaxes(a[::-1, ::-1], 0, 1),
}


#: a background image is considered misplaced (and is fitted to the cells instead) when
#: fewer than this fraction of the cells land on it...
_MIN_CELLS_ON_IMAGE = 0.5
#: ...or when it's this many times larger or smaller than the area the cells span
_MAX_IMAGE_SIZE_RATIO = 10.0
#: images are strided down to about this many px (longest side) to find the tissue on them
_FIT_IMAGE_SIDE = 512


def _image_near_cells(x0: float, y0: float, dw: float, dh: float, coords: np.ndarray) -> bool:
    """Whether an image placed at [x0, x0+dw] x [y0, y0+dh] overlaps the cells well
    enough to be trusted as is."""
    if len(coords) == 0:
        return True
    xs, ys = coords[:, 0], coords[:, 1]
    on_image = ((xs >= x0) & (xs <= x0 + dw) & (ys >= y0) & (ys <= y0 + dh)).mean()
    cells_size = max(float(np.ptp(xs)), float(np.ptp(ys)))
    if cells_size > 0:
        ratio = max(dw, dh) / cells_size
        if not (1 / _MAX_IMAGE_SIZE_RATIO <= ratio <= _MAX_IMAGE_SIZE_RATIO):
            return False
    return on_image >= _MIN_CELLS_ON_IMAGE


def _tissue_weights(img: np.ndarray) -> tuple[np.ndarray, int]:
    """Per-pixel weights (on an image strided down by the returned step) for how much
    each pixel looks like tissue rather than background: its distance in brightness from
    the image's border, which is assumed to be background (works for both brightfield
    and fluorescence images)."""
    step = max(1, int(np.ceil(max(img.shape[0], img.shape[1]) / _FIT_IMAGE_SIDE)))
    small = np.nan_to_num(np.asarray(img[::step, ::step], dtype=np.float32))
    if small.ndim == 3:
        small = small[..., :3].mean(axis=-1)
    border = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]])
    weights = np.abs(small - np.median(border))
    if weights.max() <= 0:
        return np.ones_like(small), step
    # Drops faint noise in the background, which would otherwise pull the fit toward
    # the image's center.
    weights[weights < 0.1 * weights.max()] = 0.0
    return weights, step


def _fit_image_to_cells(img: np.ndarray, coords: np.ndarray) -> tuple[float, float, float, float]:
    """(x0, y0, dw, dh) placing `img` so the tissue on it has the same mean and std
    (per axis) as the cells' coordinates."""
    weights, step = _tissue_weights(img)
    total = weights.sum()
    rows = (np.arange(weights.shape[0]) + 0.5) * step
    cols = (np.arange(weights.shape[1]) + 0.5) * step
    row_w, col_w = weights.sum(axis=1) / total, weights.sum(axis=0) / total
    placement = []
    for pos, w, cell_vals, n_px in ((cols, col_w, coords[:, 0], img.shape[1]),
                                   (rows, row_w, coords[:, 1], img.shape[0])):
        px_mean = float((pos * w).sum())
        px_std = float(np.sqrt((((pos - px_mean) ** 2) * w).sum()))
        cell_mean, cell_std = float(cell_vals.mean()), float(cell_vals.std())
        scale = cell_std / px_std if px_std > 0 and cell_std > 0 else 1.0
        placement.append((cell_mean - px_mean * scale, n_px * scale))
    (x0, dw), (y0, dh) = placement
    return x0, y0, dw, dh


def background_image_data(img: np.ndarray, scalef: float, orientation: str = "As stored",
                          coords: np.ndarray | None = None) -> tuple[dict, bool]:
    """ColumnDataSource data placing `img` (reoriented per IMAGE_ORIENTATIONS) so its
    pixels line up with obsm['spatial']. The figure's y axis is flipped, so row 0 of
    the array lands at the top. When `coords` (the cells being drawn) are given and the
    image doesn't land anywhere near them, it's instead stretched/moved so its tissue has
    the same mean and std as the cells; the returned flag says whether that happened."""
    img = IMAGE_ORIENTATIONS[orientation](img)
    h, w = img.shape[0], img.shape[1]
    x0, y0, dw, dh = 0.0, 0.0, w / scalef, h / scalef
    fitted = coords is not None and not _image_near_cells(x0, y0, dw, dh, coords)
    if fitted:
        x0, y0, dw, dh = _fit_image_to_cells(img, coords)
    return {
        "image": [image_to_rgba(img)], "x": [x0], "y": [y0], "dw": [dw], "dh": [dh],
    }, fitted
