"""The 'Align image' tab: moves a sample's cells onto a tissue image that doesn't line up
with them (see alignment.py for what applying does to the data).

The cells are drawn as hexagons colored by total counts (or spots per hexagon, or a gene),
over the image. The image only turns in 90° steps or mirrors; the cells are shifted,
scaled and rotated - by dragging on the plot, by typing numbers, or by fitting landmark
pairs (a spot on the cells, then the same spot on the image). Dragging is done in the
browser, which moves the hexagons it already has: nothing is sent until the drag ends.
"""
from __future__ import annotations

import contextlib
import html

import numpy as np
import panel as pn
from bokeh.events import Pan, PanEnd, PanStart, Tap
from bokeh.models import ColumnDataSource, CustomJS, PanTool, WheelZoomTool
from bokeh.palettes import Viridis256

from . import alignment, hexbin, spatial_plot
from .color_controls import HINT_STYLE
from .colors import GRAY_RGBA, pack_rgb

#: drag modes: radio label -> what a drag on the plot does (the JS reads the value)
_MODES = {
    "Pan the view": "pan",
    "Move the cells": "move",
    "Rotate & scale the cells": "rotate",
    "Place landmarks": "landmarks",
}
_TOTAL_COUNTS, _SPOTS, _GENE = "Total counts", "Spots per hexagon", "A gene"
#: a click of a button that removes regions must be followed by another within this long (ms)
_CONFIRM_MS = 4000
#: with this many landmark pairs or more, a mirrored image is looked for
_MIRROR_CHECK_PAIRS = 3
#: ...and suggested when it fits at least this many times better
_MIRROR_GAIN = 2.0
#: the cells' color scale spans these percentiles of the values shown
_COLOR_PERCENTILES = (1, 99)
#: hexagonal binnings kept, so going back to a size or sample doesn't bin again
_BINNINGS_KEPT = 4
_DATA_COLOR, _IMAGE_COLOR = "#ff7f0e", "#17becf"
_VIRIDIS = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in Viridis256])


def build_alignment_page(state) -> pn.viewable.Viewable:
    dataset = state.spatial
    images = spatial_plot.find_background_images(dataset.adata)
    if not images:
        return pn.pane.HTML(
            "<h3>No tissue image to align to</h3>No image was found under <code>uns['spatial']</code>.",
            margin=(10, 20),
        )

    # -- widgets -----------------------------------------------------------------------
    n_samples = len(dataset.sample_ids)
    sample_select = pn.widgets.Select(name="Sample / FOV", options=list(dataset.sample_ids),
                                      sizing_mode="stretch_width", visible=n_samples > 1)
    image_select = pn.widgets.Select(name="Image", options=list(images), sizing_mode="stretch_width")
    orientation = pn.widgets.Select(
        name="Turn / mirror the image", options=list(spatial_plot.IMAGE_ORIENTATIONS), value="As stored",
        sizing_mode="stretch_width",
        description="The image only turns in 90° steps or mirrors; the cells do the rest.",
    )
    show_select = pn.widgets.Select(
        name="Color the cells by", options=[_TOTAL_COUNTS, _SPOTS, _GENE], value=_TOTAL_COUNTS,
        sizing_mode="stretch_width",
        description="Pick whatever shows the tissue's outline and structure best.",
    )
    gene_input = pn.widgets.TextInput(name="Gene", placeholder="e.g. a marker of a clear structure",
                                      visible=False, sizing_mode="stretch_width")
    hex_size = pn.widgets.EditableFloatSlider(name=hexbin.EDGE_LABEL, start=0.1, end=1.0, value=0.5,
                                              sizing_mode="stretch_width")
    hex_size_hint = pn.pane.HTML("Starts at the Spatial tab's size. " + hexbin.EDGE_DESCRIPTION,
                                 styles=HINT_STYLE, sizing_mode="stretch_width", margin=(0, 5, 4, 5))
    opacity = pn.widgets.FloatSlider(name="Cell opacity (lower to see the image)", start=0.0, end=1.0,
                                     step=0.05, value=0.7, sizing_mode="stretch_width")
    mode = pn.widgets.RadioButtonGroup(name="Dragging on the plot", options=list(_MODES), value="Move the cells",
                                       orientation="vertical", button_type="primary", button_style="outline",
                                       sizing_mode="stretch_width")
    mode_hint = pn.pane.HTML("", styles=HINT_STYLE, sizing_mode="stretch_width", margin=(0, 5, 4, 5))
    x_input = pn.widgets.FloatInput(name="Center x", sizing_mode="stretch_width")
    y_input = pn.widgets.FloatInput(name="Center y", sizing_mode="stretch_width")
    scale_input = pn.widgets.FloatInput(name="Scale", start=1e-9, step=0.01, sizing_mode="stretch_width")
    angle_input = pn.widgets.FloatInput(name="Rotation (° clockwise)", step=0.5, sizing_mode="stretch_width")
    reset_btn = pn.widgets.Button(name="Start over", button_type="light", sizing_mode="stretch_width",
                                  description="Puts the cells back at the image's center, unrotated")

    lm_hint = pn.pane.HTML(
        "Choose <b>Place landmarks</b>, click a spot you recognize on the cells, then the same spot "
        "on the image (the cells hide until you do). Each pair moves the cells to fit: 1 pair "
        "shifts them, 2 or more also scale and rotate them. Spread the pairs out.",
        styles=HINT_STYLE, sizing_mode="stretch_width", margin=(0, 5, 4, 5),
    )
    lm_status = pn.pane.HTML("", sizing_mode="stretch_width", margin=(0, 5))
    mirror_btn = pn.widgets.Button(name="", button_type="warning", visible=False, sizing_mode="stretch_width")
    undo_lm = pn.widgets.Button(name="Undo last point", button_type="light", sizing_mode="stretch_width")
    clear_lm = pn.widgets.Button(name="Clear landmarks", button_type="light", sizing_mode="stretch_width")

    apply_btn = pn.widgets.Button(name="Apply alignment", button_type="primary", sizing_mode="stretch_width",
                                  margin=(8, 5, 4, 5))
    undo_btn = pn.widgets.Button(name="Put the cells back where they were", button_type="light",
                                 sizing_mode="stretch_width", visible=False)
    apply_note = pn.pane.HTML("", styles=HINT_STYLE, sizing_mode="stretch_width", margin=(0, 5, 4, 5))
    status = pn.pane.HTML("", sizing_mode="stretch_width", margin=(0, 5))

    # -- the plot ----------------------------------------------------------------------
    fig = spatial_plot.make_figure(title=None)
    fig.toolbar.active_scroll = fig.select_one(WheelZoomTool)
    pan_tool = fig.select_one(PanTool)
    bg_source = ColumnDataSource(data={"image": [], "x": [], "y": [], "dw": [], "dh": []})
    image_renderer = fig.image_rgba(image="image", x="x", y="y", dw="dw", dh="dh", source=bg_source, level="image")
    # The view frames the image; the cells moving never pulls it along.
    fig.x_range.renderers = [image_renderer]
    fig.y_range.renderers = [image_renderer]
    # Every source with px / py (original coordinates) has its x / y moved by the browser.
    empty = {"px": [], "py": [], "x": [], "y": []}
    cells = ColumnDataSource(data={**empty, "fill_color": []})
    center = ColumnDataSource(data=dict(empty))
    pairs_src = ColumnDataSource(data={**empty, "ix": [], "iy": [], "label": []})
    pending_src = ColumnDataSource(data=dict(empty))
    hexes = fig.ngon(x="x", y="y", n=6, radius=1.0, angle=0.0, source=cells, fill_color="fill_color",
                     line_color=None, fill_alpha=opacity.value)
    fig.scatter(x="x", y="y", source=center, marker="cross", size=18, line_color="white", line_width=3)
    fig.scatter(x="x", y="y", source=center, marker="cross", size=16, line_color="black", line_width=1.5)
    fig.segment(x0="x", y0="y", x1="ix", y1="iy", source=pairs_src, line_color="white", line_width=2)
    fig.scatter(x="x", y="y", source=pairs_src, size=11, fill_color=_DATA_COLOR, line_color="white", line_width=1.5)
    fig.scatter(x="ix", y="iy", source=pairs_src, marker="square", size=10, fill_color=_IMAGE_COLOR,
                line_color="white", line_width=1.5)
    for x, y in (("x", "y"), ("ix", "iy")):
        fig.text(x=x, y=y, text="label", source=pairs_src, x_offset=8, y_offset=-8, text_font_size="12px",
                 text_font_style="bold", text_color="black", text_outline_color="white")
    fig.scatter(x="x", y="y", source=pending_src, size=14, fill_color=_DATA_COLOR, line_color="black", line_width=2)

    params = ColumnDataSource(data={"cx": [0.0], "cy": [0.0], "s": [1.0], "theta": [0.0], "d0x": [0.0],
                                    "d0y": [0.0], "base_r": [1.0], "mode": [_MODES[mode.value]]})
    js_args = {"params": params, "srcs": [cells, center, pairs_src, pending_src], "hexes": hexes}
    params.js_on_change("data", CustomJS(args=js_args, code=_PLACE_JS + _ON_PARAMS_JS))
    for event, phase in ((PanStart, "start"), (Pan, "move"), (PanEnd, "end")):
        fig.js_on_event(event, CustomJS(args=js_args, code=_PLACE_JS + f'const PHASE = "{phase}";' + _DRAG_JS))

    # -- per-sample geometry and colors --------------------------------------------------
    bases: dict[str, tuple] = {}  # sample -> (spacing, anchor, extent) of its original coordinates
    binnings: dict[tuple[str, float], hexbin.HexBinning] = {}  # (sample, width in cell spacings) ->
    totals_cache: dict[str, np.ndarray] = {}
    # The Spatial tab's hexagons to start with: the width the load page pooled spots with, else
    # the automatic one. In cell spacings, so it's the same on the original coordinates.
    hex_factor = {"value": dataset.analysis_width or dataset.default_hex_factor()}

    def sample_base(sample: str):
        if sample not in bases:
            coords = alignment.original_coords(dataset, sample)
            spacing = spatial_plot.estimate_lattice(coords).spacing
            extent = (float(np.ptp(coords[:, 0])), float(np.ptp(coords[:, 1]))) if len(coords) else (0.0, 0.0)
            anchor = (coords.min(axis=0) + coords.max(axis=0)) / 2 if len(coords) else np.zeros(2)
            bases[sample] = (spacing, (float(anchor[0]), float(anchor[1])), extent)
        return bases[sample]

    def get_geometry(sample: str):
        """(hexagons over the sample's original coordinates, anchor, extent)."""
        spacing, anchor, extent = sample_base(sample)
        key = (sample, hex_factor["value"])
        if key not in binnings:
            if len(binnings) >= _BINNINGS_KEPT:
                binnings.pop(next(iter(binnings)))
            binnings[key] = hexbin.make_hex_binning(alignment.original_coords(dataset, sample), key[1] * spacing)
        return binnings[key], anchor, extent

    def show_hex_size(sample: str):
        """The size slider in the sample's own units, at the current width."""
        spacing = sample_base(sample)[0]
        with hold_sync():
            hex_size.param.update(**hexbin.edge_input_range(spacing),
                                  value=hexbin.edge_length(hex_factor["value"], spacing))

    def on_hex_size(event):
        if syncing["on"]:
            return
        sample = sample_select.value
        hex_factor["value"] = hexbin.width_factor(event.new, sample_base(sample)[0])
        with pn.io.hold():
            binning = get_geometry(sample)[0]
            set_sim(get_sim(), base_r=binning.width / np.sqrt(3))
            draw_cells()

    def cell_totals() -> tuple[np.ndarray, str]:
        totals = dataset.cell_totals()
        if totals is not None:
            return totals, "total counts"
        if "x" not in totals_cache:  # no raw counts kept: .X's own row sums
            totals_cache["x"] = np.asarray(dataset.adata.X.sum(axis=1)).ravel()
        return totals_cache["x"], "total expression"

    def hex_values(sample: str) -> tuple[np.ndarray | None, str]:
        binning = get_geometry(sample)[0]
        idx = dataset.sample_indices(sample)
        if show_select.value == _SPOTS:
            return binning.counts.astype(float), "spots per hexagon"
        if show_select.value == _GENE:
            gene = gene_input.value.strip()
            values = dataset.bin_gene_values(gene, idx, binning) if gene else None
            return values, html.escape(gene)
        totals, what = cell_totals()
        return np.log1p(binning.sum(totals[idx])), f"{what} (log)"

    def colors_for(values: np.ndarray | None, n: int) -> np.ndarray:
        if values is None:
            return np.full(n, GRAY_RGBA, dtype=np.uint32)
        finite = np.isfinite(values)
        out = np.full(n, GRAY_RGBA, dtype=np.uint32)
        if finite.any():
            lo, hi = np.percentile(values[finite], _COLOR_PERCENTILES)
            t = np.clip((values[finite] - lo) / (hi - lo), 0, 1) if hi > lo else np.full(finite.sum(), 0.5)
            rgb = _VIRIDIS[np.round(t * 255).astype(int)]
            out[finite] = pack_rgb(rgb[:, 0], rgb[:, 1], rgb[:, 2])
        return out

    # -- the image -----------------------------------------------------------------------
    image_cache: dict[tuple[str, str], dict] = {}

    def stored_size() -> tuple[float, float]:
        img, scalef = images[image_select.value]
        return img.shape[1] / scalef, img.shape[0] / scalef

    def orient() -> np.ndarray:
        """Stored image -> as drawn (2x3)."""
        return alignment.orientation_matrix(orientation.value, *stored_size())

    def draw_image():
        key = (image_select.value, orientation.value)
        if key not in image_cache:
            img, scalef = images[image_select.value]
            image_cache[key] = spatial_plot.background_image_data(img, scalef, orientation.value)
        bg_source.data = image_cache[key]

    def drawn_size() -> tuple[float, float]:
        d = image_cache[(image_select.value, orientation.value)]
        return float(d["dw"][0]), float(d["dh"][0])

    # -- the transform: params.data is the one copy, the inputs follow it ------------------
    syncing = {"on": False}

    def get_sim() -> alignment.Similarity:
        p = params.data
        return alignment.Similarity((p["cx"][0], p["cy"][0]), p["s"][0], p["theta"][0], (p["d0x"][0], p["d0y"][0]))

    def set_sim(sim: alignment.Similarity, base_r: float | None = None):
        p = params.data
        with hold_sync():
            params.data = {
                "cx": [float(sim.center[0])], "cy": [float(sim.center[1])], "s": [float(sim.scale)],
                "theta": [float(sim.angle)], "d0x": [float(sim.anchor[0])], "d0y": [float(sim.anchor[1])],
                "base_r": [float(base_r if base_r is not None else p["base_r"][0])], "mode": [_MODES[mode.value]],
            }
            show_inputs(sim)
        refresh_fit_text()

    @contextlib.contextmanager
    def hold_sync():
        was, syncing["on"] = syncing["on"], True
        try:
            yield
        finally:
            syncing["on"] = was

    def show_inputs(sim: alignment.Similarity):
        with hold_sync():
            x_input.value = round(sim.center[0], 3)
            y_input.value = round(sim.center[1], 3)
            scale_input.value = float(f"{sim.scale:.6g}")
            angle_input.value = round(float(np.degrees(sim.angle)), 3)

    def on_params(attr, old, new):
        if not syncing["on"]:  # the end of a drag in the browser
            show_inputs(get_sim())
            refresh_fit_text()

    params.on_change("data", on_params)

    def on_inputs(event):
        if syncing["on"] or not scale_input.value:
            return
        sim = get_sim()
        set_sim(alignment.Similarity((float(x_input.value), float(y_input.value)), float(scale_input.value),
                                     float(np.radians(angle_input.value)), sim.anchor))

    for w in (x_input, y_input, scale_input, angle_input):
        w.param.watch(on_inputs, "value")

    def starting_sim(sample: str, from_record: bool = True) -> alignment.Similarity:
        """Where the cells start: as last aligned (if `from_record`), else centered on the
        image and scaled to it (or left at scale 1 when the image's scale factor already
        makes them fit)."""
        binning, anchor, extent = get_geometry(sample)
        rec = alignment.records(dataset).get(sample) if from_record else None
        if rec is not None and rec[0] == image_select.value:
            M = alignment.compose(orient(), rec[1])
            if np.linalg.det(M[:, :2]) > 0:
                return alignment.Similarity.from_matrix(M, anchor)
        dw, dh = drawn_size()
        ratio = max(extent[0] / dw if dw else 0, extent[1] / dh if dh else 0)
        has_scalef = "no scale factor" not in image_select.value
        scale = 1.0 if (has_scalef and 0.2 <= ratio <= 1.5) or ratio <= 0 else 0.8 / ratio
        return alignment.Similarity((dw / 2, dh / 2), scale, 0.0, anchor)

    # -- landmarks -------------------------------------------------------------------------
    # (point on the original cells, the same point on the stored image): kept on the stored
    # image so they stay on its content when it's turned or mirrored.
    landmarks: dict = {"pairs": [], "pending": None}

    def draw_landmarks():
        sim = get_sim()
        pairs = landmarks["pairs"]
        P = np.array([p for p, _ in pairs]).reshape(-1, 2)
        Q = alignment.apply_matrix(orient(), np.array([q for _, q in pairs]).reshape(-1, 2))
        X = sim.apply(P) if len(P) else P
        pairs_src.data = {"px": P[:, 0], "py": P[:, 1], "x": X[:, 0], "y": X[:, 1], "ix": Q[:, 0], "iy": Q[:, 1],
                          "label": [str(i + 1) for i in range(len(pairs))]}
        pending = landmarks["pending"]
        if pending is None:
            pending_src.data = dict(empty)
        else:
            x, y = sim.apply(np.array([pending]))[0]
            pending_src.data = {"px": [pending[0]], "py": [pending[1]], "x": [x], "y": [y]}
        hexes.visible = pending is None  # picking the image's point: nothing in the way
        undo_lm.disabled = clear_lm.disabled = not pairs and pending is None
        refresh_fit_text()

    def landmark_arrays():
        pairs = landmarks["pairs"]
        P = np.array([p for p, _ in pairs])
        return P, np.array([q for _, q in pairs])

    def fit_landmarks():
        pairs = landmarks["pairs"]
        if not pairs:
            return
        P, Q_stored = landmark_arrays()
        Q = alignment.apply_matrix(orient(), Q_stored)
        sim = get_sim()
        if len(pairs) == 1 or np.ptp(P, axis=0).max() == 0:  # just shift
            shift = (Q - sim.apply(P)).mean(axis=0)
            sim = alignment.Similarity((sim.center[0] + shift[0], sim.center[1] + shift[1]), sim.scale, sim.angle,
                                       sim.anchor)
        else:
            M, _ = alignment.fit_similarity(P, Q)
            sim = alignment.Similarity.from_matrix(M, sim.anchor)
        set_sim(sim)
        draw_landmarks()

    def better_mirror() -> tuple[str, float] | None:
        """(the mirrored orientation fitting the landmarks best, how many times better it
        fits than the current one) if that's at least _MIRROR_GAIN."""
        if len(landmarks["pairs"]) < _MIRROR_CHECK_PAIRS:
            return None
        P, Q_stored = landmark_arrays()
        W, H = stored_size()

        def fit_error(name):
            return alignment.fit_similarity(P, alignment.apply_matrix(alignment.orientation_matrix(name, W, H), Q_stored))[1]

        current_sign = np.sign(np.linalg.det(orient()[:, :2]))
        mirrored = [o for o in spatial_plot.IMAGE_ORIENTATIONS
                    if np.sign(np.linalg.det(alignment.orientation_matrix(o, W, H)[:, :2])) != current_sign]
        best = min(mirrored, key=fit_error)
        gain = fit_error(orientation.value) / max(fit_error(best), 1e-12)
        return (best, gain) if gain >= _MIRROR_GAIN else None

    def refresh_fit_text():
        pairs = landmarks["pairs"]
        lines = []
        if landmarks["pending"] is not None:
            lines.append(f"<b>Now click the same spot on the image</b> (pair {len(pairs) + 1}).")
        if pairs:
            P, Q_stored = landmark_arrays()
            Q = alignment.apply_matrix(orient(), Q_stored)
            err = alignment.rms(get_sim().matrix(), P, Q)
            dw, dh = drawn_size()
            lines.insert(0, f"{len(pairs)} landmark pair{'s' * (len(pairs) > 1)} · the cells' points are "
                            f"<b>{err:,.4g}</b> pts from the image's on average "
                            f"({100 * err / max(dw, dh):.2g}% of the image's size)")
        lm_status.object = "<br>".join(f'<span style="font-size:13px">{line}</span>' for line in lines)
        suggestion = better_mirror()
        mirror_btn.visible = suggestion is not None
        if suggestion is not None:
            mirror_btn.name = f"The image looks mirrored - use '{suggestion[0]}' (fits {suggestion[1]:.0f}x better)"
            mirror_btn.tags = [suggestion[0]]

    def on_tap(event):
        if _MODES[mode.value] != "landmarks":
            return
        point = (event.x, event.y)
        if landmarks["pending"] is None:
            landmarks["pending"] = tuple(get_sim().invert(point))
        else:
            q = alignment.apply_matrix(alignment.invert_matrix(orient()), np.array([point]))[0]
            landmarks["pairs"].append((np.array(landmarks["pending"]), q))
            landmarks["pending"] = None
            fit_landmarks()
        draw_landmarks()

    fig.on_event(Tap, on_tap)

    def on_undo_landmark(event):
        if landmarks["pending"] is not None:
            landmarks["pending"] = None
        elif landmarks["pairs"]:
            landmarks["pairs"].pop()
            if len(landmarks["pairs"]) >= 2:
                fit_landmarks()
        draw_landmarks()

    def on_clear_landmarks(event):
        landmarks.update(pairs=[], pending=None)
        draw_landmarks()

    def on_mirror(event):
        if mirror_btn.tags:
            orientation.value = mirror_btn.tags[0]

    undo_lm.on_click(on_undo_landmark)
    clear_lm.on_click(on_clear_landmarks)
    mirror_btn.on_click(on_mirror)

    # -- drawing -------------------------------------------------------------------------
    header = pn.pane.HTML("", sizing_mode="stretch_width", margin=(6, 10, 0, 10),
                          styles={"font-size": "14px", "line-height": "1.5"})

    shown = {"what": ""}

    def refresh_header(what: str | None = None):
        if what is not None:
            shown["what"] = what
        sample = sample_select.value
        where = f"<b>{html.escape(sample)}</b> · " if n_samples > 1 else ""
        n = get_geometry(sample)[0].n_bins
        aligned = sample in alignment.records(dataset)
        header.object = (f"{where}{n:,} hexagons colored by <b>{shown['what']}</b> · "
                         + ("<span style='color:#2a7'>aligned</span>" if aligned else "not aligned yet"))

    def draw_cells():
        sample = sample_select.value
        binning = get_geometry(sample)[0]
        values, what = hex_values(sample)
        cells.data = {"px": binning.centers[:, 0], "py": binning.centers[:, 1],
                      **dict(zip("xy", get_sim().apply(binning.centers).T)),
                      "fill_color": colors_for(values, binning.n_bins)}
        refresh_header(what)

    def recolor(*events):
        gene_input.visible = show_select.value == _GENE
        sample = sample_select.value
        values, what = hex_values(sample)
        cells.data["fill_color"] = colors_for(values, get_geometry(sample)[0].n_bins)
        refresh_header(what)

    def start(*events):
        """A new sample or image: the cells back at their starting place, no landmarks."""
        sample = sample_select.value
        with pn.io.hold():
            draw_image()
            show_hex_size(sample)
            binning, anchor, _ = get_geometry(sample)
            landmarks.update(pairs=[], pending=None)
            sim = starting_sim(sample)
            set_sim(sim, base_r=binning.width / np.sqrt(3))
            # The black cross: the point the cells turn and scale around.
            center.data = {"px": [anchor[0]], "py": [anchor[1]], "x": [sim.center[0]], "y": [sim.center[1]]}
            draw_cells()
            draw_landmarks()
            refresh_apply()

    def on_sample(event):
        rec = alignment.records(dataset).get(event.new)
        with pn.io.hold():
            if rec is not None and rec[0] in images and rec[0] != image_select.value:
                image_select.value = rec[0]  # starts it, through on_image
            else:
                start()

    def on_orientation(event):
        """The cells stay where they are on screen; landmarks follow the image."""
        with pn.io.hold():
            draw_image()
            if landmarks["pairs"]:
                fit_landmarks()
            draw_landmarks()

    sample_select.param.watch(on_sample, "value")
    image_select.param.watch(start, "value")
    orientation.param.watch(on_orientation, "value")
    show_select.param.watch(recolor, "value")
    gene_input.param.watch(recolor, "value")
    hex_size.param.watch(on_hex_size, "value_throttled")
    reset_btn.on_click(lambda event: set_sim(starting_sim(sample_select.value, from_record=False)))

    def on_opacity(event):
        hexes.glyph.fill_alpha = event.new

    opacity.param.watch(on_opacity, "value")

    def use_mode(label: str):
        value = _MODES[label]
        # Moving / rotating takes over dragging; otherwise it pans the view.
        fig.toolbar.active_drag = None if value in ("move", "rotate") else pan_tool
        mode_hint.object = {
            "pan": "Dragging pans the view; the mouse wheel zooms.",
            "move": "Drag anywhere to move the cells. The mouse wheel zooms.",
            "rotate": "Drag around the black cross to turn the cells; dragging away from it or toward it "
                      "scales them. Hold Shift to only turn them. Start far from the cross for finer control.",
            "landmarks": "Click a spot on the cells, then the same spot on the image. Dragging pans the view.",
        }[value]
        with hold_sync():
            params.data = {**params.data, "mode": [value]}

    mode.param.watch(lambda event: use_mode(event.new), "value")

    # -- applying -------------------------------------------------------------------------
    armed = {"button": None, "timer": None, "label": ""}

    def regions_here() -> int:
        regions = state.regions
        return len(regions.in_sample(sample_select.value)) if regions is not None else 0

    def refresh_apply():
        sample = sample_select.value
        rec = alignment.records(dataset).get(sample)
        undo_btn.visible = rec is not None
        n = regions_here()
        note = ("Moves this sample's cells for good (the Spatial tab, Save .h5ad) - the original "
                "coordinates are kept as <code>obsm['spatial_original']</code>.")
        if rec is not None:
            note = f"Aligned to <b>{html.escape(rec[0])}</b>. Applying again replaces that. " + note
        if n:
            note += (f"<br><b style='color:#b45309'>⚠ This sample has {n} drawn region{'s' * (n > 1)}.</b> "
                     "Their outlines would no longer match the moved cells, so they will be removed.")
        apply_note.object = note

    def confirmed(button: pn.widgets.Button, label: str) -> bool:
        """With regions on the sample, a first click only asks for a second one."""
        n = regions_here()
        if not n or armed["button"] is button:
            disarm()
            return True
        disarm()
        button.name = f"Click again to confirm - removes {n} region{'s' * (n > 1)}"
        button.button_type = "danger"
        armed["button"] = button
        armed["label"] = label
        armed["timer"] = pn.state.curdoc.add_timeout_callback(disarm, _CONFIRM_MS)
        return False

    def disarm():
        button = armed["button"]
        if button is None:
            return
        if armed["timer"] is not None:
            with contextlib.suppress(ValueError):
                pn.state.curdoc.remove_timeout_callback(armed["timer"])
        button.name = armed["label"]
        button.button_type = "primary" if button is apply_btn else "light"
        armed.update(button=None, timer=None)

    def finish(sample: str, removed: int, message: str):
        for listener in state.alignment_listeners:
            listener(sample)
        if removed:
            message += f" Removed {removed} region{'s' * (removed > 1)} drawn on it."
        status.object = f'<span style="font-size:13px">{message}</span>'
        refresh_apply()
        refresh_header()

    def on_apply(event):
        if not confirmed(apply_btn, "Apply alignment"):
            return
        sample = sample_select.value
        M = alignment.compose(alignment.invert_matrix(orient()), get_sim().matrix())
        removed = state.regions.remove_sample(sample) if state.regions is not None else 0
        alignment.apply(dataset, sample, M, image_select.value)
        finish(sample, removed, f"Aligned <b>{html.escape(sample)}</b> to <b>{html.escape(image_select.value)}</b>"
                                " - the Spatial tab now shows it on this image.")

    def on_undo(event):
        if not confirmed(undo_btn, "Put the cells back where they were"):
            return
        sample = sample_select.value
        removed = state.regions.remove_sample(sample) if state.regions is not None else 0
        alignment.undo(dataset, sample)
        finish(sample, removed, f"Put <b>{html.escape(sample)}</b>'s cells back where they were.")

    apply_btn.on_click(on_apply)
    undo_btn.on_click(on_undo)
    if state.regions is not None:
        state.regions.listeners.append(refresh_apply)

    # -- layout ---------------------------------------------------------------------------
    use_mode(mode.value)
    start()
    rec = alignment.records(dataset).get(sample_select.value)
    if rec is not None and rec[0] in images and rec[0] != image_select.value:
        image_select.value = rec[0]

    def section(title: str) -> pn.pane.HTML:
        return pn.pane.HTML(f"<b>{title}</b>", margin=(12, 5, 0, 5), styles={"font-size": "14px"})

    sidebar = pn.Column(
        section("1. What to align"), sample_select, image_select, orientation,
        section("2. What the cells show"), show_select, gene_input, hex_size, hex_size_hint, opacity,
        section("3. Move the cells"), mode, mode_hint,
        pn.Row(x_input, y_input, sizing_mode="stretch_width", margin=0),
        pn.Row(scale_input, angle_input, sizing_mode="stretch_width", margin=0), reset_btn,
        section("Landmarks (faster)"), lm_hint, lm_status, mirror_btn,
        pn.Row(undo_lm, clear_lm, sizing_mode="stretch_width", margin=0),
        section("4. Apply"), apply_note, apply_btn, undo_btn, status,
        width=380, sizing_mode="stretch_height", scroll=True,
    )
    plot_area = pn.Column(header, pn.pane.Bokeh(fig, sizing_mode="stretch_both"), sizing_mode="stretch_both", margin=0)
    return pn.Row(sidebar, plot_area, sizing_mode="stretch_both")


#: Defines place(cx, cy, s, t): moves every source's x / y from its px / py (original
#: coordinates) by that transform, in place - the browser redraws them without sending
#: anything back - and sizes / turns the hexagons to match.
_PLACE_JS = """
function place(cx, cy, s, t) {
    const p = params.data, dx = p.d0x[0], dy = p.d0y[0];
    const c = Math.cos(t) * s, sn = Math.sin(t) * s;
    for (const src of srcs) {
        const d = src.data, px = d.px, py = d.py, x = d.x, y = d.y;
        for (let i = 0; i < px.length; i++) {
            const u = px[i] - dx, v = py[i] - dy;
            x[i] = cx + c * u - sn * v;
            y[i] = cy + sn * u + c * v;
        }
        src.change.emit();
    }
    // Bokeh's angles turn counter-clockwise on screen; t turns the cells clockwise.
    hexes.glyph.radius = {value: p.base_r[0] * s};
    hexes.glyph.angle = {value: -t};
}
"""

#: The transform changed (from Python, or at the end of a drag).
_ON_PARAMS_JS = """
const p = params.data;
place(p.cx[0], p.cy[0], p.s[0], p.theta[0]);
"""

#: A drag on the plot (PHASE: start / move / end) in the move or rotate mode: moves the
#: cells live, and only sends the new transform to Python when the drag ends.
_DRAG_JS = """
const p = params.data, mode = p.mode[0];
if (mode !== "move" && mode !== "rotate") return;
const drags = window._svAlignDrag || (window._svAlignDrag = {});
if (PHASE === "start") {
    drags[params.id] = {x: cb_obj.x, y: cb_obj.y, cx: p.cx[0], cy: p.cy[0], s: p.s[0], t: p.theta[0], now: null};
    return;
}
const d = drags[params.id];
if (d == null) return;
if (PHASE === "end") {
    delete drags[params.id];
    if (d.now != null) {
        params.data = {...p, cx: [d.now[0]], cy: [d.now[1]], s: [d.now[2]], theta: [d.now[3]]};
    }
    return;
}
let cx = d.cx, cy = d.cy, s = d.s, t = d.t;
if (mode === "move") {
    cx += cb_obj.x - d.x;
    cy += cb_obj.y - d.y;
} else {
    const r0 = Math.hypot(d.x - d.cx, d.y - d.cy), r1 = Math.hypot(cb_obj.x - d.cx, cb_obj.y - d.cy);
    if (r0 > 0 && r1 > 0) {
        t = d.t + Math.atan2(cb_obj.y - d.cy, cb_obj.x - d.cx) - Math.atan2(d.y - d.cy, d.x - d.cx);
        if (!(cb_obj.modifiers && cb_obj.modifiers.shift)) s = d.s * r1 / r0;
    }
}
d.now = [cx, cy, s, t];
place(cx, cy, s, t);
"""
