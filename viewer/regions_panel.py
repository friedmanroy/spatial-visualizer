"""The Spatial tab's "Regions & differential expression" panel: draw regions on the
plot with the lasso, list / rename / recolor / delete them, gray out everything else,
save them as an obs field, and compare them with differential expression (results in
the DE pop-up, de_dialog.DEDialog).
"""
from __future__ import annotations

import asyncio
import html
import sys
import traceback

import numpy as np
import panel as pn
from bokeh.events import SelectionGeometry
from bokeh.models import ColumnDataSource, LassoSelectTool, PanTool

from . import de
from . import regions as regions_mod
from .color_controls import HINT_STYLE, _SWATCH_CSS, _SWATCH_SIZE
from .colors import GRAY_RGBA

_NEW_REGION = "A new region"
#: a click of "Clear all" must be followed by another within this long (ms) to clear
_CONFIRM_MS = 4000


class RegionsPanel:
    """`get_sample`: () -> the sample shown; `on_change`: () -> None, recolors the plot
    after regions change; `add_field`: (column) -> None, offers a new obs column for
    coloring and colors by it; `color_by_genes`: (genes) -> None."""

    def __init__(self, state, fig, get_sample, on_change, add_field, color_by_genes):
        self.state = state
        self.dataset = state.spatial
        self.get_sample = get_sample
        self.on_change = on_change
        self.add_field = add_field
        self.regions = regions_mod.Regions(self.dataset)
        self.regions.listeners.append(self._on_regions_change)
        state.de_dialog.on_color_genes = color_by_genes

        # -- drawing on the plot ---------------------------------------------------
        # The lasso selects nothing itself (no renderers): its polygon is sent here and
        # tested against every cell of the sample, whatever the plot is drawing.
        self.lasso = LassoSelectTool(renderers=[], description="Lasso: draw a region")
        fig.add_tools(self.lasso)
        self._pan = fig.select_one(PanTool)
        self._fig = fig
        fig.on_event(SelectionGeometry, self._on_lasso)
        self._outline_source = ColumnDataSource({"xs": [], "ys": [], "color": []})
        self._label_source = ColumnDataSource({"x": [], "y": [], "name": [], "color": []})
        # A white halo under each colored outline, so it shows on cells of its own color too.
        self._halos = fig.patches("xs", "ys", source=self._outline_source, fill_alpha=0,
                                  line_color="white", line_width=5)
        self._outlines = fig.patches("xs", "ys", source=self._outline_source, fill_color="color",
                                     fill_alpha=0.08, line_color="color", line_width=2.5)
        self._labels = fig.text("x", "y", text="name", source=self._label_source, text_color="color",
                                text_font_size="13px", text_font_style="bold", text_align="center",
                                text_baseline="middle", text_outline_color="white")

        # -- the sidebar card ------------------------------------------------------
        self.draw_toggle = pn.widgets.Toggle(
            name="✎ Draw regions (lasso)", button_type="primary", button_style="outline",
            sizing_mode="stretch_width", margin=(8, 5, 2, 5),
        )
        draw_hint = pn.pane.HTML(
            "While on, dragging on the plot draws a region instead of panning (the lasso is also "
            "in the plot's toolbar).", styles=HINT_STYLE, sizing_mode="stretch_width", margin=(0, 5, 2, 5),
        )
        self.draw_toggle.param.watch(self._on_draw_toggle, "value")
        self.lasso_target = pn.widgets.Select(
            name="Each lasso draws", options=[_NEW_REGION], value=_NEW_REGION, sizing_mode="stretch_width",
            description="Pick a region to grow it with the next lassos, e.g. to cover a ring or "
                        "separate patches of the same tissue type.",
        )
        self.list_box = pn.Column(sizing_mode="stretch_width", margin=(4, 0))
        self.only_regions = pn.widgets.Checkbox(name="Gray out cells outside the regions", value=False,
                                                margin=(6, 5, 2, 5))
        self.show_outlines = pn.widgets.Checkbox(name="Show region outlines", value=True, margin=(2, 5))
        self.only_regions.param.watch(lambda e: self.on_change(), "value")
        self.show_outlines.param.watch(self._on_show_outlines, "value")
        self.save_btn = pn.widgets.Button(
            name=f"Save as field '{regions_mod.FIELD_NAME}'", sizing_mode="stretch_width", margin=(5, 5),
            description="Adds the regions as a field to color by (here and on the scRNA / snRNA tab)",
        )
        self.clear_btn = pn.widgets.Button(name="Clear all", button_type="light", width=110, margin=(5, 5))
        self.save_btn.on_click(self._on_save)
        self.clear_btn.on_click(self._on_clear)
        self._clear_armed = None  # timeout callback while waiting for a confirming click
        self.status = pn.pane.HTML("", sizing_mode="stretch_width", margin=(0, 5), styles=HINT_STYLE)

        # differential expression
        self.comparison = pn.widgets.Select(name="Compare", options=list(de.COMPARISONS), value=de.VS_REST,
                                            sizing_mode="stretch_width")
        self.region_a = pn.widgets.Select(name="Region A", options=[], sizing_mode="stretch_width")
        self.region_b = pn.widgets.Select(name="Region B", options=[], sizing_mode="stretch_width")
        self.method = pn.widgets.Select(name="Test", options=list(de.METHODS), sizing_mode="stretch_width")
        self.comparison.param.watch(self._update_de_widgets, "value")
        self.de_btn = pn.widgets.Button(name="Run differential expression", button_type="primary",
                                        sizing_mode="stretch_width", margin=(8, 5, 4, 5))
        self.de_btn.on_click(self._on_run_de)
        self.last_btn = pn.widgets.Button(name="Show last results", button_type="light",
                                          sizing_mode="stretch_width", margin=(0, 5, 4, 5), visible=False)
        self.last_btn.on_click(lambda e: state.de_dialog.open())
        self.de_status = pn.pane.HTML("", sizing_mode="stretch_width", margin=(0, 5), styles=HINT_STYLE)
        de_hint = pn.pane.HTML(
            f"Ranks the genes that differ between the cells of the compared regions "
            f"(scanpy's rank_genes_groups on the normalized data, background genes left out; "
            f"at most {de.DEFAULT_MAX_CELLS:,} randomly chosen cells per side).",
            styles=HINT_STYLE, sizing_mode="stretch_width", margin=(0, 5, 4, 5),
        )
        self.empty_hint = pn.pane.HTML(
            "No regions yet: turn on <b>Draw regions</b> and drag around an area of the tissue. "
            "Regions are saved with this file and come back next time it's opened.",
            styles=HINT_STYLE, sizing_mode="stretch_width", margin=(4, 5),
        )
        heading = lambda text: pn.pane.HTML(f"<b>{text}</b>", margin=(10, 5, 0, 5), styles={"font-size": "13px"})
        self.card = pn.Card(
            self.draw_toggle, draw_hint, self.lasso_target, self.empty_hint, self.list_box,
            self.only_regions, self.show_outlines,
            pn.Row(self.save_btn, self.clear_btn, sizing_mode="stretch_width", margin=0),
            self.status,
            heading("Differential expression"), de_hint,
            self.comparison, self.region_a, self.region_b, self.method, self.de_btn, self.last_btn,
            self.de_status,
            title="Regions & differential expression", collapsed=not self.regions.items,
            sizing_mode="stretch_width", margin=(4, 5),
        )
        self.refresh()

    # -- used by the Spatial tab -----------------------------------------------------
    def mask_colors(self, colors: np.ndarray, idx: np.ndarray, binning=None) -> np.ndarray:
        """`colors` (of the sample's cells `idx`, or of its bins) with everything outside
        the regions grayed out, if that's on and the sample has any regions."""
        sample = self.get_sample()
        if not self.only_regions.value or not self.regions.in_sample(sample) or len(colors) == 0:
            return colors
        inside = self.regions.mask(sample)[idx]
        if binning is not None:  # a bin is in if most of its cells are
            inside = binning.mean(inside.astype(float)) >= 0.5
        return np.where(inside, colors, GRAY_RGBA)

    def header_html(self) -> str:
        """One line for the header above the plot listing the sample's regions, or ''."""
        sample = self.get_sample()
        here = self.regions.in_sample(sample)
        if not self.regions.items:
            return ""
        parts = [f"<span style='color:{r.color}'>■</span> <b>{html.escape(r.name)}</b> ({len(r.cells):,})"
                 for r in here]
        elsewhere = len(self.regions.items) - len(here)
        text = "Regions: " + (", ".join(parts) if parts else "none on this sample")
        if elsewhere:
            text += f" · {elsewhere} on other samples"
        return f'<span style="color:#555;font-size:13px">{text}</span>'

    def refresh(self):
        """Redraws the outlines and the list for the sample shown."""
        sample = self.get_sample()
        here = self.regions.in_sample(sample)
        xs, ys, colors = [], [], []
        lx, ly, names, label_colors = [], [], [], []
        for r in here:
            for pxs, pys in r.polygons:
                xs.append(np.asarray(pxs)), ys.append(np.asarray(pys)), colors.append(r.color)
            # Name at the middle of the region's largest polygon.
            pxs, pys = max(r.polygons, key=lambda p: np.ptp(p[0]) * np.ptp(p[1]))
            lx.append(float(np.mean([min(pxs), max(pxs)]))), ly.append(float(np.mean([min(pys), max(pys)])))
            names.append(r.name), label_colors.append(r.color)
        self._outline_source.data = {"xs": xs, "ys": ys, "color": colors}
        self._label_source.data = {"x": lx, "y": ly, "name": names, "color": label_colors}

        self.list_box[:] = [self._region_row(r) for r in self.regions.items]
        has = bool(self.regions.items)
        self.empty_hint.visible = not has
        for w in (self.only_regions, self.show_outlines, self.save_btn, self.clear_btn):
            w.disabled = not has
        options = [_NEW_REGION] + [f"Add to {r.name}" for r in here]
        keep = self.lasso_target.value if self.lasso_target.value in options else _NEW_REGION
        self.lasso_target.options = options
        self.lasso_target.value = keep
        names_all = self.regions.names()
        for w, default in ((self.region_a, 0), (self.region_b, 1)):
            current = w.value
            w.options = names_all
            if current not in names_all and names_all:
                w.value = names_all[min(default, len(names_all) - 1)]
        if self.region_a.value == self.region_b.value and len(names_all) > 1:
            self.region_b.value = next(n for n in names_all if n != self.region_a.value)
        self._update_de_widgets()

    # -- callbacks -----------------------------------------------------------------
    def _region_row(self, r: regions_mod.Region) -> pn.viewable.Viewable:
        picker = pn.widgets.ColorPicker(value=r.color, name="", width=_SWATCH_SIZE, height=_SWATCH_SIZE,
                                        margin=(14, 6, 0, 8), stylesheets=[_SWATCH_CSS])
        picker.param.watch(lambda e, n=r.name: self.regions.set_color(n, e.new), "value")
        name = pn.widgets.TextInput(value=r.name, width=130, margin=(5, 4))

        def on_rename(event, old=r.name):
            if not self.regions.rename(old, event.new):
                event.obj.value = old

        name.param.watch(on_rename, "value")
        where = "" if len(self.dataset.sample_ids) == 1 else f" · {html.escape(r.sample)}"
        count = pn.pane.HTML(f"{len(r.cells):,} cells{where}", margin=(12, 4, 0, 4),
                             styles={"font-size": "12px", "color": "#555"}, sizing_mode="stretch_width")
        delete = pn.widgets.Button(name="✕", width=32, button_type="light", margin=(5, 5),
                                   description=f"Delete {r.name}")
        delete.on_click(lambda e, n=r.name: self.regions.remove(n))
        return pn.Row(picker, name, count, delete, sizing_mode="stretch_width", margin=0)

    def _on_regions_change(self):
        with pn.io.hold():
            self.refresh()
            self.on_change()

    def _on_draw_toggle(self, event):
        self._fig.toolbar.active_drag = self.lasso if event.new else self._pan
        if event.new:
            self.card.collapsed = False

    def _on_lasso(self, event):
        geometry = event.geometry or {}
        if not event.final or geometry.get("type") != "poly":
            return
        xs, ys = list(geometry.get("x", [])), list(geometry.get("y", []))
        if len(xs) < 3:
            return
        target = self.lasso_target.value
        to = target.removeprefix("Add to ") if target != _NEW_REGION else None
        region = self.regions.add_polygon(self.get_sample(), xs, ys, to=to)
        self.card.collapsed = False
        self.status.object = (f"{'Grew' if to else 'Drew'} <b>{html.escape(region.name)}</b>: "
                              f"{len(region.cells):,} cells." if len(region.cells)
                              else f"<b>{html.escape(region.name)}</b> has no cells yet - draw around some cells.")

    def _on_show_outlines(self, event):
        self._outlines.visible = self._halos.visible = self._labels.visible = event.new

    def _on_save(self, event):
        col = regions_mod.FIELD_NAME
        adata = self.dataset.adata
        adata.obs[col] = self.regions.as_field()
        adata.uns[f"{col}_colors"] = [r.color for r in self.regions.items]
        self.add_field(col)
        self.status.object = (f"Saved as the field <b>{col}</b> (where regions overlap, a cell gets the "
                              f"later one). Save again after changing the regions.")

    def _on_clear(self, event):
        doc = pn.state.curdoc
        if self._clear_armed is None:
            self.clear_btn.name = "Click to confirm"
            self.clear_btn.button_type = "danger"
            self._clear_armed = doc.add_timeout_callback(self._disarm_clear, _CONFIRM_MS)
            return
        doc.remove_timeout_callback(self._clear_armed)
        self._disarm_clear()
        self.regions.clear()
        self.status.object = "Cleared all regions."

    def _disarm_clear(self):
        self._clear_armed = None
        self.clear_btn.name = "Clear all"
        self.clear_btn.button_type = "light"

    def _update_de_widgets(self, *events):
        mode = self.comparison.value
        self.region_a.visible = self.region_b.visible = mode == de.A_VS_B
        n = len(self.regions.items)
        needed = {de.VS_REST: 1, de.VS_OTHERS: 2, de.A_VS_B: 2}[mode]
        self.de_btn.disabled = n < needed
        self.de_status.object = ("" if n >= needed else
                                 f"Draw {'a region' if needed == 1 else 'at least two regions'} first.")

    async def _on_run_de(self, event):
        mode = self.comparison.value
        try:
            comparisons = de.comparisons(self.regions, mode, self.region_a.value, self.region_b.value)
        except ValueError as exc:
            self.de_status.object = html.escape(str(exc))
            return
        method = de.METHODS[self.method.value]
        results, errors = [], []
        n = len(comparisons)
        busy = self.state.busy
        try:
            async with busy.show("Comparing regions (differential expression)...",
                                 f"{n} comparison{'s' if n != 1 else ''}, {self.method.value}"):
                for group, g_cells, ref, r_cells in comparisons:
                    busy.step(f"{group} vs {ref}")
                    try:
                        results.append(await asyncio.to_thread(de.run, self.dataset, group, g_cells, ref, r_cells,
                                                               method))
                    except ValueError as exc:
                        errors.append(str(exc))
                await busy.step_now("Showing the results")
                if results:
                    self.state.de_dialog.show(results, errors)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            self.de_status.object = f"<b>Differential expression failed:</b> {html.escape(str(exc))}"
            return
        self.last_btn.visible = bool(results)
        self.de_status.object = "<br>".join(html.escape(e) for e in errors)
