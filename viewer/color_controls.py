"""Shared 'Field' / 'Gene & Pathway' coloring controls, reused by the Spatial page and
the scRNA/snRNA embedding page so the two don't duplicate ~150 lines of widget wiring.

The Gene & Pathway section mirrors the TNBC viewer's layout closely: one shared
Gene/Pathway mode for all three channels, each channel's controls stacked vertically
(not side by side) so they stay within the sidebar's width, with a per-channel Clear
button and a small local-vs-whole-dataset histogram under each channel's name.
"""
from __future__ import annotations

import numpy as np
import panel as pn

from . import colors as colors_mod
from . import histogram as histogram_mod
from . import pathways as pathways_mod

GRAY = colors_mod.GRAY

_CHANNEL_LABEL_COLOR = {"R": "#ff4444", "G": "#44cc44", "B": "#4488ff"}
_CHANNEL_NAME = {"R": "Red", "G": "Green", "B": "Blue"}
_WIDGET_WIDTH = 300
#: left inset shared by every top-level row in the sidebar, so fields/buttons/
#: checkboxes all start flush against the same edge instead of drifting per widget.
_INSET = 10


class ColorControls:
    def __init__(self, get_dataset, get_pathways, on_change, get_active_indices=None):
        self.get_dataset = get_dataset  # () -> Dataset | None
        self.get_pathways = get_pathways  # () -> dict
        self.on_change = on_change  # () -> None
        # () -> np.ndarray | None - positional indices, into the full dataset, of the
        # cells currently on screen (e.g. the active sample). None means "no subset,
        # everything shown" - skips the local-vs-global histogram overlay.
        self.get_active_indices = get_active_indices or (lambda: None)

        self.mode = pn.widgets.RadioButtonGroup(
            options=["Field", "Gene & Pathway"], value="Field", button_type="primary",
            margin=(5, 0, 5, _INSET),
        )
        self.field_select = pn.widgets.Select(
            name="Color by field", options=[], margin=(5, 10, 5, _INSET),
        )
        self.value_picker_box = pn.Column(sizing_mode="stretch_both", margin=0)
        self.select_all_btn = pn.widgets.Button(name="Select all", width=110, margin=(5, 5, 5, 0))
        self.deselect_all_btn = pn.widgets.Button(name="Deselect all", width=110, margin=5)
        self._value_checkboxes: dict[str, pn.widgets.Checkbox] = {}
        # Rebuilding N checkbox/color-picker widgets from scratch is the expensive part
        # of switching fields (each one is its own Bokeh model that has to be created
        # and shipped to the browser) - caching the built picker per field means
        # switching back to an already-visited field is instant.
        self._value_picker_cache: dict[str, tuple] = {}

        self.expr_mode = pn.widgets.RadioButtonGroup(
            options=["Gene", "Pathway"], value="Gene", width=_WIDGET_WIDTH, margin=(5, 0, 5, _INSET)
        )

        self.channel_widgets: dict[str, dict] = {}
        channel_blocks = [self.expr_mode]
        for label in ("R", "G", "B"):
            gene_input = pn.widgets.AutocompleteInput(
                name=f"{label} gene", placeholder="e.g. Col1a1", width=_WIDGET_WIDTH - 60,
                options=[], min_characters=2, restrict=True, case_sensitive=False,
            )
            pathway_select = pn.widgets.Select(
                name=f"{label} pathway", options=[""], value="", visible=False, width=_WIDGET_WIDTH - 60
            )
            clear_btn = pn.widgets.Button(name="Clear", width=60, margin=(18, 0, 0, 5))
            info_pane = pn.pane.HTML("", visible=False, width=_WIDGET_WIDTH, margin=(0, 0, 0, _INSET))
            hist_pane = pn.pane.HTML("", visible=False, width=_WIDGET_WIDTH, margin=(0, 0, 0, _INSET))
            self.channel_widgets[label] = dict(
                gene=gene_input, pathway=pathway_select, info=info_pane, hist=hist_pane,
            )
            color = _CHANNEL_LABEL_COLOR[label]
            channel_blocks.append(
                pn.Column(
                    pn.pane.HTML(
                        f'<div style="color:{color};font-weight:bold;font-size:12px;'
                        f'margin-top:6px">{_CHANNEL_NAME[label]} channel</div>',
                        margin=(0, 0, 0, _INSET),
                    ),
                    pn.Row(gene_input, pathway_select, clear_btn, margin=(0, 0, 0, _INSET)),
                    hist_pane,
                    info_pane,
                    margin=0,
                )
            )
            clear_btn.on_click(lambda e, lbl=label: self._clear_channel(lbl))
        self.gene_pathway_section = pn.Column(*channel_blocks, visible=False)

        self.field_select.param.watch(self._on_field_change, "value")
        self.mode.param.watch(self._on_mode_change, "value")
        self.expr_mode.param.watch(self._on_expr_mode_change, "value")
        self.select_all_btn.on_click(lambda e: self._set_all(True))
        self.deselect_all_btn.on_click(lambda e: self._set_all(False))
        for w in self.channel_widgets.values():
            w["gene"].param.watch(self._fire, "value")
            w["pathway"].param.watch(self._fire, "value")

        self.layout = pn.Column(
            self.mode,
            self.field_select,
            pn.Row(self.select_all_btn, self.deselect_all_btn, margin=(0, 0, 0, _INSET)),
            self.value_picker_box,
            self.gene_pathway_section,
            scroll=True,
            sizing_mode="stretch_both",
        )

    def refresh_for_dataset(self):
        dataset = self.get_dataset()
        if dataset is None:
            return
        # Changing .options can make Panel auto-correct .value, which synchronously
        # re-enters our own watchers (e.g. redrawing the Bokeh plot) while the widget's
        # own property change is still being patched into the document - if the pane is
        # already attached to a live session document, that reentrant patch raises a
        # Bokeh "document lock" error. `pn.io.hold()` batches all of this into one patch.
        with pn.io.hold():
            self._value_picker_cache = {}  # a new/changed dataset invalidates cached pickers
            options = dataset.field_options()
            self.field_select.options = options
            if options and self.field_select.value not in options:
                self.field_select.value = options[0]
            else:
                self._rebuild_value_picker()
            pathway_names = sorted(self.get_pathways().keys())
            gene_names = list(dataset.adata.var_names)
            for w in self.channel_widgets.values():
                w["pathway"].options = [""] + pathway_names
                w["gene"].options = gene_names

    def _on_mode_change(self, event):
        # Several widget properties change here, each of which can synchronously
        # re-enter our own watchers (rebuilding the value picker, redrawing the plot).
        # Once this pane is attached to a live session document, an unheld nested
        # document patch like that raises a Bokeh document-lock error - pn.io.hold()
        # batches everything below into a single patch instead.
        with pn.io.hold():
            field_mode = event.new == "Field"
            self.field_select.visible = field_mode
            self.select_all_btn.visible = field_mode
            self.deselect_all_btn.visible = field_mode
            self.value_picker_box.visible = field_mode
            self.gene_pathway_section.visible = not field_mode
            self._fire()

    def _on_expr_mode_change(self, event):
        with pn.io.hold():
            is_gene = event.new == "Gene"
            for w in self.channel_widgets.values():
                w["gene"].visible = is_gene
                w["pathway"].visible = not is_gene
            self._fire()

    def _clear_channel(self, label: str):
        with pn.io.hold():
            w = self.channel_widgets[label]
            w["gene"].value = ""
            w["pathway"].value = ""

    def _on_field_change(self, event):
        with pn.io.hold():
            self._rebuild_value_picker()
            self._fire()

    def _rebuild_value_picker(self):
        dataset = self.get_dataset()
        field = self.field_select.value
        if dataset is None or not field:
            self._value_checkboxes = {}
            self.value_picker_box[:] = []
            return

        cached = self._value_picker_cache.get(field)
        if cached is not None:
            widget, checkboxes = cached
            self._value_checkboxes = checkboxes
            self.value_picker_box[:] = [widget]
            return

        spec = dataset.classify(field)
        self._value_checkboxes = {}
        rows = []
        for cat in spec.categories:
            color = spec.colors.get(cat, GRAY)
            # The swatch itself *is* the color control - clicking it opens the
            # browser's native color wheel, no separate widget needed.
            picker = pn.widgets.ColorPicker(value=color, name="", width=26, height=26, margin=(3, 8, 3, _INSET))
            picker.param.watch(lambda e, s=spec, f=field, c=cat: self._on_color_change(s, f, c, e.new), "value")
            cb = pn.widgets.Checkbox(name=cat, value=True, margin=(7, 5, 5, 0))
            cb.param.watch(self._fire, "value")
            rows.append(pn.Row(picker, cb, margin=0))
            self._value_checkboxes[cat] = cb
        widget = pn.Column(*rows, scroll=True, sizing_mode="stretch_both", margin=0)

        self._value_picker_cache[field] = (widget, self._value_checkboxes)
        self.value_picker_box[:] = [widget]

    def _on_color_change(self, spec, field: str, category: str, new_color: str):
        spec.colors[category] = new_color
        dataset = self.get_dataset()
        if dataset is not None and dataset.color_cache is not None:
            dataset.color_cache.set_color(field, category, new_color)
        self._fire()

    def _set_all(self, value: bool):
        with pn.io.hold():
            for cb in self._value_checkboxes.values():
                cb.value = value

    def _active_labels(self) -> set:
        return {cat for cat, cb in self._value_checkboxes.items() if cb.value}

    def _fire(self, *events):
        with pn.io.hold():
            self.on_change()

    def compute_colors(self) -> np.ndarray:
        dataset = self.get_dataset()
        if dataset is None:
            return np.array([], dtype=object)
        n = dataset.adata.n_obs
        if self.mode.value == "Field":
            if not self.field_select.value:
                return np.full(n, GRAY, dtype=object)
            spec = dataset.classify(self.field_select.value)
            active = self._active_labels()
            return colors_mod.field_fill_colors(spec, active)
        is_gene = self.expr_mode.value == "Gene"
        channels = {}
        channel_values = {}
        pathways = self.get_pathways()
        for label, w in self.channel_widgets.items():
            if is_gene:
                gene = w["gene"].value.strip()
                values = dataset.gene_values(gene) if gene else None
                channels[label] = (values, False)
            else:
                pname = w["pathway"].value
                values = pathways_mod.score_pathway(dataset, pname, pathways) if pname else None
                channels[label] = (values, True)
            channel_values[label] = values
        self._update_pathway_info(pathways, is_gene)
        self._update_histograms(channel_values)
        return colors_mod.compute_rgb_colors(n, channels)

    def _update_histograms(self, channel_values: dict[str, np.ndarray | None]):
        active_idx = self.get_active_indices()
        for label, values in channel_values.items():
            hist_pane = self.channel_widgets[label]["hist"]
            if values is None:
                hist_pane.object = ""
                hist_pane.visible = False
                continue
            local_values = values[active_idx] if active_idx is not None else values
            hist_pane.object = histogram_mod.render_histogram(local_values, values, _CHANNEL_LABEL_COLOR[label])
            hist_pane.visible = True

    def _update_pathway_info(self, pathways, is_gene: bool):
        for label, w in self.channel_widgets.items():
            pname = w["pathway"].value
            if not is_gene and pname:
                info = pathways.get(pname, {})
                genes = info.get("genes", [])
                w["info"].object = (
                    f"<div style='font-size:11px;color:#444;margin:2px 0 4px 0;line-height:1.35'>"
                    f"<i>{info.get('source', '')}</i><br>{info.get('description', '')}</div>"
                    f"<details style='font-size:10px;color:#666;margin:0 0 6px 0'>"
                    f"<summary style='cursor:pointer'>Genes ({len(genes)})</summary>"
                    f"<div style='margin-top:4px;font-family:monospace;line-height:1.4;"
                    f"word-break:break-word'>{', '.join(genes)}</div></details>"
                )
                w["info"].visible = True
            else:
                w["info"].object = ""
                w["info"].visible = False
