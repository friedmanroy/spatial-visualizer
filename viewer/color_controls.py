"""Shared 'Field' / 'Gene & Pathway' coloring controls, used by the Spatial page.

The Gene & Pathway section mirrors the TNBC viewer's layout closely: one shared
Gene/Pathway mode for all three channels, each channel's controls stacked vertically
(not side by side) so they stay within the sidebar's width, with a per-channel Clear
button and a small local-vs-whole-dataset histogram under each channel's name.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import PrintfTickFormatter

from . import background
from . import colors as colors_mod
from . import fields as fields_mod
from . import histogram as histogram_mod
from . import hover
from . import pathways as pathways_mod
from .state import ALL_SAMPLES

GRAY = colors_mod.GRAY

_CHANNEL_LABEL_COLOR = {"R": "#ff4444", "G": "#44cc44", "B": "#4488ff"}
_CHANNEL_NAME = {"R": "Red", "G": "Green", "B": "Blue"}
#: width (px) of the colored "● Red" / "● Green" / "● Blue" label in front of each channel
_CHANNEL_LABEL_WIDTH = 62
#: small gray explanatory text used under headers and for summaries
HINT_STYLE = {"font-size": "12px", "color": "#555", "line-height": "1.35"}
#: continuous fields are offered among the genes as "<name> (field)" - a suffix, so
#: typing the field's name in the search box finds it
_FIELD_SUFFIX = " (field)"
_GENE_HINT = (
    "Each channel colors cells by one gene: brighter = higher expression. "
    "Use up to three channels; overlapping expression mixes the colors. "
    f"Continuous fields (e.g. total counts, cell-type confidence, and the first principal components "
    f"PC1-PC5 of the expression) can go in a channel too - listed as <i>name{_FIELD_SUFFIX}</i>. "
    "On the Spatial tab, click a cell to show its 3 most expressed."
)
_PATHWAY_HINT = (
    "Each channel colors cells by a pathway score: its color where the pathway's genes "
    "are higher than comparable control genes, the opposite color where lower."
)
_PATHWAY_PROMPT = "Choose a pathway…"
#: how many of the most highly variable genes the gene dropdown offers
_N_DROPDOWN_GENES = 200
_DROPDOWN_PROMPT = f"or pick a continuous field / top-{_N_DROPDOWN_GENES} variable gene"
#: left inset shared by every top-level row in the sidebar, so fields/buttons/
#: checkboxes all start flush against the same edge instead of drifting per widget.
_INSET = 1
#: how many hexagon value arrays of other samples (for the histograms) are kept
_BIN_VALUES_CACHE_SIZE = 128
#: Side length (px) of each category's color swatch button, and the vertical space
#: (px) left between neighbouring swatches in the category list.
_SWATCH_SIZE = 18
_SWATCH_GAP = 8
#: Strips the browser's default frame/padding around a native <input type="color"> so
#: the chosen color fills the whole swatch button instead of a small inner rectangle.
#: The explicit size overrides Bokeh's own min-height for inputs, which otherwise keeps
#: the swatch tall regardless of the widget's `height`.
_SWATCH_CSS = f"""
:host {{ height: {_SWATCH_SIZE}px !important; min-height: 0 !important; }}
input[type="color"] {{
  padding: 0; border: 1px solid rgba(0, 0, 0, 0.25); border-radius: 4px;
  width: {_SWATCH_SIZE}px !important; height: {_SWATCH_SIZE}px !important;
  min-height: 0 !important; cursor: pointer; background: none;
}}
input[type="color"]::-webkit-color-swatch-wrapper {{ padding: 0; }}
input[type="color"]::-webkit-color-swatch {{ border: none; border-radius: 3px; }}
input[type="color"]::-moz-color-swatch {{ border: none; border-radius: 3px; }}
"""


#: the color range slider is placed so its track spans the histogram's value axis
#: (Bokeh's slider track is inset by about this much (px) from the widget's edges)
_SLIDER_TRACK_INSET = 1
_SLIDER_LEFT = _INSET + int(histogram_mod.AXIS_INSET_PX) - _SLIDER_TRACK_INSET
_SLIDER_WIDTH = histogram_mod.WIDTH_PX - 2 * int(histogram_mod.AXIS_INSET_PX) + 2 * _SLIDER_TRACK_INSET



def section_label(text: str) -> pn.pane.HTML:
    """Small bold heading above a group of controls (e.g. a toggle), so each group
    says what it's for."""
    return pn.pane.HTML(f"<b>{text}</b>", margin=(8, 5, 0, _INSET + 3), styles={"font-size": "13px"})


def keep_filled(*widgets):
    """Number inputs can be emptied (value None), which the code reading them doesn't
    expect: an emptied box gets its previous value back. Call right after creating the
    widgets, so this watcher runs before any other."""
    def restore(event):
        if event.new is None and event.old is not None:
            widget = event.obj
            widget.value = event.old
            # Panel doesn't echo a value back to the browser while it's handling that same
            # change from it, so the box would still look empty: re-send it on the next tick.
            doc = pn.state.curdoc
            if doc is not None and doc.session_context is not None:
                doc.add_next_tick_callback(lambda: widget.param.trigger("value"))

    for w in widgets:
        w.param.watch(restore, "value")


def _pathway_options(names: list[str]) -> dict[str, str]:
    return {_PATHWAY_PROMPT: "", **{n: n for n in names}}


class ColorControls:
    def __init__(self, get_dataset, get_pathways, on_change, get_active_indices=None, get_hex_width=None):
        self.get_dataset = get_dataset  # () -> Dataset | None
        self.get_pathways = get_pathways  # () -> dict
        self.on_change = on_change  # () -> None
        # () -> np.ndarray | None - positional indices, into the full dataset, of the
        # cells currently on screen (e.g. the active sample). None means "no subset,
        # everything shown" - skips the local-vs-global histogram overlay.
        self.get_active_indices = get_active_indices or (lambda: None)
        # () -> width (in cell spacings) of the hexagons drawn, or None: with hexagons
        # drawn, the histograms show the hexagons' values, as colored.
        self.get_hex_width = get_hex_width or (lambda: None)

        # Outline style: the active choice is filled, the others white - so which mode
        # is on can be read at a glance (two shades of blue were hard to tell apart).
        self.mode = pn.widgets.RadioButtonGroup(
            options=["Field", "Gene & Pathway"], value="Field", button_type="primary",
            button_style="outline", sizing_mode="stretch_width", margin=(5, 5, 5, _INSET),
        )
        self.field_select = pn.widgets.Select(
            name="Field", options=[], sizing_mode="stretch_width", margin=(5, 5, 0, _INSET),
        )
        self.select_all_btn = pn.widgets.Button(
            name="Select all", button_type="light", width=90, height=28, margin=(2, 2, 2, _INSET),
        )
        self.deselect_all_btn = pn.widgets.Button(
            name="Select none", button_type="light", width=90, height=28, margin=2,
        )
        # e.g. "12 categories · 14,036 cells" - what the list below covers.
        self.field_summary = pn.pane.HTML("", margin=(0, 5, 2, _INSET + 3), styles=HINT_STYLE)
        self.field_header = pn.Column(
            pn.Row(self.select_all_btn, self.deselect_all_btn, margin=0),
            self.field_summary, margin=0, sizing_mode="stretch_width",
        )
        # The category list is the one part of the sidebar that grows to fill the
        # remaining height, and it scrolls on its own (no nested scrollbars).
        self.value_picker_box = pn.Column(sizing_mode="stretch_both", margin=0)
        self._value_checkboxes: dict[str, pn.widgets.Checkbox] = {}
        self._value_rows: dict[str, pn.Row] = {}  # category -> its swatch + checkbox row
        # Fields with more than fields.CHECKBOX_MAX categories get this searchable
        # multi-select instead (empty = every category shown).
        self._multi: pn.widgets.MultiChoice | None = None
        # Other samples' hexagon values for the histograms: key -> (cell values used, bin values).
        self._bin_values_cache: dict = {}
        # bumped on every histogram update, so a background overlay for an older one is dropped
        self._hist_generation = 0
        # Rebuilding N checkbox/color-picker widgets from scratch is the expensive part
        # of switching fields (each one is its own Bokeh model that has to be created
        # and shipped to the browser) - caching the built picker per field means
        # switching back to an already-visited field is instant.
        self._value_picker_cache: dict[str, tuple] = {}

        self.expr_mode = pn.widgets.RadioButtonGroup(
            options=["Gene", "Pathway"], value="Gene", button_type="primary", button_style="outline",
            sizing_mode="stretch_width", margin=(5, 5, 5, _INSET),
        )
        self.expr_hint = pn.pane.HTML(_GENE_HINT, margin=(0, 5, 4, _INSET), styles=HINT_STYLE)
        self.filter_note = pn.pane.HTML(
            "", visible=False, sizing_mode="stretch_width", margin=(0, 5, 6, _INSET),
            styles={**HINT_STYLE, "color": "#7a4b00", "background": "#fff4dc",
                    "padding": "4px 6px", "border-radius": "4px"},
        )

        self.channel_widgets: dict[str, dict] = {}
        self._hold_fire = False  # set while several channels change at once (set_genes)
        channel_blocks = [self.expr_mode, self.expr_hint, self.filter_note]
        for label in ("R", "G", "B"):
            gene_input = pn.widgets.AutocompleteInput(
                name="", placeholder="Search any gene, e.g. Col1a1", sizing_mode="stretch_width",
                options=[], min_characters=2, restrict=True, case_sensitive=False, margin=(2, 2, 2, 2),
            )
            pathway_select = pn.widgets.Select(
                name="", options={_PATHWAY_PROMPT: ""}, value="", visible=False,
                sizing_mode="stretch_width", margin=(2, 2, 2, 2),
            )
            # Browsable alternative to searching: the most highly variable genes, most variable first.
            gene_dropdown = pn.widgets.Select(
                name="", options={_DROPDOWN_PROMPT: ""}, value="",
                sizing_mode="stretch_width", margin=(0, 2, 2, _CHANNEL_LABEL_WIDTH + _INSET + 2),
            )
            clear_btn = pn.widgets.Button(
                name="✕", button_type="light", width=32, height=32, margin=(2, 5, 2, 0),
                description=f"Clear the {_CHANNEL_NAME[label].lower()} channel",
            )
            info_pane = pn.pane.HTML("", visible=False, sizing_mode="stretch_width", margin=(0, 5, 0, _INSET))
            # (Fixed width: the color range slider under it lines up with its value axis.)
            hist_pane = pn.pane.HTML("", visible=False, width=histogram_mod.WIDTH_PX, margin=(0, 5, 0, _INSET))
            # The values the channel's colors run between, on the histogram's axis: a lower
            # top for genes / pathways whose few high values would leave everything else
            # dim, or a bottom of 0 for pathway scores' values above it alone.
            range_slider = pn.widgets.RangeSlider(
                name="Color range", start=0.0, end=1.0, value=(0.0, 1.0), step=0.01, visible=False,
                format=PrintfTickFormatter(format="%.3g"), bar_color=_CHANNEL_LABEL_COLOR[label],
                width=_SLIDER_WIDTH, margin=(0, 5, 4, _SLIDER_LEFT),
            )
            self.channel_widgets[label] = dict(
                gene=gene_input, gene_dropdown=gene_dropdown, pathway=pathway_select, info=info_pane, hist=hist_pane,
                range=range_slider, chosen_range=None,  # (low, high) the user set, else None: automatic
            )
            color = _CHANNEL_LABEL_COLOR[label]
            channel_label = pn.pane.HTML(
                f'<div style="color:{color};font-weight:bold;font-size:13px">'
                f'● {_CHANNEL_NAME[label]}</div>',
                width=_CHANNEL_LABEL_WIDTH, margin=(10, 0, 0, _INSET),
            )
            channel_blocks.append(
                pn.Column(
                    pn.Row(channel_label, gene_input, pathway_select, clear_btn, margin=0,
                           sizing_mode="stretch_width"),
                    gene_dropdown,
                    hist_pane,
                    range_slider,
                    info_pane,
                    margin=(0, 0, 6, 0),
                    sizing_mode="stretch_width",
                )
            )
            clear_btn.on_click(lambda e, lbl=label: self._clear_channel(lbl))
        self.gene_pathway_section = pn.Column(
            *channel_blocks, visible=False, scroll=True, sizing_mode="stretch_both", margin=0,
        )

        self.field_select.param.watch(self._on_field_change, "value")
        self.mode.param.watch(self._on_mode_change, "value")
        self.expr_mode.param.watch(self._on_expr_mode_change, "value")
        self.select_all_btn.on_click(lambda e: self._set_all(True))
        self.deselect_all_btn.on_click(lambda e: self._set_all(False))
        for w in self.channel_widgets.values():
            # A new gene / pathway starts at its own full range (watched first: before recoloring).
            w["gene"].param.watch(lambda e, w=w: self._reset_range(w), "value")
            w["pathway"].param.watch(lambda e, w=w: self._reset_range(w), "value")
            w["gene"].param.watch(self._fire, "value")
            w["gene"].param.watch(lambda e, w=w: self._sync_dropdown(w), "value")
            w["gene_dropdown"].param.watch(lambda e, w=w: self._on_dropdown_pick(w, e.new), "value")
            w["pathway"].param.watch(self._fire, "value")
            w["range"].param.watch(lambda e, w=w: self._on_range_pick(w, e.new), "value_throttled")

        self.layout = pn.Column(
            section_label("Color cells by"),
            self.mode,
            self.field_select,
            self.field_header,
            self.value_picker_box,
            self.gene_pathway_section,
            sizing_mode="stretch_both",
            margin=0,
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
            self._bin_values_cache = {}
            options = dataset.field_options()
            self.field_select.options = options
            if options and self.field_select.value not in options:
                self.field_select.value = options[0]
            else:
                self._rebuild_value_picker()
            pathway_names = sorted(self.get_pathways().keys())
            for w in self.channel_widgets.values():
                w["pathway"].options = _pathway_options(pathway_names)
            self.refresh_gene_suggestions()

    def refresh_gene_suggestions(self):
        """Re-fills each channel's gene search options and dropdown (continuous fields,
        then the most variable genes) - e.g. after the background-gene filter
        (Dataset.gene_filter) changed, or a field was added."""
        dataset = self.get_dataset()
        if dataset is None:
            return
        fields = list(self._field_channels(dataset))
        search_options = list(dataset.adata.var_names) + fields
        dropdown_options = {_DROPDOWN_PROMPT: ""}
        dropdown_options.update({f: f for f in fields})
        dropdown_options.update({g: g for g in dataset.top_variable_genes(_N_DROPDOWN_GENES)})
        with pn.io.hold():
            for w in self.channel_widgets.values():
                w["gene"].options = search_options
                w["gene_dropdown"].options = dropdown_options
                self._sync_dropdown(w)

    # -- continuous fields shown like genes ----------------------------------------
    @staticmethod
    def _field_channels(dataset) -> dict[str, str]:
        """{"<name> (field)": obs column} for the continuous fields offered for coloring."""
        return {f"{c}{_FIELD_SUFFIX}": c for c in dataset.field_options()
                if dataset.classify(c).kind == "binned_continuous"}

    @staticmethod
    def _field_column(dataset, name: str) -> str | None:
        """The obs column behind a "<name> (field)" channel entry, else None (a gene)."""
        if not name.endswith(_FIELD_SUFFIX):
            return None
        col = name[:-len(_FIELD_SUFFIX)]
        if col in dataset.adata.obs.columns and dataset.classify(col).kind == "binned_continuous":
            return col
        return None

    def _channel_values(self, dataset, name: str) -> np.ndarray | None:
        """Per-cell values of a gene channel entry: a gene's expression, or a continuous
        field's values (NaN where missing)."""
        col = self._field_column(dataset, name)
        if col is None:
            return dataset.gene_values(name)
        return pd.to_numeric(dataset.adata.obs[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)

    def _bin_channel_values(self, dataset, name: str, values: np.ndarray, idx: np.ndarray, binning, mask=None):
        """Per-bin values of a gene channel entry: a gene as Dataset.bin_gene_values makes
        it, a field as its cells' mean (over cells with a value, and in `mask` if given)."""
        if self._field_column(dataset, name) is None:
            return dataset.bin_gene_values(name, idx, binning, mask)
        has_value = np.isfinite(values[idx])
        return binning.mean(values[idx], has_value if mask is None else mask & has_value)

    @staticmethod
    def _bin_field_means(dataset, spec, idx, binning, active) -> tuple[np.ndarray, np.ndarray]:
        """(each bin's value-range code, its mean) for a continuous field: the mean over the
        bin's cells (of the sample's cells `idx`) with a value in one of the `active` ranges,
        placed in the field's ranges. Code -1 and NaN for bins without any."""
        values = pd.to_numeric(dataset.adata.obs[spec.name], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        active_codes = np.array([c in active for c in spec.categories] + [False])
        selected = active_codes[spec.codes[idx]] & np.isfinite(values[idx])
        means = binning.mean(values[idx], selected)
        codes = np.minimum(np.searchsorted(spec.bin_edges, means), len(spec.categories) - 1)
        return np.where(np.isfinite(means), codes, -1).astype(np.int32), means

    @staticmethod
    def _pathway_values(dataset, name: str, pathways, binning):
        """What a pathway channel needs before binning: per-cell scores - or, for hexagons
        scored from their pooled counts (Dataset.pools_hexagons), just the pathway's genes,
        since the hexagons are scored themselves. None if too few genes match."""
        if binning is not None and dataset.pools_hexagons:
            return pathways_mod.pathway_genes(dataset, name, pathways)
        return pathways_mod.score_pathway(dataset, name, pathways)

    @staticmethod
    def _bin_pathway_values(dataset, name: str, values, idx, binning, mask, pathways):
        """Per-bin values of a pathway channel (see _pathway_values): each hexagon's score
        on its pooled profile, or its cells' mean score. A coarser display level's hexagons
        get their analysis hexagons' scores averaged (over their cells), rather than being
        pooled and scored again."""
        if isinstance(values, tuple):
            base = binning.base
            scores = pathways_mod.score_pathway_bins(dataset, name, pathways, idx, base or binning, mask)
            if base is None or scores is None:
                return scores
            per_cell = scores[base.cell_bin]
            scored = np.isfinite(per_cell)
            return binning.mean(per_cell, scored if mask is None else scored & mask)
        return binning.mean(values[idx], mask)

    def refresh_field_options(self, changed_field: str, select: bool = False):
        """Re-reads the field list after an obs column was added/recomputed (e.g. Leiden),
        optionally switching to Field mode and coloring by that column."""
        dataset = self.get_dataset()
        if dataset is None:
            return
        with pn.io.hold():
            self._value_picker_cache.pop(changed_field, None)  # its categories may have changed
            self.field_select.options = dataset.field_options()
            self.refresh_gene_suggestions()  # a continuous field can go in a gene channel too
            if select:
                self.mode.value = "Field"
            if select and self.field_select.value != changed_field:
                self.field_select.value = changed_field  # its watcher rebuilds + recolors
            elif self.field_select.value == changed_field:
                self._rebuild_value_picker()
                self._fire()

    def refresh_pathway_options(self, changed_name: str | None = None):
        """Re-reads the pathway list after the Pathways tab added/changed/removed one,
        recoloring if a channel is showing the pathway that changed."""
        pathway_names = sorted(self.get_pathways().keys())
        with pn.io.hold():
            for w in self.channel_widgets.values():
                if w["pathway"].value not in pathway_names:
                    w["pathway"].value = ""
                w["pathway"].options = _pathway_options(pathway_names)
            if changed_name and any(w["pathway"].value == changed_name for w in self.channel_widgets.values()):
                self._fire()

    def _on_mode_change(self, event):
        # Several widget properties change here, each of which can synchronously
        # re-enter our own watchers (rebuilding the value picker, redrawing the plot).
        # Once this pane is attached to a live session document, an unheld nested
        # document patch like that raises a Bokeh document-lock error - pn.io.hold()
        # batches everything below into a single patch instead.
        with pn.io.hold():
            field_mode = event.new == "Field"
            self.field_select.visible = field_mode
            self.field_header.visible = field_mode
            self.value_picker_box.visible = field_mode
            self.gene_pathway_section.visible = not field_mode
            self._fire()

    def _on_expr_mode_change(self, event):
        with pn.io.hold():
            is_gene = event.new == "Gene"
            for w in self.channel_widgets.values():
                w["gene"].visible = is_gene
                w["gene_dropdown"].visible = is_gene
                w["pathway"].visible = not is_gene
            self.expr_hint.object = _GENE_HINT if is_gene else _PATHWAY_HINT
            self._fire()

    def _on_dropdown_pick(self, w: dict, gene: str):
        # Only push a real pick into the search box; the gene box's own watcher
        # does the recoloring, so the dropdown never fires on its own.
        if gene and w["gene"].value != gene:
            w["gene"].value = gene

    def _sync_dropdown(self, w: dict):
        """Mirrors the searched gene in the dropdown (or resets it to the prompt)."""
        dataset = self.get_dataset()
        gene = w["gene"].value.strip()
        matched = None
        if dataset is not None and gene:
            matched = gene if self._field_column(dataset, gene) is not None else dataset.match_gene(gene)
        w["gene_dropdown"].value = matched if matched in w["gene_dropdown"].values else ""

    def _reset_range(self, w: dict):
        """Back to the automatic color range (a new gene / pathway has its own values)."""
        w["chosen_range"] = None

    def _on_range_pick(self, w: dict, value):
        # A handle dropped next to 0 is put on it: 0 is where pathway scores change color,
        # and hard to hit exactly by dragging.
        near = 2 * w["range"].step
        low, high = (0.0 if abs(v) <= near else float(v) for v in value)
        w["chosen_range"] = (low, high)
        if (low, high) != tuple(value):
            w["range"].value = (low, high)
        self._fire()

    def _chosen_ranges(self) -> dict:
        return {label: w["chosen_range"] for label, w in self.channel_widgets.items()}

    @staticmethod
    def _set_slider(slider, start: float, end: float, value, chosen: bool):
        """The slider's bounds and handles at once: a range the user `chosen` widens the
        bounds to fit, any other is cut to them. Setting them doesn't recolor: only
        dragging does (value_throttled)."""
        if end <= start:
            end = start + 1.0
        low, high = value
        if chosen:
            start, end = min(start, low), max(end, high)
        else:
            low, high = min(max(low, start), end), min(max(high, start), end)
        slider.param.update(start=start, end=end, step=(end - start) / 200, value=(low, high))

    def _show_ranges(self, used: dict):
        """Each channel's slider handles at the range it's colored over (the automatic one
        unless the user picked one)."""
        for label, w in self.channel_widgets.items():
            if label in used:
                slider = w["range"]
                self._set_slider(slider, slider.start, slider.end, used[label], w["chosen_range"] is not None)

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
            self._multi = None
            self.value_picker_box[:] = []
            self.field_summary.object = ""
            return

        cached = self._value_picker_cache.get(field)
        if cached is not None:
            widget, checkboxes, rows, multi = cached
            self._value_checkboxes = checkboxes
            self._value_rows = rows
            self._multi = multi
            self._show_select_buttons()
            self.value_picker_box[:] = [widget]
            self.update_category_counts()
            return

        spec = dataset.classify(field)
        self._value_checkboxes = {}
        rows = {}
        if len(spec.categories) > fields_mod.CHECKBOX_MAX:
            # A checkbox + color swatch each would be thousands of widgets: a searchable
            # multi-select of the categories to show instead (their colors stay the defaults).
            self._multi = pn.widgets.MultiChoice(
                name="Show only these (type to search; empty = all)", options=list(spec.categories),
                value=[], sizing_mode="stretch_width", margin=(5, 5, 5, _INSET),
            )
            self._multi.param.watch(self._fire, "value")
            widget = pn.Column(self._multi, sizing_mode="stretch_width", margin=0)
            self._value_rows = rows
            self._value_picker_cache[field] = (widget, self._value_checkboxes, rows, self._multi)
            self._show_select_buttons()
            self.value_picker_box[:] = [widget]
            self.update_category_counts()
            return
        self._multi = None
        self._show_select_buttons()
        for cat in spec.categories:
            color = spec.colors.get(cat, GRAY)
            # The swatch itself *is* the color control - clicking it opens the
            # browser's native color wheel, no separate widget needed.
            picker = pn.widgets.ColorPicker(
                value=color, name="", width=_SWATCH_SIZE, height=_SWATCH_SIZE,
                margin=(_SWATCH_GAP // 2, 8, _SWATCH_GAP // 2, _INSET + 3),
                stylesheets=[_SWATCH_CSS],
            )
            picker.param.watch(lambda e, s=spec, f=field, c=cat: self._on_color_change(s, f, c, e.new), "value")
            cb = pn.widgets.Checkbox(
                name=cat, value=True, margin=(_SWATCH_GAP // 2, 5, _SWATCH_GAP // 2, 0),
            )
            cb.param.watch(self._fire, "value")
            rows[cat] = pn.Row(picker, cb, margin=0)
            self._value_checkboxes[cat] = cb
        self._value_rows = rows
        widget = pn.Column(*rows.values(), scroll=True, sizing_mode="stretch_both", margin=0)

        self._value_picker_cache[field] = (widget, self._value_checkboxes, rows, None)
        self.value_picker_box[:] = [widget]
        self.update_category_counts()

    def _show_select_buttons(self):
        """Select all / none only apply to the checkbox list (empty already means all)."""
        self.select_all_btn.visible = self.deselect_all_btn.visible = self._multi is None

    def update_category_counts(self):
        """Puts each category's cell count - among the cells currently shown (e.g. the
        active sample) - next to its name, plus a one-line summary above the list."""
        dataset = self.get_dataset()
        field = self.field_select.value
        if dataset is None or not field or (not self._value_checkboxes and self._multi is None):
            return
        spec = dataset.classify(field)
        active_idx = self.get_active_indices()
        codes = spec.codes if active_idx is None else spec.codes[active_idx]
        counts = np.bincount(codes[codes >= 0], minlength=len(spec.categories))
        with pn.io.hold():
            for cat, count in zip(spec.categories, counts):
                if cat in self._value_checkboxes:
                    self._value_checkboxes[cat].name = f"{cat}  ({count:,})"
            # Categories: biggest first (stable, so ties keep their natural order).
            # Value ranges of a continuous field keep their natural low-to-high order.
            if spec.kind == "categorical" and self._multi is None:
                order = np.argsort(-counts, kind="stable")
                rows = [self._value_rows[spec.categories[i]] for i in order]
                if self.value_picker_box.objects:
                    self.value_picker_box[0].objects = rows
            noun = "value ranges" if spec.kind == "binned_continuous" else "categories"
            where = "in this sample" if active_idx is not None else "in total"
            self.field_summary.object = f"{len(spec.categories)} {noun} · cell counts {where}"

    def _on_color_change(self, spec, field: str, category: str, new_color: str):
        spec.colors[category] = new_color
        dataset = self.get_dataset()
        if dataset is not None and dataset.color_cache is not None:
            dataset.color_cache.set_color(field, category, new_color)
        self._fire()

    def _set_all(self, value: bool):
        with pn.io.hold():
            if self._multi is not None:
                self._multi.value = []  # empty = all shown
            for cb in self._value_checkboxes.values():
                cb.value = value

    def coloring_summary_html(self) -> str:
        """One short HTML phrase saying what the plot is colored by right now, for the
        header above the plot (so it's readable without looking at the sidebar)."""
        if self.mode.value == "Field":
            field = self.field_select.value
            return f"Colored by <b>{html.escape(field)}</b>" if field else "No field to color by"
        is_gene = self.expr_mode.value == "Gene"
        parts = []
        for label, w in self.channel_widgets.items():
            name = w["gene"].value.strip() if is_gene else w["pathway"].value
            if name:
                parts.append(
                    f'<span style="color:{_CHANNEL_LABEL_COLOR[label]};font-weight:bold">'
                    f'{_CHANNEL_NAME[label]}</span> = {html.escape(name)}'
                )
        if not parts:
            kind = "gene" if is_gene else "pathway"
            return f"<i>Nothing colored yet: choose a {kind} for at least one channel in the sidebar</i>"
        summary = ("Genes: " if is_gene else "Pathway scores: ") + " · ".join(parts)
        f = self.category_filter()
        if f is not None:
            field, n_selected, n_total = f
            summary += (f" · only cells in {n_selected} of {n_total} selected "
                        f"<b>{html.escape(field)}</b> categories")
        return summary

    def active_labels(self) -> set:
        """The current field's selected categories (value ranges)."""
        return self._active_labels()

    def _active_labels(self) -> set:
        if self._multi is not None:
            return set(self._multi.value) or set(self._multi.options)
        return {cat for cat, cb in self._value_checkboxes.items() if cb.value}

    def _n_categories(self) -> int:
        return len(self._multi.options) if self._multi is not None else len(self._value_checkboxes)

    def in_gene_view(self) -> bool:
        return self.mode.value == "Gene & Pathway" and self.expr_mode.value == "Gene"

    def set_genes(self, genes: list[str]):
        """Puts `genes` in the red, green and blue channels (in that order), recoloring once.
        Channels beyond the given genes are cleared, so no earlier gene lingers."""
        self._hold_fire = True
        try:
            for i, w in enumerate(self.channel_widgets.values()):
                w["gene"].value = genes[i] if i < len(genes) else ""
        finally:
            self._hold_fire = False
        self._fire()

    def _fire(self, *events):
        if self._hold_fire:
            return
        with pn.io.hold():
            self.on_change()

    def compute_colors(self, idx: np.ndarray | None = None, binning=None, histograms: bool = True) -> np.ndarray:
        """Packed colors (see colors.pack_rgb) for every cell, or just the cells `idx`
        (e.g. a sample's) - or, given those cells' hexbin.HexBinning, for every bin: the
        bin's gene/pathway value, or its most common category among the selected ones.
        Gene / pathway colors are scaled over what is colored (those cells, or bins).
        `histograms`: also redraw the channels' histograms (from what's colored)."""
        dataset = self.get_dataset()
        if dataset is None:
            return np.array([], dtype=np.uint32)
        if self.mode.value == "Field":
            return self._field_colors(dataset, idx, binning)
        return self._expression_colors(dataset, idx, binning, self._expression(dataset, idx, binning), histograms)

    def render(self, idx: np.ndarray | None = None, binning=None, with_counts: bool = False,
               histograms: bool = True):
        """(compute_colors(idx, binning, histograms), *hover_data(idx, binning), has_counts(idx,
        binning) if `with_counts` else None) - with the gene / pathway channels computed once for all."""
        dataset = self.get_dataset()
        if dataset is None:
            return self.compute_colors(idx, binning), *self.hover_data(idx, binning), None
        if self.mode.value == "Field":
            return (self.compute_colors(idx, binning), *self.hover_data(idx, binning),
                    self.has_counts(idx, binning) if with_counts else None)
        ex = self._expression(dataset, idx, binning)
        return (self._expression_colors(dataset, idx, binning, ex, histograms),
                *self._expression_hover(binning, ex),
                self.has_counts(idx, binning, ex) if with_counts else None)

    def has_counts(self, idx: np.ndarray | None = None, binning=None, ex: dict | None = None) -> np.ndarray:
        """Per drawn cell (of `idx`, or all) or bin of `binning`: does it have counts for
        what's shown? In the Gene view, a nonzero value for at least one gene shown
        (continuous fields in a channel aren't counts, so they don't decide); otherwise - the
        Field or Pathway view, or only fields in the channels - any counts at all. A NaN
        value (a bin without a cell of the selected categories) counts as having some, so
        such bins stay drawn, gray."""
        dataset = self.get_dataset()
        if self.mode.value != "Field" and self.expr_mode.value == "Gene":
            if ex is None:
                ex = self._expression(dataset, idx, binning)
            genes = [ex["drawn"][label] for label in self.channel_widgets
                     if ex["drawn"][label] is not None and self._field_column(dataset, ex["names"][label]) is None]
            if genes:
                out = np.zeros(len(genes[0]), dtype=bool)
                for values in genes:
                    out |= np.asarray(values) != 0  # NaN != 0
                return out
        totals = dataset.spot_totals()
        if binning is not None:
            return binning.sum(totals[idx]) != 0
        return (totals if idx is None else totals[idx]) != 0

    @staticmethod
    def _n_drawn(dataset, idx, binning) -> int:
        if binning is not None:
            return binning.n_bins
        return dataset.adata.n_obs if idx is None else len(idx)

    def _field_colors(self, dataset, idx, binning) -> np.ndarray:
        if not self.field_select.value:
            return np.full(self._n_drawn(dataset, idx, binning), colors_mod.GRAY_RGBA, dtype=np.uint32)
        spec = dataset.classify(self.field_select.value)
        active = self._active_labels()
        if binning is None:
            return colors_mod.field_fill_colors(spec, active, None if idx is None else spec.codes[idx])
        if spec.kind == "binned_continuous":
            codes, _ = self._bin_field_means(dataset, spec, idx, binning, active)
            # The mean can fall in a range that isn't selected: its cells were.
            return colors_mod.field_fill_colors(spec, set(spec.categories), codes)
        active_codes = np.array([c in active for c in spec.categories] + [False])
        codes = spec.codes[idx]
        codes = np.where(active_codes[codes], codes, -1)  # code -1 hits the trailing False
        return colors_mod.field_fill_colors(spec, active, binning.majority(codes))

    def _expression(self, dataset, idx, binning) -> dict:
        """The Gene & Pathway channels as drawn - what both the colors and the tooltip are
        made from, so one recolor computes them once. Per channel label: `names` (gene /
        pathway, '' if unused), `raw` (per-cell values, or a pathway's genes when hexagons
        are scored themselves; None if unused / unknown), `bipolar` and `drawn` (one value
        per drawn cell or bin; for bins, only over `cell_mask`'s cells, as colored)."""
        is_gene = self.expr_mode.value == "Gene"
        # Categories deselected under Field stay gray here too, so an expression pattern
        # can be narrowed down to specific cell groups.
        mask = self.category_filter_mask()
        cell_mask = None if mask is None else (mask if idx is None else mask[idx])
        pathways = self.get_pathways()
        names, raw, bipolar, drawn = {}, {}, {}, {}
        for label, w in self.channel_widgets.items():
            if is_gene:
                name = w["gene"].value.strip()
                values = self._channel_values(dataset, name) if name else None
                # Principal components are centered on 0: + and - get opposite colors.
                bipolar[label] = self._field_column(dataset, name) in dataset.signed_fields
            else:
                name = w["pathway"].value
                values = self._pathway_values(dataset, name, pathways, binning) if name else None
                bipolar[label] = True
            names[label], raw[label] = name, values
            if values is None:
                drawn[label] = None
            elif binning is not None:
                drawn[label] = (
                    self._bin_channel_values(dataset, name, values, idx, binning, cell_mask) if is_gene
                    else self._bin_pathway_values(dataset, name, values, idx, binning, cell_mask, pathways))
            else:
                drawn[label] = values if idx is None else values[idx]
        return {"is_gene": is_gene, "names": names, "raw": raw, "bipolar": bipolar, "drawn": drawn,
                "mask": mask, "cell_mask": cell_mask, "pathways": pathways}

    def _expression_colors(self, dataset, idx, binning, ex: dict, histograms: bool) -> np.ndarray:
        n = self._n_drawn(dataset, idx, binning)
        mask, cell_mask = ex["mask"], ex["cell_mask"]
        self._update_filter_note()
        self._update_pathway_info(ex["pathways"], ex["is_gene"])
        channels = {label: (ex["drawn"][label], ex["bipolar"][label]) for label in self.channel_widgets}
        if binning is not None:
            # Bins without any cell of a selected category have no mean (NaN).
            shown = None if cell_mask is None else np.bincount(
                binning.cell_bin, weights=cell_mask, minlength=binning.n_bins) > 0
            # Bins the hexagon filter leaves out stay gray, whatever is colored.
            kept = dataset.kept_bins(idx, binning)
            if not kept.all():
                shown = kept if shown is None else shown & kept
                channels = {label: (None if values is None else np.where(kept, values, np.nan), bipolar)
                            for label, (values, bipolar) in channels.items()}
            if histograms:
                self._update_bin_histograms(ex["raw"], channels, ex["names"], ex["is_gene"], binning, mask)
        else:
            if histograms:
                self._update_histograms(ex["raw"], mask)
            shown = cell_mask
        ranges, used = self._chosen_ranges(), {}
        if shown is None:
            colors = colors_mod.compute_rgb_colors(n, channels, ranges, used)
        else:
            # Scale colors over the shown cells/bins only, so the contrast reflects them.
            colors = np.full(n, colors_mod.GRAY_RGBA, dtype=np.uint32)
            if shown.any():
                colors[shown] = colors_mod.compute_rgb_colors(int(shown.sum()), {
                    label: (None if values is None else values[shown], bipolar)
                    for label, (values, bipolar) in channels.items()
                }, ranges, used)
        if histograms:  # (the main drawing, not the zoomed-in layer)
            self._show_ranges(used)
        return colors

    def hover_data(self, idx: np.ndarray | None = None, binning=None) -> tuple[dict, dict]:
        """(hover columns to put in the plot's data source, meta for the tooltip's JS) -
        one value per drawn point: every cell, the cells `idx`, or `binning`'s bins.
        Only the columns the current mode uses are returned. See viewer/hover.py.
        In Gene & Pathway mode the meta's "top" is None: the most expressed genes are
        slow to compute, so the caller fills them in (hover.top_gene_columns)."""
        dataset = self.get_dataset()
        if dataset is None:
            return {}, {"kind": "none"}
        if self.mode.value != "Field":
            return self._expression_hover(binning, self._expression(dataset, idx, binning))
        binned = binning is not None
        cols = {"hv_n": binning.counts.astype(np.int32)} if binned else {}
        take = (lambda a: a) if idx is None else (lambda a: a[idx])
        field = self.field_select.value
        if not field:
            return cols, {"kind": "none"}
        spec = dataset.classify(field)
        codes = take(spec.codes)
        conf_col = f"{field}_confidence"
        has_conf = conf_col in dataset.adata.obs.columns
        conf = take(dataset.adata.obs[conf_col].to_numpy(dtype=np.float32)) if has_conf else None
        continuous = binned and spec.kind == "binned_continuous"
        if continuous:  # as drawn: the mean over each bin's cells in the selected ranges
            codes, means = self._bin_field_means(dataset, spec, idx, binning, self._active_labels())
            cols["hv_mean"] = means.astype(np.float32)
        elif binned:  # as drawn: the most common *selected* category of each bin
            active = self._active_labels()
            active_codes = np.array([c in active for c in spec.categories] + [False])
            codes = np.where(active_codes[codes], codes, -1)
            maj, share = binning.majority_share(codes)
            cols["hv_share"] = share.astype(np.float32)
            if has_conf:  # mean confidence of the bin's cells that have its type
                agree = (codes == maj[binning.cell_bin]) & ~np.isnan(conf)
                n = np.bincount(binning.cell_bin, weights=agree, minlength=binning.n_bins)
                tot = np.bincount(binning.cell_bin, weights=np.where(agree, conf, 0), minlength=binning.n_bins)
                with np.errstate(invalid="ignore", divide="ignore"):
                    conf = np.where(n > 0, tot / n, np.nan)
            codes = maj
        cols["hv_code"] = codes.astype(np.int32)
        if has_conf:
            cols["hv_conf"] = np.asarray(conf, dtype=np.float32)
        meta = {"kind": "field", "field": field, "labels": list(spec.categories),
                "has_conf": has_conf, "binned": binned, "continuous": continuous}
        return cols, meta

    def _expression_hover(self, binning, ex: dict) -> tuple[dict, dict]:
        binned = binning is not None
        cols = {"hv_n": binning.counts.astype(np.int32)} if binned else {}
        channels = []
        for label in self.channel_widgets:
            values = ex["drawn"][label]
            if values is None:
                continue
            if not binned and ex["cell_mask"] is not None:
                values = np.where(ex["cell_mask"], values, np.nan)
            cols[f"hv_{label.lower()}"] = np.asarray(values, dtype=np.float32)
            channels.append([label, ex["names"][label]])
        return cols, {"kind": "channels" if channels else "none", "channels": channels, "binned": binned,
                      "top": None}

    def category_filter(self) -> tuple[str, int, int] | None:
        """(field, number of selected categories, number of categories) when some
        categories of the current field are deselected, else None (nothing filtered)."""
        field = self.field_select.value
        n_selected = len(self._active_labels())
        if not field or n_selected == self._n_categories():
            return None
        return field, n_selected, self._n_categories()

    def category_filter_mask(self) -> np.ndarray | None:
        """Per-cell: is it in a selected category of the current field? None if no
        category is deselected."""
        if self.category_filter() is None:
            return None
        spec = self.get_dataset().classify(self.field_select.value)
        active = self._active_labels()
        active_codes = np.array([c in active for c in spec.categories] + [False])
        return active_codes[spec.codes]  # code -1 (missing) hits the trailing False

    def _update_filter_note(self):
        f = self.category_filter()
        if f is None:
            self.filter_note.visible = False
            return
        field, n_selected, n_total = f
        self.filter_note.object = (
            f"Only cells in the <b>{n_selected} of {n_total}</b> selected "
            f"<b>{html.escape(field)}</b> categories are colored; the rest are gray. "
            "Change the selection under <b>Field</b>."
        )
        self.filter_note.visible = True

    def _update_histograms(self, channel_values: dict[str, np.ndarray | None], mask=None):
        self._hist_generation += 1
        active_idx = self.get_active_indices()
        if mask is not None:  # only the cells actually colored count as "shown"
            active_idx = np.flatnonzero(mask) if active_idx is None else active_idx[mask[active_idx]]
        for label, values in channel_values.items():
            if values is None:
                self._show_histogram(label, None, None, "")
                continue
            local_values = values[active_idx] if active_idx is not None else values
            # Says what the two overlaid distributions are, right under the plot.
            if active_idx is not None:
                shown = "colored cells" if mask is not None else "this sample"
                self._show_histogram(label, local_values, values, shown, "all cells")
            else:
                self._show_histogram(label, local_values, values, "distribution across all cells")

    def _update_bin_histograms(self, channel_values, bin_channels, names, is_gene, binning, mask=None):
        """Histograms of the hexagons' values, as colored: this sample's drawn bins over
        every sample's bins of the same width (each made like the drawn ones). Other
        samples' bins not made yet are made in the background (the first time a gene is
        shown, that can mean binning every sample): until then, this sample's alone."""
        self._hist_generation += 1
        generation = self._hist_generation
        dataset = self.get_dataset()
        width = self.get_hex_width()
        # (The drawn sample's binning is made already; the others' aren't built here.) The
        # samples side by side are every sample already.
        all_drawn = (width is not None and dataset.has_hex_binning(ALL_SAMPLES, width)
                     and dataset.hex_binning(ALL_SAMPLES, width) is binning)
        others = [] if all_drawn else [sid for sid in dataset.sample_ids if width is not None and not (
            dataset.has_hex_binning(sid, width) and dataset.hex_binning(sid, width) is binning)]
        # Other samples' bins only change with the gene / pathway, width and selected
        # categories, so they're cached rather than rebuilt on every recolor.
        mask_key = None if mask is None else (self.field_select.value, frozenset(self._active_labels()))
        pathways = self.get_pathways()
        missing = {}  # cache key -> (label's raw values, sample, name) still to compute
        pending = {}  # label -> (local values, cache keys of the other samples' parts)
        for label, values in channel_values.items():
            bin_values = bin_channels[label][0]
            if values is None or bin_values is None:
                self._show_histogram(label, None, None, "")
                continue
            local = bin_values[np.isfinite(bin_values)]  # NaN: no cell of a selected category
            if not others:
                what = "colored hexagons" if mask is not None else "hexagons"
                self._show_histogram(label, local, None, f"distribution across {what}")
                continue
            keys = []
            for sid in others:
                key = (is_gene, names[label], width, sid, mask_key, dataset.hex_filter)
                cached = self._bin_values_cache.get(key)
                # Stale after e.g. a recomputed pathway score, or an edited pathway's genes.
                if cached is None or not (cached[0] is values or isinstance(values, tuple) and cached[0] == values):
                    missing[key] = (values, sid, names[label])
                keys.append(key)
            pending[label] = (local, keys)
        shown = "colored hexagons" if mask is not None else "this sample's hexagons"

        def show_all():
            for label, (local, keys) in pending.items():
                everything = np.concatenate([local] + [self._bin_values_cache[key][1] for key in keys])
                self._show_histogram(label, local, everything[np.isfinite(everything)], shown,
                                     "all samples' hexagons")

        if not missing:
            show_all()
            return
        for label, (local, _) in pending.items():
            self._show_histogram(label, local, None, shown)

        def compute():
            results = {}
            for key, (values, sid, name) in missing.items():
                idx = dataset.sample_indices(sid)
                b = dataset.hex_binning(sid, width)
                cell_mask = None if mask is None else mask[idx]
                result = (self._bin_channel_values(dataset, name, values, idx, b, cell_mask) if is_gene
                          else self._bin_pathway_values(dataset, name, values, idx, b, cell_mask, pathways))
                results[key] = (values, np.where(dataset.kept_bins(idx, b), result, np.nan))
            return results

        def apply(results):
            if results is None or generation != self._hist_generation:  # failed, or recolored since
                return
            for key, value in results.items():
                self._bin_values_cache.pop(key, None)
                if len(self._bin_values_cache) >= _BIN_VALUES_CACHE_SIZE:
                    self._bin_values_cache.pop(next(iter(self._bin_values_cache)))
                self._bin_values_cache[key] = value
            if all(key in self._bin_values_cache for _, keys in pending.values() for key in keys):
                show_all()

        background.run_later(compute, apply)

    def _show_histogram(self, label: str, local_values, global_values, shown: str, overall: str | None = None):
        """Draws a channel's histogram (hidden if `local_values` is None), captioned with
        what `local_values` are and, when overlaid, what `global_values` are."""
        w = self.channel_widgets[label]
        hist_pane, slider = w["hist"], w["range"]
        if local_values is None:
            hist_pane.object = ""
            hist_pane.visible = slider.visible = False
            return
        start, end = histogram_mod.value_bounds(local_values, global_values)
        chosen = w["chosen_range"]
        self._set_slider(slider, start, end, chosen or slider.value, chosen is not None)
        slider.visible = True
        color = _CHANNEL_LABEL_COLOR[label]
        caption = (shown if overall is None else
                   f'<span style="color:{color}">■</span> {shown} &nbsp; '
                   f'<span style="color:#9e9e9e">■</span> {overall}')
        hist_pane.object = (
            histogram_mod.render_histogram(local_values, global_values, color)
            + f'<div style="font-size:11px;color:#555;margin:-2px 0 2px 0">{caption}</div>'
        )
        hist_pane.visible = True

    def _update_pathway_info(self, pathways, is_gene: bool):
        for label, w in self.channel_widgets.items():
            pname = w["pathway"].value
            if not is_gene and pname:
                info = pathways.get(pname, {})
                genes = [html.escape(str(g)) for g in info.get("genes", [])]
                w["info"].object = (
                    f"<div style='font-size:11px;color:#444;margin:2px 0 4px 0;line-height:1.35'>"
                    f"<i>{html.escape(str(info.get('source', '')))}</i><br>"
                    f"{html.escape(str(info.get('description', '')))}</div>"
                    f"<details style='font-size:10px;color:#666;margin:0 0 6px 0'>"
                    f"<summary style='cursor:pointer'>Genes ({len(genes)})</summary>"
                    f"<div style='margin-top:4px;font-family:monospace;line-height:1.4;"
                    f"word-break:break-word'>{', '.join(genes)}</div></details>"
                )
                w["info"].visible = True
            else:
                w["info"].object = ""
                w["info"].visible = False
