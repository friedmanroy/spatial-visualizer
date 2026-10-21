"""Shared 'Field' / 'Gene & Pathway' coloring controls, reused by the Spatial page and
the scRNA/snRNA embedding page so the two don't duplicate ~150 lines of widget wiring.

The Gene & Pathway section mirrors the TNBC viewer's layout closely: one shared
Gene/Pathway mode for all three channels, each channel's controls stacked vertically
(not side by side) so they stay within the sidebar's width, with a per-channel Clear
button and a small local-vs-whole-dataset histogram under each channel's name.
"""
from __future__ import annotations

import html

import numpy as np
import panel as pn

from . import colors as colors_mod
from . import histogram as histogram_mod
from . import hover
from . import pathways as pathways_mod

GRAY = colors_mod.GRAY

_CHANNEL_LABEL_COLOR = {"R": "#ff4444", "G": "#44cc44", "B": "#4488ff"}
_CHANNEL_NAME = {"R": "Red", "G": "Green", "B": "Blue"}
#: width (px) of the colored "● Red" / "● Green" / "● Blue" label in front of each channel
_CHANNEL_LABEL_WIDTH = 62
#: small gray explanatory text used under headers and for summaries
HINT_STYLE = {"font-size": "12px", "color": "#555", "line-height": "1.35"}
_GENE_HINT = (
    "Each channel colors cells by one gene: brighter = higher expression. "
    "Use up to three channels; overlapping expression mixes the colors. "
    "On the Spatial tab, click a cell to show its 3 most expressed."
)
_PATHWAY_HINT = (
    "Each channel colors cells by a pathway score: its color where the pathway's genes "
    "are higher than comparable control genes, the opposite color where lower."
)
_PATHWAY_PROMPT = "Choose a pathway…"
#: how many of the most highly variable genes the gene dropdown offers
_N_DROPDOWN_GENES = 200
_DROPDOWN_PROMPT = f"or pick a top-{_N_DROPDOWN_GENES} variable gene"
#: left inset shared by every top-level row in the sidebar, so fields/buttons/
#: checkboxes all start flush against the same edge instead of drifting per widget.
_INSET = 1
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


def section_label(text: str) -> pn.pane.HTML:
    """Small bold heading above a group of controls (e.g. a toggle), so each group
    says what it's for."""
    return pn.pane.HTML(f"<b>{text}</b>", margin=(8, 5, 0, _INSET + 3), styles={"font-size": "13px"})


def _pathway_options(names: list[str]) -> dict[str, str]:
    return {_PATHWAY_PROMPT: "", **{n: n for n in names}}


class ColorControls:
    def __init__(self, get_dataset, get_pathways, on_change, get_active_indices=None):
        self.get_dataset = get_dataset  # () -> Dataset | None
        self.get_pathways = get_pathways  # () -> dict
        self.on_change = on_change  # () -> None
        # () -> np.ndarray | None - positional indices, into the full dataset, of the
        # cells currently on screen (e.g. the active sample). None means "no subset,
        # everything shown" - skips the local-vs-global histogram overlay.
        self.get_active_indices = get_active_indices or (lambda: None)

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
            hist_pane = pn.pane.HTML("", visible=False, sizing_mode="stretch_width", margin=(0, 5, 0, _INSET))
            self.channel_widgets[label] = dict(
                gene=gene_input, gene_dropdown=gene_dropdown, pathway=pathway_select, info=info_pane, hist=hist_pane,
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
            w["gene"].param.watch(self._fire, "value")
            w["gene"].param.watch(lambda e, w=w: self._sync_dropdown(w), "value")
            w["gene_dropdown"].param.watch(lambda e, w=w: self._on_dropdown_pick(w, e.new), "value")
            w["pathway"].param.watch(self._fire, "value")

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
            options = dataset.field_options()
            self.field_select.options = options
            if options and self.field_select.value not in options:
                self.field_select.value = options[0]
            else:
                self._rebuild_value_picker()
            pathway_names = sorted(self.get_pathways().keys())
            gene_names = list(dataset.adata.var_names)
            dropdown_options = {_DROPDOWN_PROMPT: ""}
            dropdown_options.update({g: g for g in dataset.top_variable_genes(_N_DROPDOWN_GENES)})
            for w in self.channel_widgets.values():
                w["pathway"].options = _pathway_options(pathway_names)
                w["gene"].options = gene_names
                w["gene_dropdown"].options = dropdown_options
                self._sync_dropdown(w)

    def refresh_field_options(self, changed_field: str, select: bool = False):
        """Re-reads the field list after an obs column was added/recomputed (e.g. Leiden),
        optionally switching to Field mode and coloring by that column."""
        dataset = self.get_dataset()
        if dataset is None:
            return
        with pn.io.hold():
            self._value_picker_cache.pop(changed_field, None)  # its categories may have changed
            self.field_select.options = dataset.field_options()
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
        matched = dataset.match_gene(gene) if dataset is not None and gene else None
        w["gene_dropdown"].value = matched if matched in w["gene_dropdown"].values else ""

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
            self.field_summary.object = ""
            return

        cached = self._value_picker_cache.get(field)
        if cached is not None:
            widget, checkboxes, rows = cached
            self._value_checkboxes = checkboxes
            self._value_rows = rows
            self.value_picker_box[:] = [widget]
            self.update_category_counts()
            return

        spec = dataset.classify(field)
        self._value_checkboxes = {}
        rows = {}
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

        self._value_picker_cache[field] = (widget, self._value_checkboxes, rows)
        self.value_picker_box[:] = [widget]
        self.update_category_counts()

    def update_category_counts(self):
        """Puts each category's cell count - among the cells currently shown (e.g. the
        active sample) - next to its name, plus a one-line summary above the list."""
        dataset = self.get_dataset()
        field = self.field_select.value
        if dataset is None or not field or not self._value_checkboxes:
            return
        spec = dataset.classify(field)
        active_idx = self.get_active_indices()
        codes = spec.codes if active_idx is None else spec.codes[active_idx]
        counts = np.bincount(codes[codes >= 0], minlength=len(spec.categories))
        with pn.io.hold():
            for cat, count in zip(spec.categories, counts):
                self._value_checkboxes[cat].name = f"{cat}  ({count:,})"
            # Categories: biggest first (stable, so ties keep their natural order).
            # Value ranges of a continuous field keep their natural low-to-high order.
            if spec.kind == "categorical":
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

    def _active_labels(self) -> set:
        return {cat for cat, cb in self._value_checkboxes.items() if cb.value}

    def in_gene_view(self) -> bool:
        return self.mode.value == "Gene & Pathway" and self.expr_mode.value == "Gene"

    def set_genes(self, genes: list[str]):
        """Puts `genes` in the red, green and blue channels (in that order), recoloring once."""
        self._hold_fire = True
        try:
            for w, gene in zip(self.channel_widgets.values(), genes):
                w["gene"].value = gene
        finally:
            self._hold_fire = False
        self._fire()

    def _fire(self, *events):
        if self._hold_fire:
            return
        with pn.io.hold():
            self.on_change()

    def compute_colors(self, idx: np.ndarray | None = None, binning=None) -> np.ndarray:
        """Packed colors (see colors.pack_rgb) for every cell, or - given a sample's cell indices `idx` and its
        hexbin.HexBinning - for every bin of that sample: the bin's mean gene/pathway
        value, or its most common category among the selected ones."""
        dataset = self.get_dataset()
        if dataset is None:
            return np.array([], dtype=np.uint32)
        n = dataset.adata.n_obs if binning is None else binning.n_bins
        if self.mode.value == "Field":
            if not self.field_select.value:
                return np.full(n, colors_mod.GRAY_RGBA, dtype=np.uint32)
            spec = dataset.classify(self.field_select.value)
            active = self._active_labels()
            if binning is None:
                return colors_mod.field_fill_colors(spec, active)
            active_codes = np.array([c in active for c in spec.categories] + [False])
            codes = spec.codes[idx]
            codes = np.where(active_codes[codes], codes, -1)  # code -1 hits the trailing False
            return colors_mod.field_fill_colors(spec, active, binning.majority(codes))
        is_gene = self.expr_mode.value == "Gene"
        # Categories deselected under Field stay gray here too, so an expression pattern
        # can be narrowed down to specific cell groups.
        mask = self.category_filter_mask()
        self._update_filter_note()
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
        self._update_histograms(channel_values, mask)
        if binning is not None:
            cell_mask = None if mask is None else mask[idx]
            channels = {
                label: (None if values is None else binning.mean(values[idx], cell_mask), bipolar)
                for label, (values, bipolar) in channels.items()
            }
            # Bins without any cell of a selected category have no mean (NaN).
            shown = None if cell_mask is None else np.bincount(
                binning.cell_bin, weights=cell_mask, minlength=binning.n_bins) > 0
        else:
            shown = mask
        if shown is None:
            return colors_mod.compute_rgb_colors(n, channels)
        # Scale colors over the shown cells/bins only, so the contrast reflects them.
        colors = np.full(n, colors_mod.GRAY_RGBA, dtype=np.uint32)
        if shown.any():
            colors[shown] = colors_mod.compute_rgb_colors(int(shown.sum()), {
                label: (None if values is None else values[shown], bipolar)
                for label, (values, bipolar) in channels.items()
            })
        return colors

    def hover_data(self, idx: np.ndarray | None = None, binning=None) -> tuple[dict, dict]:
        """(hover columns to put in the plot's data source, meta for the tooltip's JS) -
        one value per drawn point: every cell, the cells `idx`, or `binning`'s bins.
        Only the columns the current mode uses are returned. See viewer/hover.py."""
        dataset = self.get_dataset()
        if dataset is None:
            return {}, {"kind": "none"}
        binned = binning is not None
        cols = {"hv_n": binning.counts.astype(np.int32)} if binned else {}
        take = (lambda a: a) if idx is None else (lambda a: a[idx])
        if self.mode.value == "Field":
            field = self.field_select.value
            if not field:
                return cols, {"kind": "none"}
            spec = dataset.classify(field)
            codes = take(spec.codes)
            conf_col = f"{field}_confidence"
            has_conf = conf_col in dataset.adata.obs.columns
            conf = take(dataset.adata.obs[conf_col].to_numpy(dtype=np.float32)) if has_conf else None
            if binned:  # as drawn: the most common *selected* category of each bin
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
                    "has_conf": has_conf, "binned": binned}
            return cols, meta
        is_gene = self.expr_mode.value == "Gene"
        mask = self.category_filter_mask()
        pathways = self.get_pathways()
        channels = []
        for label, w in self.channel_widgets.items():
            name = w["gene"].value.strip() if is_gene else w["pathway"].value
            values = None
            if name:
                values = (dataset.gene_values(name) if is_gene
                          else pathways_mod.score_pathway(dataset, name, pathways))
            if values is None:
                continue
            values = np.asarray(values, dtype=np.float32)
            if binned:
                values = binning.mean(take(values), None if mask is None else take(mask))
            else:
                values = take(np.where(mask, values, np.nan) if mask is not None else values)
            cols[f"hv_{label.lower()}"] = np.asarray(values, dtype=np.float32)
            channels.append([label, name])
        # Each cell's (or bin's) most expressed genes, as codes into a list of just the
        # genes that occur, so the tooltip's labels stay short.
        top = dataset.top_genes(idx, binning, k=hover.N_TOP_GENES)
        used, codes = np.unique(top, return_inverse=True)
        codes = codes.reshape(top.shape).astype(np.int32)
        offset = 1 if len(used) and used[0] == -1 else 0  # -1 (none) is the smallest
        codes -= offset
        for j in range(top.shape[1]):
            cols[f"hv_t{j}"] = codes[:, j]
        names = [str(dataset.adata.var_names[g]) for g in used[offset:]]
        return cols, {"kind": "channels" if channels else "none", "channels": channels, "binned": binned,
                      "top": names}

    def category_filter(self) -> tuple[str, int, int] | None:
        """(field, number of selected categories, number of categories) when some
        categories of the current field are deselected, else None (nothing filtered)."""
        field = self.field_select.value
        n_selected = len(self._active_labels())
        if not field or n_selected == len(self._value_checkboxes):
            return None
        return field, n_selected, len(self._value_checkboxes)

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
        active_idx = self.get_active_indices()
        if mask is not None:  # only the cells actually colored count as "shown"
            active_idx = np.flatnonzero(mask) if active_idx is None else active_idx[mask[active_idx]]
        for label, values in channel_values.items():
            hist_pane = self.channel_widgets[label]["hist"]
            if values is None:
                hist_pane.object = ""
                hist_pane.visible = False
                continue
            local_values = values[active_idx] if active_idx is not None else values
            color = _CHANNEL_LABEL_COLOR[label]
            # Says what the two overlaid distributions are, right under the plot.
            if active_idx is not None:
                shown = "colored cells" if mask is not None else "this sample"
                caption = (f'<span style="color:{color}">■</span> {shown} &nbsp; '
                           f'<span style="color:#9e9e9e">■</span> all cells')
            else:
                caption = "distribution across all cells"
            hist_pane.object = (
                histogram_mod.render_histogram(local_values, values, color)
                + f'<div style="font-size:11px;color:#555;margin:-2px 0 2px 0">{caption}</div>'
            )
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
