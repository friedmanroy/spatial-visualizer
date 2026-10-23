"""A full-page pop-up showing differential-expression results between regions: per
comparison, a volcano plot and a sortable / searchable table of genes, with buttons to
color the tissue by the top genes and to download the table as CSV.

Placed once in the app's root (like the busy overlay) and filled by `show()`.
"""
from __future__ import annotations

import html
import io
import re

import numpy as np
import panel as pn
from bokeh.models import ColumnDataSource, HoverTool, ScientificFormatter, TapTool
from bokeh.plotting import figure

from . import de
from .error_dialog import OVERLAY_STYLES_BASE

#: below the busy overlay's, so a long step started from here still shows on top
_STYLES = {**OVERLAY_STYLES_BASE, "z-index": "99990", "padding": "3vh 3vw", "box-sizing": "border-box"}
_BOX_STYLES = {
    "background": "white", "padding": "16px 20px", "border-radius": "8px",
    "box-shadow": "0 4px 24px rgba(0,0,0,0.35)", "max-width": "1250px", "overflow": "auto",
}
_UP, _DOWN, _NS = "#d62728", "#1f77b4", "#c7c7c7"
#: how many genes the "Color by top..." buttons put in the Gene channels
_TOP = 3
#: how many of the strongest up / down genes are labeled in the volcano plot
_LABELED = 6
#: -log10(adj. p) is capped here (a p-value of 0 would otherwise be infinitely high)
_MAX_NEG_LOG_P = 300.0
_NOTE = ("Cells within one drawn region are not independent replicates, so read the p-values as a "
         "ranking of genes rather than as calibrated significance.")


class DEDialog:
    def __init__(self):
        # (genes) -> None: colors the tissue by these genes (set by the Spatial tab).
        self.on_color_genes = None
        self._title = pn.pane.HTML("", sizing_mode="stretch_width", margin=(0, 5))
        close_btn = pn.widgets.Button(name="✕ Close", width=90, margin=(0, 5))
        close_btn.on_click(lambda e: self.hide())
        self._tabs = pn.Tabs(sizing_mode="stretch_both", dynamic=True)
        box = pn.Column(
            pn.Row(self._title, close_btn, sizing_mode="stretch_width", margin=0),
            pn.pane.HTML(f"<span style='font-size:12px;color:#666'>{_NOTE}</span>",
                         sizing_mode="stretch_width", margin=(0, 5, 4, 5)),
            self._tabs,
            sizing_mode="stretch_both", styles=_BOX_STYLES,
        )
        # Shown/hidden through `display`, as the other overlays (see error_dialog).
        self.layout = pn.Column(box, styles={**_STYLES, "display": "none"})
        self.has_results = False

    def show(self, results: list[de.DEResult], errors: list[str] = ()):
        method = next((name for name, m in de.METHODS.items() if results and m == results[0].method), "")
        n = len(results)
        self._title.object = (
            f"<h3 style='margin:0'>Differential expression"
            f"<span style='font-weight:normal;color:#555;font-size:14px'> · "
            f"{n} comparison{'s' if n != 1 else ''} · {html.escape(method)}</span></h3>"
        )
        tabs = [(r.group if len({x.reference for x in results}) == 1 or n == 1 else r.title, self._result_view(r))
                for r in results]
        for msg in errors:
            tabs.append(("Skipped", pn.pane.HTML(f"<p>{html.escape(msg)}</p>", sizing_mode="stretch_width")))
        self._tabs[:] = tabs
        self._tabs.active = 0
        self.has_results = True
        self.open()

    def open(self):
        self.layout.styles = {**_STYLES, "display": "flex"}

    def hide(self):
        self.layout.styles = {**_STYLES, "display": "none"}

    def _color_by(self, genes: list[str]):
        if genes and self.on_color_genes is not None:
            self.on_color_genes(genes[:_TOP])
            self.hide()  # so the recolored tissue can be seen

    def _result_view(self, r: de.DEResult) -> pn.viewable.Viewable:
        t = r.table
        n_up, n_down = int((t["Direction"] == "up").sum()), int((t["Direction"] == "down").sum())

        def cells(n, total):
            return f"{total:,} cells" if n == total else f"a random {n:,} of its {total:,} cells"

        summary = pn.pane.HTML(
            f"<b>{html.escape(r.group)}</b> ({cells(r.n_group, r.n_group_total)}) vs "
            f"<b>{html.escape(r.reference)}</b> ({cells(r.n_reference, r.n_reference_total)}).<br>"
            f"<span style='color:{_UP}'><b>{n_up}</b> genes up</span> and "
            f"<span style='color:{_DOWN}'><b>{n_down}</b> down</span> in {html.escape(r.group)} "
            f"(adj. p &lt; {de.PADJ_THRESHOLD:g}, |log2 FC| ≥ {de.LFC_THRESHOLD:g}).",
            sizing_mode="stretch_width", margin=(4, 5), styles={"font-size": "13px"},
        )
        table = pn.widgets.Tabulator(
            t[["Gene", "log2 FC", "Adj. p-value", "% in group", "% in reference", "Direction"]],
            show_index=False, disabled=True, selectable="checkbox", pagination="local", page_size=14,
            sizing_mode="stretch_both", min_height=300, layout="fit_data_stretch",
            formatters={"log2 FC": {"type": "money", "precision": 2},
                        "Adj. p-value": ScientificFormatter(precision=2)},
            header_filters={"Gene": {"type": "input", "func": "like", "placeholder": "Search"},
                            "Direction": {"type": "list", "func": "=", "valuesLookup": True}},
        )
        volcano = self._volcano(r, on_pick=lambda gene: self._color_by([gene]))

        up_btn = pn.widgets.Button(name=f"Color by top {_TOP} up", button_type="primary", margin=(5, 5))
        down_btn = pn.widgets.Button(name=f"Color by top {_TOP} down", margin=(5, 5))
        picked_btn = pn.widgets.Button(name="Color by checked genes", margin=(5, 5), disabled=True,
                                       description=f"Check up to {_TOP} genes in the table")
        up_btn.disabled = n_up == 0
        down_btn.disabled = n_down == 0
        up_btn.on_click(lambda e: self._color_by(r.top("up", _TOP)))
        down_btn.on_click(lambda e: self._color_by(r.top("down", _TOP)))

        def picked() -> list[str]:
            return [table.value["Gene"].iloc[i] for i in table.selection]

        def on_selection(event):
            n = len(event.new)
            picked_btn.disabled = n == 0
            picked_btn.name = (f"Color by checked genes ({n})" if n <= _TOP
                               else f"Color by the first {_TOP} checked genes")

        table.param.watch(on_selection, "selection")
        picked_btn.on_click(lambda e: self._color_by(picked()))
        filename = re.sub(r"[^\w.-]+", "_", f"DE_{r.group}_vs_{r.reference}") + ".csv"
        download = pn.widgets.FileDownload(
            callback=lambda: io.StringIO(r.table.to_csv(index=False)), filename=filename,
            label="Download CSV", button_type="light", margin=(5, 5), width=140,
        )
        buttons = pn.Row(up_btn, down_btn, picked_btn, download, sizing_mode="stretch_width", margin=0)
        hint = pn.pane.HTML(
            "Click a point in the volcano plot to color the tissue by that gene; check genes in the "
            "table (search with the box under <i>Gene</i>) to color by them.",
            sizing_mode="stretch_width", margin=(0, 5), styles={"font-size": "12px", "color": "#666"},
        )
        return pn.Column(
            summary, buttons, hint,
            pn.Row(volcano, table, sizing_mode="stretch_both", margin=0),
            sizing_mode="stretch_both", margin=0,
        )

    @staticmethod
    def _volcano(r: de.DEResult, on_pick) -> pn.viewable.Viewable:
        t = r.table
        neg_log_p = np.minimum(-np.log10(np.maximum(t["Adj. p-value"].to_numpy(float), 1e-300)), _MAX_NEG_LOG_P)
        color = np.select([t["Direction"] == "up", t["Direction"] == "down"], [_UP, _DOWN], _NS)
        source = ColumnDataSource({
            "x": t["log2 FC"].to_numpy(float), "y": neg_log_p, "gene": t["Gene"].astype(str).to_numpy(),
            "color": color, "padj": t["Adj. p-value"].to_numpy(float),
        })
        fig = figure(width=420, height=380, tools="pan,wheel_zoom,box_zoom,reset,save",
                     x_axis_label=f"log2 fold change ({r.group} / reference)",
                     y_axis_label="-log10 adjusted p-value", toolbar_location="above")
        dots = fig.scatter("x", "y", source=source, color="color", size=5, alpha=0.75,
                           nonselection_alpha=0.75, selection_line_color="black")
        fig.add_tools(HoverTool(renderers=[dots], tooltips=[("gene", "@gene"), ("log2 FC", "@x{0.00}"),
                                                            ("adj. p", "@padj{0.00e+0}")]))
        fig.add_tools(TapTool(renderers=[dots]))
        for value in (de.LFC_THRESHOLD, -de.LFC_THRESHOLD):
            fig.line([value, value], [0, float(neg_log_p.max(initial=1))], color="#999", line_dash="dashed")
        # Label the strongest few genes each way, on the side facing the middle of the plot.
        for direction, align, offset in (("up", "right", -5), ("down", "left", 5)):
            rows = t.index[t["Gene"].isin(r.top(direction, _LABELED))]
            fig.text(t.loc[rows, "log2 FC"].to_numpy(float), neg_log_p[rows],
                     text=t.loc[rows, "Gene"].astype(str).tolist(), text_font_size="10px", text_align=align,
                     text_baseline="middle", x_offset=offset, text_color="#333")

        def on_tap(attr, old, new):
            if new:
                on_pick(source.data["gene"][new[0]])
                source.selected.indices = []

        source.selected.on_change("indices", on_tap)
        return pn.pane.Bokeh(fig, margin=(0, 10, 0, 0))
