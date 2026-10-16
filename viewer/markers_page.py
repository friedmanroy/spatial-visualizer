"""The 'Cell types' tab: the marker gene sets that cell typing uses, in an editable
table - PanglaoDB's sets for the chosen species/tissue plus the user's own. Gene lists
can be edited, sets switched off, new cell types added, a set viewed as a pathway score,
and cell typing re-run on an existing clustering to see the effect right away.

Edits are saved to cache/custom_markers.json (see celltyping.load_custom_markers).
"""
from __future__ import annotations

import html
import sys
import traceback

import pandas as pd
import panel as pn

from . import celltyping
from . import clustering
from . import pathways as pathways_mod
from .celltype_controls import results_html
from .color_controls import HINT_STYLE
from .pathways_page import parse_genes

#: prefix for marker sets added as pathways ("Show as pathway")
PATHWAY_PREFIX = "Cell type: "
_COLUMNS = ["On", "Cell type", "Markers", "Measured", "Not measured", "Source"]


def build_markers_page(state) -> pn.viewable.Viewable:
    typing = state.celltyping
    dataset = state.spatial
    rows: dict[str, celltyping.MarkerSet] = {}  # current table rows by cell type

    settings = pn.pane.HTML("", sizing_mode="stretch_width", styles={"font-size": "13px"})
    download_btn = pn.widgets.Button(name="Download the PanglaoDB marker genes", button_type="primary",
                                     width=280, visible=False)
    status = pn.pane.HTML("", sizing_mode="stretch_width", styles={"font-size": "13px"})
    table = pn.widgets.Tabulator(
        pd.DataFrame(columns=_COLUMNS), show_index=False, selectable="checkbox",
        # fit_columns: the fixed-width columns keep their size, Markers takes the rest
        sizing_mode="stretch_width", height=430, layout="fit_columns",
        header_filters={"Cell type": {"type": "input", "func": "like", "placeholder": "search"},
                        "Markers": {"type": "input", "func": "like", "placeholder": "search"},
                        "Source": {"type": "input", "func": "like", "placeholder": "filter"}},
        editors={"On": {"type": "tickCross"}, "Cell type": None, "Markers": {"type": "textarea"},
                 "Measured": None, "Not measured": None, "Source": None},
        formatters={"On": {"type": "tickCross"}, "Not measured": {"type": "html"},
                    "Markers": {"type": "textarea"}},
        widths={"On": 60, "Cell type": 190, "Measured": 100, "Not measured": 200, "Source": 150},
        text_align={"Measured": "right"},
    )

    def measured_info(genes: list[str]) -> tuple[str, str]:
        if dataset is None:
            return "", ""
        missing = [g for g in genes if dataset.match_gene(g) is None]
        found = len(genes) - len(missing)
        flag = " ⚠" if found < celltyping.MIN_MARKERS else ""
        gray = (f"<span style='color:#999'>{html.escape(', '.join(missing))}</span>" if missing else "")
        return f"{found} of {len(genes)}{flag}", gray

    def refresh(*events):
        has_markers = typing._markers is not None
        download_btn.visible = not has_markers
        settings.object = (
            f"Showing the marker sets cell typing would use for: <b>{html.escape(typing.settings_label)}</b>. "
            "Species, tissue and which sets to use are chosen in the <b>Leiden clustering &amp; cell "
            "types</b> panel on the Spatial tab."
            + ("" if has_markers else " <i>PanglaoDB isn't downloaded yet, so only your own sets are listed.</i>")
        )
        rows.clear()
        records = []
        for m in typing.marker_table():
            rows[m.name] = m
            measured, missing = measured_info(m.genes)
            records.append({"On": m.enabled, "Cell type": m.name, "Markers": ", ".join(m.genes),
                            "Measured": measured, "Not measured": missing, "Source": m.origin})
        table.value = pd.DataFrame(records, columns=_COLUMNS)
        table.selection = []

    def say(text: str):
        status.object = text

    def on_edit(event):
        name = table.value.iloc[event.row]["Cell type"]
        m = rows.get(name)
        if m is None:
            return
        custom = celltyping.load_custom_markers()
        if event.column == "On":
            disabled = set(custom["disabled"])
            (disabled.discard if event.value else disabled.add)(name)
            custom["disabled"] = sorted(disabled)
            say(f"{'Switched on' if event.value else 'Switched off'} <b>{html.escape(name)}</b>.")
        elif event.column == "Markers":
            genes = parse_genes(str(event.value))
            if not genes:
                say("<b>A marker set needs at least one gene</b> - switch it off instead to leave it out.")
                refresh()
                return
            if m.panglao_genes is not None and [g.upper() for g in genes] == [g.upper() for g in m.panglao_genes]:
                custom["types"].pop(name, None)  # back to PanglaoDB's own list
            else:
                custom["types"][name] = {"genes": genes, "edited_from_panglao": m.panglao_genes is not None}
            measured, _ = measured_info(genes)
            say(f"Updated <b>{html.escape(name)}</b>: {len(genes)} genes ({measured} measured in this data).")
        else:
            return
        celltyping.save_custom_markers(custom)
        refresh()

    table.on_edit(on_edit)

    def selected() -> list[celltyping.MarkerSet]:
        names = [table.value.iloc[i]["Cell type"] for i in table.selection]
        return [rows[n] for n in names if n in rows]

    reset_btn = pn.widgets.Button(name="Reset to PanglaoDB", width=170,
                                  description="Undo your edits to the selected PanglaoDB sets.")
    delete_btn = pn.widgets.Button(name="Delete", button_type="danger", width=100,
                                   description="Delete the selected sets you added (PanglaoDB sets can only be switched off).")
    pathway_btn = pn.widgets.Button(name="Show as pathway", width=160,
                                    description="Score the selected sets like pathways, to color the tissue by them.")

    def on_reset(event):
        custom = celltyping.load_custom_markers()
        done = [m.name for m in selected() if m.origin == celltyping.ORIGIN_EDITED]
        for name in done:
            custom["types"].pop(name, None)
        celltyping.save_custom_markers(custom)
        say(f"Reset {len(done)} set(s) to PanglaoDB." if done else "Select one or more <i>edited</i> PanglaoDB sets first.")
        refresh()

    def on_delete(event):
        custom = celltyping.load_custom_markers()
        chosen = selected()
        added = [m.name for m in chosen if m.origin == celltyping.ORIGIN_ADDED]
        for name in added:
            custom["types"].pop(name, None)
            custom["disabled"] = [n for n in custom["disabled"] if n != name]
        celltyping.save_custom_markers(custom)
        skipped = len(chosen) - len(added)
        say((f"Deleted {len(added)} set(s). " if added else "")
            + (f"{skipped} PanglaoDB set(s) can't be deleted - untick <b>On</b> to leave them out." if skipped else "")
            or "Select one or more sets first.")
        refresh()

    def on_pathway(event):
        chosen = selected()
        if not chosen:
            say("Select one or more sets first.")
            return
        for m in chosen:
            pathways_mod.save_custom_pathway(
                PATHWAY_PREFIX + m.name, m.genes,
                f"Marker genes of {m.name} ({m.origin}), added from the Cell types tab.",
            )
        state.pathways = pathways_mod.load_pathways(state.pathways_path)
        for m in chosen:
            for ds in (state.spatial, state.embedding):
                if ds is not None:
                    ds.pathway_score_cache.pop(PATHWAY_PREFIX + m.name, None)
            for listener in state.pathway_listeners:
                listener(PATHWAY_PREFIX + m.name)
        names = ", ".join(f"<b>{html.escape(PATHWAY_PREFIX + m.name)}</b>" for m in chosen)
        say(f"Added {names} as pathway(s): on the Spatial tab choose <b>Gene &amp; Pathway</b> → "
            "<b>Pathway</b> to color the tissue by them. They're listed on the Pathways tab.")

    reset_btn.on_click(on_reset)
    delete_btn.on_click(on_delete)
    pathway_btn.on_click(on_pathway)

    # ---- add a cell type ---------------------------------------------------------------
    name_input = pn.widgets.TextInput(name="Cell type name", placeholder="e.g. Tumor-associated macrophages",
                                      sizing_mode="stretch_width")
    genes_input = pn.widgets.TextAreaInput(name="Marker genes (separated by commas or new lines)",
                                           placeholder="CD68, CD163\nMRC1\nAPOE", height=110,
                                           sizing_mode="stretch_width")
    add_btn = pn.widgets.Button(name="Add cell type", button_type="primary", width=140)

    def on_add(event):
        name = name_input.value.strip()
        genes = parse_genes(genes_input.value)
        if not name or not genes:
            say("Please give the cell type a name and at least one marker gene.")
            return
        custom = celltyping.load_custom_markers()
        existing = rows.get(name)
        panglao = existing.panglao_genes if existing is not None else None
        custom["types"][name] = {"genes": genes, "edited_from_panglao": panglao is not None}
        custom["disabled"] = [n for n in custom["disabled"] if n != name]
        celltyping.save_custom_markers(custom)
        measured, _ = measured_info(genes)
        verb = "Replaced PanglaoDB's markers for" if panglao is not None else (
            "Updated" if existing is not None else "Added")
        warn = (f" <b>Only sets with at least {celltyping.MIN_MARKERS} measured genes are scored.</b>"
                if dataset is not None and int(measured.split(" ")[0]) < celltyping.MIN_MARKERS else "")
        say(f"{verb} <b>{html.escape(name)}</b>: {len(genes)} genes ({measured} measured in this data).{warn}"
            + ("" if typing.source.value != "panglao" else
               " Note: the Leiden panel is set to <i>PanglaoDB only</i>, so your sets aren't used until you change it."))
        name_input.value = genes_input.value = ""
        refresh()

    add_btn.on_click(on_add)

    # ---- re-run cell typing ----------------------------------------------------------
    rerun_select = pn.widgets.Select(name="Clustering", options=[], width=260)
    rerun_btn = pn.widgets.Button(name="Re-run cell typing", button_type="success", width=170, align="end")
    results = pn.pane.HTML("", sizing_mode="stretch_width")

    def refresh_clusterings(*events):
        options = clustering.clustering_columns(dataset.adata) if dataset is not None else []
        rerun_select.options = options
        if options and rerun_select.value not in options:
            rerun_select.value = options[-1]
        rerun_btn.disabled = not options

    async def on_rerun(event):
        if not rerun_select.value or state.run_celltyping is None:
            return
        await state.run_celltyping(rerun_select.value)

    def on_result(result):
        results.object = results_html(result)
        refresh_clusterings()

    rerun_btn.on_click(on_rerun)
    state.celltype_listeners.append(on_result)
    state.field_listeners.append(lambda ds, field: refresh_clusterings())
    if dataset is not None and dataset.celltype_results:
        results.object = results_html(list(dataset.celltype_results.values())[-1])

    async def on_download(event):
        try:
            await typing.ensure_markers()
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            say(f"<b>Couldn't download PanglaoDB:</b> {html.escape(str(exc))}")
        refresh()

    download_btn.on_click(on_download)
    for widget in (typing.species, typing.tissue, typing.include_common, typing.source):
        widget.param.watch(refresh, "value")
    typing.tissue.param.watch(refresh, "options")  # filled in once PanglaoDB is downloaded
    refresh()
    refresh_clusterings()

    hint = lambda text: pn.pane.HTML(text, styles=HINT_STYLE, sizing_mode="stretch_width")
    return pn.Column(
        "### Marker gene sets for cell typing",
        hint(
            "Each row is a cell type and the marker genes that identify it. <b>Edit a gene list</b> by "
            "double-clicking it, <b>untick On</b> to leave a type out, or tick rows on the left to reset, "
            f"delete or view them as a pathway. Types need at least {celltyping.MIN_MARKERS} of their "
            "markers measured in this data to be scored (⚠ = fewer); unmeasured markers are listed in gray. "
            f"Your changes are saved on this computer (<code>{html.escape(celltyping.CUSTOM_MARKERS_PATH)}</code>) "
            "and used for every dataset."
        ),
        settings,
        download_btn,
        table,
        pn.Row(reset_btn, delete_btn, pathway_btn),
        status,
        pn.layout.Divider(),
        "### Add a cell type",
        hint("Using the name of an existing type replaces its markers."),
        name_input,
        genes_input,
        add_btn,
        pn.layout.Divider(),
        "### Re-run cell typing",
        hint("Labels an existing clustering again with the current marker sets - no need to re-run "
             "Leiden. The result also appears on the Spatial tab."),
        pn.Row(rerun_select, rerun_btn),
        results,
        sizing_mode="stretch_width",
        max_width=1100,
        margin=(10, 20),
        scroll=True,
    )
