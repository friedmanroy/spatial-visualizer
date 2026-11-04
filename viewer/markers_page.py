"""The 'Cell types' tab: the user's own marker gene sets for cell typing, in an editable
table (the chosen database's sets aren't listed - a set named like one of them replaces
it). Gene lists can be edited, sets switched off, new cell types added, a set viewed as a
pathway score, marker databases imported from a downloaded file (CellMarker 2.0, TISCH2,
...), and cell typing re-run on an existing clustering to see the effect right away.

Edits are saved to cache/custom_markers.json (see celltyping.load_custom_markers).
"""
from __future__ import annotations

import asyncio
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
from .file_picker import build_path_picker
from .pathways_page import parse_genes

#: prefix for marker sets added as pathways ("Show as pathway")
PATHWAY_PREFIX = "Cell type: "
_COLUMNS = ["On", "Cell type", "Markers", "Negative markers", "Measured", "Not measured", "Source"]


def build_markers_page(state) -> pn.viewable.Viewable:
    typing = state.celltyping
    dataset = state.spatial
    rows: dict[str, celltyping.MarkerSet] = {}  # current table rows by cell type

    settings = pn.pane.HTML("", sizing_mode="stretch_width", styles={"font-size": "13px"})
    switched_off = pn.pane.HTML("", sizing_mode="stretch_width", styles={"font-size": "13px"})
    switch_on_btn = pn.widgets.Button(name="Switch them back on", width=170, visible=False)
    status = pn.pane.HTML("", sizing_mode="stretch_width", styles={"font-size": "13px"})
    table = pn.widgets.Tabulator(
        pd.DataFrame(columns=_COLUMNS), show_index=False, selectable="checkbox",
        sizing_mode="stretch_width", height=430, layout="fit_columns",
        header_filters={"Cell type": {"type": "input", "func": "like", "placeholder": "search"},
                        "Markers": {"type": "input", "func": "like", "placeholder": "search"},
                        "Source": {"type": "input", "func": "like", "placeholder": "filter"}},
        editors={"On": {"type": "tickCross"}, "Cell type": None, "Markers": {"type": "textarea"},
                 "Negative markers": {"type": "textarea"},
                 "Measured": None, "Not measured": None, "Source": None},
        formatters={"On": {"type": "tickCross"}, "Not measured": {"type": "html"},
                    "Markers": {"type": "textarea"}, "Negative markers": {"type": "textarea"}},
        # percentages, so Markers stays readable on a laptop-width window too
        widths={"On": 50, "Cell type": "15%", "Markers": "26%", "Negative markers": "12%", "Measured": 90,
                "Not measured": "12%", "Source": "12%"},
        text_align={"Measured": "right"},
        configuration={"placeholder": "No marker sets of your own yet - add one below."},
    )

    def measured_info(genes: list[str]) -> tuple[str, str]:
        if dataset is None:
            return "", ""
        missing = [g for g in genes if dataset.match_gene(g) is None]
        found = len(genes) - len(missing)
        flag = " ⚠" if found < celltyping.MIN_MARKERS else ""
        gray = (f"<span style='color:#999'>{html.escape(', '.join(missing))}</span>" if missing else "")
        return f"{found} of {len(genes)}{flag}", gray

    def database_types() -> dict[str, list[str]]:
        """The chosen database's types for the current species / tissue (empty if not loaded)."""
        if typing._markers is None:
            return {}
        db = typing.db
        net = celltyping.marker_sets(typing._markers, typing.species.value, typing.tissue.value,
                                     db.common_tissues if typing.include_common.value else ())
        return {t: list(g) for t, g in net.groupby("source", sort=True)["target"]}

    def origin_label(m: celltyping.MarkerSet) -> str:
        return f"Replaces {typing.db.name}'s" if m.origin == celltyping.ORIGIN_EDITED else "Added by you"

    def refresh(*events):
        use = {"both": "used together with", "database": "<b>not</b> used - only", "custom": "used instead of"}
        settings.object = (
            f"Cell typing currently uses: <b>{html.escape(typing.settings_label)}</b>. Your sets are "
            f"{use[typing.source.value]} the <b>{html.escape(typing.db.name)}</b> database"
            + ("'s sets" if typing.source.value != "database" else "")
            + ". The database, species, tissue and which sets to use are chosen in the <b>Leiden "
            "clustering &amp; cell types</b> panel on the Spatial tab."
        )
        custom = celltyping.load_custom_markers()
        rows.clear()
        records = []
        for m in celltyping.custom_sets(custom, database_types()):
            rows[m.name] = m
            measured, missing = measured_info(m.genes)
            records.append({"On": m.enabled, "Cell type": m.name, "Markers": ", ".join(m.genes),
                            "Negative markers": ", ".join(m.negative),
                            "Measured": measured, "Not measured": missing, "Source": origin_label(m)})
        table.value = pd.DataFrame(records, columns=_COLUMNS)
        table.selection = []
        # Database types switched off from this table before it listed only your own sets
        off = [n for n in custom["disabled"] if n not in custom["types"]]
        switched_off.object = (
            f"{len(off)} database cell type(s) are switched off: {html.escape(', '.join(off))}." if off else "")
        switch_on_btn.visible = bool(off)

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
            custom["types"][name] = {"genes": genes, "negative": m.negative}
            measured, _ = measured_info(genes)
            say(f"Updated <b>{html.escape(name)}</b>: {len(genes)} genes ({measured} measured in this data).")
        elif event.column == "Negative markers":
            negative = parse_genes(str(event.value))
            custom["types"][name] = {"genes": m.genes, "negative": negative}
            say(f"Updated <b>{html.escape(name)}</b>'s negative markers: "
                + (", ".join(html.escape(g) for g in negative) if negative else "none")
                + ". Only the ULM scoring method uses them.")
        else:
            return
        celltyping.save_custom_markers(custom)
        refresh()

    table.on_edit(on_edit)

    def selected() -> list[celltyping.MarkerSet]:
        names = [table.value.iloc[i]["Cell type"] for i in table.selection]
        return [rows[n] for n in names if n in rows]

    delete_btn = pn.widgets.Button(name="Delete", button_type="danger", width=100,
                                   description="Delete the selected sets (one replacing a database set "
                                               "brings back the database's markers).")
    pathway_btn = pn.widgets.Button(name="Show as pathway", width=160,
                                    description="Score the selected sets like pathways, to color the tissue by them.")

    def on_delete(event):
        chosen = selected()
        if not chosen:
            say("Select one or more sets first.")
            return
        custom = celltyping.load_custom_markers()
        names = {m.name for m in chosen}
        for name in names:
            custom["types"].pop(name, None)
        custom["disabled"] = [n for n in custom["disabled"] if n not in names]
        celltyping.save_custom_markers(custom)
        say(f"Deleted {len(names)} set(s).")
        refresh()

    def on_switch_on(event):
        custom = celltyping.load_custom_markers()
        custom["disabled"] = [n for n in custom["disabled"] if n in custom["types"]]
        celltyping.save_custom_markers(custom)
        say("Switched the database's cell types back on.")
        refresh()

    def on_pathway(event):
        chosen = selected()
        if not chosen:
            say("Select one or more sets first.")
            return
        for m in chosen:
            pathways_mod.save_custom_pathway(
                PATHWAY_PREFIX + m.name, m.genes,
                f"Marker genes of {m.name} ({origin_label(m).lower()}), added from the Cell types tab.",
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

    switch_on_btn.on_click(on_switch_on)
    delete_btn.on_click(on_delete)
    pathway_btn.on_click(on_pathway)

    # ---- add a cell type ---------------------------------------------------------------
    name_input = pn.widgets.TextInput(name="Cell type name", placeholder="e.g. Tumor-associated macrophages",
                                      sizing_mode="stretch_width")
    genes_input = pn.widgets.TextAreaInput(name="Marker genes (separated by commas or new lines)",
                                           placeholder="CD68, CD163\nMRC1\nAPOE", height=110,
                                           sizing_mode="stretch_width")
    negative_input = pn.widgets.TextAreaInput(
        name="Negative markers (optional): genes this type should not express",
        placeholder="CD3E, MS4A1", height=60, sizing_mode="stretch_width",
    )
    add_btn = pn.widgets.Button(name="Add cell type", button_type="primary", width=140)

    def on_add(event):
        name = name_input.value.strip()
        genes = parse_genes(genes_input.value)
        if not name or not genes:
            say("Please give the cell type a name and at least one marker gene.")
            return
        custom = celltyping.load_custom_markers()
        existing = name in custom["types"]
        replaces = name in database_types()
        custom["types"][name] = {"genes": genes, "negative": parse_genes(negative_input.value)}
        custom["disabled"] = [n for n in custom["disabled"] if n != name]
        celltyping.save_custom_markers(custom)
        measured, _ = measured_info(genes)
        verb = "Updated" if existing else (
            f"Replaced {html.escape(typing.db.name)}'s markers for" if replaces else "Added")
        warn = (f" <b>Only sets with at least {celltyping.MIN_MARKERS} measured genes are scored.</b>"
                if dataset is not None and int(measured.split(" ")[0]) < celltyping.MIN_MARKERS else "")
        say(f"{verb} <b>{html.escape(name)}</b>: {len(genes)} genes ({measured} measured in this data).{warn}"
            + ("" if typing.source.value != "database" else
               " Note: the Leiden panel is set to <i>Database only</i>, so your sets aren't used until you change it."))
        name_input.value = genes_input.value = negative_input.value = ""
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

    # ---- import a marker database from a file -------------------------------------------
    import_path, import_browse, _ = build_path_picker(
        "Marker table (.xlsx, .csv, .tsv)", placeholder="/path/to/Cell_marker_Human.xlsx",
        file_pattern="*.*", busy=state.busy)
    import_path.sizing_mode = "stretch_width"
    import_name = pn.widgets.TextInput(name="Name it", placeholder="e.g. CellMarker 2.0", width=220)
    import_btn = pn.widgets.Button(name="Import", button_type="primary", width=100, align="end")
    import_status = pn.pane.HTML("", sizing_mode="stretch_width", styles={"font-size": "13px"})
    imported_select = pn.widgets.Select(name="Imported databases", options={}, width=260)
    remove_btn = pn.widgets.Button(name="Remove", button_type="danger", width=100, align="end")
    imported_row = pn.Row(imported_select, remove_btn, visible=False)

    def refresh_imported():
        options = {db.name: key for key, db in celltyping.all_databases().items() if db.resource is None}
        imported_select.options = options
        if options and imported_select.value not in options.values():
            imported_select.value = next(iter(options.values()))
        imported_row.visible = bool(options)

    async def on_import(event):
        path, name = import_path.value.strip(), import_name.value.strip()
        if not path or not name:
            import_status.object = "Please choose a file and give the database a name."
            return
        try:
            async with state.busy.show("Importing the marker table...", path):
                db, about = await asyncio.to_thread(celltyping.import_marker_file, path, name)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            import_status.object = f"<b>Couldn't import it:</b> {html.escape(str(exc))}"
            return
        import_status.object = (f"Imported <b>{html.escape(name)}</b>. {html.escape(about)} "
                                "It's now chosen as the marker database in the Leiden panel.")
        import_path.value = import_name.value = ""
        typing.refresh_databases(select=db.key)
        refresh_imported()

    def on_remove(event):
        key = imported_select.value
        if not key:
            return
        celltyping.remove_imported(key)
        typing.refresh_databases()
        import_status.object = "Removed it."
        refresh_imported()

    import_btn.on_click(on_import)
    remove_btn.on_click(on_remove)

    for widget in (typing.database, typing.species, typing.tissue, typing.include_common, typing.source,
                   typing.method):
        widget.param.watch(refresh, "value")
    typing.tissue.param.watch(refresh, "options")  # filled in once the database is downloaded
    refresh()
    refresh_clusterings()
    refresh_imported()

    hint = lambda text: pn.pane.HTML(text, styles=HINT_STYLE, sizing_mode="stretch_width")
    return pn.Column(
        "### Your marker gene sets",
        hint(
            "Each row is a cell type you added and the marker genes that identify it. <b>Edit a gene list</b> by "
            "double-clicking it, <b>untick On</b> to leave a type out, or tick rows on the left to "
            f"delete them or view them as a pathway. Types need at least {celltyping.MIN_MARKERS} of their "
            "markers measured in this data to be scored (⚠ = fewer); unmeasured markers are listed in gray. "
            "<b>Negative markers</b> (genes a type should <i>not</i> express, e.g. CD3E for B cells) count "
            "against it with the ULM method (the others leave them out); the databases have none, so add your own. "
            f"Your sets are saved on this computer (<code>{html.escape(celltyping.CUSTOM_MARKERS_PATH)}</code>) "
            "and used for every dataset."
        ),
        settings,
        table,
        pn.Row(delete_btn, pathway_btn),
        status,
        pn.Row(switched_off, switch_on_btn),
        pn.layout.Divider(),
        "### Add a cell type",
        hint("Using the name of one of your sets replaces its markers; using the name of a database "
             "cell type replaces the database's markers for it."),
        name_input,
        genes_input,
        negative_input,
        add_btn,
        pn.layout.Divider(),
        "### Import a marker database",
        hint("For databases that can't be downloaded automatically - e.g. <b>CellMarker 2.0</b>'s "
             "<code>Cell_marker_Human.xlsx</code> (or _Mouse) from its Download page (only its curated "
             "markers are kept), or a <b>TISCH2</b> dataset's differentially expressed genes table "
             f"(each cell type's top {celltyping.IMPORT_TOP_GENES} genes are kept). Any table with a "
             "cell-type column and a gene column works; a tissue column, if present, fills the Tissue list."),
        pn.Row(import_path, import_browse, import_name, import_btn, sizing_mode="stretch_width"),
        import_status,
        imported_row,
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
