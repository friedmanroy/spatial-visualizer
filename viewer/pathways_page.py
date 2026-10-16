"""The 'Pathways' tab: lets the user add their own gene sets (from a comma / line
separated gene list), which are saved to cache/custom_pathways.json and immediately
offered in the Gene & Pathway coloring on the other tabs.
"""
from __future__ import annotations

import re

import panel as pn

from . import pathways as pathways_mod

#: at most this many unmatched genes are listed by name in the status message
_MAX_LISTED_MISSING = 20


def parse_genes(text: str) -> list[str]:
    """Gene symbols separated by commas, semicolons, new lines, tabs or spaces, in the
    given order with (case-insensitive) duplicates and stray quotes removed."""
    genes: dict[str, str] = {}
    for token in re.split(r"[,;\s]+", text):
        gene = token.strip("\"'")
        if gene and gene.upper() not in genes:
            genes[gene.upper()] = gene
    return list(genes.values())


def build_pathways_page(state) -> pn.viewable.Viewable:
    name_input = pn.widgets.TextInput(name="Pathway name", placeholder="e.g. My hypoxia genes")
    description_input = pn.widgets.TextAreaInput(name="Description (optional)", height=70)
    genes_input = pn.widgets.TextAreaInput(
        name="Genes (separated by commas or new lines)",
        placeholder="Hif1a, Vegfa, Slc2a1\nPgk1\nLdha",
        height=220,
    )
    save_btn = pn.widgets.Button(name="Add pathway", button_type="primary", width=140)
    status = pn.pane.Markdown("")

    custom_list = pn.pane.Markdown("")
    remove_select = pn.widgets.Select(name="Remove a custom pathway", options=[])
    remove_btn = pn.widgets.Button(name="Remove", button_type="danger", width=100, align="end")

    def refresh_custom_list():
        custom = pathways_mod.load_custom_pathways()
        names = sorted(custom)
        custom_list.object = "\n".join(
            f"- **{n}** ({len(custom[n].get('genes', []))} genes)" for n in names
        ) or "_No custom pathways yet._"
        remove_select.options = names
        remove_select.disabled = remove_btn.disabled = not names

    def pathways_changed(name: str):
        state.pathways = pathways_mod.load_pathways(state.pathways_path)
        for ds in (state.spatial, state.embedding):
            if ds is not None:
                ds.pathway_score_cache.pop(name, None)  # its genes may have changed
        for listener in state.pathway_listeners:
            listener(name)
        refresh_custom_list()

    def match_report(genes: list[str]) -> str:
        ds = state.spatial
        if ds is None:
            return ""
        missing = [g for g in genes if ds.match_gene(g) is None]
        found = len(genes) - len(missing)
        report = f"\n\n{found} of {len(genes)} genes were found in the loaded spatial data."
        if missing:
            listed = ", ".join(missing[:_MAX_LISTED_MISSING])
            more = f" and {len(missing) - _MAX_LISTED_MISSING} more" if len(missing) > _MAX_LISTED_MISSING else ""
            report += f" Not found: {listed}{more}."
        if found < 2:
            report += " **At least 2 genes must be found for the pathway to be scored.**"
        return report

    def on_save(event):
        name = name_input.value.strip()
        genes = parse_genes(genes_input.value)
        if not name:
            status.object = "_Please give the pathway a name._"
            return
        if name.startswith("_"):
            status.object = "_Pathway names can't start with an underscore._"
            return
        if len(genes) < 2:
            status.object = "_Please enter at least 2 genes._"
            return
        if name in pathways_mod.load_pathways(state.pathways_path, include_custom=False):
            status.object = (
                f"**{name}** already exists in the default pathways file - please choose a different name."
            )
            return
        replaced = name in pathways_mod.load_custom_pathways()
        pathways_mod.save_custom_pathway(name, genes, description_input.value.strip())
        pathways_changed(name)
        status.object = (
            f"{'Updated' if replaced else 'Added'} **{name}** ({len(genes)} genes). "
            "It's now available in the Gene & Pathway coloring." + match_report(genes)
        )
        name_input.value = description_input.value = genes_input.value = ""

    def on_remove(event):
        name = remove_select.value
        if not name:
            return
        pathways_mod.remove_custom_pathway(name)
        pathways_changed(name)
        status.object = f"Removed **{name}**."

    save_btn.on_click(on_save)
    remove_btn.on_click(on_remove)
    refresh_custom_list()
    # Pathways can also be added elsewhere (Cell types tab -> "Show as pathway").
    state.pathway_listeners.append(lambda name: refresh_custom_list())

    return pn.Column(
        "### Add a pathway",
        "Paste a list of genes to use as a gene set in the **Gene & Pathway** coloring. "
        "Adding a pathway with the name of one you added before replaces it. Your pathways are "
        f"saved on this computer (`{pathways_mod.CUSTOM_PATHWAYS_PATH}`) and loaded "
        "automatically next time, alongside the default ones.",
        name_input,
        description_input,
        genes_input,
        save_btn,
        status,
        pn.layout.Divider(),
        "### Your pathways",
        custom_list,
        pn.Row(remove_select, remove_btn),
        width=560,
        margin=(10, 20),
    )
