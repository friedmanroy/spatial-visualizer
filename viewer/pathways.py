"""Loading pathways.json (default: viewer/data/pathways.json) and scoring gene sets against whichever AnnData is active.

Pathways the user adds on the Pathways tab are kept separately in
cache/custom_pathways.json and merged in on load, so the default pathways.json is never
modified (it lives inside the installed package, which the launchers replace on update).

Ported from the TNBC viewer's scanpy score_genes pattern, generalized to take the
target AnnData as a parameter (so the same scoring works for the spatial data and any
scRNA/snRNA companion data).
"""
from __future__ import annotations

import json
import os

import numpy as np
import scanpy as sc

from .cache import CACHE_DIR

CUSTOM_PATHWAYS_PATH = os.path.join(CACHE_DIR, "custom_pathways.json")
CUSTOM_SOURCE = "Custom (added in the viewer)"


def _read_pathway_file(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def load_pathways(path: str, include_custom: bool = True) -> dict:
    """The pathways file at `path` plus (unless include_custom=False) the user's custom
    pathways; on a name clash the pathways file wins."""
    pathways = _read_pathway_file(path)
    if include_custom:
        for name, info in load_custom_pathways().items():
            pathways.setdefault(name, info)
    return pathways


def load_custom_pathways() -> dict:
    return _read_pathway_file(CUSTOM_PATHWAYS_PATH)


def _write_custom_pathways(custom: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = CUSTOM_PATHWAYS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(custom, f, indent=2, ensure_ascii=False)
    os.replace(tmp, CUSTOM_PATHWAYS_PATH)  # never leave a half-written file behind


def save_custom_pathway(name: str, genes: list[str], description: str = ""):
    custom = load_custom_pathways()
    custom[name] = {"source": CUSTOM_SOURCE, "description": description, "genes": genes}
    _write_custom_pathways(custom)


def remove_custom_pathway(name: str):
    custom = load_custom_pathways()
    if custom.pop(name, None) is not None:
        _write_custom_pathways(custom)


def score_pathway(dataset, pathway_name: str, pathways: dict) -> np.ndarray | None:
    """Returns a per-cell score array (cached on `dataset`), or None if too few genes match."""
    cache = dataset.pathway_score_cache
    if pathway_name in cache:
        return cache[pathway_name]
    info = pathways.get(pathway_name)
    if info is None:
        return None
    # Match case-insensitively so mouse-cased pathway gene lists (Col1a1) still
    # resolve against a human-cased dataset (COL1A1), or vice versa.
    matched = (dataset.match_gene(g) for g in info.get("genes", []))
    genes = sorted({g for g in matched if g is not None})
    if len(genes) < 2:
        cache[pathway_name] = None
        return None
    adata = dataset.adata
    score_col = f"_pathway_score_{pathway_name}"
    sc.tl.score_genes(
        adata, gene_list=genes, score_name=score_col,
        ctrl_size=min(50, max(len(genes), 10)), random_state=0,
    )
    score = np.asarray(adata.obs[score_col].to_numpy(), dtype=np.float64)
    del adata.obs[score_col]
    cache[pathway_name] = score
    return score
