"""Loading data/pathways.json and scoring gene sets against whichever AnnData is active.

Ported from the TNBC viewer's scanpy score_genes pattern, generalized to take the
target AnnData as a parameter (so the same scoring works for the spatial data and any
scRNA/snRNA companion data).
"""
from __future__ import annotations

import json
import os

import numpy as np
import scanpy as sc


def load_pathways(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def score_pathway(dataset, pathway_name: str, pathways: dict) -> np.ndarray | None:
    """Returns a per-cell score array (cached on `dataset`), or None if too few genes match."""
    cache = dataset.pathway_score_cache
    if pathway_name in cache:
        return cache[pathway_name]
    info = pathways.get(pathway_name)
    if info is None:
        return None
    genes = [g for g in info.get("genes", []) if g in dataset.adata.var_names]
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
