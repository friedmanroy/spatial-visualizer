"""Rudimentary marker-based cell typing of clusters (e.g. Leiden), using decoupler with
PanglaoDB's cell-type marker genes.

Each cluster is summarised by its cells' mean expression; every gene is then
standardised across clusters, so a marker set scores high in a cluster where its genes
are *relatively* high. decoupler's univariate linear model (ULM) scores each PanglaoDB
cell type against each cluster, and the best-scoring type is suggested - or "Unknown"
when no type scores significantly.

Each cell also gets a confidence (0-1) in its cluster's suggested type: the same ULM
scores computed per cell, turned into shares with a softmax over all scored types -
near 1 when the cell's markers clearly point to that type, low when another type's
markers fit it as well or better. These are suggestions to check against known
markers, not a validated annotation.

PanglaoDB is downloaded once (through OmniPath) and kept in cache/ for offline use.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import cache

ALL_TISSUES = "All tissues"
#: cell types found in almost any tissue - offered alongside a chosen tissue's own types
COMMON_ORGANS = ("Immune system", "Blood", "Vasculature", "Connective tissue")
UNKNOWN = "Unknown"
#: a cell type needs at least this many of its markers measured to be scored at all -
#: low, since targeted spatial panels often hold only a handful of any type's markers
MIN_MARKERS = 3
#: PanglaoDB markers detected in fewer than this fraction of that cell type's cells are
#: left out (the threshold decoupler's own cell-type tutorial uses)
MIN_SENSITIVITY = 0.5
#: a suggestion needs a ULM p-value below this, else the cluster is "Unknown"
MAX_PVALUE = 0.05
#: how many candidates per cluster are kept for display
N_CANDIDATES = 3

_MARKERS_CACHE = os.path.join(cache.CACHE_DIR, "panglaodb_markers.csv.gz")


def load_markers() -> pd.DataFrame:
    """PanglaoDB's marker table (human gene symbols, with per-species flags), from
    cache/ if present, else downloaded through decoupler/OmniPath and cached."""
    if os.path.exists(_MARKERS_CACHE):
        return pd.read_csv(_MARKERS_CACHE)
    import decoupler as dc  # imported lazily: it's slow to import and only needed here

    try:
        markers = dc.op.resource("PanglaoDB", organism="human")
    except Exception as exc:
        raise RuntimeError(
            "Couldn't download the PanglaoDB marker genes (needs an internet connection "
            "the first time; after that they're kept in cache/)."
        ) from exc
    os.makedirs(cache.CACHE_DIR, exist_ok=True)
    markers.to_csv(_MARKERS_CACHE, index=False)
    return markers


def markers_available_offline() -> bool:
    return os.path.exists(_MARKERS_CACHE)


def tissue_options(markers: pd.DataFrame) -> list[str]:
    return [ALL_TISSUES] + sorted(markers["organ"].dropna().unique())


def guess_species(var_names) -> str:
    """'mouse' if gene symbols are mostly mouse-cased (Col1a1), else 'human' (COL1A1).
    Ensembl IDs are recognised by prefix (ENSMUSG = mouse, ENSG = human)."""
    names = [str(v) for v in var_names[:2000] if any(ch.isalpha() for ch in str(v))]
    if not names:
        return "human"
    ensembl_mouse = sum(n.upper().startswith("ENSMUS") for n in names)
    ensembl_human = sum(n.upper().startswith("ENSG") for n in names)
    if ensembl_mouse + ensembl_human > len(names) / 2:
        return "mouse" if ensembl_mouse > ensembl_human else "human"
    mouse_like = sum(n[0].isupper() and n[1:] != n[1:].upper() for n in names if len(n) > 1)
    return "mouse" if mouse_like > len(names) / 2 else "human"


def _as_bool(col: pd.Series) -> pd.Series:
    if col.dtype == bool:
        return col
    return col.astype(str).str.lower().isin(("true", "1"))


def marker_sets(markers: pd.DataFrame, species: str, tissue: str, include_common: bool) -> pd.DataFrame:
    """(cell_type, gene) pairs to score: canonical, sensitive markers seen in `species`,
    from `tissue` (plus the ubiquitous immune/blood/vessel/connective types if asked)."""
    keep = (
        _as_bool(markers[species])
        & _as_bool(markers["canonical_marker"])
        & (pd.to_numeric(markers[f"{species}_sensitivity"], errors="coerce") > MIN_SENSITIVITY)
    )
    if tissue != ALL_TISSUES:
        organs = {tissue, *COMMON_ORGANS} if include_common else {tissue}
        keep &= markers["organ"].isin(organs)
    net = markers.loc[keep, ["cell_type", "genesymbol"]].drop_duplicates()
    return net.rename(columns={"cell_type": "source", "genesymbol": "target"})


@dataclass
class ClusterCall:
    cluster: str
    n_cells: int
    label: str  # suggested cell type, or UNKNOWN
    candidates: list[tuple[str, float, int]] = field(default_factory=list)  # (type, score, markers measured)
    mean_confidence: float = float("nan")  # mean per-cell confidence in `label` (NaN if Unknown)


@dataclass
class CellTypingResult:
    cluster_col: str
    column: str  # obs column holding each cell's suggested type
    confidence_column: str  # obs column holding each cell's confidence (0-1) in that type
    species: str
    tissue: str
    n_types_scored: int
    calls: list[ClusterCall]


def _cluster_means(adata, codes: np.ndarray, n_clusters: int) -> np.ndarray:
    valid = codes >= 0
    cells = np.flatnonzero(valid)
    indicator = sp.csr_matrix(
        (np.ones(len(cells)), (codes[valid], cells)), shape=(n_clusters, adata.n_obs),
    )
    sizes = np.asarray(indicator.sum(axis=1)).ravel()
    mean = sp.diags(1.0 / np.maximum(sizes, 1)) @ indicator @ adata.X
    return np.asarray(mean.todense()) if sp.issparse(mean) else np.asarray(mean)


def annotate_clusters(dataset, cluster_col: str, markers: pd.DataFrame, species: str,
                      tissue: str, include_common: bool = True, report=lambda text: None) -> CellTypingResult:
    """Suggests a cell type per cluster of obs[`cluster_col`], stores each cell's
    suggestion as obs[f"{cluster_col}_celltype"], and returns the per-cluster calls."""
    import decoupler as dc

    adata = dataset.adata
    clusters = adata.obs[cluster_col].astype("category")
    names = [str(c) for c in clusters.cat.categories]
    codes = clusters.cat.codes.to_numpy()

    # PanglaoDB uses human symbols; match them to this dataset's own spelling (case-
    # insensitively, which also covers most mouse symbols) and drop unmeasured ones.
    report("Matching PanglaoDB marker genes to this data's genes")
    net = marker_sets(markers, species, tissue, include_common)
    net["target"] = [dataset.match_gene(g) for g in net["target"]]
    net = net.dropna(subset=["target"]).drop_duplicates()
    measured = net.groupby("source").size()
    net = net[net["source"].isin(measured[measured >= MIN_MARKERS].index)]
    if net.empty:
        raise ValueError(
            f"None of the {tissue.lower()} cell types has at least {MIN_MARKERS} of its "
            "marker genes measured in this data - try 'All tissues', or the gene panel "
            "may be too small for marker-based typing."
        )

    genes = sorted(set(net["target"]))
    report(f"Scoring {net['source'].nunique()} cell types in each of the {len(names)} clusters")
    gene_idx = [adata.var_names.get_loc(g) for g in genes]
    means = pd.DataFrame(_cluster_means(adata[:, gene_idx], codes, len(names)), index=names, columns=genes)
    spread = means.std(axis=0)
    means = means.loc[:, spread > 0]
    scaled = (means - means.mean(axis=0)) / spread[means.columns]
    scores, pvalues = dc.mt.ulm(data=scaled, net=net, tmin=MIN_MARKERS)

    sizes = np.bincount(codes[codes >= 0], minlength=len(names))
    calls = []
    for i, name in enumerate(names):
        ranked = scores.loc[name].sort_values(ascending=False)
        top = ranked.index[0]
        significant = ranked.iloc[0] > 0 and pvalues.loc[name, top] < MAX_PVALUE
        calls.append(ClusterCall(
            cluster=name, n_cells=int(sizes[i]), label=top if significant else UNKNOWN,
            candidates=[(t, float(ranked[t]), int(measured[t])) for t in ranked.index[:N_CANDIDATES]],
        ))

    column = f"{cluster_col}_celltype"
    label_of = {c.cluster: c.label for c in calls}
    cell_labels = np.array([label_of.get(str(v)) if v == v else None for v in clusters.to_numpy()], dtype=object)
    present = sorted({c.label for c in calls} - {UNKNOWN}) + ([UNKNOWN] if UNKNOWN in label_of.values() else [])
    adata.obs[column] = pd.Categorical(cell_labels, categories=present)

    report(f"Scoring each cell's confidence in its type ({adata.n_obs:,} cells)")
    confidence = _cell_confidence(dc, adata, genes, net, cell_labels)
    confidence_column = f"{column}_confidence"
    adata.obs[confidence_column] = confidence
    for i, call in enumerate(calls):
        in_cluster = codes == i
        if call.label != UNKNOWN and in_cluster.any():
            call.mean_confidence = float(np.nanmean(confidence[in_cluster]))
    return CellTypingResult(cluster_col, column, confidence_column, species, tissue,
                            int(net["source"].nunique()), calls)


def _cell_confidence(dc, adata, genes: list[str], net: pd.DataFrame, cell_labels: np.ndarray) -> np.ndarray:
    """Per cell: softmax share of its assigned type among all scored types' per-cell ULM
    scores (NaN for cells without a type). Genes are divided by their spread across
    cells (not centred, so sparse data stays sparse) so no gene dominates by scale."""
    import anndata as ad

    X = adata[:, genes].X
    X = sp.csr_matrix(X, dtype=np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)
    mean = np.asarray(X.mean(axis=0)).ravel()
    mean_sq = np.asarray((X.multiply(X) if sp.issparse(X) else X * X).mean(axis=0)).ravel()
    spread = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    spread[spread == 0] = 1.0
    X = X @ sp.diags(1.0 / spread) if sp.issparse(X) else X / spread
    cells = ad.AnnData(sp.csr_matrix(X) if sp.issparse(X) else X)
    cells.var_names = genes
    # Batches keep memory bounded on million-cell data (each batch is densified).
    dc.mt.ulm(data=cells, net=net, tmin=MIN_MARKERS, bsize=50_000)
    scores = cells.obsm["score_ulm"]
    S = np.nan_to_num(scores.to_numpy(dtype=np.float64))
    S -= S.max(axis=1, keepdims=True)
    share = np.exp(S)
    share /= share.sum(axis=1, keepdims=True)

    type_index = {t: j for j, t in enumerate(scores.columns)}
    confidence = np.full(adata.n_obs, np.nan, dtype=np.float32)
    for label in set(cell_labels) - {None, UNKNOWN}:
        if label in type_index:
            rows = np.flatnonzero(cell_labels == label)
            confidence[rows] = share[rows, type_index[label]]
    return confidence
