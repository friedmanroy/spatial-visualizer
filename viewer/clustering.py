"""Shared scanpy preprocessing (HVG -> scale -> PCA -> neighbor graph) used both for the
PCA/UMAP embedding and for optional Leiden clustering at load time.

Always works on a highly-variable-gene subset *copy*, so the dataset's own .X (used
elsewhere for gene-expression coloring) is never mutated by scaling.

Sub-cellular spots (Xenium transcripts, Open-ST, Visium HD...) are too sparse to cluster
or type one by one, so everything here can instead work on hexagonal bins, and each spot
gets its bin's result. A bin's profile is its spots' summed raw counts, normalized and
log1p'd as one unit (see hex_profiles) - not an average of the spots' own log values.

Broadly expressed "background" genes (mitochondrial, ribosomal, or detected almost
everywhere) can be left out of the genes PCA / UMAP / Leiden use - see GeneFilter.
"""
from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

DEFAULT_N_PCS = 10
DEFAULT_N_NEIGHBORS = 15
DEFAULT_LEIDEN_RESOLUTION = 1.0
#: what clustering / PCA / UMAP run on (labels of the choice shown in the UI)
HEXAGONS, CELLS = "Hexagons", "Individual cells"
#: gene-name prefixes (case-insensitive) of mitochondrial and ribosomal-protein genes
MITO_PREFIXES = ("MT-", "MT.")
RIBO_PREFIXES = ("RPS", "RPL", "MRPS", "MRPL")


@dataclass(frozen=True)
class GeneFilter:
    """Background genes to leave out of PCA / UMAP / Leiden (and cell typing)."""
    mito: bool = True
    ribo: bool = False
    #: leave out genes detected (> 0) in more than this fraction of the cells / hexagons
    #: being analysed; None = keep them
    max_detected: float | None = None

    def describe(self) -> str:
        parts = (["mitochondrial"] if self.mito else []) + (["ribosomal"] if self.ribo else [])
        if self.max_detected is not None:
            parts.append(f"detected in >{self.max_detected:.0%}")
        return ("without " + ", ".join(parts) + " genes") if parts else "all genes"


NO_FILTER = GeneFilter(mito=False)


def background_genes(adata, gene_filter: GeneFilter | None) -> np.ndarray:
    """Boolean mask over adata.var_names of the genes `gene_filter` leaves out. The
    detection rate is measured on `adata` itself (cells, or hexagon profiles)."""
    out = np.zeros(adata.n_vars, dtype=bool)
    if gene_filter is None:
        return out
    upper = np.array([str(g).upper() for g in adata.var_names])
    if gene_filter.mito:
        out |= np.char.startswith(upper, MITO_PREFIXES[0]) | np.char.startswith(upper, MITO_PREFIXES[1])
    if gene_filter.ribo:
        for prefix in RIBO_PREFIXES:
            out |= np.char.startswith(upper, prefix)
    if gene_filter.max_detected is not None and adata.n_obs:
        X = adata.X
        detected = (np.asarray((X > 0).sum(axis=0)).ravel() if sp.issparse(X)
                    else np.asarray((np.asarray(X) > 0).sum(axis=0)).ravel())
        out |= detected / adata.n_obs > gene_filter.max_detected
    return out


#: how many stored values `looks_like_raw_counts` inspects (enough to be reliable, cheap on huge files)
_RAW_CHECK_SAMPLE = 100_000


def looks_like_raw_counts(adata) -> bool:
    """True if .X holds non-negative whole numbers (judged on a sample of its values),
    i.e. raw counts that still need normalizing. Normalized / log1p data has fractions."""
    X = adata.X
    values = X.data if sp.issparse(X) else np.asarray(X).ravel()
    values = values[:_RAW_CHECK_SAMPLE]
    if values.size == 0:
        return False
    return bool(np.all(values >= 0) and np.all(values == np.round(values)))


def normalize_if_raw(adata) -> bool:
    """normalize_total + log1p, but only if .X still looks like raw counts, so loading
    an already-normalized file doesn't log-transform it a second time. Returns whether
    it normalized."""
    if not looks_like_raw_counts(adata):
        return False
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return True


def normalize_keeping_counts(adata):
    """normalize_total + log1p on .X (whatever it holds), returning the values from
    before - kept so hexagonal bins can sum raw counts before normalizing (hex_profiles).
    A sparse .X keeps only a copy of its values; the index arrays are shared."""
    X = adata.X
    if sp.issparse(X):
        X = adata.X = sp.csr_matrix(X)
        counts = sp.csr_matrix((X.data.copy(), X.indices, X.indptr), shape=X.shape)
    else:
        counts = np.array(X, copy=True)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return counts


def _no_report(text: str):
    pass


def neighbors_graph(adata, n_pcs: int, n_neighbors: int, report=_no_report,
                    gene_filter: GeneFilter | None = None):
    """Returns a scaled HVG-subset copy of `adata` with PCA and a neighbor graph
    computed, leaving out the background genes `gene_filter` names. Also stores the PCA
    on `adata.obsm['X_pca']`."""
    keep = ~background_genes(adata, gene_filter)
    if keep.sum() < 2:
        raise ValueError("Leaving out the background genes leaves fewer than 2 genes - "
                         "relax the background-gene filter.")
    if "highly_variable" in adata.var.columns:  # the file's own choice of genes
        hvg_mask = adata.var["highly_variable"].to_numpy(dtype=bool) & keep
    else:
        n_top = min(2000, int(keep.sum()))
        report(f"Finding the {n_top:,} most variable genes")
        hvg = sc.pp.highly_variable_genes(adata[:, keep], n_top_genes=n_top, flavor="seurat", inplace=False)
        hvg_mask = np.zeros(adata.n_vars, dtype=bool)
        hvg_mask[np.flatnonzero(keep)] = hvg["highly_variable"].to_numpy(dtype=bool)
    if hvg_mask.sum() < 2:
        hvg_mask = keep

    unit = adata.uns.get("unit", "cells")  # hex_profiles' bins say "hexagons"
    report(f"Scaling the expression of {int(hvg_mask.sum()):,} genes in {adata.n_obs:,} {unit}")
    tmp = adata[:, hvg_mask].copy()
    sc.pp.scale(tmp, max_value=10)
    n_comps = max(2, min(n_pcs, tmp.n_vars - 1, tmp.n_obs - 1))
    report(f"Computing PCA ({n_comps} components)")
    sc.pp.pca(tmp, n_comps=n_comps, svd_solver="arpack")
    adata.obsm["X_pca"] = tmp.obsm["X_pca"]

    n_neigh = max(2, min(n_neighbors, tmp.n_obs - 1))
    report(f"Building the neighbor graph ({n_neigh} neighbors per {unit[:-1]})")
    sc.pp.neighbors(tmp, n_neighbors=n_neigh, n_pcs=n_comps)
    return tmp


def clustering_columns(adata) -> list[str]:
    """obs columns holding Leiden clusterings made by the viewer (not the cell-type /
    confidence columns derived from them)."""
    return [c for c in adata.obs.columns
            if c.startswith("leiden_res") and not c.endswith(("_celltype", "_confidence"))]


def leiden_column_name(resolution: float, hex_width: float | None = None) -> str:
    name = f"leiden_res{resolution:g}"
    return name if hex_width is None else f"{name}_hex{hex_width:g}"


def _leiden_labels(adata, resolution: float, n_pcs: int, n_neighbors: int, report=_no_report,
                   gene_filter: GeneFilter | None = None) -> np.ndarray:
    tmp = neighbors_graph(adata, n_pcs, n_neighbors, report, gene_filter)
    report(f"Running Leiden (resolution {resolution:g})")
    sc.tl.leiden(tmp, resolution=resolution, key_added="leiden", flavor="igraph",
                 n_iterations=2, directed=False, random_state=0)
    return tmp.obs["leiden"].to_numpy()


def _as_categorical(labels: np.ndarray) -> pd.Categorical:
    """Cluster labels ('0', '1', ... or None) with their categories in numeric order."""
    present = {label for label in labels if label is not None}
    return pd.Categorical(labels, categories=sorted(present, key=int))


def run_leiden(dataset, resolution: float, n_pcs: int = DEFAULT_N_PCS,
               n_neighbors: int = DEFAULT_N_NEIGHBORS, report=_no_report,
               gene_filter: GeneFilter | None = None) -> str:
    """Leiden-clusters the dataset's individual cells and stores the labels as a
    categorical obs column, returning that column's name (one per resolution, so
    different runs can be compared)."""
    adata = dataset.adata
    col = leiden_column_name(resolution)
    adata.obs[col] = _as_categorical(_leiden_labels(adata, resolution, n_pcs, n_neighbors, report, gene_filter))
    dataset.hex_clusterings.pop(col, None)
    return col


def hex_profiles(dataset, width_factor: float, genes: list[str] | None = None):
    """(AnnData with one row per hexagonal bin - `width_factor` cell spacings wide - of
    every sample; [(cell indices, HexBinning)] per sample, in the same order as the rows).
    obs holds each bin's sample and cell count, obsm['spatial'] its center.

    If the app normalized the data (dataset.counts holds the values from before), a bin's
    profile is its cells' summed counts, then normalize_total + log1p over the bins - the
    bin is treated as one deeper-sequenced spot. Otherwise .X was used as-is, and a bin
    holds its cells' mean. `genes` keeps only those columns (after normalizing, which
    needs every gene's counts)."""
    adata = dataset.adata
    summed = dataset.counts is not None
    X = dataset.counts if summed else adata.X
    profiles, binnings, samples, counts, centers = [], [], [], [], []
    for sid in dataset.sample_ids:
        idx = dataset.sample_indices(sid)
        binning = dataset.hex_binning(sid, width_factor)
        weights = sp.diags(binning.counts.astype(np.float32)) @ binning.mean_matrix if summed else binning.mean_matrix
        profile = weights @ X[idx]
        profiles.append(sp.csr_matrix(profile, dtype=np.float32) if sp.issparse(profile) else np.asarray(profile))
        binnings.append((idx, binning))
        samples += [sid] * binning.n_bins
        counts.append(binning.counts)
        centers.append(binning.centers)
    stacked = sp.vstack(profiles, format="csr") if sp.issparse(profiles[0]) else np.vstack(profiles)
    bins = ad.AnnData(stacked)
    bins.var_names = adata.var_names
    if summed:
        sc.pp.normalize_total(bins)
        sc.pp.log1p(bins)
    if genes is not None:
        bins = bins[:, genes].copy()
    bins.obs["sample"] = pd.Categorical(samples, categories=dataset.sample_ids)
    bins.obs["n_cells"] = np.concatenate(counts)
    bins.obs_names = [f"hex{i}" for i in range(bins.n_obs)]
    bins.obsm["spatial"] = np.vstack(centers)
    bins.uns["unit"] = "hexagons"
    return bins, binnings


def hex_profile_verb(dataset) -> str:
    """How hex_profiles makes the bins' profiles, for progress messages."""
    return ("Summing counts and normalizing" if dataset.counts is not None
            else "Averaging expression")


def bins_to_cells(bin_values: np.ndarray, binnings, n_obs: int, fill=None) -> np.ndarray:
    """Gives every cell its bin's value (cells outside every sample get `fill`)."""
    out = np.full(n_obs, fill, dtype=object if fill is None else bin_values.dtype)
    offset = 0
    for idx, binning in binnings:
        out[idx] = bin_values[offset + binning.cell_bin]
        offset += binning.n_bins
    return out


def cells_to_bins(codes: np.ndarray, binnings) -> np.ndarray:
    """Each bin's most common non-negative code among its cells (-1 if none)."""
    return np.concatenate([binning.majority(codes[idx]) for idx, binning in binnings])


def run_leiden_on_hexbins(dataset, width_factor: float, resolution: float,
                          n_pcs: int = DEFAULT_N_PCS,
                          n_neighbors: int = DEFAULT_N_NEIGHBORS, report=_no_report,
                          gene_filter: GeneFilter | None = None) -> tuple[str, int]:
    """Leiden-clusters the hexagonal bins (`width_factor` cell spacings wide) of every
    sample (each bin's profile as in hex_profiles), then gives every cell
    its bin's cluster - the way to cluster sub-cellular spots, and far faster than
    per-cell clustering on million-cell data. Stores the labels as an obs column
    (remembering the bin width in dataset.hex_clusterings, so cell typing can work on
    the same bins) and returns (column name, number of bins clustered)."""
    adata = dataset.adata
    report(f"{hex_profile_verb(dataset)} within hexagonal bins ({len(dataset.sample_ids)} sample(s))")
    bins, binnings = hex_profiles(dataset, width_factor)
    bin_labels = _leiden_labels(bins, resolution, n_pcs, n_neighbors, report, gene_filter)
    report(f"Giving each cell its bin's cluster ({bins.n_obs:,} bins)")
    # Cells outside every sample (missing sample value) stay unlabeled.
    cell_labels = bins_to_cells(np.asarray(bin_labels, dtype=object), binnings, adata.n_obs)
    col = leiden_column_name(resolution, width_factor)
    adata.obs[col] = _as_categorical(cell_labels)
    dataset.hex_clusterings[col] = width_factor
    return col, bins.n_obs
