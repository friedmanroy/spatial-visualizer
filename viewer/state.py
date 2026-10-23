"""Shared application state: loaded AnnData(s), FOV/sample column, and per-dataset caches.

Mirrors the TNBC viewer's use of module-level state for widget callbacks, but generic:
no column name is ever assumed to exist - everything is looked up through `fields.py`.
"""
from __future__ import annotations

import numpy as np
import scanpy as sc
import scipy.sparse as sp

from . import clustering
from . import fields as fields_mod
from . import hexbin
from . import spatial_plot

#: how many recent top-genes results Dataset.top_genes keeps
_TOP_GENES_CACHE_SIZE = 8


def top_k_per_row(X, k: int, chunk: int = 10_000) -> np.ndarray:
    """(n_rows, k) column indices of each row's k largest values, largest first; -1
    where a row has fewer than k positive values. Sparse rows are padded to the longest
    row's nonzero count (per chunk) rather than densified over all columns."""
    n = X.shape[0]
    out = np.full((n, k), -1, dtype=np.int32)
    for start in range(0, n, chunk):
        block = X[start:start + chunk]
        m = block.shape[0]
        if sp.issparse(block):
            block = sp.csr_matrix(block)
            lens = np.diff(block.indptr)
            width = max(int(lens.max(initial=0)), k)
            vals = np.full((m, width), -np.inf, dtype=np.float32)
            cols = np.full((m, width), -1, dtype=np.int32)
            rows = np.repeat(np.arange(m), lens)
            pos = np.arange(block.nnz) - np.repeat(block.indptr[:-1], lens)
            vals[rows, pos] = block.data
            cols[rows, pos] = block.indices
        else:
            vals = np.asarray(block, dtype=np.float32)
            cols = np.broadcast_to(np.arange(vals.shape[1], dtype=np.int32), vals.shape)
            if vals.shape[1] < k:
                pad = k - vals.shape[1]
                vals = np.pad(vals, ((0, 0), (0, pad)), constant_values=-np.inf)
                cols = np.pad(cols, ((0, 0), (0, pad)), constant_values=-1)
        top = np.argpartition(-vals, k - 1, axis=1)[:, :k] if vals.shape[1] > k else np.argsort(-vals, axis=1)
        top_vals = np.take_along_axis(vals, top, axis=1)
        order = np.argsort(-top_vals, axis=1, kind="stable")
        top = np.take_along_axis(top, order, axis=1)
        top_vals = np.take_along_axis(top_vals, order, axis=1)
        top_cols = np.take_along_axis(cols, top, axis=1)
        top_cols[~(top_vals > 0)] = -1
        out[start:start + m] = top_cols
    return out


class Dataset:
    """Wraps one AnnData plus the caches used by the coloring/plotting machinery.

    Used identically for the main spatial AnnData and for a companion scRNA/snRNA
    AnnData (or a spatial-derived embedding reusing the same object).
    """

    def __init__(self, adata, name: str = "dataset"):
        self.adata = adata
        self.name = name
        self.sample_col: str | None = None
        self.sample_ids: list[str] = []
        self._sample_indices: dict[str, np.ndarray] = {}
        self._lattice_cache: dict[str, spatial_plot.Lattice] = {}
        self._binning_cache: dict[tuple[str, float], hexbin.HexBinning] = {}
        self._field_cache: dict[str, fields_mod.FieldSpec] = {}
        self._gene_cache: dict[str, np.ndarray] = {}
        self._var_upper_lookup: dict[str, str] | None = None
        self._variable_ranking: np.ndarray | None = None  # var indices, most variable first
        self._top_genes_cache: dict[tuple, np.ndarray] = {}
        # Background genes (clustering.GeneFilter) left out of the "most expressed" genes.
        self.gene_filter = None
        self._background_masks: dict = {}
        self.pathway_score_cache: dict[str, np.ndarray | None] = {}
        # obs column -> celltyping.CellTypingResult, for showing per-cluster suggestions.
        self.celltype_results: dict = {}
        # obs column -> bin width (cell spacings) of clusterings made on hexagonal bins,
        # so cell typing can score those same bins rather than single cells / spots.
        self.hex_clusterings: dict[str, float] = {}
        # For a dataset of hexagon profiles made from another one (the scRNA tab's
        # hexagon PCA/UMAP): (that Dataset, [(cell indices, HexBinning)] per sample).
        self.hex_source: tuple | None = None
        # None means "no restriction, offer every eligible field"; otherwise only
        # these column names are offered as "color by" options (set on the landing
        # page so irrelevant obs columns don't clutter the field picker).
        self.allowed_fields: set[str] | None = None
        # Set right after construction (landing page) once the source file path is
        # known; carries any custom category colors saved from a previous session.
        self.color_cache = None  # viewer.cache.ColorCache | None

    # -- fields / categories -------------------------------------------------
    def classify(self, colname: str) -> fields_mod.FieldSpec:
        if colname not in self._field_cache:
            spec = fields_mod.classify_field(self.adata, colname)
            if self.color_cache is not None:
                spec.colors.update(self.color_cache.get_field_colors(colname))
            self._field_cache[colname] = spec
        return self._field_cache[colname]

    def add_field(self, colname: str):
        """Registers an obs column computed after loading (e.g. Leiden clusters):
        drops any stale classification of it and makes sure it's offered for coloring."""
        self._field_cache.pop(colname, None)
        self.classify(colname)
        if self.allowed_fields is not None:
            self.allowed_fields.add(colname)

    def field_options(self, exclude: tuple[str, ...] = ()) -> list[str]:
        options = fields_mod.list_field_options(self.adata, exclude=exclude)
        if self.allowed_fields is not None:
            options = [o for o in options if o in self.allowed_fields]
        return options

    # -- sample / FOV splitting -----------------------------------------------
    def set_sample_column(self, colname: str | None):
        self.sample_col = colname
        self._sample_indices = {}
        self._lattice_cache = {}
        self._binning_cache = {}
        if colname is None:
            self.sample_ids = ["all cells"]
            self._sample_indices["all cells"] = np.arange(self.adata.n_obs)
            return
        col = self.adata.obs[colname]
        values = col.cat.categories if hasattr(col, "cat") else sorted(col.dropna().unique())
        self.sample_ids = [str(v) for v in values]
        codes = col.astype(str).to_numpy()
        for sid in self.sample_ids:
            self._sample_indices[sid] = np.where(codes == sid)[0]

    def sample_indices(self, sample_id: str) -> np.ndarray:
        return self._sample_indices[sample_id]

    def sample_coords(self, sample_id: str) -> np.ndarray:
        return self.adata.obsm["spatial"][self.sample_indices(sample_id)]

    def largest_sample_size(self) -> int:
        return max(len(idx) for idx in self._sample_indices.values())

    def sample_lattice(self, sample_id: str) -> spatial_plot.Lattice:
        if sample_id not in self._lattice_cache:
            self._lattice_cache[sample_id] = spatial_plot.estimate_lattice(self.sample_coords(sample_id))
        return self._lattice_cache[sample_id]

    def hex_binning(self, sample_id: str, width_factor: float) -> hexbin.HexBinning:
        """Hexagonal bins `width_factor` cell spacings wide over one sample's cells
        (shared by the Spatial view and hexagon-level Leiden clustering)."""
        key = (sample_id, width_factor)
        if key not in self._binning_cache:
            width = width_factor * self.sample_lattice(sample_id).spacing
            self._binning_cache[key] = hexbin.make_hex_binning(self.sample_coords(sample_id), width)
        return self._binning_cache[key]

    # -- gene expression -------------------------------------------------------
    def match_gene(self, gene: str) -> str | None:
        """Resolves a gene symbol to this dataset's actual var_names spelling,
        matching case-insensitively so mouse (Col1a1) and human (COL1A1) gene
        symbol conventions both work regardless of which one a gene list (e.g.
        pathways.json) happens to be written in."""
        if gene in self.adata.var_names:
            return gene
        if self._var_upper_lookup is None:
            self._var_upper_lookup = {v.upper(): v for v in self.adata.var_names}
        return self._var_upper_lookup.get(gene.upper())

    def top_variable_genes(self, n: int) -> list[str]:
        """The `n` most highly variable genes, most variable (normalized dispersion) first,
        leaving out background genes. Computed with inplace=False so it never touches
        var['highly_variable'], which PCA/clustering use with their own (larger) gene count."""
        if self._variable_ranking is None:
            hvg = sc.pp.highly_variable_genes(self.adata, flavor="seurat", inplace=False)
            dispersion = hvg["dispersions_norm"].to_numpy(dtype=float)
            ranked = np.argsort(-np.nan_to_num(dispersion, nan=-np.inf), kind="stable")
            self._variable_ranking = ranked[np.isfinite(dispersion[ranked])]
        ranked = self._variable_ranking[~self.background_mask()[self._variable_ranking]]
        return [str(self.adata.var_names[i]) for i in ranked[:n]]

    def gene_values(self, gene: str) -> np.ndarray | None:
        # Index the raw matrix directly (bypassing AnnData's __getitem__, which
        # rebuilds a whole view/obs/var subset just to read one column) - this is
        # what makes repeated single-gene lookups fast on large datasets.
        matched = self.match_gene(gene)
        if matched is None:
            return None
        if matched not in self._gene_cache:
            gene_idx = self.adata.var_names.get_loc(matched)
            col = self.adata.X[:, gene_idx]
            arr = np.asarray(col.todense()).ravel() if sp.issparse(col) else np.asarray(col).ravel()
            self._gene_cache[matched] = arr
        return self._gene_cache[matched]


    def background_mask(self) -> np.ndarray:
        """Boolean mask over var_names of the background genes self.gene_filter leaves
        out (detection rates measured over all cells)."""
        if self.gene_filter not in self._background_masks:
            self._background_masks[self.gene_filter] = clustering.background_genes(self.adata, self.gene_filter)
        return self._background_masks[self.gene_filter]

    def top_genes(self, rows: np.ndarray | None = None, binning=None, k: int = 5) -> np.ndarray:
        """(n, k) var indices of the k most expressed genes of each cell in `rows` (all
        cells if None) - or, given the binning of those cells, of each bin (its cells'
        mean) - leaving out background genes. -1 where fewer than k genes are expressed.
        Recent results are cached, so recoloring doesn't recompute them."""
        key = (k, None if rows is None else hash(rows.tobytes()), None if binning is None else id(binning),
               self.gene_filter)
        if key not in self._top_genes_cache:
            X = self.adata.X if rows is None else self.adata.X[rows]
            background = self.background_mask()
            if background.any():  # zero those genes' columns
                X = X @ sp.diags((~background).astype(np.float32))
            if binning is not None:
                X = binning.mean_matrix @ X
            if len(self._top_genes_cache) >= _TOP_GENES_CACHE_SIZE:
                self._top_genes_cache.pop(next(iter(self._top_genes_cache)))
            self._top_genes_cache[key] = top_k_per_row(X, k)
        return self._top_genes_cache[key]


class AppState:
    def __init__(self):
        self.spatial: Dataset | None = None
        self.embedding: Dataset | None = None
        self.pathways: dict = {}
        self.pathways_path: str = ""
        # Called with a pathway name whenever the Pathways tab adds/changes/removes one.
        self.pathway_listeners: list = []
        # Called with (dataset, field name) whenever an obs column is added to a
        # dataset from inside the viewer (e.g. re-running Leiden clustering).
        self.field_listeners: list = []
        # Full-page "working..." overlay for long steps (viewer.busy_overlay.BusyOverlay).
        self.busy = None
        # Cell-typing settings (celltype_controls.CellTypingControls), shared by the
        # landing page and the Spatial tab so choices carry over.
        self.celltyping = None
        # Background genes left out of PCA / UMAP / Leiden and cell typing
        # (gene_filter_controls.GeneFilterControls), shared like the cell-typing settings.
        self.gene_filter = None
        # Called with each new celltyping.CellTypingResult (e.g. to refresh the Cell types tab).
        self.celltype_listeners: list = []
        # async (cluster_col) -> None: cell-types a clustering of the spatial data and
        # shows the result (set by the Spatial tab, used by the Cell types tab).
        self.run_celltyping = None
