"""Shared application state: loaded AnnData(s), FOV/sample column, and per-dataset caches.

Mirrors the TNBC viewer's use of module-level state for widget callbacks, but generic:
no column name is ever assumed to exist - everything is looked up through `fields.py`.
"""
from __future__ import annotations

import numpy as np
import scanpy as sc
import scipy.sparse as sp

from . import fields as fields_mod
from . import hexbin
from . import spatial_plot


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
        self._top_variable_genes: dict[int, list[str]] = {}
        self.pathway_score_cache: dict[str, np.ndarray | None] = {}
        # obs column -> celltyping.CellTypingResult, for showing per-cluster suggestions.
        self.celltype_results: dict = {}
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
        """The `n` most highly variable genes, most variable (normalized dispersion) first.
        Computed with inplace=False so it never touches var['highly_variable'], which
        PCA/clustering use with their own (larger) gene count."""
        if n not in self._top_variable_genes:
            hvg = sc.pp.highly_variable_genes(self.adata, flavor="seurat", inplace=False)
            hvg.index = self.adata.var_names
            ranked = hvg["dispersions_norm"].dropna().sort_values(ascending=False, kind="stable")
            self._top_variable_genes[n] = [str(g) for g in ranked.index[:n]]
        return self._top_variable_genes[n]

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
        # Called with each new celltyping.CellTypingResult (e.g. to refresh the Cell types tab).
        self.celltype_listeners: list = []
        # async (cluster_col) -> None: cell-types a clustering of the spatial data and
        # shows the result (set by the Spatial tab, used by the Cell types tab).
        self.run_celltyping = None
