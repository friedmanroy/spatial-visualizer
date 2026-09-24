"""Shared application state: loaded AnnData(s), FOV/sample column, and per-dataset caches.

Mirrors the TNBC viewer's use of module-level state for widget callbacks, but generic:
no column name is ever assumed to exist - everything is looked up through `fields.py`.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import fields as fields_mod


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
        self._field_cache: dict[str, fields_mod.FieldSpec] = {}
        self._gene_cache: dict[str, np.ndarray] = {}
        self._var_upper_lookup: dict[str, str] | None = None
        self.pathway_score_cache: dict[str, np.ndarray | None] = {}
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

    def field_options(self, exclude: tuple[str, ...] = ()) -> list[str]:
        options = fields_mod.list_field_options(self.adata, exclude=exclude)
        if self.allowed_fields is not None:
            options = [o for o in options if o in self.allowed_fields]
        return options

    # -- sample / FOV splitting -----------------------------------------------
    def set_sample_column(self, colname: str | None):
        self.sample_col = colname
        self._sample_indices = {}
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
