"""Shared application state: loaded AnnData(s), FOV/sample column, and per-dataset caches.

Mirrors the TNBC viewer's use of module-level state for widget callbacks, but generic:
no column name is ever assumed to exist - everything is looked up through `fields.py`.
"""
from __future__ import annotations

import threading

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata.io import read_elem

from . import clustering
from . import fields as fields_mod
from . import hexbin
from . import seurat
from . import spatial_plot

#: how many recent top-genes results Dataset.top_genes keeps
_TOP_GENES_CACHE_SIZE = 8
#: hexagonal binnings of at most this many bin widths are kept (each holds per-cell
#: arrays for every sample it was made for); the least recently used width goes first
_BINNING_WIDTHS_KEPT = 3
#: top_k_per_row works on chunks of at most about this many values (rows x padded width)
_TOP_K_BUDGET = 16_000_000
#: the most variable genes of spots kept as raw counts are ranked on this many random spots
_HVG_SAMPLE = 100_000
#: the obsm entries the viewer reads; loading leaves the others out of memory
_OBSM_USED = ("spatial",)


def read_viewer_file(path: str):
    """The AnnData the viewer works on, from an .h5ad file or a Seurat object saved as .rds."""
    return seurat.read_seurat_rds(path) if seurat.is_rds(path) else read_viewer_h5ad(path)


def read_viewer_h5ad(path: str):
    """An AnnData of the parts of an .h5ad the viewer uses: X, obs, var, uns and _OBSM_USED.
    Nothing here reads layers, raw, obsp, varp, varm or other obsm entries, and on huge files
    those can outweigh X itself. Files in the old layout, not readable piece by piece, are read whole."""
    with h5py.File(path, "r") as f:
        if not all(isinstance(f.get(k), h5py.Group) and f[k].attrs.get("encoding-type") == "dataframe"
                   for k in ("obs", "var")) or not isinstance(f.get("obsm", f), h5py.Group):
            return sc.read_h5ad(path)
        # X last: reading spot names briefly takes far more memory than holding them does.
        obs, var = read_elem(f["obs"]), read_elem(f["var"])
        obsm = {k: read_elem(f["obsm"][k]) for k in _OBSM_USED if k in f.get("obsm", ())}
        uns = read_elem(f["uns"]) if "uns" in f else {}
        X = read_elem(f["X"]) if "X" in f else None
    return ad.AnnData(X=X, obs=obs, var=var, obsm=obsm, uns=uns)


def top_k_per_row(X, k: int, exclude: np.ndarray | None = None) -> np.ndarray:
    """(n_rows, k) column indices of each row's k largest values, largest first; -1
    where a row has fewer than k positive values. Columns where the boolean `exclude`
    is True are never picked. Sparse rows are padded to the longest row's nonzero count
    rather than densified over all columns, and rows are taken in chunks sized so that
    padding stays within _TOP_K_BUDGET values (summed hexagons are nearly dense)."""
    n = X.shape[0]
    out = np.full((n, k), -1, dtype=np.int32)
    if sp.issparse(X):
        X = sp.csr_matrix(X)
        width = max(int(np.diff(X.indptr).max(initial=0)), k)
    else:
        width = max(X.shape[1], k)
    chunk = max(1, _TOP_K_BUDGET // width)
    for start in range(0, n, chunk):
        block = X[start:start + chunk]
        m = block.shape[0]
        if sp.issparse(block):
            lens = np.diff(block.indptr)
            width = max(int(lens.max(initial=0)), k)
            vals = np.full((m, width), -np.inf, dtype=np.float32)
            cols = np.full((m, width), -1, dtype=np.int32)
            rows = np.repeat(np.arange(m), lens)
            pos = np.arange(block.nnz) - np.repeat(block.indptr[:-1], lens)
            vals[rows, pos] = block.data
            cols[rows, pos] = block.indices
            if exclude is not None:
                vals[(cols >= 0) & exclude[np.maximum(cols, 0)]] = -np.inf
        else:
            vals = np.array(block, dtype=np.float32)  # a copy: excluded columns get overwritten
            if exclude is not None:
                vals[:, exclude] = -np.inf
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

    """

    def __init__(self, adata, name: str = "dataset"):
        self.adata = adata
        self.name = name
        # .X from before the app normalized it (normalize_total + log1p), or None if it
        # didn't: hexagonal bins sum these counts and then normalize (clustering.hex_profiles).
        # For huge samples only the bins are normalized: .X stays raw and this is .X itself.
        self.counts = None
        # Whether the app normalized each cell / spot's .X (not just the hexagons').
        self.normalized_spots = False
        # Log-normalized copy of .X while .X stays raw counts (see expression()).
        self._lognorm = None
        self.sample_col: str | None = None
        self.sample_ids: list[str] = []
        self._sample_indices: dict[str, np.ndarray] = {}
        self.split_lock = threading.RLock()  # hold to change the split, or to read it across several calls
        self._lattice_cache: dict[str, spatial_plot.Lattice] = {}
        self._binning_cache: dict[tuple[str, float], hexbin.HexBinning] = {}
        self._binning_lock = threading.RLock()  # binnings are slow and big: never two at once
        self._field_cache: dict[str, fields_mod.FieldSpec] = {}
        self._gene_cache: dict[str, np.ndarray] = {}
        self._count_cache: dict[str, np.ndarray] = {}  # genes' columns of self.counts
        self._cell_totals: np.ndarray | None = None  # each cell's total of self.counts
        # Which hexagons are analysed and whether they're normalized (hexbin.HexFilter);
        # its `normalize` is the load page's choice for cells as well as hexagons.
        self.hex_filter = hexbin.HexFilter()
        # Hexagon width (cell spacings) Leiden and cell typing run on, chosen on the load
        # page for samples too large to analyse spot by spot; None = on the cells themselves.
        self.analysis_width: float | None = None
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
        # obs column -> the hexbin.HexFilter its hexagons were clustered with.
        self.hex_cluster_filters: dict[str, hexbin.HexFilter] = {}
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

    def remove_field(self, colname: str):
        """Drops an obs column made inside the viewer (e.g. cell types of a clustering
        that was just recomputed), along with everything derived from it."""
        if colname in self.adata.obs.columns:
            del self.adata.obs[colname]
        self._field_cache.pop(colname, None)
        self.celltype_results.pop(colname, None)
        if self.allowed_fields is not None:
            self.allowed_fields.discard(colname)

    def field_options(self, exclude: tuple[str, ...] = ()) -> list[str]:
        """obs columns offered for 'color by', alphabetically: those (among
        allowed_fields, if set) that aren't unusable. Uses the cached classification."""
        return sorted(
            c for c in self.adata.obs.columns
            if c not in exclude and (self.allowed_fields is None or c in self.allowed_fields)
            and self.classify(c).kind != "unusable"
        )

    # -- sample / FOV splitting -----------------------------------------------
    def set_sample_column(self, colname: str | None):
        with self.split_lock:
            if self.sample_ids and colname == self.sample_col:
                return  # already split this way: keep the measured lattices and binnings
            # Built aside and swapped in together: other threads may be reading the split.
            if colname is None:
                sample_ids = ["all cells"]
                indices = {"all cells": np.arange(self.adata.n_obs)}
            else:
                col = self.adata.obs[colname]
                values = col.cat.categories if hasattr(col, "cat") else sorted(col.dropna().unique())
                sample_ids = [str(v) for v in values]
                # Grouping codes rather than comparing every spot's label with every sample's.
                codes = col.cat.codes.to_numpy() if hasattr(col, "cat") else pd.Categorical(col, categories=values).codes
                by_sample = np.argsort(codes, kind="stable")  # stable: each sample's spots stay in order
                ends = np.cumsum(np.bincount(codes[codes >= 0], minlength=len(values)))
                groups = np.split(by_sample[len(codes) - ends[-1]:] if len(ends) else by_sample[:0], ends[:-1])
                indices = dict(zip(sample_ids, groups))
            self._lattice_cache = {}
            self._binning_cache = {}
            self.sample_col, self.sample_ids, self._sample_indices = colname, sample_ids, indices

    def sample_indices(self, sample_id: str) -> np.ndarray:
        return self._sample_indices[sample_id]

    def sample_coords(self, sample_id: str) -> np.ndarray:
        return self.adata.obsm["spatial"][self.sample_indices(sample_id)]

    def largest_sample_size(self) -> int:
        return max(len(idx) for idx in self._sample_indices.values())

    def largest_sample(self) -> str:
        """The sample with the most cells."""
        return max(self._sample_indices, key=lambda sid: len(self._sample_indices[sid]))

    def sample_extent(self, sample_id: str) -> tuple[float, float]:
        """(width, height) the sample's cells span, in coordinate units."""
        coords = self.sample_coords(sample_id)
        if len(coords) == 0:
            return 0.0, 0.0
        return float(np.ptp(coords[:, 0])), float(np.ptp(coords[:, 1]))

    def default_hex_factor(self) -> float:
        """The automatic hexagon width, in cell spacings, from the largest sample."""
        largest = self.largest_sample()
        return hexbin.default_width_factor(self.sample_extent(largest), self.sample_lattice(largest).spacing)

    def sample_lattice(self, sample_id: str) -> spatial_plot.Lattice:
        if sample_id not in self._lattice_cache:
            self._lattice_cache[sample_id] = spatial_plot.estimate_lattice(self.sample_coords(sample_id))
        return self._lattice_cache[sample_id]

    def hex_binning(self, sample_id: str, width_factor: float) -> hexbin.HexBinning:
        """Hexagonal bins `width_factor` cell spacings wide over one sample's cells
        (shared by the Spatial view and hexagon-level Leiden clustering)."""
        key = (sample_id, width_factor)
        with self._binning_lock:
            if key in self._binning_cache:
                self._binning_cache[key] = self._binning_cache.pop(key)  # most recently used last
            else:
                widths = list(dict.fromkeys(w for _, w in self._binning_cache))  # least recent first
                if width_factor not in widths and len(widths) >= _BINNING_WIDTHS_KEPT:
                    for old in [k for k in self._binning_cache if k[1] == widths[0]]:
                        del self._binning_cache[old]
                width = width_factor * self.sample_lattice(sample_id).spacing
                self._binning_cache[key] = hexbin.make_hex_binning(self.sample_coords(sample_id), width)
            return self._binning_cache[key]

    def has_hex_binning(self, sample_id: str, width_factor: float) -> bool:
        """Whether hex_binning would answer right away."""
        return (sample_id, width_factor) in self._binning_cache

    def cell_totals(self, counts=None) -> np.ndarray | None:
        """Each cell's total of self.counts - or, while the app hasn't kept any yet, of
        `counts`, the matrix they will be kept from (None if neither)."""
        counts = self.counts if self.counts is not None else counts
        if counts is None:
            return None
        if self._cell_totals is None:
            self._cell_totals = np.asarray(counts.sum(axis=1)).ravel()
        return self._cell_totals

    def kept_bins(self, idx: np.ndarray, binning, hex_filter: hexbin.HexFilter | None = None) -> np.ndarray:
        """Boolean mask over `binning`'s bins (of the sample's cells `idx`): those `hex_filter`
        (by default self.hex_filter) analyses."""
        totals = self.cell_totals()
        return (hex_filter or self.hex_filter).keep(binning, None if totals is None else binning.sum(totals[idx]))

    def kept_binning(self, sample_id: str, width_factor: float, hex_filter: hexbin.HexFilter | None = None):
        """(the sample's cells in analysed bins, a binning of only those bins and cells)."""
        idx = self.sample_indices(sample_id)
        binning = self.hex_binning(sample_id, width_factor)
        cells, kept = binning.subset(self.kept_bins(idx, binning, hex_filter))
        return idx[cells], kept

    # -- gene expression -------------------------------------------------------
    @property
    def raw_spots(self) -> bool:
        """.X still holds raw counts because the app normalized only the hexagons
        (samples too large to normalize spot by spot)."""
        return self.counts is not None and not self.normalized_spots

    def expression(self):
        """An AnnData with this dataset's genes whose .X is log-normalized expression:
        the dataset's own AnnData, or - while .X stays raw counts (raw_spots) - a
        normalize_total (if hex_filter.normalize) + log1p copy of it, made the first time it's needed and kept
        (it doubles the memory .X takes). For per-spot analyses: Leiden on cells,
        pathway scores, differential expression of spots."""
        if not self.raw_spots:
            return self.adata
        if self._lognorm is None:
            self._lognorm = self.log_counts(self.adata.X.copy())
        return self._lognorm

    def log_counts(self, X):
        """An AnnData of raw counts `X` (rows of this dataset's cells; modified in place)
        made the way the cells were: normalize_total if chosen, then log1p."""
        return clustering.log_normalized(X, self.adata, self.hex_filter.normalize)

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
            adata = self.adata
            if self.raw_spots:  # dispersions need log-normalized data; a sample of the spots is plenty
                if self._lognorm is not None:
                    adata = self._lognorm
                else:
                    n = adata.n_obs
                    rows = np.sort(np.random.default_rng(0).choice(n, min(n, _HVG_SAMPLE), replace=False))
                    adata = self.log_counts(adata.X[rows])
            hvg = sc.pp.highly_variable_genes(adata, flavor="seurat", inplace=False)
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

    def bin_gene_values(self, gene: str, idx: np.ndarray, binning, mask: np.ndarray | None = None):
        """`gene`'s value in each bin of `binning` (over the sample's cells `idx`), made
        the way clustering.hex_profiles makes hexagons: with the pre-normalization counts
        kept, the bin's summed counts of the gene, over its summed total counts and scaled
        to the median total of the sample's analysed bins if hex_filter.normalize, log1p'd
        (bins the filter leaves out are NaN); otherwise its cells' mean. With a boolean
        `mask` (per cell of `idx`), only those cells count, and bins without any are NaN."""
        values = self.gene_values(gene)
        if values is None:
            return None
        if self.counts is None:
            return binning.mean(values[idx], mask)
        matched = self.match_gene(gene)
        if matched not in self._count_cache:
            col = self.counts[:, self.adata.var_names.get_loc(matched)]
            self._count_cache[matched] = (np.asarray(col.todense()).ravel() if sp.issparse(col)
                                          else np.asarray(col).ravel())
        keep = self.kept_bins(idx, binning)
        gene_sums = binning.sum(self._count_cache[matched][idx], mask)
        if self.hex_filter.normalize:
            totals = self.cell_totals()[idx]
            bin_totals = binning.sum(totals, mask)
            kept_totals = (binning.sum(totals) if mask is not None else bin_totals)[keep]
            target = float(np.median(kept_totals[kept_totals > 0])) if (kept_totals > 0).any() else 1.0
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.where(bin_totals > 0, np.log1p(gene_sums / bin_totals * target), 0.0)
        else:
            out = np.log1p(gene_sums)
        out[~keep] = np.nan
        if mask is not None:
            out[binning.sum(np.ones(len(idx)), mask) == 0] = np.nan
        return out


    def background_mask(self) -> np.ndarray:
        """Boolean mask over var_names of the background genes self.gene_filter leaves
        out (detection rates measured over all cells)."""
        if self.gene_filter not in self._background_masks:
            self._background_masks[self.gene_filter] = clustering.background_genes(self.adata, self.gene_filter)
        return self._background_masks[self.gene_filter]

    def top_genes(self, rows: np.ndarray | None = None, binning=None, k: int = 5) -> np.ndarray:
        """(n, k) var indices of the k most expressed genes of each cell in `rows` (all
        cells if None) - or, given the binning of those cells, of each bin (its cells'
        summed counts if kept, else their mean) - leaving out background genes. -1 where fewer than k genes are expressed.
        Recent results are cached, so recoloring doesn't recompute them."""
        key = (k, None if rows is None else hash(rows.tobytes()), None if binning is None else id(binning),
               self.gene_filter)
        # The binning is kept with the result: binnings can be evicted (hex_binning), and
        # a new one may then reuse the old one's id.
        cached = self._top_genes_cache.get(key)
        if cached is None or cached[0] is not binning:
            # Normalizing a bin's summed counts doesn't change their order, so ranking
            # the sums ranks the genes as the normalized bin would.
            source = self.counts if binning is not None and self.counts is not None else self.adata.X
            X = source if rows is None else source[rows]
            if binning is not None:
                X = (binning.sum_matrix if self.counts is not None else binning.mean_matrix) @ X
            self._top_genes_cache.pop(key, None)
            if len(self._top_genes_cache) >= _TOP_GENES_CACHE_SIZE:
                self._top_genes_cache.pop(next(iter(self._top_genes_cache)))
            cached = self._top_genes_cache[key] = (binning, top_k_per_row(X, k, exclude=self.background_mask()))
        return cached[1]


class AppState:
    def __init__(self):
        self.spatial: Dataset | None = None
        self.pathways: dict = {}
        self.pathways_path: str = ""
        # Called with a pathway name whenever the Pathways tab adds/changes/removes one.
        self.pathway_listeners: list = []
        # Called with (dataset, field name) whenever an obs column is added to a
        # dataset from inside the viewer (e.g. re-running Leiden clustering).
        self.field_listeners: list = []
        # Full-page "working..." overlay for long steps (viewer.busy_overlay.BusyOverlay).
        self.busy = None
        # Pop-up with differential-expression results between regions (de_dialog.DEDialog).
        self.de_dialog = None
        # Cell-typing settings (celltype_controls.CellTypingControls), shared by the
        # landing page and the Spatial tab so choices carry over.
        self.celltyping = None
        # Background genes left out of PCA / Leiden and cell typing
        # (gene_filter_controls.GeneFilterControls), shared like the cell-typing settings.
        self.gene_filter = None
        # Which hexagons are analysed and whether they're normalized
        # (hex_filter_controls.HexFilterControls), shared like the background genes.
        self.hex_filter = None
        # Called with each new celltyping.CellTypingResult (e.g. to refresh the Cell types tab).
        self.celltype_listeners: list = []
        # async (cluster_col) -> None: cell-types a clustering of the spatial data and
        # shows the result (set by the Spatial tab, used by the Cell types tab).
        self.run_celltyping = None
