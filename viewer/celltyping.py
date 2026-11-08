"""Rudimentary marker-based cell typing of clusters (e.g. Leiden), from a marker database
(PanglaoDB by default - see DATABASES) plus the user's own marker sets. Everything runs
on existing implementations (decoupler, scanpy), on single cells - or, for a clustering of
hexagonal bins (sub-cellular spots), on its hexagons - over all measured genes (minus any
background genes the user left out). One of four methods:

- AUCell (decoupler `dc.mt.aucell`): per cell, how near the top of the cell's own gene
  ranking a type's markers are.
- Module score (scanpy `sc.tl.score_genes`): per cell, the markers' mean expression minus
  that of randomly drawn control genes of similar expression.
- ULM (decoupler `dc.mt.ulm`, as in decoupler's PanglaoDB tutorial): per cell, a linear
  model of all genes' expression on "is a marker of the type" (negative markers count -1).
- Top-gene overlap (scanpy `sc.tl.rank_genes_groups` + decoupler `dc.mt.ora`): each
  cluster's own top genes (Wilcoxon, the cluster vs the rest, at most ORA_TOP_GENES with
  adjusted p < MAX_PVALUE and log2 FC >= ORA_MIN_LFC), then Fisher's exact test of which
  type's markers are over-represented among them (all scored genes as background).

For the per-cell methods, each cluster is then labelled as in decoupler's tutorial: decoupler's
`dc.tl.rankby_group` tests every type's scores in the cluster's cells against all other
cells (t-test), and the type with the highest positive statistic is suggested - or
"Unknown" when none is significant (adjusted p < MAX_PVALUE). For the top-gene overlap the
most over-represented significant type is suggested, else "Unknown".

Negative markers come only from the user's own marker sets (the databases have none);
only ULM can use them (as weight -1) - the other methods leave them out.

Each cell also gets a confidence (0-1) in its cluster's suggested type: the percentile of
its score for that type among all cells (ULM's score for the top-gene overlap, which has
no per-cell score) - near 1 when the cell expresses the type's markers more strongly than
almost any other cell. For a hexagon clustering it's computed per hexagon and every spot
gets its hexagon's. These are suggestions to check against known markers, not a
validated annotation.

Each database is downloaded through decoupler/OmniPath the first time it's chosen and
kept in cache/ for offline use. Databases OmniPath doesn't serve (CellMarker 2.0, TISCH2's
per-dataset marker tables, ...) can be imported from a file the user downloaded
(import_marker_file); they're kept, already tidied, in cache/marker_databases/.
Every database is turned into the same (cell_type, gene, tissue, species) table.
Marker sets the user adds on the Cell types tab are kept in cache/custom_markers.json; a
set named like one of the database's types replaces that type's markers.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import cache, clustering

ALL_TISSUES = "All tissues"
#: cell types found in almost any tissue - offered alongside a chosen tissue's own types
COMMON_ORGANS = ("Immune system", "Blood", "Vasculature", "Connective tissue")
UNKNOWN = "Unknown"
#: a cell type needs at least this many of its markers measured to be scored at all -
#: low, since targeted spatial panels often hold only a handful of any type's markers
MIN_MARKERS = 3
#: PanglaoDB markers detected in no more than this fraction of that cell type's cells are
#: left out (the threshold decoupler's own cell-type tutorial uses)
MIN_SENSITIVITY = 0.5
#: a suggestion needs an adjusted p-value below this (Benjamini-Hochberg, as decoupler /
#: scanpy report them), else the cluster is "Unknown"
MAX_PVALUE = 0.05
#: top-gene overlap: a cluster's top genes are at most this many genes significantly
#: higher than in the other clusters (adjusted p < MAX_PVALUE, log2 FC >= ORA_MIN_LFC)...
ORA_TOP_GENES = 100
ORA_MIN_LFC = 0.5
#: ...found on at most this many randomly chosen cells (or hexagons) per cluster
ORA_MAX_UNITS = 1000
#: how many candidates per cluster are kept for display
N_CANDIDATES = 3
#: cells (or hexagons) decoupler scores at once - bounds memory on million-cell data
BATCH_SIZE = 50_000

#: how cell types are scored (label shown in the UI -> key)
AUCELL, SCORE_GENES, ULM, ORA = "aucell", "score_genes", "ulm", "ora"
METHODS = {
    "AUCell (decoupler)": AUCELL,
    "Module score (scanpy score_genes)": SCORE_GENES,
    "ULM (decoupler's PanglaoDB tutorial)": ULM,
    "Top-gene overlap (cluster top genes vs marker sets, ORA)": ORA,
}
DEFAULT_METHOD = ULM

CUSTOM_MARKERS_PATH = os.path.join(cache.CACHE_DIR, "custom_markers.json")
#: imported marker databases: one tidied table each, plus index.json naming them
IMPORTED_DIR = os.path.join(cache.CACHE_DIR, "marker_databases")
_IMPORTED_INDEX = os.path.join(IMPORTED_DIR, "index.json")
#: how many of a cluster's strongest markers for its suggested type are listed
N_DRIVERS = 4

#: which marker sets cell typing uses (label shown in the UI -> key)
MARKER_SOURCES = {
    "Database + my marker sets": "both",
    "Database only": "database",
    "My marker sets only": "custom",
}
#: where a marker set comes from
ORIGIN_DATABASE, ORIGIN_EDITED, ORIGIN_ADDED = "database", "edited", "added"


@dataclass(frozen=True)
class MarkerDatabase:
    key: str
    name: str  # shown in the UI
    about: str  # one line on what it holds, shown under the dropdown
    resource: str | None = None  # OmniPath resource (through decoupler); None = imported from a file
    common_tissues: tuple[str, ...] = ()  # offered alongside a chosen tissue's own types
    #: "cell type" or "cell state" (CancerSEA labels clusters by functional state)
    kind: str = "cell type"


#: the databases decoupler downloads (key -> MarkerDatabase); imported ones are added at runtime
DATABASES = {
    "panglaodb": MarkerDatabase(
        "panglaodb", "PanglaoDB",
        "Canonical markers of ~180 cell types across most tissues, from scRNA-seq studies.",
        "PanglaoDB", COMMON_ORGANS),
    "celltypist": MarkerDatabase(
        "celltypist", "CellTypist (immune)",
        "Markers of ~100 immune (and some stromal) subtypes - finer T / B / myeloid cells, "
        "useful for a tumor's immune infiltrate.",
        "CellTypist"),
    "cancersea": MarkerDatabase(
        "cancersea", "CancerSEA (cancer cell states)",
        "Genes of 14 functional states of cancer cells (hypoxia, EMT, stemness, invasion, ...) - "
        "labels clusters by state rather than by cell type.",
        "CancerSEA", kind="cell state"),
}
DEFAULT_DATABASE = "panglaodb"
#: an imported table with per-gene scores (TISCH2 / Seurat / scanpy marker tables) keeps
#: each type's IMPORT_TOP_GENES highest-scoring genes with adjusted p < MAX_PVALUE
IMPORT_TOP_GENES = 50
#: CellMarker's marker sources; markers found only by single-cell sequencing are left out
#: (the rest are curated from experiments, reviews and antibody vendors)
_UNCURATED_SOURCES = {"single-cell sequencing"}


def _cache_path(db: MarkerDatabase) -> str:
    if db.resource is None:
        return os.path.join(IMPORTED_DIR, f"{db.key}.csv.gz")
    # PanglaoDB keeps the name it was cached under before other databases were offered
    return os.path.join(cache.CACHE_DIR, f"{db.key}_markers.csv.gz")


def all_databases() -> dict[str, MarkerDatabase]:
    """The built-in databases followed by the imported ones."""
    try:
        with open(_IMPORTED_INDEX) as f:
            index = json.load(f)
    except (OSError, json.JSONDecodeError):
        index = {}
    imported = {key: MarkerDatabase(key, info["name"], info.get("about", "")) for key, info in index.items()}
    return {**DATABASES, **imported}


def markers_available_offline(db: MarkerDatabase) -> bool:
    return os.path.exists(_cache_path(db))


def load_markers(db: MarkerDatabase) -> pd.DataFrame:
    """`db`'s markers as a (cell_type, gene, tissue, species) table - tissue may be
    missing, species is "human", "mouse" or "any" - from cache/ if present, else
    downloaded through decoupler/OmniPath and cached (the download as it came)."""
    path = _cache_path(db)
    if db.resource is None:
        return pd.read_csv(path)
    if os.path.exists(path):
        raw = pd.read_csv(path)
    else:
        import decoupler as dc  # imported lazily: it's slow to import and only needed here

        try:
            raw = dc.op.resource(db.resource, organism="human")
        except Exception as exc:
            raise RuntimeError(
                f"Couldn't download the {db.name} marker genes (needs an internet connection "
                "the first time; after that they're kept in cache/)."
            ) from exc
        _write_table(raw, path)
    return _TIDY[db.key](raw)


def _write_table(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"  # never leave a truncated table behind
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def _tidy_panglao(raw: pd.DataFrame) -> pd.DataFrame:
    """Canonical markers, per species, detected in more than MIN_SENSITIVITY of that
    type's cells."""
    parts = []
    for species in ("human", "mouse"):
        keep = (
            _as_bool(raw[species])
            & _as_bool(raw["canonical_marker"])
            & (pd.to_numeric(raw[f"{species}_sensitivity"], errors="coerce") > MIN_SENSITIVITY)
        )
        parts.append(pd.DataFrame({"cell_type": raw.loc[keep, "cell_type"], "gene": raw.loc[keep, "genesymbol"],
                                   "tissue": raw.loc[keep, "organ"], "species": species}))
    return pd.concat(parts, ignore_index=True)


def _tidy_celltypist(raw: pd.DataFrame) -> pd.DataFrame:
    """Subtypes (CD8 T cells, not just T cells), once per tissue they were seen in."""
    df = pd.DataFrame({"cell_type": raw["cell_subtype"], "gene": raw["genesymbol"],
                       "tissue": raw["tissues"].astype(str).str.split(";"), "species": "any"})
    return df.explode("tissue").drop_duplicates(ignore_index=True)


def _tidy_cancersea(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"cell_type": raw["state"], "gene": raw["genesymbol"], "tissue": None, "species": "any"})


_TIDY = {"panglaodb": _tidy_panglao, "celltypist": _tidy_celltypist, "cancersea": _tidy_cancersea}


def tissue_options(markers: pd.DataFrame) -> list[str]:
    return [ALL_TISSUES] + sorted(markers["tissue"].dropna().astype(str).unique())


# ---- importing a database from a file ---------------------------------------------------

#: column names recognised in an imported table (lower-cased), most preferred first
_TYPE_COLUMNS = ("cell_name", "celltype (major-lineage)", "celltype (minor-lineage)", "celltype",
                 "cell_type", "cell type", "cell_type_name", "cluster", "group", "state")
_GENE_COLUMNS = ("symbol", "gene", "genes", "gene_symbol", "genesymbol", "gene symbol", "gene_name",
                 "marker", "markers", "names", "feature")
_TISSUE_COLUMNS = ("tissue_class", "tissue", "organ", "tissue_type")
_SCORE_COLUMNS = ("log2fc", "avg_log2fc", "avg_logfc", "logfc", "logfoldchanges", "log2foldchange",
                  "log2 fold change", "scores")
_PADJ_COLUMNS = ("adjusted p-value", "adj. p-value", "p_val_adj", "padj", "pvals_adj", "fdr", "adj.p.val",
                 "q_value", "qvalue")


def _find_column(columns: dict[str, str], candidates) -> str | None:
    """The original name of the first of `candidates` among `columns` (lower-cased -> original)."""
    return next((columns[c] for c in candidates if c in columns), None)


def _read_table(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if lower.endswith(".csv") or lower.endswith(".csv.gz"):
        return pd.read_csv(path)
    return pd.read_csv(path, sep=None, engine="python")  # .tsv, .txt: let pandas sniff the separator


def import_marker_file(path: str, name: str) -> tuple[MarkerDatabase, str]:
    """Reads a marker table downloaded by the user - CellMarker 2.0's Cell_marker_*.xlsx,
    a TISCH2 / Seurat / scanpy marker-gene table, or any table with a cell-type and a
    gene column - keeps it (tidied) as a database named `name`, and returns it with a
    one-line summary of what was kept."""
    raw = _read_table(path)
    columns = {str(c).strip().lower(): c for c in raw.columns}
    # Also accept any "Celltype (...)" column TISCH2 might name differently.
    type_col = _find_column(columns, _TYPE_COLUMNS) or next(
        (orig for low, orig in columns.items() if low.startswith("celltype")), None)
    gene_col = _find_column(columns, _GENE_COLUMNS)
    if type_col is None or gene_col is None:
        raise ValueError(
            "Couldn't tell which columns hold the cell types and the genes (found: "
            + ", ".join(str(c) for c in raw.columns)
            + "). Name them e.g. 'cell_type' and 'gene'."
        )
    notes = []
    df = raw
    source_col = columns.get("marker_source")
    if source_col is not None:  # CellMarker 2.0: keep the curated markers
        curated = ~df[source_col].astype(str).str.strip().str.lower().isin(_UNCURATED_SOURCES)
        if curated.any() and not curated.all():
            notes.append(f"left out {int((~curated).sum()):,} markers found only by single-cell sequencing")
            df = df[curated]
    score_col = _find_column(columns, _SCORE_COLUMNS)
    if score_col is not None:  # a differential-expression table: each type's top genes
        df = df[pd.to_numeric(df[score_col], errors="coerce") > 0]
        padj_col = _find_column(columns, _PADJ_COLUMNS)
        if padj_col is not None:
            df = df[pd.to_numeric(df[padj_col], errors="coerce") < MAX_PVALUE]
        df = (df.assign(_score=pd.to_numeric(df[score_col], errors="coerce"))
              .sort_values("_score", ascending=False).groupby(type_col, sort=False).head(IMPORT_TOP_GENES))
        notes.append(f"kept each type's top {IMPORT_TOP_GENES} genes by {score_col}"
                     + (f" with {padj_col} < {MAX_PVALUE:g}" if padj_col is not None else ""))
    df = df.dropna(subset=[type_col, gene_col])
    tissue_col = _find_column(columns, _TISSUE_COLUMNS)
    species_col = columns.get("species")
    tidy = pd.DataFrame({
        "cell_type": df[type_col].astype(str).str.strip(),
        # some tables list several genes in one cell ("CD3D, CD3E")
        "gene": df[gene_col].astype(str).str.split(r"[,;/\s]+"),
        "tissue": df[tissue_col].astype(str).str.strip() if tissue_col is not None else None,
        "species": (df[species_col].astype(str).str.strip().str.lower()
                    .where(lambda s: s.isin(("human", "mouse")), "any") if species_col is not None else "any"),
    }).explode("gene")
    tidy["gene"] = tidy["gene"].str.strip("[]()'\" ")
    tidy = tidy[(tidy["gene"] != "") & (tidy["gene"].str.lower() != "nan")
                & (tidy["cell_type"] != "") & (tidy["cell_type"].str.lower() != "nan")]
    tidy = tidy.drop_duplicates(ignore_index=True)
    if tidy.empty:
        raise ValueError("No cell-type markers were left in this file" + (f" ({'; '.join(notes)})" if notes else "") + ".")

    key = "file-" + (re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "markers")
    about = (f"Imported from {os.path.basename(path)}: {tidy['cell_type'].nunique():,} cell types, "
             f"{len(tidy):,} markers" + (f" ({'; '.join(notes)})" if notes else "") + ".")
    db = MarkerDatabase(key, name, about)
    _write_table(tidy, _cache_path(db))
    index = {k: {"name": d.name, "about": d.about} for k, d in all_databases().items() if d.resource is None}
    index[key] = {"name": name, "about": about}
    cache.write_json_atomic(_IMPORTED_INDEX, index)
    return db, about


def remove_imported(key: str):
    index = {k: {"name": d.name, "about": d.about}
             for k, d in all_databases().items() if d.resource is None and k != key}
    cache.write_json_atomic(_IMPORTED_INDEX, index)
    try:
        os.remove(os.path.join(IMPORTED_DIR, f"{key}.csv.gz"))
    except OSError:
        pass


#: how many randomly chosen gene names guess_species checks for mouse casing
SPECIES_SAMPLE = 100


def _mouse_cased(name: str) -> bool:
    """True for a mouse-style symbol (Col1a1, mt-Co1): lowercase letters beyond human
    names' "orf" (C1orf112), and not a probe / control name (NegControlProbe_00042)."""
    if "_" in name or "control" in name.lower() or "blank" in name.lower():
        return False
    return any(ch.islower() for ch in name.replace("orf", ""))


def guess_species(var_names) -> str:
    """'mouse' if any of SPECIES_SAMPLE randomly chosen gene symbols is mouse-cased
    (Col1a1), else 'human' (COL1A1) - files often upper-case their first genes, so
    those alone mislead. Ensembl IDs are recognised by prefix (ENSMUSG = mouse, ENSG = human)."""
    names = [str(v) for v in var_names if any(ch.isalpha() for ch in str(v))]
    if not names:
        return "human"
    rng = np.random.default_rng(0)
    sample = [names[i] for i in rng.choice(len(names), min(SPECIES_SAMPLE, len(names)), replace=False)]
    ensembl_mouse = sum(n.upper().startswith("ENSMUS") for n in sample)
    ensembl_human = sum(n.upper().startswith("ENSG") for n in sample)
    if ensembl_mouse + ensembl_human > len(sample) / 2:
        return "mouse" if ensembl_mouse > ensembl_human else "human"
    return "mouse" if any(_mouse_cased(n) for n in sample) else "human"


def _as_bool(col: pd.Series) -> pd.Series:
    if col.dtype == bool:
        return col
    return col.astype(str).str.lower().isin(("true", "1"))


def marker_sets(markers: pd.DataFrame, species: str, tissue: str, common_tissues=()) -> pd.DataFrame:
    """(source = cell type, target = gene) pairs to score: markers seen in `species` (all
    of them if the database lists none for it - human symbols still match most mouse
    genes case-insensitively), from `tissue` plus `common_tissues`."""
    species_ok = markers["species"].isin((species, "any"))
    keep = species_ok if species_ok.any() else pd.Series(True, index=markers.index)
    if tissue != ALL_TISSUES:
        keep &= markers["tissue"].isin({tissue, *common_tissues})
    net = markers.loc[keep, ["cell_type", "gene"]].drop_duplicates()
    return net.rename(columns={"cell_type": "source", "gene": "target"})


# ---- the user's own marker sets ------------------------------------------------------

def load_custom_markers() -> dict:
    """{"types": {name: {"genes": [...], "negative": [...]}} (the user's sets - one named
    like a database type replaces its markers; "negative" = genes the type should not
    express, optional), "disabled": [names switched off]}."""
    try:
        with open(CUSTOM_MARKERS_PATH) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        data = {}
    return {"types": dict(data.get("types", {})), "disabled": list(data.get("disabled", []))}


def save_custom_markers(data: dict):
    cache.write_json_atomic(CUSTOM_MARKERS_PATH, data)


@dataclass
class MarkerSet:
    name: str
    genes: list[str]
    origin: str  # ORIGIN_DATABASE / ORIGIN_EDITED (the user's set replacing a database one) / ORIGIN_ADDED
    enabled: bool
    database_genes: list[str] | None  # the database's own list, None if not a database type
    negative: list[str] = field(default_factory=list)  # genes the type should not express


def marker_table(markers: pd.DataFrame | None, db: MarkerDatabase, species: str, tissue: str,
                 include_common: bool, custom: dict, source: str) -> list[MarkerSet]:
    """Every marker set cell typing would consider with these settings (switched-off
    ones included, with enabled=False): the database's for the species/tissue, with the
    user's sets replacing same-named ones and the rest appended - or only one of the two."""
    database: dict[str, list[str]] = {}
    if markers is not None and source != "custom":
        net = marker_sets(markers, species, tissue, db.common_tissues if include_common else ())
        database = {t: list(g) for t, g in net.groupby("source", sort=True)["target"]}
    # "Database only" means the database untouched - the user's switched-off types too.
    disabled = set(custom["disabled"]) if source != "database" else set()
    rows = {name: MarkerSet(name, genes, ORIGIN_DATABASE, name not in disabled, genes)
            for name, genes in database.items()}
    if source != "database":
        rows.update({m.name: m for m in custom_sets(custom, database)})
    return sorted(rows.values(), key=lambda m: m.name.lower())


def custom_sets(custom: dict, database: dict[str, list[str]] | None = None) -> list[MarkerSet]:
    """The user's own marker sets; those named like a type in `database` (type -> genes)
    are marked as replacing it."""
    database = database or {}
    disabled = set(custom["disabled"])
    return sorted(
        (MarkerSet(name, list(info.get("genes", [])), ORIGIN_EDITED if name in database else ORIGIN_ADDED,
                   name not in disabled, database.get(name), list(info.get("negative", [])))
         for name, info in custom["types"].items()),
        key=lambda m: m.name.lower(),
    )


def net_from_table(rows: list[MarkerSet]) -> pd.DataFrame:
    """(source, target, weight) rows of the enabled sets: weight 1 for a marker, -1 for
    a negative marker (a gene listed as both counts as a marker)."""
    pairs = ([(m.name, g, 1.0) for m in rows if m.enabled for g in m.genes]
             + [(m.name, g, -1.0) for m in rows if m.enabled for g in m.negative])
    return pd.DataFrame(pairs, columns=["source", "target", "weight"]).drop_duplicates(["source", "target"])


@dataclass
class ClusterCall:
    cluster: str
    n_cells: int
    label: str  # suggested cell type, or UNKNOWN
    candidates: list[tuple[str, float, int]] = field(default_factory=list)  # (type, score, markers measured)
    mean_confidence: float = float("nan")  # mean per-cell confidence in `label` (NaN if Unknown)
    drivers: list[tuple[str, float]] = field(default_factory=list)  # (gene, standardised level) behind `label`
    n_bins: int | None = None  # hexagons in the cluster, for a hexagon clustering


@dataclass
class CellTypingResult:
    cluster_col: str
    column: str  # obs column holding each cell's suggested type
    confidence_column: str  # obs column holding each cell's confidence (0-1) in that type
    species: str
    tissue: str
    n_types_scored: int
    calls: list[ClusterCall]
    hex_width: float | None = None  # bin width (cell spacings) when hexagons were typed
    background: str = ""  # which background genes were left out (GeneFilter.describe)
    method: str = DEFAULT_METHOD  # one of METHODS' values


def _cluster_means(X, codes: np.ndarray, n_clusters: int) -> np.ndarray:
    """Mean row of X (cells or hexagons x genes) per cluster code (-1 = none)."""
    valid = codes >= 0
    rows = np.flatnonzero(valid)
    indicator = sp.csr_matrix(
        (np.ones(len(rows)), (codes[valid], rows)), shape=(n_clusters, X.shape[0]),
    )
    sizes = np.asarray(indicator.sum(axis=1)).ravel()
    mean = sp.diags(1.0 / np.maximum(sizes, 1)) @ indicator @ X
    return np.asarray(mean.todense()) if sp.issparse(mean) else np.asarray(mean)


def annotate_clusters(dataset, cluster_col: str, net: pd.DataFrame, species: str,
                      tissue: str, report=lambda text: None,
                      gene_filter: clustering.GeneFilter | None = None,
                      method: str = DEFAULT_METHOD) -> CellTypingResult:
    """Suggests a cell type per cluster of obs[`cluster_col`] from the (cell type, gene,
    weight) rows in `net` (see marker_table / net_from_table; weight -1 = negative
    marker), scored with `method` (see METHODS), stores each cell's suggestion as
    obs[f"{cluster_col}_celltype"], and returns the per-cluster calls. A clustering of
    hexagonal bins (dataset.hex_clusterings) is typed on those bins, not on single
    cells / spots. Genes `gene_filter` counts as background are left out.
    `tissue` is only a label for the results."""
    import anndata as ad
    import decoupler as dc

    adata = dataset.adata
    clusters = adata.obs[cluster_col].astype("category")
    names = [str(c) for c in clusters.cat.categories]
    codes = clusters.cat.codes.to_numpy()
    sizes = np.bincount(codes[codes >= 0], minlength=len(names))
    if (sizes > 0).sum() < 2:
        raise ValueError("Cell typing compares the clusters with each other, so it needs at least 2 "
                         "clusters - try a higher Leiden resolution.")

    # What gets scored: hexagons for a hexagon clustering, else the cells themselves -
    # with every measured gene (log-normalized), less the background genes.
    width = dataset.hex_clusterings.get(cluster_col)
    binnings = None
    if width is not None:
        report(f"{clustering.hex_profile_verb(dataset)} within hexagonal bins ({width:g} cell spacings wide)")
        units, binnings = clustering.hex_profiles(dataset, width, None, dataset.hex_cluster_filters.get(cluster_col))
        unit_codes = clustering.cells_to_bins(codes, binnings)
    else:
        units, unit_codes = dataset.expression(), codes
    keep = ~clustering.background_genes(units, gene_filter)
    genes = list(np.asarray(units.var_names)[keep])
    scored = ad.AnnData(units.X if keep.all() else units.X[:, np.flatnonzero(keep)],
                        obs=pd.DataFrame(index=[str(i) for i in range(units.n_obs)]),
                        var=pd.DataFrame(index=genes))

    # The databases mostly use human symbols; match them to this dataset's own spelling
    # (case-insensitively, which also covers most mouse symbols) and drop unmeasured ones.
    report("Matching the marker genes to this data's genes")
    net = net.copy()
    if "weight" not in net.columns:
        net["weight"] = 1.0
    net["target"] = [dataset.match_gene(g) for g in net["target"]]
    net = net[net["target"].isin(set(genes))].drop_duplicates(["source", "target"])
    if method != ULM:  # only ULM can weigh negative markers
        net = net[net["weight"] > 0]
    # Only (positive) markers count towards MIN_MARKERS: a type isn't identified by
    # what it lacks alone.
    measured = net[net["weight"] > 0].groupby("source").size()
    net = net[net["source"].isin(measured[measured >= MIN_MARKERS].index)]
    if net.empty:
        raise ValueError(
            f"None of the cell types considered has at least {MIN_MARKERS} of its marker "
            "genes measured in this data - try 'All tissues', check the marker sets on the "
            "Cell types tab, or the gene panel may be too small for marker-based typing."
        )
    types = sorted(net["source"].unique())
    marker_genes = sorted(set(net.loc[net["weight"] > 0, "target"]))

    unit_word = "hexagons" if width is not None else "cells"
    method_name = {v: k for k, v in METHODS.items()}[method]
    report(f"Scoring {len(types)} cell types in each of the {scored.n_obs:,} {unit_word} ({method_name})")
    if method == ORA:
        tables, standout = _ora_tables(dc, scored, net, unit_codes, names, report)
        # No per-cell score of its own: the confidence uses ULM's.
        report(f"Scoring each {unit_word[:-1]}'s confidence in its type ({scored.n_obs:,} {unit_word})")
        unit_scores = _decoupler_scores(dc.mt.ulm, scored, net)
    else:
        if method == SCORE_GENES:
            unit_scores = _module_scores(scored, net, types)
        else:
            unit_scores = _decoupler_scores(dc.mt.aucell if method == AUCELL else dc.mt.ulm, scored, net)
        report(f"Testing each cluster's scores against all other {unit_word} (decoupler rankby_group)")
        tables = _rank_by_cluster(dc, unit_scores, unit_codes, names)
        # The type's markers that stand out most in each cluster: cluster means of the
        # marker genes, standardised across clusters.
        means = pd.DataFrame(_cluster_means(scored[:, marker_genes].X, unit_codes, len(names)),
                             index=names, columns=marker_genes)
        standout = (means - means.mean(axis=0)) / means.std(axis=0).replace(0, 1)

    n_bins = np.bincount(unit_codes[unit_codes >= 0], minlength=len(names)) if width is not None else None
    calls = []
    for i, name in enumerate(names):
        table = tables.get(name)
        label = UNKNOWN
        candidates = []
        if table is not None and len(table):
            best = table.iloc[0]
            if best["score"] > 0 and best["padj"] < MAX_PVALUE:
                label = str(best["type"])
            candidates = [(str(r["type"]), float(r["score"]), int(measured[r["type"]]))
                          for _, r in table.head(N_CANDIDATES).iterrows()]
        drivers = []
        if label != UNKNOWN:
            type_genes = [g for g in net.loc[(net["source"] == label) & (net["weight"] > 0), "target"]
                          if g in standout.columns]
            levels = standout.loc[name, type_genes].sort_values(ascending=False)
            drivers = [(g, float(v)) for g, v in levels.items() if v > 0][:N_DRIVERS]
        calls.append(ClusterCall(
            cluster=name, n_cells=int(sizes[i]), label=label, candidates=candidates,
            drivers=drivers, n_bins=None if n_bins is None else int(n_bins[i]),
        ))

    column = f"{cluster_col}_celltype"
    label_of = {c.cluster: c.label for c in calls}
    cell_labels = np.array([label_of.get(str(v)) if v == v else None for v in clusters.to_numpy()], dtype=object)
    present = sorted({c.label for c in calls} - {UNKNOWN}) + ([UNKNOWN] if UNKNOWN in label_of.values() else [])
    adata.obs[column] = pd.Categorical(cell_labels, categories=present)

    unit_labels = np.array([calls[c].label if c >= 0 else None for c in unit_codes], dtype=object)
    unit_conf = _percentile_confidence(unit_scores, unit_labels)
    confidence = (unit_conf if binnings is None
                  else clustering.bins_to_cells(unit_conf, binnings, adata.n_obs, fill=np.nan))
    confidence_column = f"{column}_confidence"
    adata.obs[confidence_column] = confidence
    for i, call in enumerate(calls):
        in_cluster = unit_codes == i
        if call.label != UNKNOWN and in_cluster.any():
            call.mean_confidence = float(np.nanmean(unit_conf[in_cluster]))
    return CellTypingResult(cluster_col, column, confidence_column, species, tissue,
                            len(types), calls, hex_width=width,
                            background="" if gene_filter is None else gene_filter.describe(),
                            method=method)


def _decoupler_scores(func, scored, net: pd.DataFrame) -> pd.DataFrame:
    """Units x types scores from a decoupler method (dc.mt.aucell / dc.mt.ulm) on `scored`
    (an AnnData of units x genes); NaN for units without any expression, which decoupler
    leaves out."""
    out = func(data=scored, net=net, tmin=MIN_MARKERS, bsize=BATCH_SIZE)
    result = scored if out is None else out  # decoupler returns a copy when it drops empty rows
    key = f"score_{func.name}"
    scores = result.obsm[key].reindex(scored.obs_names)
    for k in [k for k in scored.obsm if k.endswith(f"_{func.name}")]:
        del scored.obsm[k]
    return scores


def _module_scores(scored, net: pd.DataFrame, types: list[str]) -> pd.DataFrame:
    """Units x types scanpy module scores (sc.tl.score_genes, its default control genes)."""
    import anndata as ad
    import scanpy as sc

    work = ad.AnnData(scored.X, obs=pd.DataFrame(index=scored.obs_names), var=pd.DataFrame(index=scored.var_names))
    scores = {}
    for t in types:
        markers = list(net.loc[net["source"] == t, "target"])
        sc.tl.score_genes(work, markers, score_name="_score", use_raw=False, random_state=0)
        scores[t] = work.obs.pop("_score").to_numpy()
    return pd.DataFrame(scores, index=scored.obs_names)


def _rank_by_cluster(dc, unit_scores: pd.DataFrame, unit_codes: np.ndarray,
                     names: list[str]) -> dict[str, pd.DataFrame]:
    """Per cluster: every type's (type, score = t statistic, padj) from decoupler's
    rankby_group - the cluster's units against all others - best first. Clusters with
    fewer than 2 scored units can't be tested and get no table."""
    import anndata as ad

    valid = (unit_codes >= 0) & unit_scores.notna().all(axis=1).to_numpy()
    counts = np.bincount(unit_codes[valid], minlength=len(names))
    valid &= np.isin(unit_codes, np.flatnonzero(counts >= 2))
    groups = [names[c] for c in unit_codes[valid]]
    if len(set(groups)) < 2:
        raise ValueError("Too few cells / hexagons with marker expression to compare the clusters.")
    data = ad.AnnData(unit_scores.to_numpy()[valid],
                      obs=pd.DataFrame({"cluster": pd.Categorical(groups)}, index=unit_scores.index[valid]),
                      var=pd.DataFrame(index=unit_scores.columns.astype(str)))
    ranked = dc.tl.rankby_group(data, groupby="cluster", reference="rest", method="t-test_overestim_var")
    return {str(g): df.rename(columns={"name": "type", "stat": "score"})
            .sort_values("score", ascending=False)[["type", "score", "padj"]].reset_index(drop=True)
            for g, df in ranked.groupby("group")}


def _ora_tables(dc, scored, net: pd.DataFrame, unit_codes: np.ndarray, names: list[str], report):
    """(per cluster: every type's (type, score = log odds ratio, padj) from decoupler's ORA
    of the cluster's top genes, best first; clusters x marker genes: the cluster's
    Wilcoxon score for its top genes, 0 for the rest). All of `scored`'s genes are the
    background."""
    import anndata as ad
    import scanpy as sc

    rng = np.random.default_rng(0)
    picked = []
    for i in range(len(names)):
        rows = np.flatnonzero(unit_codes == i)
        if len(rows) > ORA_MAX_UNITS:
            rows = rng.choice(rows, ORA_MAX_UNITS, replace=False)
        if len(rows) >= 2:  # a group of one can't be ranked
            picked.append(rows)
    rows = np.sort(np.concatenate(picked)) if picked else np.zeros(0, dtype=int)
    groups = [names[c] for c in unit_codes[rows]]
    tested = [n for n in names if n in set(groups)]
    genes = list(scored.var_names)
    marker_genes = sorted(set(net["target"]))
    standout = pd.DataFrame(0.0, index=names, columns=marker_genes)
    tables: dict[str, pd.DataFrame] = {}
    if len(tested) < 2:
        return tables, standout
    report(f"Finding each cluster's top genes ({len(tested)} clusters, {len(rows):,} cells / hexagons, "
           f"{len(genes):,} genes; scanpy rank_genes_groups, Wilcoxon)")
    sub = ad.AnnData(
        scored.X[rows],
        obs=pd.DataFrame({"cluster": pd.Categorical(groups, categories=tested)},
                         index=[str(i) for i in range(len(rows))]),
        var=pd.DataFrame(index=genes),
    )
    sub.uns["log1p"] = {"base": None}  # the units are log-normalized expression
    sc.tl.rank_genes_groups(sub, "cluster", method="wilcoxon", use_raw=False, tie_correct=True)
    report("Testing which marker sets are over-represented among each cluster's top genes (decoupler ORA)")
    for name in tested:
        df = sc.get.rank_genes_groups_df(sub, group=name)
        top = df[(df["pvals_adj"] < MAX_PVALUE) & (df["logfoldchanges"] >= ORA_MIN_LFC)]
        top = top.sort_values("scores", ascending=False).head(ORA_TOP_GENES)
        if top.empty:
            continue
        # ORA takes each row's n_up highest values as the "selected" genes: give the top
        # genes descending ranks and every other gene 0.
        values = pd.Series(0.0, index=genes)
        values[list(top["names"])] = np.arange(len(top), 0, -1, dtype=float)
        es, padj = dc.mt.ora(data=values.to_frame(name).T, net=net, tmin=MIN_MARKERS,
                             n_up=len(top), n_bg=0, empty=False)
        tables[name] = (pd.DataFrame({"type": es.columns, "score": es.iloc[0].to_numpy(),
                                      "padj": padj.iloc[0].to_numpy()})
                        .sort_values(["score", "padj"], ascending=[False, True]).reset_index(drop=True))
        hits = [g for g in marker_genes if g in set(top["names"])]
        standout.loc[name, hits] = top.set_index("names").loc[hits, "scores"].to_numpy()
    return tables, standout


def _percentile_confidence(unit_scores: pd.DataFrame, unit_labels: np.ndarray) -> np.ndarray:
    """Per unit: the percentile (0-1) of its score for its assigned type among all units'
    scores for that type (NaN for units without a type or a score)."""
    confidence = np.full(len(unit_scores), np.nan, dtype=np.float32)
    for label in set(unit_labels) - {None, UNKNOWN}:
        if label in unit_scores.columns:
            percentile = unit_scores[label].rank(pct=True).to_numpy()
            rows = unit_labels == label
            confidence[rows] = percentile[rows]
    return confidence
