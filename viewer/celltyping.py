"""Rudimentary marker-based cell typing of clusters (e.g. Leiden), from PanglaoDB's
cell-type marker genes plus the user's own marker sets, with one of two methods:

- ULM (decoupler): each cluster is summarised by its cells' mean expression - or, for a
  clustering of hexagonal bins (sub-cellular spots), by its bins' mean - and every gene
  is then standardised across clusters, so a marker set scores high in a cluster where
  its genes are *relatively* high. decoupler's univariate linear model scores each cell
  type against each cluster, and the best-scoring type is suggested - or "Unknown" when
  no type scores significantly.
- ScType (Ianevski et al. 2022): every gene is standardised across the cells (or
  hexagons); a type's score in a cell adds up its markers' levels - each weighted by
  how specific it is, i.e. how few of the scored types share it - and subtracts its
  negative markers' (genes the type should *not* express). A cluster gets the type with
  the highest mean score over its cells, or "Unknown" if that's below SCTYPE_MIN_SCORE.
- Top-gene overlap (over-representation analysis): each cluster's own top genes are
  found first, without any marker list (Wilcoxon, the cluster vs all other clusters, over
  every gene), then a hypergeometric test asks which type's markers are over-represented
  among them; the most significant type is suggested (adjusted p < MAX_PVALUE), else
  "Unknown". Negative markers aren't used.

Negative markers come only from the user's own marker sets (PanglaoDB has none); ULM
uses them too, as markers with weight -1.

Each cell also gets a confidence (0-1) in its cluster's suggested type: the method's
scores computed per cell (ULM's for the top-gene overlap, which has no per-cell score),
turned into shares with a softmax over all scored types -
near 1 when the cell's markers clearly point to that type, low when another type's
markers fit it as well or better. For a hexagon clustering this is scored per hexagon
(single spots hold too few transcripts) and every spot gets its hexagon's confidence.
These are suggestions to check against known markers, not a validated annotation.

PanglaoDB is downloaded once (through OmniPath) and kept in cache/ for offline use.
Marker sets the user edits or adds on the Cell types tab are kept in
cache/custom_markers.json - only the differences from PanglaoDB (added types, changed
gene lists, switched-off types), so a fresh PanglaoDB download never loses them.
"""
from __future__ import annotations

import json
import os
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
#: PanglaoDB markers detected in fewer than this fraction of that cell type's cells are
#: left out (the threshold decoupler's own cell-type tutorial uses)
MIN_SENSITIVITY = 0.5
#: a suggestion needs a ULM p-value below this, else the cluster is "Unknown" (decoupler
#: already adjusts each cluster's p-values across the cell types tried, Benjamini-Hochberg)
MAX_PVALUE = 0.05
#: ScType calls a cluster "Unknown" when its best type's mean score per cell (or
#: hexagon) is below this - ScType's own rule (summed score under a quarter of the cells)
SCTYPE_MIN_SCORE = 0.25
#: top-gene overlap: a cluster's top genes are at most this many genes significantly
#: higher than in the other clusters (adjusted p < MAX_PVALUE, log2 FC >= ORA_MIN_LFC)...
ORA_TOP_GENES = 100
ORA_MIN_LFC = 0.5
#: ...found on at most this many randomly chosen cells (or hexagons) per cluster
ORA_MAX_UNITS = 1000
#: how many candidates per cluster are kept for display
N_CANDIDATES = 3

#: how cell types are scored (label shown in the UI -> key)
ULM, SCTYPE, ORA = "ulm", "sctype", "ora"
METHODS = {
    "ULM (decoupler, on cluster averages)": ULM,
    "ScType (specific + negative markers)": SCTYPE,
    "Top-gene overlap (each cluster's top genes vs marker sets)": ORA,
}

_MARKERS_CACHE = os.path.join(cache.CACHE_DIR, "panglaodb_markers.csv.gz")
CUSTOM_MARKERS_PATH = os.path.join(cache.CACHE_DIR, "custom_markers.json")
#: how many of a cluster's strongest markers for its suggested type are listed
N_DRIVERS = 4

#: which marker sets cell typing uses (label shown in the UI -> key)
MARKER_SOURCES = {
    "PanglaoDB + my edits": "both",
    "PanglaoDB only": "panglao",
    "My marker sets only": "custom",
}
#: where a marker set comes from, as shown in the Cell types table
ORIGIN_PANGLAO, ORIGIN_EDITED, ORIGIN_ADDED = "PanglaoDB", "PanglaoDB, edited", "Added by you"


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
    tmp = _MARKERS_CACHE + ".tmp"  # never leave a truncated table behind
    markers.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, _MARKERS_CACHE)
    return markers


def markers_available_offline() -> bool:
    return os.path.exists(_MARKERS_CACHE)


def tissue_options(markers: pd.DataFrame) -> list[str]:
    return [ALL_TISSUES] + sorted(markers["organ"].dropna().unique())


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


# ---- the user's own marker sets ------------------------------------------------------

def load_custom_markers() -> dict:
    """{"types": {name: {"genes": [...], "negative": [...]}} (added types and edited
    PanglaoDB types; "negative" = genes the type should not express, optional),
    "disabled": [names switched off]}."""
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
    origin: str  # ORIGIN_PANGLAO / ORIGIN_EDITED / ORIGIN_ADDED
    enabled: bool
    panglao_genes: list[str] | None  # PanglaoDB's own list (for "reset"), None if added
    negative: list[str] = field(default_factory=list)  # genes the type should not express


def marker_table(markers: pd.DataFrame | None, species: str, tissue: str, include_common: bool,
                 custom: dict, source: str) -> list[MarkerSet]:
    """Every marker set cell typing would consider with these settings (switched-off
    ones included, with enabled=False): PanglaoDB's for the species/tissue, with the
    user's edits applied and their added types appended - or only the user's sets."""
    panglao: dict[str, list[str]] = {}
    if markers is not None:
        net = marker_sets(markers, species, tissue, include_common)
        panglao = {t: list(g) for t, g in net.groupby("source", sort=True)["target"]}
    # "PanglaoDB only" means PanglaoDB untouched - the user's switched-off types too.
    disabled = set(custom["disabled"]) if source != "panglao" else set()
    rows: dict[str, MarkerSet] = {}
    if source != "custom":
        for name, genes in panglao.items():
            rows[name] = MarkerSet(name, genes, ORIGIN_PANGLAO, name not in disabled, genes)
    for name, info in custom["types"].items():
        genes = list(info.get("genes", []))
        negative = list(info.get("negative", []))
        if name in panglao:
            if source != "panglao":
                rows[name] = MarkerSet(name, genes, ORIGIN_EDITED, name not in disabled, panglao[name], negative)
        elif source != "panglao":
            # An edited PanglaoDB type outside the chosen tissue still counts as one of
            # "your" sets; a type you added is always included.
            origin = ORIGIN_EDITED if info.get("edited_from_panglao") else ORIGIN_ADDED
            rows[name] = MarkerSet(name, genes, origin, name not in disabled, None, negative)
    return sorted(rows.values(), key=lambda m: m.name.lower())


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
    method: str = ULM  # ULM or SCTYPE


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
                      gene_filter: clustering.GeneFilter | None = None, method: str = ULM) -> CellTypingResult:
    """Suggests a cell type per cluster of obs[`cluster_col`] from the (cell type, gene,
    weight) rows in `net` (see marker_table / net_from_table; weight -1 = negative
    marker), scored with `method` (ULM or SCTYPE), stores each cell's suggestion as
    obs[f"{cluster_col}_celltype"], and returns the per-cluster calls. A clustering of
    hexagonal bins (dataset.hex_clusterings) is typed on those bins, not on single
    cells / spots. Marker genes `gene_filter` counts as background are left out.
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

    # PanglaoDB uses human symbols; match them to this dataset's own spelling (case-
    # insensitively, which also covers most mouse symbols) and drop unmeasured ones.
    report("Matching the marker genes to this data's genes")
    net = net.copy()
    if "weight" not in net.columns:
        net["weight"] = 1.0
    net["target"] = [dataset.match_gene(g) for g in net["target"]]
    net = net.dropna(subset=["target"]).drop_duplicates(["source", "target"])
    genes = sorted(set(net["target"]))

    # What gets scored: hexagons for a hexagon clustering, else the cells themselves.
    width = dataset.hex_clusterings.get(cluster_col)
    binnings = None
    full = None  # every gene of the scored units: the top-gene overlap looks beyond the markers
    if not genes:
        units, unit_codes = None, codes
    elif width is not None:
        report(f"{clustering.hex_profile_verb(dataset)} within hexagonal bins ({width:g} cell spacings wide)")
        bins, binnings = clustering.hex_profiles(dataset, width, None if method == ORA else genes)
        if method == ORA:
            full, units = bins, bins[:, genes].copy()
        else:
            units = bins
        unit_codes = clustering.cells_to_bins(codes, binnings)
    else:
        # Log-normalized expression even where .X is kept as raw counts (Dataset.expression).
        expr = dataset.expression()
        if method == ORA:
            full = expr
        units, unit_codes = ad.AnnData(expr.X[:, [adata.var_names.get_loc(g) for g in genes]]), codes
    if units is not None:
        units.var_names = genes
        background = clustering.background_genes(units, gene_filter)
        if background.any():
            units = units[:, ~background].copy()
            genes = list(units.var_names)
            net = net[net["target"].isin(genes)]
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
    keep = units.var_names.isin(set(net["target"]))
    units = units[:, keep].copy()
    genes = list(units.var_names)

    unit_word = "hexagons" if width is not None else "cells"
    method_name = {v: k for k, v in METHODS.items()}[method]
    report(f"Scoring {net['source'].nunique()} cell types in each of the {len(names)} clusters ({method_name})")
    if method == SCTYPE:
        unit_scores, types, gene_mean, gene_spread = _sctype_unit_scores(units.X, genes, net)
        scores, standout, confident = _sctype_cluster_scores(
            unit_scores, types, units.X, gene_mean, gene_spread, genes, unit_codes, names)
    elif method == ORA:
        keep_genes = ~clustering.background_genes(full, gene_filter)
        scores, standout, confident = _ora_cluster_scores(full, keep_genes, net, unit_codes, names, report)
    else:
        scores, standout, confident = _ulm_cluster_scores(dc, units.X, genes, net, unit_codes, names)

    n_bins = np.bincount(unit_codes[unit_codes >= 0], minlength=len(names)) if width is not None else None
    calls = []
    for i, name in enumerate(names):
        ranked = scores.loc[name].sort_values(ascending=False)
        label = ranked.index[0] if confident[name] else UNKNOWN
        drivers = []
        if label != UNKNOWN:  # the type's markers that stand out most in this cluster
            type_genes = [g for g in net.loc[(net["source"] == label) & (net["weight"] > 0), "target"]
                          if g in standout.columns]
            levels = standout.loc[name, type_genes].sort_values(ascending=False)
            drivers = [(g, float(v)) for g, v in levels.items() if v > 0][:N_DRIVERS]
        calls.append(ClusterCall(
            cluster=name, n_cells=int(sizes[i]), label=label,
            candidates=[(t, float(ranked[t]), int(measured[t])) for t in ranked.index[:N_CANDIDATES]],
            drivers=drivers, n_bins=None if n_bins is None else int(n_bins[i]),
        ))

    column = f"{cluster_col}_celltype"
    label_of = {c.cluster: c.label for c in calls}
    cell_labels = np.array([label_of.get(str(v)) if v == v else None for v in clusters.to_numpy()], dtype=object)
    present = sorted({c.label for c in calls} - {UNKNOWN}) + ([UNKNOWN] if UNKNOWN in label_of.values() else [])
    adata.obs[column] = pd.Categorical(cell_labels, categories=present)

    report(f"Scoring each {unit_word[:-1]}'s confidence in its type ({units.n_obs:,} {unit_word})")
    unit_labels = np.array([calls[c].label if c >= 0 else None for c in unit_codes], dtype=object)
    if method == SCTYPE:
        # Rows without any marker expression get no confidence, as with ULM.
        X = units.X
        has_markers = (np.diff(sp.csr_matrix(X).indptr) > 0) if sp.issparse(X) else (np.asarray(X) != 0).any(axis=1)
        unit_conf = _softmax_confidence(unit_scores, types, unit_labels, has_markers)
    else:
        unit_conf = _cell_confidence(dc, units, net, unit_labels)
    confidence = (unit_conf if binnings is None
                  else clustering.bins_to_cells(unit_conf, binnings, adata.n_obs, fill=np.nan))
    confidence_column = f"{column}_confidence"
    adata.obs[confidence_column] = confidence
    for i, call in enumerate(calls):
        in_cluster = unit_codes == i
        if call.label != UNKNOWN and in_cluster.any():
            call.mean_confidence = float(np.nanmean(unit_conf[in_cluster]))
    return CellTypingResult(cluster_col, column, confidence_column, species, tissue,
                            int(net["source"].nunique()), calls, hex_width=width,
                            background="" if gene_filter is None else gene_filter.describe(),
                            method=method)


def _ulm_cluster_scores(dc, X, genes: list[str], net: pd.DataFrame, unit_codes: np.ndarray,
                        names: list[str]):
    """(clusters x types ULM scores, clusters x genes standardised levels, per cluster:
    whether its best type scores significantly). Genes are standardised across the
    clusters' mean profiles."""
    means = pd.DataFrame(_cluster_means(X, unit_codes, len(names)), index=names, columns=genes)
    spread = means.std(axis=0)
    means = means.loc[:, spread > 0]
    if means.shape[1] == 0:
        raise ValueError("None of the measured marker genes differs between the clusters, so "
                         "there's nothing to tell cell types apart by.")
    scaled = (means - means.mean(axis=0)) / spread[means.columns]
    scores, pvalues = dc.mt.ulm(data=scaled, net=net, tmin=MIN_MARKERS)
    top = scores.idxmax(axis=1)
    confident = pd.Series([scores.loc[c, t] > 0 and pvalues.loc[c, t] < MAX_PVALUE for c, t in top.items()],
                          index=scores.index)
    return scores, scaled, confident


def _ora_cluster_scores(full, keep: np.ndarray, net: pd.DataFrame, unit_codes: np.ndarray,
                        names: list[str], report):
    """(clusters x types -log10 adjusted p-value of its markers' overlap with the
    cluster's top genes, clusters x marker genes: the cluster's Wilcoxon score for its top
    genes (0 for the rest), per cluster: whether its best type is significant).

    `full` holds every gene of the units (cells or hexagons); genes where `keep` is False
    (background) are left out of both the top genes and the gene universe."""
    import anndata as ad
    import scanpy as sc
    from scipy.stats import false_discovery_control, hypergeom

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
    gene_names = np.asarray(full.var_names)[keep]

    top: dict[str, pd.Series] = {}  # cluster -> its top genes' Wilcoxon scores
    if len(tested) >= 2:
        report(f"Finding each cluster's top genes ({len(tested)} clusters, {len(rows):,} cells / hexagons, "
               f"{int(keep.sum()):,} genes)")
        sub = ad.AnnData(
            full.X[rows][:, np.flatnonzero(keep)],
            obs=pd.DataFrame({"cluster": pd.Categorical(groups, categories=tested)},
                             index=[str(i) for i in range(len(rows))]),
            var=pd.DataFrame(index=gene_names),
        )
        sub.uns["log1p"] = {"base": None}  # the units are log-normalized expression
        sc.tl.rank_genes_groups(sub, "cluster", method="wilcoxon", use_raw=False, tie_correct=True)
        for name in tested:
            df = sc.get.rank_genes_groups_df(sub, group=name)
            df = df[(df["pvals_adj"] < MAX_PVALUE) & (df["logfoldchanges"] >= ORA_MIN_LFC)]
            top[name] = df.sort_values("scores", ascending=False).head(ORA_TOP_GENES).set_index("names")["scores"]

    report("Testing which marker sets are over-represented among each cluster's top genes")
    types = sorted(net["source"].unique())
    markers = {t: set(net.loc[(net["source"] == t) & (net["weight"] > 0), "target"]) for t in types}
    n_markers = np.array([len(markers[t]) for t in types])
    universe = int(keep.sum())
    scores = pd.DataFrame(0.0, index=names, columns=types)
    standout = pd.DataFrame(0.0, index=names, columns=sorted(set(net["target"])))
    confident = pd.Series(False, index=names)
    for name, gene_scores in top.items():
        drawn = set(gene_scores.index)
        if not drawn:
            continue
        overlap = np.array([len(markers[t] & drawn) for t in types])
        padj = false_discovery_control(hypergeom.sf(overlap - 1, universe, n_markers, len(drawn)))
        # Equally significant types (common after adjusting) are ranked by their overlap.
        scores.loc[name] = -np.log10(np.maximum(padj, 1e-300)) + 1e-6 * overlap
        best = int(np.argmax(scores.loc[name].to_numpy()))
        confident[name] = bool(padj[best] < MAX_PVALUE and overlap[best] > 0)
        hits = [g for g in standout.columns if g in drawn]
        standout.loc[name, hits] = gene_scores[hits].to_numpy()
    return scores, standout, confident


def _gene_mean_spread(X) -> tuple[np.ndarray, np.ndarray]:
    """Each column's mean and standard deviation over the rows of X (sparse or dense),
    without densifying; a spread of 0 becomes 1 (the column is then 0 once centred)."""
    mean = np.asarray(X.mean(axis=0), dtype=np.float64).ravel()
    mean_sq = np.asarray((X.multiply(X) if sp.issparse(X) else np.asarray(X) ** 2).mean(axis=0),
                         dtype=np.float64).ravel()
    spread = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    spread[spread == 0] = 1.0
    return mean, spread


def _sctype_unit_scores(X, genes: list[str], net: pd.DataFrame):
    """ScType scores of every row of X (cells or hexagons x `genes`): (rows x types array,
    types, the genes' means, the genes' spreads). Per type: the sum of its markers'
    standardised levels, each weighted by its specificity, over sqrt(number of markers),
    minus the same for its negative markers (unweighted). Specificity is 1 for a marker
    of one type and falls linearly to 0 for one shared by every type scored.
    Standardising is folded into the weights, so a sparse X is never densified."""
    X = sp.csr_matrix(X, dtype=np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)
    mean, spread = _gene_mean_spread(X)
    types = sorted(net["source"].unique())
    positive = net[net["weight"] > 0]
    n_types = len(types)
    shared_by = positive.groupby("target")["source"].nunique()
    specificity = (n_types - shared_by) / (n_types - 1) if n_types > 1 else shared_by * 0 + 1.0
    gene_index = {g: i for i, g in enumerate(genes)}
    W = np.zeros((len(genes), n_types))
    for j, t in enumerate(types):
        rows = net[net["source"] == t]
        pos = [g for g in rows.loc[rows["weight"] > 0, "target"] if g in gene_index]
        neg = [g for g in rows.loc[rows["weight"] < 0, "target"] if g in gene_index]
        for g in pos:
            W[gene_index[g], j] += specificity[g] / np.sqrt(len(pos))
        for g in neg:
            W[gene_index[g], j] -= 1.0 / np.sqrt(len(neg))
    # ((X - mean) / spread) @ W == X @ Wz - mean @ Wz, with Wz = W / spread
    Wz = W / spread[:, None]
    scores = np.asarray(X @ Wz) - mean @ Wz
    return scores, types, mean, spread


def _sctype_cluster_scores(unit_scores: np.ndarray, types: list[str], X, gene_mean, gene_spread,
                           genes: list[str], unit_codes: np.ndarray, names: list[str]):
    """(clusters x types mean ScType score per unit, clusters x genes mean standardised
    level, per cluster: whether its best type reaches SCTYPE_MIN_SCORE)."""
    valid = unit_codes >= 0
    rows = np.flatnonzero(valid)
    indicator = sp.csr_matrix((np.ones(len(rows)), (unit_codes[valid], rows)),
                              shape=(len(names), unit_scores.shape[0]))
    n_units = np.asarray(indicator.sum(axis=1)).ravel()
    scores = pd.DataFrame(np.asarray(indicator @ unit_scores) / np.maximum(n_units, 1)[:, None],
                          index=names, columns=types)
    standout = pd.DataFrame((_cluster_means(X, unit_codes, len(names)) - gene_mean) / gene_spread,
                            index=names, columns=genes)
    confident = (scores.max(axis=1) >= SCTYPE_MIN_SCORE) & pd.Series(n_units > 0, index=names)
    return scores, standout, confident


def _softmax_confidence(S: np.ndarray, types: list[str], unit_labels: np.ndarray,
                        scored: np.ndarray) -> np.ndarray:
    """Per row of S (units x types scores): the softmax share of the row's assigned type
    among all types (NaN for rows without a type, or not `scored`)."""
    S = np.nan_to_num(np.asarray(S, dtype=np.float64))
    S = S - S.max(axis=1, keepdims=True)
    share = np.exp(S)
    share /= share.sum(axis=1, keepdims=True)
    type_index = {t: j for j, t in enumerate(types)}
    confidence = np.full(S.shape[0], np.nan, dtype=np.float32)
    for label in set(unit_labels) - {None, UNKNOWN}:
        if label in type_index:
            rows = np.flatnonzero((unit_labels == label) & scored)
            confidence[rows] = share[rows, type_index[label]]
    return confidence


def _cell_confidence(dc, units, net: pd.DataFrame, cell_labels: np.ndarray) -> np.ndarray:
    """Per row of `units` (cells or hexagons x marker genes): softmax share of its
    assigned type among all scored types' ULM scores (NaN for rows without a type).
    Genes are divided by their spread across rows (not centred, so sparse data stays
    sparse) so no gene dominates by scale."""
    import anndata as ad

    genes = list(units.var_names)
    X = units.X
    X = sp.csr_matrix(X, dtype=np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)
    _, spread = _gene_mean_spread(X)
    X = X @ sp.diags(1.0 / spread) if sp.issparse(X) else X / spread
    cells = ad.AnnData(sp.csr_matrix(X) if sp.issparse(X) else X)
    cells.var_names = genes
    # Batches keep memory bounded on million-cell data (each batch is densified).
    cells.obs_names = [str(i) for i in range(cells.n_obs)]
    # Rows without any marker expression are dropped and a repaired copy returned;
    # they get no confidence (NaN) rather than an even split across types.
    repaired = dc.mt.ulm(data=cells, net=net, tmin=MIN_MARKERS, bsize=50_000)
    scores = (cells if repaired is None else repaired).obsm["score_ulm"].reindex(cells.obs_names)
    scored = scores.notna().any(axis=1).to_numpy()
    return _softmax_confidence(scores.to_numpy(dtype=np.float64), list(scores.columns), cell_labels, scored)
