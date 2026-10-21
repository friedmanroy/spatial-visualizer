# Spatial Transcriptomics Visualizer

An interactive [Panel](https://panel.holoviz.org/) + [Bokeh](https://bokeh.org/) viewer for
spatial transcriptomics data, built to work with **any** AnnData `.h5ad` file rather than one
specific dataset. Nothing about the obs schema is hardcoded: after loading a file you choose
which obs column separates samples/fields-of-view, which fields are offered for coloring, and
color everything either by an obs field or by gene expression / gene-set (pathway) scores.

## Features

- **Generic field coloring** - pick any obs column; categorical fields get one checkbox +
  clickable color swatch per category (click a swatch to open the color picker), continuous
  fields are automatically binned into 10 groups with equal numbers of cells (so long-tailed
  fields such as total counts still spread across the colors).
- **Cell shapes** - draw cells as hexagons, squares or circles. **Auto** (the default)
  detects per sample whether the spots sit on a hexagonal grid (e.g. Visium), a square grid
  (e.g. binned Visium HD / Stereo-seq) or neither (e.g. segmented cells, Slide-seq -> circles).
  Markers are sized (and rotated) from the spacing between neighboring cells so they tile the
  tissue, and a slider makes them smaller or bigger.
- **Fields-of-view / samples** - pick which obs column splits the data into samples, then
  browse between them with a dropdown or Prev/Next.
- **Huge samples** - samples with more than a million cells are drawn as hexagonal bins
  (about 25,000 per sample by default, adjustable under **Display settings**): each bin shows
  its cells' mean expression, or their most common selected category. Zoom in until at most
  50,000 cells are in view (adjustable) and the cells there are drawn individually (as circles
  by default), colored per cell by expression; clusters and cell types computed on the
  hexagons color each cell like its hexagon.
- **Top genes in view** - the header above the plot lists the 4 most expressed genes among
  the cells currently in view, updating as you pan and zoom.
- **Hover tooltips** - hovering a cell (or hex bin) shows its category in the current field
  (with the confidence for suggested cell types), or its values for the genes / pathways shown.
- **Leiden clustering** - at load time or from the **Leiden clustering & cell types** panel,
  using scanpy's standard pipeline (HVGs, PCA, neighbors, `sc.tl.leiden`). By default it
  clusters **hexagons** (each one's mean expression; width adjustable) and every spot gets its
  hexagon's cluster - the way to cluster sub-cellular spots (Xenium transcripts, Open-ST,
  Visium HD...), and much faster on huge samples. Choose **Individual cells** for segmented
  single-cell data.
- **Background genes** - mitochondrial (`MT-`), ribosomal-protein (`RPS`/`RPL`) and genes
  detected in almost every cell / hexagon can be left out of clustering, PCA/UMAP, cell typing
  and the "most expressed genes" shown (mitochondrial genes are left out by default).
- **Cell-type suggestions** - optionally, each Leiden cluster gets a suggested cell type from
  PanglaoDB marker genes, scored with [decoupler](https://decoupler.readthedocs.io) (ULM on
  the clusters' standardised mean expression). A hexagon clustering is typed on its hexagons,
  and each spot's confidence is its hexagon's. The species (human or mouse) is preset from the gene names (and can be changed); pick the
  tissue to keep off-tissue types out; a table shows each cluster's suggestion, score and
  runners-up. A rough first guess to check against known markers - not a validated
  annotation. PanglaoDB is downloaded once (needs internet) and then kept in `cache/`.
- **Your own marker sets** - a **Cell types** tab lists the marker gene sets cell typing uses
  (PanglaoDB's for the chosen species/tissue) in an editable table: change a type's genes,
  switch types off, add your own cell types, reset edits, or view a set as a pathway score.
  It shows how many of each set's markers are measured in your data, and can re-run cell
  typing on an existing clustering right away. The suggestions table lists the markers
  behind each call. Your changes are saved in `cache/custom_markers.json` and used for every
  dataset; the Leiden panel picks PanglaoDB only, your sets only, or both.
- **Gene & Pathway view** - RGB-channel overlay of up to three genes or gene-set (pathway)
  scores at once, with autocomplete gene search, a description + gene list for each pathway,
  and a small histogram of the score distribution under each channel.
- **Custom pathways** - a **Pathways** tab where you paste a comma- or line-separated gene
  list to add your own gene set. It's immediately available in the Gene & Pathway coloring,
  and saved to `cache/custom_pathways.json` so it's loaded automatically next time (the
  default `pathways.json` is left untouched).
- **scRNA / snRNA companion page** - load a second `.h5ad` file, or compute PCA/UMAP on demand
  from the spatial data itself - by default on its hexagons (one point per hexagon, colored by
  its most common category), or on individual cells - with configurable number of PCs and neighbors. Uses scanpy's
  own `sc.pp.pca` / `sc.pp.neighbors` / `sc.tl.umap`.
- **Persistent choices** - custom category colors, which fields are enabled, and the
  sample/FOV column are cached to
  `cache/` inside the app's folder (gitignored) and reloaded automatically the next time you open the same file.
- **Native file browser** - "Browse..." opens the OS's own file picker.
- **Loading overlay** - long steps (loading, clustering, PCA/UMAP...) gray out the page with a
  checklist of the step's stages and a live timer for each.
- **Error dialog** - any unexpected error shows a popup with the traceback instead of silently
  freezing the page.

## Getting started - the easy way (no Python needed)

For people who just want to use the app. No Python, no terminal, no admin rights.

1. Unzip the `spatial-visualizer` folder you were sent, and put it wherever you like.
2. Inside it, double-click:
   - **Mac:** `Start Spatial Visualizer (Mac).command`
   - **Windows:** `Start Spatial Visualizer (Windows).bat`
3. A small text window opens and the app appears in your browser.
   - The **first launch** downloads everything it needs and can take a few minutes. Later
     launches are much faster.
   - Closing the browser tab stops the app.

**First time on a Mac:** macOS may say the file "cannot be opened because it is from an
unidentified developer". Right-click (or Control-click) the file, choose **Open**, then click
**Open** again. On newer macOS versions you may instead need to go to
**System Settings -> Privacy & Security** and click **Open Anyway**. You only need to do this once.

If that doesn't work (or you'd rather skip it), open **Terminal**, type `source ` (with a
space after it), drag the `install_vis.bash` file from this folder into the Terminal window,
and press Enter. This one-time setup unblocks the launcher, and then double-clicking works.

**First time on Windows:** if a blue "Windows protected your PC" box appears, click
**More info -> Run anyway**. You only need to do this once.

<details>
<summary>What the launcher actually does</summary>

It installs [uv](https://docs.astral.sh/uv/) into your user folder (once). uv then provides a
private copy of Python and the required packages. Nothing is installed system-wide, and any
Python you already have is left untouched.

The launcher always runs the version of the app **in this folder**; to update, get a newer zip.

Saved colors and field choices are kept in the `cache/` folder inside this folder. To keep
them when switching to a newer zip, copy `cache/` into the new folder.
To remove everything: delete this folder, run `uv cache clean`, then delete uv itself (see
the uv docs).
</details>

### Sharing the app (maintainers)

Send people a zip of the project. The simplest clean way (only tracked files, so no `venv/`,
`cache/` or `.h5ad` data, and the Mac launcher stays executable) is:

```bash
git archive --format=zip --prefix=spatial-visualizer/ -o ~/Desktop/spatial-visualizer.zip HEAD
```

`git archive` only includes committed changes. Users run the copy in their zip, so to update
them you send a new zip.

## Getting started - for developers

Requires Python 3.12+.

```bash
git clone git@github.com:friedmanroy/spatial-visualizer.git
cd spatial-visualizer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt     # or: pip install -e .
```

Dependencies are pinned in `pyproject.toml`. `requirements.txt` just installs this package
(in editable mode) from there.

## Usage

```bash
python st_visualizer.py     # or, after installing: spatial-visualizer
```

(Or double-click a `Start Spatial Visualizer` launcher, see above.) This opens a browser tab with a landing page. From there:

1. Enter (or browse to) the path of a spatial `.h5ad` file - it must have coordinates in
   `adata.obsm['spatial']`.
2. Once loaded, pick which obs column separates samples/fields-of-view (leave it as **None**
   if the file is a single sample; the choice is remembered for next time), optionally point at a `pathways.json` file (defaults to
   the bundled `viewer/data/pathways.json`), and choose which obs fields should be offered for
   coloring.
3. Click **Open viewer**. The **Spatial** tab shows each cell as a hexagon, square or circle (auto-detected)
   (sized so they tile the tissue by default, adjustable with the **Cell size** slider) with per-sample
   navigation and the Field / Gene & Pathway coloring controls described above.
4. The **scRNA / snRNA** tab lets you load a companion single-cell dataset, or compute a
   PCA/UMAP embedding directly from the spatial data on demand (this can take a while for large
   datasets, hence "on demand" rather than automatic).

Closing the browser tab shuts the server down automatically.

## Data

- `viewer/data/pathways.json` - the gene-set definitions used by the Gene & Pathway view. Each entry
  has a `source`, a `description`, and a list of `genes` (edit or extend freely - the app reads
  this file at startup; launcher users can instead point the app at their own copy).
- The bundled Open-ST/RCTD `.h5ad` used during development is **not** included in this
  repository (too large for git) - point the app at your own `.h5ad` file instead.

## Project layout

```
st_visualizer.py         entry point (`python st_visualizer.py`)
pyproject.toml           package metadata + pinned dependencies (`spatial-visualizer` command)
Start Spatial Visualizer (Mac).command / (Windows).bat   double-click launchers
viewer/
  cli.py                 starts the server (shared by the script and the installed command)
  app.py                 assembles the landing page and the tabbed main viewer
  landing.py              landing page: file path, sample/FOV column, field selection
  spatial_page.py          "Spatial" tab
  embedding_page.py        "scRNA / snRNA" tab
  pathways_page.py         "Pathways" tab: add/remove custom gene sets
  color_controls.py        shared Field / Gene & Pathway coloring controls
  fields.py                 generic obs-column classification (categorical/binned/unusable)
  colors.py                 palette assignment, RGB blending
  histogram.py               gene/pathway score distribution histograms
  pathways.py                pathways.json + custom pathway loading, score_genes scoring
  clustering.py              normalization check, PCA/neighbors, Leiden (per cell or per hex bin)
  hexbin.py                  hexagonal binning (huge samples, sub-cellular spots)
  gene_filter_controls.py    "leave out background genes" widgets
  celltyping.py              PanglaoDB + decoupler cell-type suggestions for clusters
  celltype_controls.py       species / tissue widgets and the suggestions table
  markers_page.py            "Cell types" tab: edit / add marker gene sets
  busy_overlay.py            full-page "working..." overlay
  spatial_plot.py            spatial rendering (hexagon/square/circle glyphs sized from cell spacing)
  state.py                   AppState / Dataset - loaded AnnData(s) + caches
  cache.py                   persists per-dataset color/field choices to cache/
  file_picker.py              native OS file-picker integration
  error_dialog.py             global error popup
  data/pathways.json        example gene-set definitions (bundled with the package)
```
