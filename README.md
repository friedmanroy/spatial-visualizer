# Spatial Transcriptomics Visualizer

An interactive [Panel](https://panel.holoviz.org/) + [Bokeh](https://bokeh.org/) viewer for
spatial transcriptomics data, built to work with **any** AnnData `.h5ad` file rather than one
specific dataset. Nothing about the obs schema is hardcoded: after loading a file you choose
which obs column separates samples/fields-of-view, which fields are offered for coloring, and
color everything either by an obs field or by gene expression / gene-set (pathway) scores.

## Features

- **Generic field coloring** - pick any obs column; categorical fields get one checkbox +
  clickable color swatch per category (click a swatch to open the color picker), continuous
  fields are automatically binned into 10 groups.
- **Fields-of-view / samples** - pick which obs column splits the data into samples, then
  browse between them with a dropdown or Prev/Next.
- **Gene & Pathway view** - RGB-channel overlay of up to three genes or gene-set (pathway)
  scores at once, with autocomplete gene search, a description + gene list for each pathway,
  and a small histogram of the score distribution under each channel.
- **scRNA / snRNA companion page** - load a second `.h5ad` file, or compute PCA/UMAP on demand
  from the spatial data itself, with configurable number of PCs and neighbors. Uses scanpy's
  own `sc.pp.pca` / `sc.pp.neighbors` / `sc.tl.umap`.
- **Persistent choices** - custom category colors and which fields are enabled are cached to
  `cache/` (gitignored) and reloaded automatically the next time you open the same file.
- **Native file browser** - "Browse..." opens the OS's own file picker.
- **Error dialog** - any unexpected error shows a popup with the traceback instead of silently
  freezing the page.

## Requirements

- Python 3.11+
- A `pathways.json` file describing gene sets (an example is included at `data/pathways.json`)

## Installation

```bash
git clone git@github.com:friedmanroy/spatial-visualizer.git
cd spatial-visualizer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python st_visualizer.py
```

This opens a browser tab with a landing page. From there:

1. Enter (or browse to) the path of a spatial `.h5ad` file - it must have coordinates in
   `adata.obsm['spatial']`.
2. Once loaded, pick which obs column separates samples/fields-of-view (or check the box if
   the file has only one), optionally point at a `pathways.json` file (defaults to
   `data/pathways.json` if present), and choose which obs fields should be offered for
   coloring.
3. Click **Open viewer**. The **Spatial** tab shows the hex/point rendering with per-sample
   navigation and the Field / Gene & Pathway coloring controls described above.
4. The **scRNA / snRNA** tab lets you load a companion single-cell dataset, or compute a
   PCA/UMAP embedding directly from the spatial data on demand (this can take a while for large
   datasets, hence "on demand" rather than automatic).

Closing the browser tab shuts the server down automatically.

## Data

- `data/pathways.json` - the gene-set definitions used by the Gene & Pathway view. Each entry
  has a `source`, a `description`, and a list of `genes` (edit or extend freely - the app reads
  this file at startup).
- The bundled Open-ST/RCTD `.h5ad` used during development is **not** included in this
  repository (too large for git) - point the app at your own `.h5ad` file instead.

## Project layout

```
st_visualizer.py         entry point (`python st_visualizer.py`)
viewer/
  app.py                 assembles the landing page and the tabbed main viewer
  landing.py              landing page: file path, sample/FOV column, field selection
  spatial_page.py          "Spatial" tab
  embedding_page.py        "scRNA / snRNA" tab
  color_controls.py        shared Field / Gene & Pathway coloring controls
  fields.py                 generic obs-column classification (categorical/binned/unusable)
  colors.py                 palette assignment, RGB blending
  histogram.py               gene/pathway score distribution histograms
  pathways.py                pathways.json loading + scanpy score_genes scoring
  spatial_plot.py            spatial rendering (Bokeh circle glyph)
  state.py                   AppState / Dataset - loaded AnnData(s) + caches
  cache.py                   persists per-dataset color/field choices to cache/
  file_picker.py              native OS file-picker integration
  error_dialog.py             global error popup
data/
  pathways.json              example gene-set definitions
```
