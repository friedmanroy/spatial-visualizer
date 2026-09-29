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
- **Persistent choices** - custom category colors, which fields are enabled, and the
  sample/FOV column are cached to
  `cache/` inside the app's folder (gitignored) and reloaded automatically the next time you open the same file.
- **Native file browser** - "Browse..." opens the OS's own file picker.
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

Which version of the app runs:
- If the app's GitHub page can be reached, the launcher runs the **latest version** from there,
  so you get updates automatically.
- Otherwise (the repository is private, or you're offline), it runs the version **in this folder**.

Saved colors and field choices are kept in the `cache/` folder inside this folder in both
cases. To keep them when switching to a newer zip, copy `cache/` into the new folder.
To remove everything: delete this folder, run `uv cache clean`, then delete uv itself (see
the uv docs).
</details>

### Sharing the app (maintainers)

Send people a zip of the project. The simplest clean way (only tracked files, so no `venv/`,
`cache/` or `.h5ad` data, and the Mac launcher stays executable) is:

```bash
git archive --format=zip --prefix=spatial-visualizer/ -o ~/Desktop/spatial-visualizer.zip HEAD
```

`git archive` only includes committed changes. Before the repository is public, users run
the copy in their zip, so to update them you send a new zip. After the repository is public
(the launchers check `SOURCE`, the `main` branch's zip on GitHub), every launch picks up
whatever is on `main` automatically. Users don't need to do anything, and they keep using the
zip they already have. To have users follow a more stable branch instead, change `SOURCE` in
both launchers before sending the zip.

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
3. Click **Open viewer**. The **Spatial** tab shows the hex/point rendering with per-sample
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
  data/pathways.json        example gene-set definitions (bundled with the package)
```
