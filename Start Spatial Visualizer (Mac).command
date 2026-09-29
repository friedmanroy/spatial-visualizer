#!/bin/bash
# Double-click to start the Spatial Transcriptomics Visualizer (macOS).
#
# If the GitHub repository is reachable (i.e. public), each launch runs the latest
# version from it. Otherwise (private repo, or no internet) it runs the copy of the app
# in this folder. Either way uv provides a private Python + packages - nothing is
# installed system-wide and no admin rights are needed.

SOURCE="https://github.com/friedmanroy/spatial-visualizer/archive/refs/heads/main.zip"
PYTHON_VERSION="3.13"

HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# Saved colors/fields stay in this folder whether the online or the local copy runs.
export SPATIAL_VISUALIZER_CACHE_DIR="$HERE/cache"

fail() {
    echo
    echo "Something went wrong (see the messages above)."
    read -n 1 -s -r -p "Press any key to close this window..."
    exit 1
}

if ! command -v uv >/dev/null 2>&1; then
    echo "First-time setup: installing uv (the tool that runs the app)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || fail
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "Starting the Spatial Transcriptomics Visualizer."
echo "The first launch downloads everything it needs and can take a few minutes."
echo "The app opens in your browser. Closing the browser tab stops it."
echo

if curl -sfIL --max-time 10 "$SOURCE" >/dev/null 2>&1; then
    echo "Using the latest version from GitHub."
    uvx --managed-python --python "$PYTHON_VERSION" \
        --refresh-package spatial-visualizer \
        --from "$SOURCE" spatial-visualizer || fail
else
    echo "Using the version in this folder."
    uv run --managed-python --python "$PYTHON_VERSION" \
        --project "$HERE" spatial-visualizer || fail
fi
