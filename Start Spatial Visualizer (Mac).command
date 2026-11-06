#!/bin/bash
# Double-click to start the Spatial Transcriptomics Visualizer (macOS).
#
# Runs the copy of the app in this folder. uv provides a private Python + packages -
# nothing is installed system-wide and no admin rights are needed. --frozen installs
# exactly the versions in uv.lock, so every computer gets the same environment.

PYTHON_VERSION="3.13"

HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# Saved colors/fields/regions are kept in this folder's cache/.
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
echo "The app opens in your browser. Closing the browser tab stops it (after about 30 s)."
echo

echo "Using the version in this folder."
uv run --frozen --managed-python --python "$PYTHON_VERSION" \
    --project "$HERE" spatial-visualizer || fail
