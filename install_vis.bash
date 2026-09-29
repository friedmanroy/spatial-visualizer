# One-time setup for the Spatial Transcriptomics Visualizer on macOS (also works on Linux).
#
# Run it from Terminal, in either of these ways:
#     source install_vis.bash
#     bash install_vis.bash
#
# It unblocks the double-click launcher (macOS flags files that came from a download and
# may refuse to open them), makes the launcher executable, and installs uv if needed.
# After that, double-click "Start Spatial Visualizer (Mac).command" to start the app.
#
# Written to work when sourced from zsh (the macOS default) as well as bash, so it uses
# a function with `return` instead of `exit`, which would close the user's Terminal.

_install_vis() {
    local here
    if [ -n "${BASH_SOURCE:-}" ]; then
        here="${BASH_SOURCE[0]}"
    elif [ -n "${ZSH_VERSION:-}" ]; then
        eval 'here="${(%):-%x}"'
    else
        here="$0"
    fi
    here="$(cd "$(dirname "$here")" && pwd)" || return 1
    local launcher="$here/Start Spatial Visualizer (Mac).command"

    echo "Setting up the Spatial Transcriptomics Visualizer in:"
    echo "    $here"
    echo

    if [ "$(uname)" = "Darwin" ]; then
        echo "- Unblocking downloaded files..."
        xattr -dr com.apple.quarantine "$here" 2>/dev/null
    fi

    if [ -f "$launcher" ]; then
        echo "- Making the launcher executable..."
        chmod +x "$launcher" || return 1
    else
        echo "Can't find \"$launcher\" - is this script in the app's folder?"
        return 1
    fi

    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if command -v uv >/dev/null 2>&1; then
        echo "- uv is already installed."
    else
        echo "- Installing uv (the tool that runs the app)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh || return 1
    fi

    echo
    echo "Done! To start the app, double-click:"
    echo "    Start Spatial Visualizer (Mac).command"
    echo "(the first start downloads everything it needs and can take a few minutes)."
}

_install_vis
unset -f _install_vis
