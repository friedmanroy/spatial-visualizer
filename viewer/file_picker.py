"""A text path input paired with a native OS file-picker dialog, so a file can either
be typed by path or found by browsing - reused by the landing page and the scRNA/snRNA
companion-file prompt. The dialog pops up as its own window, which only works because
the app runs locally (`python st_visualizer.py`) with a real display, not in a
headless/remote environment.

The dialog is run in a separate process (awaited from a worker thread) rather than
in-process: a blocking dialog on the server's event-loop thread freezes the whole page
until it closes, and on macOS an in-process Tk window often opens hidden behind the
browser, so the page appears frozen for good.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import traceback

import panel as pn

# Fallback for non-macOS: plain Tk dialog, in its own process so it owns its main thread.
_TK_SCRIPT = r"""
import sys
import tkinter as tk
from tkinter import filedialog

pattern, label = sys.argv[1], sys.argv[2]
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
root.update()
path = filedialog.askopenfilename(
    title="Select a file",
    filetypes=[(f"{label} files", pattern), ("All files", "*.*")],
)
root.destroy()
sys.stdout.write(path or "")
"""

# `activate` brings osascript itself to the front so the dialog isn't hidden behind the browser.
_OSASCRIPT = 'activate\nPOSIX path of (choose file with prompt "Select a file")'


def _pick_file_blocking(file_pattern: str) -> str | None:
    if sys.platform == "darwin":
        cmd = ["osascript", "-e", _OSASCRIPT]
    else:
        label = file_pattern.lstrip("*.").upper() or "Data"
        cmd = [sys.executable, "-c", _TK_SCRIPT, file_pattern, label]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # osascript exits non-zero with error -128 when the user cancels - not a failure.
        if "-128" not in result.stderr:
            print(f"File dialog failed: {result.stderr.strip()}", file=sys.stderr)
        return None
    return result.stdout.strip() or None


def build_path_picker(
    label: str, placeholder: str = "/path/to/data.h5ad", file_pattern: str = "*.h5ad"
) -> tuple[pn.widgets.TextInput, pn.widgets.Button, pn.viewable.Viewable]:
    """Returns (text input, browse button, both side by side). Callers short on width
    can ignore the combined row and lay out the input and button themselves."""
    path_input = pn.widgets.TextInput(name=label, placeholder=placeholder)
    # align="end" puts the button at the bottom of the row, level with the text box
    # rather than with the text box's title above it.
    browse_btn = pn.widgets.Button(name="Browse...", width=110, align="end")

    async def on_browse(event):
        browse_btn.disabled = True  # avoid stacking several dialogs
        try:
            path = await asyncio.to_thread(_pick_file_blocking, file_pattern)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return
        finally:
            browse_btn.disabled = False
        if path:
            path_input.value = path

    browse_btn.on_click(on_browse)

    return path_input, browse_btn, pn.Row(path_input, browse_btn)
