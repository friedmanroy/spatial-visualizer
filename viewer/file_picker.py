"""A text path input paired with a native OS file-picker dialog, so a file can either
be typed by path or found by browsing - reused by the landing page and the scRNA/snRNA
companion-file prompt. The dialog pops up as its own window, which only works because
the app runs locally (`python st_visualizer.py`) with a real display, not in a
headless/remote environment.

The dialog opens in the folder of the path already typed in, or else the folder a file
was last picked from (remembered across sessions in cache/settings.json), instead of
wherever the app happened to be started.

The dialog is run in a separate process (awaited from a worker thread) rather than
in-process: a blocking dialog on the server's event-loop thread freezes the whole page
until it closes, and on macOS an in-process Tk window often opens hidden behind the
browser, so the page appears frozen for good.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
import traceback

import panel as pn

from . import cache

# Fallback for non-macOS: plain Tk dialog, in its own process so it owns its main thread.
_TK_SCRIPT = r"""
import sys
import tkinter as tk
from tkinter import filedialog

pattern, label, initial_dir = sys.argv[1], sys.argv[2], sys.argv[3]
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
root.update()
path = filedialog.askopenfilename(
    title="Select a file",
    initialdir=initial_dir or None,
    filetypes=[(f"{label} files", pattern), ("All files", "*.*")],
)
root.destroy()
sys.stdout.write(path or "")
"""

# `activate` brings osascript itself to the front so the dialog isn't hidden behind the browser.
# The start folder comes in as an argument (not pasted into the script), so no quoting issues.
_OSASCRIPT = 'activate\nPOSIX path of (choose file with prompt "Select a file")'
_OSASCRIPT_IN_DIR = """on run argv
activate
POSIX path of (choose file with prompt "Select a file" default location (POSIX file (item 1 of argv)))
end run"""


def _start_directory(current_value: str) -> str | None:
    """The folder of the path already typed in (if it exists), else the last-used folder."""
    if current_value:
        folder = os.path.dirname(os.path.abspath(os.path.expanduser(current_value)))
        if os.path.isdir(folder):
            return folder
    return cache.get_last_directory()


def _pick_file_blocking(file_pattern: str, start_dir: str | None) -> str | None:
    if sys.platform == "darwin":
        cmd = ["osascript", "-e", _OSASCRIPT_IN_DIR, start_dir] if start_dir else ["osascript", "-e", _OSASCRIPT]
    else:
        label = file_pattern.lstrip("*.").upper() or "Data"
        cmd = [sys.executable, "-c", _TK_SCRIPT, file_pattern, label, start_dir or ""]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # osascript exits non-zero with error -128 when the user cancels - not a failure.
        if "-128" not in result.stderr:
            print(f"File dialog failed: {result.stderr.strip()}", file=sys.stderr)
        return None
    return result.stdout.strip() or None


def remember_folder_of(path: str):
    """Makes the next file dialog (this session or a later one) start in `path`'s folder.
    Also called after a typed-in path loads successfully."""
    try:
        cache.set_last_directory(os.path.dirname(os.path.abspath(os.path.expanduser(path))))
    except OSError:  # remembering the folder is a convenience - never fail because of it
        traceback.print_exc(file=sys.stderr)


def build_path_picker(
    label: str, placeholder: str = "/path/to/data.h5ad", file_pattern: str = "*.h5ad", busy=None,
) -> tuple[pn.widgets.TextInput, pn.widgets.Button, pn.viewable.Viewable]:
    """Returns (text input, browse button, both side by side). Callers short on width
    can ignore the combined row and lay out the input and button themselves. With a
    BusyOverlay `busy`, the page is grayed out while the dialog is open."""
    path_input = pn.widgets.TextInput(name=label, placeholder=placeholder)
    # align="end" puts the button at the bottom of the row, level with the text box
    # rather than with the text box's title above it.
    browse_btn = pn.widgets.Button(name="Browse...", width=110, align="end")

    async def on_browse(event):
        browse_btn.disabled = True  # avoid stacking several dialogs
        try:
            start_dir = _start_directory(path_input.value.strip())
            async with (busy.show(
                "Waiting for you to choose a file...",
                "The file dialog is a separate window - if you don't see it, it may be behind the browser.",
            ) if busy is not None else contextlib.nullcontext()):
                path = await asyncio.to_thread(_pick_file_blocking, file_pattern, start_dir)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return
        finally:
            browse_btn.disabled = False
        if path:
            path_input.value = path
            remember_folder_of(path)

    browse_btn.on_click(on_browse)

    return path_input, browse_btn, pn.Row(path_input, browse_btn)
