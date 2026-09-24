"""A text path input paired with a native OS file-picker dialog, so a file can either
be typed by path or found by browsing - reused by the landing page and the scRNA/snRNA
companion-file prompt. Uses tkinter's file dialog, which pops up as its own window;
this only works because the app runs locally (`python st_visualizer.py`) with a real
display, not in a headless/remote environment.
"""
from __future__ import annotations

import sys
import traceback

import panel as pn


def _pick_file_native(file_pattern: str) -> str | None:
    import tkinter as tk
    from tkinter import filedialog

    label = file_pattern.lstrip("*.").upper() or "Data"
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(
            title="Select a file",
            filetypes=[(f"{label} files", file_pattern), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return path or None


def build_path_picker(
    label: str, placeholder: str = "/path/to/data.h5ad", file_pattern: str = "*.h5ad"
) -> tuple[pn.widgets.TextInput, pn.viewable.Viewable]:
    path_input = pn.widgets.TextInput(name=label, placeholder=placeholder)
    browse_btn = pn.widgets.Button(name="Browse...", width=110)

    def on_browse(event):
        try:
            path = _pick_file_native(file_pattern)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return
        if path:
            path_input.value = path

    browse_btn.on_click(on_browse)

    return path_input, pn.Row(path_input, browse_btn)
