#!/usr/bin/env python
"""Generic spatial transcriptomics visualizer.

Usage:
    python st_visualizer.py

(Equivalent to the `spatial-visualizer` command installed by `pip install .`.)
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from viewer.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
