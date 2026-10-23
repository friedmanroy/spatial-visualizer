"""Persists per-dataset UI choices (custom category colors, which fields are offered,
which obs column splits samples/FOVs) to a small JSON file under cache/, keyed by the
source .h5ad path, so they carry over between sessions on the same file.

The cache/ folder lives in the app's own folder: the double-click launchers point it at
the folder they sit in (even when running the latest version from GitHub, whose code lives
elsewhere); a source checkout uses the repo root; otherwise the current directory.
"""
from __future__ import annotations

import hashlib
import json
import os

_SOURCE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("SPATIAL_VISUALIZER_CACHE_DIR"):
    CACHE_DIR = os.environ["SPATIAL_VISUALIZER_CACHE_DIR"]
elif os.path.exists(os.path.join(_SOURCE_ROOT, "pyproject.toml")):
    CACHE_DIR = os.path.join(_SOURCE_ROOT, "cache")
else:
    CACHE_DIR = "cache"


def _cache_path(source_path: str) -> str:
    key = hashlib.md5(os.path.abspath(source_path).encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.json")


class ColorCache:
    """One instance per loaded dataset file. Read once at load time, written
    incrementally as the user changes colors or the set of offered fields."""

    def __init__(self, source_path: str):
        self.path = _cache_path(source_path)
        self.data: dict = {}
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.data = json.load(f)
            except (OSError, json.JSONDecodeError):
                self.data = {}

    def get_field_colors(self, field: str) -> dict[str, str]:
        return self.data.get("fields", {}).get(field, {})

    def set_color(self, field: str, category: str, color: str):
        self.data.setdefault("fields", {}).setdefault(field, {})[category] = color
        self._save()

    def get_enabled_fields(self) -> list[str] | None:
        return self.data.get("enabled_fields")

    def set_enabled_fields(self, fields: list[str]):
        self.data["enabled_fields"] = sorted(fields)
        self._save()

    def get_sample_column(self) -> str | None:
        """The obs column last used to split samples/FOVs, or None (no split / never set)."""
        return self.data.get("sample_column")

    def set_sample_column(self, column: str | None):
        self.data["sample_column"] = column
        self._save()

    def get_normalize(self) -> bool | None:
        """Whether the user last chose to normalize this file (None: never chose)."""
        return self.data.get("normalize")

    def set_normalize(self, normalize: bool):
        self.data["normalize"] = normalize
        self._save()

    def get_regions(self) -> list[dict]:
        """Regions drawn on the Spatial tab (see regions.Region.to_json), oldest first."""
        return self.data.get("regions", [])

    def set_regions(self, regions: list[dict]):
        self.data["regions"] = regions
        self._save()

    def _save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)


# ---- app-wide settings (not tied to one dataset) ------------------------------------

_SETTINGS_PATH = os.path.join(CACHE_DIR, "settings.json")


def _load_settings() -> dict:
    try:
        with open(_SETTINGS_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def get_last_directory() -> str | None:
    """Folder the file picker last chose a file in (any session), if it still exists."""
    path = _load_settings().get("last_directory")
    return path if path and os.path.isdir(path) else None


def set_last_directory(path: str):
    settings = _load_settings()
    settings["last_directory"] = path
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)
