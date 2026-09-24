"""Persists per-dataset UI choices (custom category colors, which fields are offered)
to a small JSON file under cache/, keyed by the source .h5ad path, so they carry over
between sessions on the same file.
"""
from __future__ import annotations

import hashlib
import json
import os

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

    def _save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
