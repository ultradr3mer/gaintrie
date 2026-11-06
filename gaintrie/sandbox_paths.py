"""Locates the local test data (model .bin / .safetensors files): $GAINTRIE_DATA if set,
else the first existing root below — add a new machine by appending its root to `_ROOTS`.
"""
from pathlib import Path
import os

_ROOTS = [
    Path(r"D:\modelData"),  # Windows dev box
    Path("/home/deck/PycharmProjects/python-sandbox/modelCompression"),  # Steam Deck
    Path.home() / "source" / "TestData",  # Linux dev box
]


def sandbox_path(*parts: str) -> Path:
    """First existing root joined with `parts` (e.g. `sandbox_path("bins", "x.bin")`).
    Falls back to the first root if none exist, so callers that just want to
    print/skip on a missing file still get a usable Path."""
    roots = ([Path(os.environ["GAINTRIE_DATA"])] if os.environ.get("GAINTRIE_DATA") else []) + _ROOTS
    for root in roots:
        if root.exists():
            return root.joinpath(*parts)
    return roots[0].joinpath(*parts)
