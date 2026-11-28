"""Shared test setup: run from the repo root with `python -m pytest`.

Tests that read local ADNI-derived tables call data_path(), which skips the test when the data is
not on this machine (e.g. a fresh clone). No test prints subject-level values.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def data_path(rel: str) -> Path:
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"local data not available: {rel}")
    return path
