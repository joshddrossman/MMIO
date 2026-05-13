"""Compatibility shim for ``import generation``.

The full implementation lives in ``old_scripts/generation.py``. The repo
root kept this import path so ``queries``, ``agent_tools``, and ``main``
continue to work after the file was moved under ``old_scripts/``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "old_scripts" / "generation.py"
_spec = importlib.util.spec_from_file_location(__name__, _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load generation from {_PATH}")
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

for _k, _v in vars(_impl).items():
    if _k.startswith("__") and _k.endswith("__"):
        continue
    globals()[_k] = _v
