"""Shared pytest configuration for the buildgpt2 test suite.

``src`` is not a Python package (it has no ``__init__.py``), so it is not
importable out of the box. Adding it to ``sys.path`` here makes the modules
under ``src`` importable from every test module regardless of how pytest is
invoked (from the repo root, from ``tests/``, or via an IDE test runner).
"""

import sys
from pathlib import Path

# .../buildgpt2/tests/conftest.py -> .../buildgpt2
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
