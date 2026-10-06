"""conftest.py — make the project importable as `ele`.

The repo root IS the `ele` package (it contains __init__.py and the core/
sub-package). Tests import via `ele.core.*`, so Python needs the *parent*
of this directory on sys.path.
"""
import sys
from pathlib import Path

parent = str(Path(__file__).resolve().parent.parent)
if parent not in sys.path:
    sys.path.insert(0, parent)
