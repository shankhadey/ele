"""conftest.py — make the project importable as `ele` and provide
test defaults.

The repo root IS the `ele` package (it contains __init__.py and the core/
sub-package). Tests import via `ele.core.*`, so Python needs the *parent*
of this directory on sys.path.

Since the LLM judge is now mandatory at ``App`` startup, tests set a
placeholder ``OPENAI_API_KEY`` and ``EVAL_JUDGE_API_KEY`` so ``AppConfig``
validation passes without a real key. Tests that exercise the judge itself
mock ``ele.core.scoring.openai`` directly and never make real API calls.
"""
import os
import sys
from pathlib import Path

parent = str(Path(__file__).resolve().parent.parent)
if parent not in sys.path:
    sys.path.insert(0, parent)

# Set placeholder credentials so App(AppConfig()) can pass the mandatory
# judge check during tests without any real network access. Real calls
# through openai are always mocked in the individual tests.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("EVAL_JUDGE_API_KEY", "test-key-not-real")
