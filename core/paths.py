"""Central path resolution for ELE data directories.

Answer keys, run logs, and per-scenario results are sensitive: they contain
or directly reveal ground truth. To keep the public repo (scenarios +
framework) free of ground truth, these three directories live in a SEPARATE
PRIVATE repo (ELEAnswers). This module is the single source of truth for
where each directory lives so every entry point — the evaluation engine,
run.py, and the analysis/validation scripts — agrees.

Resolution order for each of answers/ logs/ results/ (first match wins):

  1. Dedicated env var:   ELE_ANSWERS_DIR / ELE_LOGS_DIR / ELE_RESULTS_DIR
  2. ELE_PRIVATE_DIR env var:      <ELE_PRIVATE_DIR>/<name>
  3. config/eval_config.json key:  <ele_private_dir>/<name>
  4. Local default:                <PROJECT_ROOT>/<name>   (backward compatible)

Relative paths from ELE_PRIVATE_DIR or the config key are resolved against
PROJECT_ROOT, so a sibling checkout can be referenced as "../ELEAnswers".

Nothing here creates directories; callers mkdir where they write.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

# PROJECT_ROOT is the repo root (the parent of this core/ package directory).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

_CONFIG_FILE = PROJECT_ROOT / "config" / "eval_config.json"

# Env var names.
_ENV_PRIVATE_DIR = "ELE_PRIVATE_DIR"
_ENV_BY_NAME = {
    "answers": "ELE_ANSWERS_DIR",
    "logs": "ELE_LOGS_DIR",
    "results": "ELE_RESULTS_DIR",
}


def project_root() -> Path:
    """Return the public repo root."""
    return PROJECT_ROOT


def _resolve(raw: str) -> Path:
    """Resolve a possibly-relative path against PROJECT_ROOT."""
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


@lru_cache(maxsize=1)
def _config_private_dir() -> Optional[str]:
    """Read the optional ``ele_private_dir`` key from eval_config.json.

    Cached because it is read on every path lookup. Returns None when the
    config file or the key is absent, or the file is unreadable.
    """
    try:
        with open(_CONFIG_FILE) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("ele_private_dir")
    return value or None


def _private_dir() -> Optional[Path]:
    """Return the configured private-repo root, or None if unset.

    ELE_PRIVATE_DIR (env) wins over the ele_private_dir config key.
    """
    env = os.environ.get(_ENV_PRIVATE_DIR)
    if env:
        return _resolve(env)
    cfg = _config_private_dir()
    if cfg:
        return _resolve(cfg)
    return None


def _data_dir(name: str) -> Path:
    """Resolve one of the data directories by name ('answers'|'logs'|'results')."""
    # 1. Dedicated per-directory env override.
    dedicated = os.environ.get(_ENV_BY_NAME[name])
    if dedicated:
        return _resolve(dedicated)
    # 2/3. Private-repo root (env or config).
    private = _private_dir()
    if private is not None:
        return private / name
    # 4. Local default — backward compatible.
    return PROJECT_ROOT / name


def answers_dir() -> Path:
    """Directory holding held-out answer-key JSON files."""
    return _data_dir("answers")


def logs_dir() -> Path:
    """Directory holding per-scenario run transcripts."""
    return _data_dir("logs")


def results_dir() -> Path:
    """Directory holding per-run scored result records."""
    return _data_dir("results")
