"""Shared helpers: configuration, logging, seeding, and results I/O.

Every stage writes its outputs through these helpers so that results/ has a single, predictable layout:
``results/metrics/<name>.json``, ``results/tables/<name>.csv`` and ``results/figures/<name>.png``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


_RUN_MODE = {"mode": "full"}


def set_run_mode(mode: str) -> None:
    """Route results I/O: ``"full"`` writes to results/, ``"dev"`` to results/dev/ (not committed)."""
    if mode not in ("dev", "full"):
        raise ValueError(f"mode must be 'dev' or 'full', got {mode!r}")
    _RUN_MODE["mode"] = mode


def get_run_mode() -> str:
    return _RUN_MODE["mode"]


def _results_dir(key: str) -> Path:
    rel = load_config()["paths"][key]
    if _RUN_MODE["mode"] == "dev":
        rel = rel.replace("results/", "results/dev/", 1)
    return repo_path(rel)


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Load the YAML config (cached). ``path`` defaults to ``configs/config.yaml``."""
    with open(path or CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def repo_path(relative: str) -> Path:
    """Resolve a config-relative path against the repository root and make sure its directory exists."""
    p = REPO_ROOT / relative
    p.mkdir(parents=True, exist_ok=True) if p.suffix == "" else p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str) -> logging.Logger:
    """Module logger with a single consistent format."""
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.environ.get("LOGLEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    return logger


def set_seed(seed: int | None = None) -> int:
    """Seed Python and NumPy global RNGs; returns the seed used. Prefer passing explicit RNGs."""
    seed = load_config()["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    return seed


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def save_json(obj: dict[str, Any], name: str) -> Path:
    """Write ``results/metrics/<name>.json``."""
    path = _results_dir("metrics_dir") / f"{name}.json"
    path.write_text(json.dumps(_to_jsonable(obj), indent=2))
    return path


def load_json(name: str) -> dict[str, Any]:
    """Read ``results/metrics/<name>.json`` (or the dev copy in dev mode)."""
    path = _results_dir("metrics_dir") / f"{name}.json"
    return json.loads(path.read_text())


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    """Write ``results/tables/<name>.csv``."""
    path = _results_dir("tables_dir") / f"{name}.csv"
    df.to_csv(path, index=index)
    return path


def figure_path(name: str) -> Path:
    """Path for ``results/figures/<name>.png``."""
    return _results_dir("figures_dir") / f"{name}.png"


@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Iterator[dict[str, float]]:
    """Context manager that records wall-clock seconds into the yielded dict under ``"seconds"``."""
    out: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = time.perf_counter() - start
        if logger is not None:
            logger.info("%s took %.1fs", label, out["seconds"])
