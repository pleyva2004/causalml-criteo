"""Download, validate, and load the Criteo Uplift v2.1 dataset.

Pipeline: Kaggle zip -> CSV -> typed parquet with a fixed, stratified train/val/test ``split`` column,
plus a development-mode parquet that is a stratified subsample of every split.

Column meanings (Diemert et al., 2021, "A Large Scale Benchmark for Individual Treatment Effect
Prediction and Uplift Modeling"):

* ``treatment`` -- randomized *assignment*: 1 = user eligible to be shown the advertiser's ads,
  0 = user held out of advertising (control). This is the randomized variable.
* ``exposure`` -- whether the user was *actually shown* at least one ad during the test. It is a
  post-randomization outcome of treatment (auction wins, user activity), so ``treatment == 0``
  implies ``exposure == 0``, but treated users are not all exposed.
* ``visit`` -- the user visited the advertiser's website during the 2-week test window.
* ``conversion`` -- the user converted (purchased) during the window; ``visit == 0`` implies
  ``conversion == 0``.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

from src.utils import get_logger, load_config, repo_path

log = get_logger(__name__)

FEATURES: list[str] = [f"f{i}" for i in range(12)]
LABELS: list[str] = ["treatment", "conversion", "visit", "exposure"]
SPLIT_CODES: dict[str, int] = {"train": 0, "val": 1, "test": 2}

Mode = Literal["dev", "full"]
Split = Literal["train", "val", "test"]


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the processed parquet is missing; tells the user how to build it."""


def download_criteo(force: bool = False) -> Path:
    """Download the Kaggle zip (~340 MB) into ``data/raw`` and extract the CSV (~3 GB).

    Uses Kaggle's public dataset download endpoint. If that fails (network policy, Kaggle changes),
    download manually: ``kaggle datasets download -d arashnic/uplift-modeling -p data/raw`` or from
    the Kaggle web page, then rerun ``make data``.
    """
    cfg = load_config()["data"]
    raw_dir = repo_path(load_config()["paths"]["raw_dir"])
    csv_path = raw_dir / cfg["csv_name"]
    zip_path = raw_dir / cfg["zip_name"]
    if csv_path.exists() and not force:
        log.info("CSV already present: %s", csv_path)
        return csv_path

    if not zip_path.exists() or force:
        log.info("Downloading %s -> %s", cfg["kaggle_url"], zip_path)
        tmp = zip_path.with_suffix(".part")
        req = urllib.request.Request(cfg["kaggle_url"], headers={"User-Agent": "causalml-criteo"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 22)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Download failed ({exc}). Download '{cfg['kaggle_dataset']}' manually from Kaggle into "
                f"{raw_dir} and rerun."
            ) from exc
        tmp.rename(zip_path)

    log.info("Extracting %s", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".csv")]
        if cfg["csv_name"] not in members:
            raise RuntimeError(f"{cfg['csv_name']} not found in zip; members: {members}")
        zf.extract(cfg["csv_name"], raw_dir)
    return csv_path


def read_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Read the raw CSV with explicit dtypes (float64 features, int8 labels)."""
    schema = {f: pa.float64() for f in FEATURES} | {c: pa.int8() for c in LABELS}
    table = pv.read_csv(csv_path, convert_options=pv.ConvertOptions(column_types=schema))
    return table.to_pandas()


def validate_raw(df: pd.DataFrame, expected_rows: int | None) -> dict[str, int]:
    """Check schema and the structural constraints of the data-generating process.

    Returns counts of any violations; raises only on schema problems that would make every
    downstream result meaningless (wrong columns, non-binary labels, wrong row count).
    """
    missing = set(FEATURES + LABELS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} rows, got {len(df):,}")
    for c in LABELS:
        bad = ~df[c].isin([0, 1])
        if bad.any():
            raise ValueError(f"Column {c} has {int(bad.sum())} non-binary values")
    checks = {
        "rows": len(df),
        "missing_values": int(df.isna().sum().sum()),
        "control_exposed": int(((df.treatment == 0) & (df.exposure == 1)).sum()),
        "conversion_without_visit": int(((df.visit == 0) & (df.conversion == 1)).sum()),
    }
    for k in ("control_exposed", "conversion_without_visit"):
        if checks[k]:
            log.warning("Structural constraint violated: %s = %d", k, checks[k])
    return checks


def assign_splits(df: pd.DataFrame, fractions: dict[str, float], seed: int) -> np.ndarray:
    """Stratified train/val/test assignment on treatment x visit x conversion.

    Each stratum is shuffled with a seeded RNG and cut at the configured fractions, so every split
    has (up to rounding) the same treatment ratio and outcome rates. Returns int8 codes (see
    ``SPLIT_CODES``).
    """
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {fractions}")
    rng = np.random.default_rng(seed)
    strata = (df["treatment"].to_numpy() * 4 + df["visit"].to_numpy() * 2 + df["conversion"].to_numpy())
    codes = np.empty(len(df), dtype=np.int8)
    cut1 = fractions["train"]
    cut2 = fractions["train"] + fractions["val"]
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        idx = idx[rng.permutation(len(idx))]
        n = len(idx)
        a, b = int(round(cut1 * n)), int(round(cut2 * n))
        codes[idx[:a]] = SPLIT_CODES["train"]
        codes[idx[a:b]] = SPLIT_CODES["val"]
        codes[idx[b:]] = SPLIT_CODES["test"]
    return codes


def stratified_subsample(df: pd.DataFrame, fraction: float, seed: int,
                         strata_cols: Sequence[str] = ("split", "treatment", "visit", "conversion")) -> pd.DataFrame:
    """Sample ``fraction`` of rows within each stratum (reproducible)."""
    rng = np.random.default_rng(seed)
    keys = df[list(strata_cols)].astype(np.int64)
    stratum = np.zeros(len(df), dtype=np.int64)
    for c in strata_cols:
        stratum = stratum * 4 + keys[c].to_numpy()
    keep = np.zeros(len(df), dtype=bool)
    for s in np.unique(stratum):
        idx = np.flatnonzero(stratum == s)
        k = int(round(fraction * len(idx)))
        keep[rng.choice(idx, size=k, replace=False)] = True
    return df.loc[keep]


def build_processed(force: bool = False) -> dict[str, int]:
    """Download (if needed), validate, split, and write the full and dev parquet files."""
    cfg = load_config()
    processed = repo_path(cfg["paths"]["processed_dir"])
    full_path = processed / cfg["data"]["full_parquet"]
    dev_path = processed / cfg["data"]["dev_parquet"]
    if full_path.exists() and dev_path.exists() and not force:
        log.info("Processed data already present in %s", processed)
        return {}

    csv_path = download_criteo()
    df = read_raw_csv(csv_path)
    checks = validate_raw(df, cfg["data"]["expected_rows"])
    log.info("Validated raw data: %s", checks)

    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    df["split"] = assign_splits(df, cfg["data"]["split_fractions"], cfg["seed"])
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), full_path, compression="zstd")

    dev = stratified_subsample(df, cfg["data"]["dev_fraction"], cfg["seed"])
    pq.write_table(pa.Table.from_pandas(dev, preserve_index=False), dev_path, compression="zstd")
    log.info("Wrote %s (%d rows) and %s (%d rows)", full_path.name, len(df), dev_path.name, len(dev))
    return checks


def load_criteo(mode: Mode = "dev", split: Split | Sequence[Split] | None = None,
                columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load the processed dataset.

    Args:
        mode: ``"dev"`` (stratified ~5% subsample, for exploration and tests) or ``"full"``.
        split: restrict to one or more of ``"train"``, ``"val"``, ``"test"``.
        columns: subset of columns to read (``split`` is always available for filtering).
    """
    cfg = load_config()
    name = cfg["data"]["dev_parquet"] if mode == "dev" else cfg["data"]["full_parquet"]
    path = repo_path(cfg["paths"]["processed_dir"]) / name
    if not path.exists():
        raise DatasetNotFoundError(f"{path} not found. Run `make data` (or `uv run causalml data`) first.")
    filters = None
    if split is not None:
        splits = [split] if isinstance(split, str) else list(split)
        filters = [("split", "in", [SPLIT_CODES[s] for s in splits])]
    cols = None if columns is None else list(dict.fromkeys([*columns]))
    return pq.read_table(path, columns=cols, filters=filters).to_pandas()


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split a frame into features X, treatment T, conversion, visit."""
    return df[FEATURES], df["treatment"], df["conversion"], df["visit"]
