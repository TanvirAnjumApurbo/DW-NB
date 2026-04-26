"""Utility helpers for experiments and reproducibility."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd


def seed_everything(seed: int = 42) -> None:
    """Seed Python and NumPy RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if absent and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def configure_logging(level: int = logging.INFO) -> None:
    """Set up consistent project logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def atomic_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write CSV atomically by temporary file then replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
